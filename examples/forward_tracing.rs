use std::collections::HashMap;

use std::sync::Arc;
use std::thread;
use std::{f32::consts::SQRT_2, fs::File, io::Read};

#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
// use packed_simd::f32x4;
use crossbeam::channel::{unbounded, Receiver, Sender};
use rand::prelude::*;
use rayon::prelude::*;

use ::math::{random_cosine_direction, SingleWavelength, XYZColor};
use lens_sampler::RadialSampler;
use optics::*;
use subcrate::parsing::*;
use subcrate::{film::Film, parsing::*};

use ::math::spectral::BOUNDED_VISIBLE_RANGE;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(short, default_value = "800")]
    pub width: usize,

    #[structopt(short, default_value = "800")]
    pub height: usize,

    #[structopt(long, default_value = "22")]
    pub threads: usize,

    #[structopt(long)]
    pub lens: String,
}

pub enum Command {
    ChangeFloat(f32),
    ChangeInt(i32),
    Advance,
}

pub fn bind(
    senders: &mut HashMap<String, Arc<Sender<Command>>>,
    receivers: &mut HashMap<String, Receiver<Command>>,
    name: String,
) {
    let (sender, receiver) = unbounded();
    senders.insert(name.clone(), Arc::new(sender));
    receivers.insert(name.clone(), receiver);
}

pub fn terminal_thread(channels: HashMap<String, Arc<Sender<Command>>>) -> impl Fn() -> () {
    move || {
        use linefeed::{Interface, ReadResult};

        let reader = Interface::new("tracer-prompt").expect("failed to create terminal interface");

        reader
            .set_prompt("tracer> ")
            .expect("failed to set terminal prompt");

        // let mut last_input = String::from("");

        while let Ok(result) = reader.read_line() {
            let input = match result {
                ReadResult::Input(input) => input,
                ReadResult::Eof => {
                    return;
                }
                ReadResult::Signal(signal) => {
                    return;
                }
            };
            reader.add_history_unique(input.clone());
            let mut chomper = input.split(" ");
            if let Some(word) = chomper.next() {
                if let Some(channel) = channels.get(word) {
                    if let Some(command_type) = chomper.next() {
                        match command_type {
                            "float" => {
                                if let Some(Ok(value)) = chomper.next().map(|e| e.parse::<f32>()) {
                                    if channel.send(Command::ChangeFloat(value)).is_err() {
                                        return;
                                    }
                                }
                            }
                            "int" => {
                                if let Some(Ok(value)) = chomper.next().map(|e| e.parse::<i32>()) {
                                    if channel.send(Command::ChangeInt(value)).is_err() {
                                        return;
                                    }
                                }
                            }
                            _ => {}
                        }
                        println!();
                    } else {
                        if channel.send(Command::Advance).is_err() {
                            return;
                        }
                        if word == "exit" {
                            return;
                        }
                    }
                } else if word == "help" {
                    for key in channels.keys() {
                        println!("key {}", key);
                    }
                }
            }
        }

        println!("Goodbye.");
    }
}
fn main() {
    use subcrate::tonemap::{sRGB, Tonemapper};
    let opt = Opt::from_args();
    println!("{:?}", opt);
    let window_width = opt.width;
    let window_height = opt.height;

    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "forward tracing",
        window_width,
        window_height,
        WindowOptions {
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut film = Film::new(window_width, window_height, XYZColor::BLACK);
    let mut window_pixels = Film::new(window_width, window_height, 0u32);

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    let width = film.width;
    let height = film.height;

    let frame_dt = 6944.0 / 1000000.0;

    let mut camera_file = File::open(format!("data/cameras/{}.txt", opt.lens)).unwrap();
    let mut camera_spec = String::new();
    camera_file.read_to_string(&mut camera_spec).unwrap();

    let (lenses, _last_ior, _last_vno) = parse_lenses_from(&camera_spec);
    let lens_assembly = LensAssembly::new(&lenses);

    let scene = get_scene("textures.toml").unwrap();

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
    let mut textures: Vec<TexStack> = Vec::new();
    for tex in scene.textures {
        textures.push(parse_texture_stack(tex.clone(), wavelength_bounds));
    }

    let original_aperture_radius = lens_assembly.aperture_radius();
    let mut heat_bias = 0.01;
    let mut heat_cap = 10.0;

    let mut aperture_radius = original_aperture_radius / 10.0; // start with small aperture
    let mut lens_zoom = 0.0;
    let mut sensor_pos = -lens_assembly.total_thickness_at(lens_zoom);
    let mut wall_position = 5000.0;
    let mut sensor_size = 35.0;
    let mut texture_scale = 1.0;

    let mut samples_per_iteration = 1usize;
    let mut total_samples = 0;
    let mut focal_distance_suggestion = None;
    let mut focal_distance_vec: Vec<f32> = Vec::new();
    let mut variance: f32 = 0.0;
    let mut stddev: f32 = 0.0;

    let direction_cache_radius_bins = 512;
    let direction_cache_wavelength_bins = 512;

    let mut direction_cache = RadialSampler::new(
        SQRT_2 * sensor_size / 2.0, // diagonal.
        direction_cache_radius_bins,
        direction_cache_wavelength_bins,
        wavelength_bounds,
        sensor_pos,
        &lens_assembly,
        lens_zoom,
        |aperture_radius, ray| bladed_aperture(aperture_radius, 6, ray),
        heat_bias,
        sensor_size,
    );

    let mut wavelength_sweep: f32 = 0.0;
    let mut wavelength_sweep_speed = 0.001;
    let mut efficiency = 0.0;
    let efficiency_heat = 0.99;
    let mut scene_mode = SceneMode::PinLight;
    let mut paused = false;

    let mut sender_map = HashMap::new();
    let mut receiver_map = HashMap::new();

    for bind_target in vec![
        "aperture",
        "wavelength_sweep",
        "wall_position",
        "film_position",
        "heat_bias",
        "heat_cap",
        "texture_scale",
        "lens_zoom",
        "sensor_size",
        "samples",
        "paused",
        "clear",
        "clear_film",
        "scene_mode",
        "printout",
        "exit",
    ] {
        bind(&mut sender_map, &mut receiver_map, bind_target.to_owned());
    }

    let handle = thread::spawn(terminal_thread(sender_map));

    'outer: while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut clear_film = false;
        let mut clear_direction_cache = false;

        for (value_name, receiver) in receiver_map.iter() {
            let (change_float, change_int, got_command) = {
                let command = receiver.try_recv();
                if command.is_err() {
                    (0.0, 0, false)
                } else {
                    match command.unwrap() {
                        Command::ChangeFloat(float) => (float, 0, true),
                        Command::ChangeInt(int) => (0.0, int, true),
                        Command::Advance => (0.0, 0, true),
                    }
                }
            };
            if !got_command {
                continue;
            }
            match value_name.as_str() {
                "aperture" => {
                    aperture_radius *= change_float.exp2();
                    heat_bias *= change_float.exp2();
                    clear_direction_cache = true;
                    println!(
                            "changed aperture radius to {}, f stop = {:?} . note: change is interpreted as 2^v, where v is the provided value",
                            aperture_radius,
                            original_aperture_radius / aperture_radius
                        );
                }
                "film_position" => {
                    clear_film = true;
                    clear_direction_cache = true;
                    total_samples = 0;
                    println!(
                        "{:?}, {:?}, offset = {}",
                        sensor_pos,
                        lens_assembly.total_thickness_at(lens_zoom),
                        sensor_pos + lens_assembly.total_thickness_at(lens_zoom)
                    );
                    println!("{:?}, {}, {}", focal_distance_suggestion, variance, stddev);
                    sensor_pos += change_float;
                    println!(
                        "{:?}, {:?}",
                        sensor_pos,
                        lens_assembly.total_thickness_at(lens_zoom)
                    );
                }
                "wall_position" => {
                    clear_film = true;
                    total_samples = 0;
                    wall_position += 10.0 * change_float;
                    println!("{:?}", wall_position);
                    println!("{:?}, {}, {}", focal_distance_suggestion, variance, stddev);
                }
                "heat_bias" => {
                    heat_bias *= 1.1f32.powf(change_float);
                    println!("{:?}", heat_bias);
                }
                "wavelength_sweep" => {
                    wavelength_sweep_speed *= 1.1f32.powf(change_float);
                    println!("{:?}", wavelength_sweep_speed);
                }
                "heat_cap" => {
                    heat_cap *= 1.1f32.powf(change_float);
                    println!("{:?}", heat_cap);
                }
                "texture_scale" => {
                    texture_scale *= 1.1f32.powf(change_float);
                    println!("{:?}", texture_scale);
                }
                "lens_zoom" => {
                    clear_film = true;
                    clear_direction_cache = true;
                    total_samples = 0;
                    lens_zoom += 0.01 * change_float;
                    println!("{:?}", lens_zoom);
                }
                "samples" => {
                    match change_int {
                        v if v > 0 => samples_per_iteration += v as usize,
                        v if v < 0 => {
                            if (-v) as usize >= samples_per_iteration {
                                samples_per_iteration = 1;
                            } else {
                                samples_per_iteration -= (-v) as usize;
                            }
                        }
                        _ => {}
                    }

                    println!("{:?}", samples_per_iteration);
                }
                "sensor_size" => {
                    clear_film = true;
                    clear_direction_cache = true;
                    total_samples = 0;
                    sensor_size *= 1.1f32.powf(change_float);
                    println!("{:?}", sensor_size);
                }
                "paused" => {
                    // got command, and reached here, thus
                    println!("switching pause state");
                    paused = !paused;
                }
                "clear" => {
                    clear_film = true;
                    clear_direction_cache = true;
                    wavelength_sweep = 0.0;
                    total_samples = 0;
                }
                "clear_film" => {
                    clear_film = true;
                }
                "scene_mode" => {
                    // do mode transition
                    scene_mode = scene_mode.cycle();
                    // if matches!(scene_mode, SceneMode::SpotLight) {
                    //     println!("skipping unimplemented scene mode {:?}", scene_mode);
                    //     scene_mode = scene_mode.cycle();
                    // }
                    println!("new mode is {:?}", scene_mode);
                }
                "printout" => {
                    println!("total samples: {}", total_samples);
                    println!("wavelength_sweep: {}", wavelength_sweep);
                    println!("sampling efficiency is {}", efficiency);
                }
                "exit" => {
                    println!("got exit command");
                    break 'outer;
                }
                _ => {}
            }
        }

        if paused {
            let pause_duration = std::time::Duration::from_nanos((frame_dt * 1_000_000.0) as u64);
            std::thread::sleep(pause_duration);

            window
                .update_with_buffer(&window_pixels.buffer, window_width, window_height)
                .unwrap();
            continue;
        }
        if clear_film {
            film.buffer
                .par_iter_mut()
                .for_each(|e| *e = XYZColor::BLACK)
        }
        if clear_direction_cache {
            direction_cache = RadialSampler::new(
                SQRT_2 * sensor_size / 2.0, // diagonal.
                direction_cache_radius_bins,
                direction_cache_wavelength_bins,
                wavelength_bounds,
                sensor_pos,
                &lens_assembly,
                lens_zoom,
                |aperture_radius, ray| bladed_aperture(aperture_radius, 6, ray),
                heat_bias,
                sensor_size,
            );
        }

        let srgb_tonemapper = sRGB::new(&film, 1.0);

        // autofocus:
        {
            let n = 25;
            let origin = Point3::new(0.0, 0.0, sensor_pos);
            let direction = Point3::new(
                0.0,
                lens_assembly.lenses.last().unwrap().housing_radius,
                0.0,
            ) - origin;
            let maximum_angle = -(direction.y() / direction.z()).atan();

            focal_distance_vec.clear();
            for i in 0..n {
                // choose angle to shoot ray from (0.0, 0.0, wall_position)
                let angle = ((i as f32 + 0.5) / n as f32) * maximum_angle;
                let ray = Ray::new(origin, Vec3::new(0.0, angle.sin(), angle.cos()));
                // println!("{:?}", ray);
                for w in 0..10 {
                    let lambda =
                        wavelength_bounds.lower + (w as f32 / 10.0) * wavelength_bounds.span();
                    let result =
                        lens_assembly.trace_forward(lens_zoom, &Input { ray, lambda }, 1.0, |e| {
                            (bladed_aperture(aperture_radius, 6, e), false)
                        });
                    if let Some(Output { ray: pupil_ray, .. }) = result {
                        let dt = (-pupil_ray.origin.y()) / pupil_ray.direction.y();
                        let point = pupil_ray.point_at_parameter(dt);
                        // println!("{:?}", point);

                        if point.z().is_finite() {
                            focal_distance_vec.push(point.z());
                        }
                    }
                }
            }
            if focal_distance_vec.len() > 0 {
                let avg: f32 =
                    focal_distance_vec.iter().sum::<f32>() / focal_distance_vec.len() as f32;
                focal_distance_suggestion = Some(avg);
                variance = focal_distance_vec
                    .iter()
                    .map(|e| (avg - *e).powf(2.0))
                    .sum::<f32>()
                    / focal_distance_vec.len() as f32;
                stddev = variance.sqrt();
            }
        }

        total_samples += samples_per_iteration;

        let (a, b) = film
            .buffer
            .par_iter_mut()
            .enumerate()
            .map(|(i, pixel)| {
                let mut sampler = RandomSampler::new();
                let px = i % width;
                let py = i / width;

                let (mut successes, mut attempts) = (0, 0);
                let lambda = wavelength_bounds.sample(random::<f32>());

                let central_point = Point3::new(
                    ((px as f32 + 0.5) / width as f32 - 0.5) * sensor_size,
                    ((py as f32 + 0.5) / height as f32 - 0.5) * sensor_size,
                    sensor_pos,
                );
                for _ in 0..samples_per_iteration {
                    let v;
                    let s0 = sampler.draw_2d();
                    let [mut x, mut y, z, _]: [f32; 4] = central_point.0.into();
                    x += (s0.x - 0.5) / width as f32 * sensor_size;
                    y += (s0.y - 0.5) / height as f32 * sensor_size;

                    let point = Point3::new(x, y, z);
                    if true {
                        // using lens symmetry sampler

                        v = direction_cache.sample(
                            lambda,
                            point,
                            sampler.draw_2d(),
                            sampler.draw_1d(),
                        );
                    } else {
                        // random cosine sampling
                        v = random_cosine_direction(sampler.draw_2d());
                    }
                    let ray = Ray::new(point, v);

                    attempts += 1;
                    // do actual tracing through lens for film sample
                    let result =
                        lens_assembly.trace_forward(lens_zoom, &Input { ray, lambda }, 1.0, |e| {
                            (bladed_aperture(aperture_radius, 6, e), false)
                        });
                    if let Some(Output {
                        ray: pupil_ray,
                        tau,
                    }) = result
                    {
                        successes += 1;

                        match scene_mode {
                            // // texture based
                            // ignore because texture scale is used across multiple of these entries
                            SceneMode::TexturedWall => {
                                let t = (wall_position - pupil_ray.origin.z())
                                    / pupil_ray.direction.z();
                                let point_at_wall = pupil_ray.point_at_parameter(t);
                                let uv = (
                                    (point_at_wall.x().abs() / texture_scale) % 1.0,
                                    (point_at_wall.y().abs() / texture_scale) % 1.0,
                                );
                                let m = textures[0].eval_at(lambda, uv);
                                let energy = tau * m * 3.0;
                                *pixel +=
                                    XYZColor::from(SingleWavelength::new(lambda, energy.into()));
                            }

                            SceneMode::PinLight => {
                                // diffuse pin lights
                                let t = (wall_position - pupil_ray.origin.z())
                                    / pupil_ray.direction.z();
                                let point_at_wall = pupil_ray.point_at_parameter(t);
                                let uv = (
                                    (point_at_wall.x().abs() / texture_scale) % 1.0,
                                    (point_at_wall.y().abs() / texture_scale) % 1.0,
                                );
                                let m = if (uv.0 - 0.5).powi(2) + (uv.1 - 0.5).powi(2) < 0.001 {
                                    // if pupil_ray.direction.z() > 0.999 {
                                    //     1.0
                                    // } else {
                                    //     0.0
                                    // }
                                    1.0
                                } else {
                                    0.0
                                };
                                let energy = tau * m * 3.0;
                                *pixel +=
                                    XYZColor::from(SingleWavelength::new(lambda, energy.into()));
                            }

                            SceneMode::SpotLight { pos, size, span } => {
                                let t = (wall_position - pupil_ray.origin.z())
                                    / pupil_ray.direction.z();
                                let point_at_wall = pupil_ray.point_at_parameter(t);
                                let uv = (
                                    (point_at_wall.x() / texture_scale),
                                    (point_at_wall.y() / texture_scale),
                                );
                                let m =
                                    if (uv.0 - pos.x()).powi(2) + (uv.1 - pos.y()).powi(2) < size {
                                        // if position matches
                                        if pupil_ray.direction.z() > span {
                                            // if direction matches
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    } else {
                                        0.0
                                    };
                                let energy = tau * m * 3.0;
                                *pixel +=
                                    XYZColor::from(SingleWavelength::new(lambda, energy.into()));
                            }
                        };
                    }
                }
                (successes, attempts)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        efficiency =
            (efficiency_heat) * efficiency + (1.0 - efficiency_heat) * (a as f32 / b as f32);

        window_pixels
            .buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel_idx, v)| {
                let y: usize = pixel_idx / width;
                let x: usize = pixel_idx - width * y;
                let (mapped, _linear) = srgb_tonemapper.map(&film, (x, y));
                let [r, g, b, _]: [f32; 4] = mapped.into();
                *v = rgb_to_u32((255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8);
            });
        window
            .update_with_buffer(&window_pixels.buffer, window_width, window_height)
            .unwrap();
    }
    handle.join().unwrap();
}
