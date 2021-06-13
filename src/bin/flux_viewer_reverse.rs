use std::{f32::consts::TAU, fs::File, io::Read};

// use crate::math::Sample2D;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
// use packed_simd::f32x4;
// use rand::prelude::*;
use rayon::prelude::*;

use crate::math::{SingleWavelength, XYZColor};
use film::Film;
// use lens_sampler::RadialSampler;
use optics::*;
use parse::*;

use crate::math::spectral::BOUNDED_VISIBLE_RANGE;
use tonemap::{sRGB, Tonemapper};

#[derive(Debug, Copy, Clone)]
pub enum Mode {
    Texture,
    SpotLight,
    PinLight,
}
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

fn main() {
    let opt = Opt::from_args();
    println!("{:?}", opt);
    let window_width = opt.width;
    let window_height = opt.height;

    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "flux viewer",
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

    let frame_dt = 6944.0 / 1000000.0;

    let mut camera_file = File::open(format!("data/cameras/{}.txt", opt.lens)).unwrap();
    let mut camera_spec = String::new();
    camera_file.read_to_string(&mut camera_spec).unwrap();

    let (lenses, _last_ior, _last_vno) = parse_lenses_from(&camera_spec);
    let lens_assembly = LensAssembly::new(&lenses);

    let scene = get_scene("textures.toml").unwrap();

    let mut textures: Vec<TexStack> = Vec::new();
    for tex in scene.textures {
        textures.push(parse_texture_stack(tex.clone()));
    }

    let original_aperture_radius = lens_assembly.aperture_radius();
    let mut heat_bias = 0.01;
    let mut heat_cap = 10.0;

    let mut aperture_radius = original_aperture_radius / 3.0; // start with small aperture
    let mut lens_zoom = 0.0;
    let mut sensor_pos = -lens_assembly.total_thickness_at(lens_zoom);
    let mut wall_position = 5000.0;
    let mut sensor_size = 35.0;
    let mut texture_scale = 30.0;

    let mut samples_per_iteration = 1000usize;
    let mut total_samples = 0;
    let mut focal_distance_suggestion = None;
    let mut focal_distance_vec: Vec<f32> = Vec::new();
    let mut variance: f32 = 0.0;
    let mut stddev: f32 = 0.0;

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;

    // let direction_cache_radius_bins = 512;
    // let direction_cache_wavelength_bins = 512;

    // let mut direction_cache = RadialSampler::new(
    //     SQRT_2 * sensor_size / 2.0, // diagonal.
    //     direction_cache_radius_bins,
    //     direction_cache_wavelength_bins,
    //     wavelength_bounds,
    //     sensor_pos,
    //     &lens_assembly,
    //     lens_zoom,
    //     |aperture_radius, ray| bladed_aperture(aperture_radius, 6, ray),
    //     heat_bias,
    //     sensor_size,
    // );

    let mut last_pressed_hotkey = Key::A;
    let mut wavelength_sweep: f32 = 0.0;
    let mut wavelength_sweep_speed = 0.001;
    let mut efficiency = 0.0;
    let efficiency_heat = 0.99;
    let mut mode = Mode::Texture;
    let mut paused = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut clear_film = false;
        let mut clear_direction_cache = false;
        let mut config_direction: f32 = 0.0;
        let keys = window.get_keys_pressed(KeyRepeat::No);

        for key in keys.unwrap_or(vec![]) {
            match key {
                Key::A => {
                    // aperture
                    println!("mode switched to aperture mode");
                    println!(
                        "{:?}, f stop = {:?}",
                        aperture_radius,
                        original_aperture_radius / aperture_radius
                    );
                    last_pressed_hotkey = Key::A;
                }
                Key::F => {
                    // Film
                    println!("mode switched to Film position (focus) mode");
                    println!(
                        "{:?}, {:?}, offset = {}",
                        sensor_pos,
                        lens_assembly.total_thickness_at(lens_zoom),
                        sensor_pos + lens_assembly.total_thickness_at(lens_zoom)
                    );
                    println!("{:?}, {}, {}", focal_distance_suggestion, variance, stddev);
                    last_pressed_hotkey = Key::F;
                }
                Key::W => {
                    // Wall
                    println!("mode switched to Wall position mode");
                    println!("{:?}", wall_position);
                    println!("{:?}, {}, {}", focal_distance_suggestion, variance, stddev);
                    last_pressed_hotkey = Key::W;
                }
                Key::R => {
                    // Wall
                    println!("mode switched to wavelength sweep speed mode");
                    println!("{:?}", wavelength_sweep_speed);
                    last_pressed_hotkey = Key::R;
                }
                Key::H => {
                    // Heat
                    println!("mode switched to Heat mode");
                    last_pressed_hotkey = Key::H;
                }
                Key::Z => {
                    // zoom
                    println!("mode switched to zoom mode");
                    last_pressed_hotkey = Key::Z;
                }
                Key::S => {
                    // samples
                    println!("mode switched to samples mode");
                    last_pressed_hotkey = Key::S;
                }

                Key::C => {
                    // heat cap
                    println!("mode switched to heat cap mode");
                    last_pressed_hotkey = Key::C;
                }
                Key::T => {
                    // heat cap
                    println!("mode switched to texture scale mode");
                    last_pressed_hotkey = Key::T;
                }
                Key::E => {
                    // film size.
                    println!("mode switched to film size mode");
                    println!("{:?}", sensor_size);
                    last_pressed_hotkey = Key::E;
                }
                Key::P => {
                    // pause simulation
                    println!("switching pause state");
                    paused = !paused;
                }
                Key::NumPadMinus | Key::NumPadPlus => {
                    // pass
                }
                _ => {
                    println!("available keys are as follows. \nA => Aperture mode\nF => Focus mode\nW => Wall position mode\nH => Heat multiplier mode\nC => Heat Cap mode\nT => texture scale mode\nR => Wavelength sweep speed mode\nE => Film Span mode. allows for artificial zoom.\nS => Samples per frame mode\nZ => Zoom mode (only affects zoomable lenses)\n")
                }
            }
        }
        if window.is_key_pressed(Key::NumPadPlus, KeyRepeat::Yes) {
            config_direction += 1.0;
        }
        if window.is_key_pressed(Key::NumPadMinus, KeyRepeat::Yes) {
            config_direction -= 1.0;
        }
        if config_direction.abs() > 0.0 {
            match last_pressed_hotkey {
                Key::A => {
                    // aperture
                    aperture_radius *= 1.1f32.powf(config_direction);
                    heat_bias *= 1.1f32.powf(config_direction);
                    clear_direction_cache = true;
                    println!(
                        "{:?}, f stop = {:?}",
                        aperture_radius,
                        original_aperture_radius / aperture_radius
                    );
                }
                Key::F => {
                    // Film
                    clear_film = true;
                    clear_direction_cache = true;
                    total_samples = 0;
                    println!("{:?}, {}, {}", focal_distance_suggestion, variance, stddev);
                    sensor_pos += 1.0 * config_direction;
                    println!(
                        "{:?}, {:?}",
                        sensor_pos,
                        lens_assembly.total_thickness_at(lens_zoom)
                    );
                }
                Key::W => {
                    // Wall

                    clear_film = true;
                    total_samples = 0;
                    wall_position += 10.0 * config_direction;
                    println!("{:?}", wall_position);
                    println!("{:?}, {}, {}", focal_distance_suggestion, variance, stddev);
                }
                Key::H => {
                    // Heat
                    heat_bias *= 1.1f32.powf(config_direction);
                    println!("{:?}", heat_bias);
                }
                Key::R => {
                    // wavelength sweep
                    wavelength_sweep_speed *= 1.1f32.powf(config_direction);
                    println!("{:?}", wavelength_sweep_speed);
                }
                Key::C => {
                    // heat cap
                    heat_cap *= 1.1f32.powf(config_direction);
                    println!("{:?}", heat_cap);
                }
                Key::T => {
                    // texture scale
                    texture_scale *= 1.1f32.powf(config_direction);
                    println!("{:?}", texture_scale);
                }
                Key::Z => {
                    clear_film = true;
                    clear_direction_cache = true;
                    total_samples = 0;
                    lens_zoom += 0.01 * config_direction;
                    println!("{:?}", lens_zoom);
                }
                Key::S => {
                    let tmp_dir = config_direction as i32;
                    match tmp_dir {
                        1 => samples_per_iteration += 1,
                        -1 => {
                            if samples_per_iteration > 1 {
                                samples_per_iteration -= 1;
                            }
                        }
                        _ => {}
                    }

                    println!("{:?}", samples_per_iteration);
                }
                Key::E => {
                    clear_film = true;
                    clear_direction_cache = true;
                    total_samples = 0;
                    sensor_size *= 1.1f32.powf(config_direction);
                    println!("{:?}", sensor_size);
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
        if window.is_key_pressed(Key::Space, KeyRepeat::Yes) {
            clear_film = true;
            clear_direction_cache = true;
            wavelength_sweep = 0.0;
            total_samples = 0;
        }
        if window.is_key_pressed(Key::V, KeyRepeat::Yes) {
            println!("total samples: {}", total_samples);
            println!("wavelength_sweep: {}", wavelength_sweep);
            println!("sampling efficiency is {}", efficiency);
        }
        if window.is_key_pressed(Key::B, KeyRepeat::No) {
            clear_film = true;
        }
        if window.is_key_pressed(Key::M, KeyRepeat::No) {
            // do mode transition
            mode = match mode {
                Mode::Texture => Mode::PinLight,
                Mode::PinLight => Mode::SpotLight,
                Mode::SpotLight => Mode::Texture,
            };
            println!("new mode is {:?}", mode);
        }
        if clear_film {
            film.buffer
                .par_iter_mut()
                .for_each(|e| *e = XYZColor::BLACK)
        }
        if clear_direction_cache {
            // direction_cache = RadialSampler::new(
            //     SQRT_2 * sensor_size / 2.0, // diagonal.
            //     direction_cache_radius_bins,
            //     direction_cache_wavelength_bins,
            //     wavelength_bounds,
            //     sensor_pos,
            //     &lens_assembly,
            //     lens_zoom,
            //     |aperture_radius, ray| bladed_aperture(aperture_radius, 6, ray),
            //     heat_bias,
            //     sensor_size,
            // );
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

        let mut sampler = RandomSampler::new();

        let (mut successes, mut attempts) = (0, 0);

        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);

        for _ in 0..samples_per_iteration {
            // ray is generated according to texture scale.
            let ray = match mode {
                // diffuse emitter texture
                Mode::Texture => {
                    // 4 possible quadrants.
                    let (rx, ry) = (
                        sampler.draw_1d().x * 2.0 - 1.0,
                        sampler.draw_1d().x * 2.0 - 1.0,
                    );

                    let (r, phi) = (
                        sampler.draw_1d().x.sqrt() * lens_assembly.lenses[0].housing_radius,
                        sampler.draw_1d().x * TAU,
                    );

                    let point_on_lens = Point3::new(r * phi.cos(), r * phi.sin(), 0.0);
                    let point_on_texture =
                        Point3::new(texture_scale * rx, texture_scale * ry, wall_position);
                    let v = (point_on_lens - point_on_texture).normalized();

                    Ray::new(point_on_texture, v)
                }
                // parallel light
                Mode::SpotLight => {
                    // 4 quadrants.

                    let (r, phi) = (
                        sampler.draw_1d().x.sqrt() * texture_scale,
                        sampler.draw_1d().x * TAU,
                    );

                    let (px, py) = (r * phi.cos(), r * phi.sin());

                    Ray::new(Point3::new(px, py, wall_position), -Vec3::Z)
                }
                Mode::PinLight => {
                    // 4 quadrants.

                    let (r, phi) = (
                        sampler.draw_1d().x.sqrt() * texture_scale,
                        sampler.draw_1d().x * TAU,
                    );

                    let (dx, dy) = (r * phi.cos(), r * phi.sin());
                    let (px, py) = (0.0, 0.0);

                    Ray::new(
                        Point3::new(px, py, wall_position),
                        Vec3::new(dx, dy, -1.0).normalized(),
                    )
                }
            };
            // println!("{:?}", ray);

            attempts += 1;
            // do actual tracing through lens for film sample
            let mut rays = Vec::new();
            let result = lens_assembly.trace_reverse_w_callback(
                lens_zoom,
                &Input { ray, lambda },
                1.04,
                |e| (bladed_aperture(aperture_radius, 6, e), false),
                |ray| rays.push(ray),
            );
            if let Some(Output {
                ray: pupil_ray,
                tau,
            }) = result
            {
                successes += 1;
                let t = (sensor_pos - pupil_ray.origin.z()) / pupil_ray.direction.z();
                let point_at_film = pupil_ray.point_at_parameter(t);
                let uv = (
                    (((point_at_film.x() / sensor_size) + 1.0) / 2.0) % 1.0,
                    (((point_at_film.y() / sensor_size) + 1.0) / 2.0) % 1.0,
                );
                film.write_at(
                    (uv.0 * window_width as f32) as usize,
                    (uv.1 * window_height as f32) as usize,
                    film.at(
                        (uv.0 * window_width as f32) as usize,
                        (uv.1 * window_height as f32) as usize,
                    ) + XYZColor::from(SingleWavelength::new(lambda, tau.into())),
                );
            }
        }
        efficiency = (efficiency_heat) * efficiency
            + (1.0 - efficiency_heat) * (successes as f32 / attempts as f32);

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
}
