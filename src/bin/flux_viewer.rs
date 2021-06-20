use std::{
    f32::consts::{PI, SQRT_2, TAU},
    fs::File,
    io::Read,
};

#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rayon::prelude::*;
use structopt::StructOpt;

extern crate line_drawing;

use crate::math::spectral::BOUNDED_VISIBLE_RANGE;
use crate::math::{
    random_cosine_direction, Point3, Ray, Sample2D, SingleWavelength, Vec3, XYZColor,
};
use crate::math::{Bounds1D, Bounds2D};
use film::Film;
use lens_sampler::RadialSampler;
use optics::*;
use parse::*;

use tonemap::{sRGB, Tonemapper};

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

pub fn ray_plane_intersection(r: Ray, o: Point3, n: Vec3) -> Option<Point3> {
    // (ro + t * dv - o) * n = 0
    // ((t * dv) + (ro - o)) * n = 0
    // (t * d) * n + (ro - o) * n = 0
    // (t * d) * n = (o - ro) * n
    // t * (d * n) = (o - ro) * n
    // t = (o - ro) * n / (d * n);

    // ||d|| = 1 if d is normalized
    // t = (o - ro) * n / (d * n)
    // and point on plane = ro + t * d;
    let t = (o - r.origin) * n / (r.direction * n);
    if t < 0.0 {
        None
    } else {
        Some(r.point_at_parameter(t))
    }
}

pub fn replace_if<T: Copy, F>(a: &mut T, b: T, condition: F)
where
    F: Fn(&T, T) -> bool,
{
    if condition(a, b) {
        *a = b;
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Mode {
    Texture,
    SpotLight,
    PinLight,
}

#[derive(Debug, Copy, Clone)]
pub enum DrawMode {
    XiaolinWu,
    Midpoint,
    Bresenham,
}

pub fn draw_rays<F>(
    film: &mut Film<XYZColor>,
    rays: &Vec<(Ray, XYZColor)>,
    draw_mode: DrawMode,
    projection_func: F,
) -> usize
where
    F: Fn(Ray, XYZColor) -> (((f32, f32), (f32, f32)), XYZColor),
{
    let (width, height) = (film.width, film.height);
    // potentially need another scaling factor based on how much z gets stretched when projected. within projection_func or after projection_func
    let mut count = 0;
    for (((x0, y0), (x1, y1)), color) in rays.iter().map(|(r, color)| projection_func(*r, *color)) {
        let (px0, py0) = (
            // (width as f32 * (line.0.x() - view_bounds.x.lower) / view_bounds.x.span()) as usize,
            // (height as f32 * (line.0.y() - view_bounds.y.lower) / view_bounds.y.span()) as usize,
            (width as f32 * x0) as usize,
            (height as f32 * y0) as usize,
        );
        let (px1, py1) = (
            // (width as f32 * (line.1.x() - view_bounds.x.lower) / view_bounds.x.span()) as usize,
            // (height as f32 * (line.1.y() - view_bounds.y.lower) / view_bounds.y.span()) as usize,
            (width as f32 * x1) as usize,
            (height as f32 * y1) as usize,
        );

        let (dx, dy) = (px1 as isize - px0 as isize, py1 as isize - py0 as isize);
        // draw single point as a point instead of going through the line algorithm
        if dx == 0 && dy == 0 {
            if px0 as usize >= width || py0 as usize >= height {
                continue;
            }
            film.buffer[py0 as usize * width + px0 as usize] += color;
            continue;
        }
        count += 1;

        let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
        match draw_mode {
            DrawMode::Midpoint => {
                for (x, y) in line_drawing::Midpoint::<f32, isize>::new(
                    (px0 as f32, py0 as f32),
                    (px1 as f32, py1 as f32),
                ) {
                    if x as usize >= width || y as usize >= height || x < 0 || y < 0 {
                        continue;
                    }
                    assert!(!b.is_nan(), "{} {}", dx, dy);
                    film.buffer[y as usize * width + x as usize] += color * b;
                }
            }
            DrawMode::XiaolinWu => {
                // let b = 1.0f32;
                for ((x, y), a) in line_drawing::XiaolinWu::<f32, isize>::new(
                    (px0 as f32, py0 as f32),
                    (px1 as f32, py1 as f32),
                ) {
                    if x as usize >= width || y as usize >= height || x < 0 || y < 0 {
                        continue;
                    }
                    assert!(!b.is_nan(), "{} {}", dx, dy);
                    film.buffer[y as usize * width + x as usize] += color * b * a;
                }
            }
            DrawMode::Bresenham => {
                for (x, y) in line_drawing::Bresenham::new(
                    (px0 as isize, py0 as isize),
                    (px1 as isize, py1 as isize),
                ) {
                    if x as usize >= width || y as usize >= height || x < 0 || y < 0 {
                        continue;
                    }
                    assert!(!b.is_nan(), "{} {}", dx, dy);
                    film.buffer[y as usize * width + x as usize] += color * b;
                }
            }
        }
    }
    return count;
}

#[derive(Debug, Copy, Clone)]
enum RayGenerationMode {
    FromSensor { forced_flat: bool },
    FromScene { forced_flat: bool },
}

#[derive(Debug, Copy, Clone)]
enum ProjectionMode {
    Orthogonal {
        scale: f32,
        normal: Vec3,
        origin: Point3,
        up: Vec3,
    },
    Projective {
        origin: Point3,
    },
}

fn main() {
    let opt = Opt::from_args();
    println!("{:?}", opt);
    let window_width = opt.width;
    let window_height = opt.height;
    let aspect_wh = window_width as f32 / window_height as f32; // for instance 16 / 9

    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .unwrap();

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
    let mut wall_position = 80.0;
    let sensor_size = 35.0;
    let mut sensor_mult = 1.0;
    let mut texture_scale = 30.0;
    let mut film_lateral_offset = 0.0;

    let max_lens_radius = lens_assembly
        .lenses
        .iter()
        .map(|l| l.housing_radius)
        .reduce(|a, b| a.max(b))
        .unwrap();
    let lens_depth = sensor_pos.abs();
    let view_bounds = Bounds2D::new(
        Bounds1D::new(-lens_depth, 0.0),
        Bounds1D::new(-max_lens_radius, max_lens_radius),
    );
    let calculated_wh_aspect = view_bounds.x.span() / view_bounds.y.span();
    let mut view_zoom = 1.0;

    view_zoom *= calculated_wh_aspect / aspect_wh;
    let mut projection_mode = ProjectionMode::Orthogonal {
        scale: max_lens_radius * view_zoom,
        origin: Point3::new(0.0, 0.0, -lens_depth / 2.0),
        normal: -Vec3::Y,
        up: Vec3::X,
    };

    let mut film = Film::new(window_width, window_height, XYZColor::BLACK);
    let mut window_pixels = Film::new(window_width, window_height, 0u32);

    let width = film.width;
    let height = film.height;
    println!(
        "suggested window sizes: {} by {} or {} by {}",
        (height as f32 * calculated_wh_aspect) as usize,
        height,
        width,
        (width as f32 / calculated_wh_aspect) as usize
    );

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

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    let frame_dt = 6944.0 / 1000000.0;

    let mut samples_per_iteration = 100usize;
    let mut total_samples = 0;
    let mut focal_distance_suggestion = None;
    let mut focal_distance_vec: Vec<f32> = Vec::new();
    let mut variance: f32 = 0.0;
    let mut stddev: f32 = 0.0;

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;

    let direction_cache_radius_bins = 512;
    let direction_cache_wavelength_bins = 512;

    let mut direction_cache = RadialSampler::new(
        SQRT_2 * sensor_mult * sensor_size / 2.0, // diagonal.
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

    let ray_generation_mode = RayGenerationMode::FromSensor { forced_flat: false };
    // let ray_generation_mode = RayGenerationMode::FromScene { forced_flat: true };
    let mut mode = Mode::SpotLight;

    let mut last_pressed_hotkey = Key::A;
    let mut wavelength_sweep: f32 = 0.0;
    let mut wavelength_sweep_speed = 0.001;
    let mut efficiency = 0.0;
    let efficiency_heat = 0.99;
    let mut paused = false;
    let mut draw_mode = DrawMode::XiaolinWu;

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
                    println!("{:?}", sensor_size * sensor_mult);
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

                    // if let ProjectionMode::Orthogonal {
                    //     normal, origin, up, ..
                    // } = projection_mode
                    // {
                    //     projection_mode = ProjectionMode::Orthogonal {
                    //         scale: view_zoom * max_lens_radius,
                    //         normal,
                    //         origin,
                    //         up,
                    //     };
                    // }
                }
                Key::W => {
                    // Wall

                    clear_film = true;
                    total_samples = 0;
                    wall_position += 1.0 * config_direction;
                    println!("{:?}", wall_position);
                    println!("{:?}, {}, {}", focal_distance_suggestion, variance, stddev);

                    // if let ProjectionMode::Orthogonal {
                    //     scale, normal, origin, up
                    // } = projection_mode
                    // {
                    //     projection_mode = ProjectionMode::Orthogonal {
                    //         scale: view_zoom * max_lens_radius,
                    //         normal,
                    //         origin,
                    //         up,
                    //     };
                    // }
                }
                Key::H => {
                    // Heat
                    heat_bias *= 1.1f32.powf(config_direction);
                    println!("{:?}", heat_bias);
                }
                Key::R => {
                    // wavelength sweep
                    // wavelength_sweep_speed *= 1.1f32.powf(config_direction);
                    film_lateral_offset += 1.0 * config_direction;
                    println!("film lateral offset = {:?}%", film_lateral_offset);
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
                    // clear_film = true;
                    // clear_direction_cache = true;
                    // total_samples = 0;
                    view_zoom *= 1.1f32.powf(config_direction);
                    if let ProjectionMode::Orthogonal {
                        normal, origin, up, ..
                    } = projection_mode
                    {
                        projection_mode = ProjectionMode::Orthogonal {
                            scale: view_zoom * max_lens_radius,
                            normal,
                            origin,
                            up,
                        };
                    }
                    println!("{:?}", view_zoom);
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
                    sensor_mult *= 1.1f32.powf(config_direction);
                    println!("{:?}", sensor_size * sensor_mult);
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
            direction_cache = RadialSampler::new(
                SQRT_2 * sensor_mult * sensor_size / 2.0, // diagonal.
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

        let (mut successes, mut attempts) = (0, 0);

        let mut collected_rays: Vec<(Ray, XYZColor)> = Vec::new();
        match ray_generation_mode {
            RayGenerationMode::FromSensor { forced_flat } => {
                match mode {
                    Mode::Texture => {
                        let mut rays: Vec<(Ray, XYZColor)> =
                            (0..opt.threads)
                                .into_par_iter()
                                .flat_map(|_t| {
                                    let mut rays = Vec::new();
                                    let mut sampler = RandomSampler::new();
                                    for _ in 0..(samples_per_iteration / opt.threads) {
                                        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                                        // ray is generated according to texture scale.
                                        let ray = if forced_flat {
                                            let point = Point3::new(
                                                (sampler.draw_1d().x - 0.5)
                                                    * sensor_size
                                                    * sensor_mult
                                                    + film_lateral_offset,
                                                0.0,
                                                sensor_pos,
                                            );
                                            // let mut v = random_cosine_direction(sampler.draw_2d());
                                            // v.0 = v.0.replace(1, 0.0);
                                            let mut v = direction_cache.sample(
                                                lambda,
                                                point,
                                                sampler.draw_2d(),
                                                sampler.draw_1d(),
                                            );
                                            v.0 = v.0.replace(1, 0.0);
                                            v = v.normalized();
                                            Ray::new(point, v)
                                        } else {
                                            let Sample2D { x, y } = sampler.draw_2d();
                                            let point = Point3::new(
                                                (x - 0.5) * sensor_size * sensor_mult
                                                    + film_lateral_offset,
                                                (y - 0.5) * sensor_size * sensor_mult,
                                                sensor_pos,
                                            );
                                            // let v = random_cosine_direction(sampler.draw_2d());
                                            let v = direction_cache.sample(
                                                lambda,
                                                point,
                                                sampler.draw_2d(),
                                                sampler.draw_1d(),
                                            );

                                            Ray::new(point, v)
                                        };

                                        // println!("{:?}", ray);

                                        // do actual tracing through lens for film sample

                                        let result = lens_assembly.trace_forward_w_callback(
                                            lens_zoom,
                                            &Input { ray, lambda },
                                            1.04,
                                            |e| (bladed_aperture(aperture_radius, 6, e), false),
                                            |ray, tau| {
                                                // the following if condition hides rays that don't have their time set. applicable when rays exit the assembly from some surface that isn't the front iris.
                                                if ray.time > 0.0 {
                                                    rays.push((
                                                        ray,
                                                        XYZColor::from(SingleWavelength::new(
                                                            lambda,
                                                            tau.into(),
                                                        )),
                                                    ));
                                                }
                                            },
                                        );
                                        if let Some(Output {
                                            ray: mut pupil_ray,
                                            mut tau,
                                        }) = result
                                        {
                                            pupil_ray.time = if let Some(point) =
                                                ray_plane_intersection(
                                                    pupil_ray,
                                                    Point3::new(0.0, 0.0, wall_position),
                                                    Vec3::Z,
                                                ) {
                                                (point - pupil_ray.origin).norm()
                                            } else {
                                                // tau = 0.0;s
                                                // flag 0
                                                // 100000.0
                                                0.0
                                            };
                                            rays.push((
                                                pupil_ray,
                                                XYZColor::from(SingleWavelength::new(
                                                    lambda,
                                                    tau.into(),
                                                )),
                                            ));
                                        }
                                    }
                                    rays
                                })
                                .collect();
                        collected_rays.extend(rays.drain(..));
                    }
                    Mode::SpotLight => {
                        let mut rays: Vec<(Ray, XYZColor)> =
                            (0..opt.threads)
                                .into_par_iter()
                                .flat_map(|t| {
                                    let mut rays = Vec::new();
                                    let mut sampler = RandomSampler::new();
                                    for _ in 0..(samples_per_iteration / opt.threads) {
                                        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                                        // ray is generated according to texture scale.
                                        let ray = if forced_flat {
                                            let point = Point3::new(
                                                (sampler.draw_1d().x - 0.5)
                                                    * sensor_size
                                                    * sensor_mult
                                                    + film_lateral_offset,
                                                0.0,
                                                sensor_pos,
                                            );
                                            // let mut v = random_cosine_direction(sampler.draw_2d());
                                            // v.0 = v.0.replace(1, 0.0);
                                            // v.normalize();
                                            // let mut v = direction_cache.sample(
                                            //     lambda,
                                            //     point,
                                            //     sampler.draw_2d(),
                                            //     sampler.draw_1d(),
                                            // );
                                            Ray::new(point, Vec3::Z)
                                        } else {
                                            let Sample2D { x, y } = sampler.draw_2d();
                                            let point = Point3::new(
                                                (x - 0.5) * sensor_size * sensor_mult
                                                    + film_lateral_offset,
                                                (y - 0.5) * sensor_size * sensor_mult,
                                                sensor_pos,
                                            );
                                            // let v = random_cosine_direction(sampler.draw_2d());
                                            // let v = direction_cache.sample(
                                            //     lambda,
                                            //     point,
                                            //     sampler.draw_2d(),
                                            //     sampler.draw_1d(),
                                            // );

                                            Ray::new(point, Vec3::Z)
                                        };

                                        // println!("{:?}", ray);

                                        // do actual tracing through lens for film sample

                                        let result = lens_assembly.trace_forward_w_callback(
                                            lens_zoom,
                                            &Input { ray, lambda },
                                            1.04,
                                            |e| (bladed_aperture(aperture_radius, 6, e), false),
                                            |ray, tau| {
                                                // the following if condition hides rays that don't have their time set. applicable when rays exit the assembly from some surface that isn't the front iris.
                                                // flag 0
                                                if ray.time > 0.0 {
                                                    rays.push((
                                                        ray,
                                                        XYZColor::from(SingleWavelength::new(
                                                            lambda,
                                                            tau.into(),
                                                        )),
                                                    ));
                                                }
                                            },
                                        );
                                        if let Some(Output {
                                            ray: mut pupil_ray,
                                            mut tau,
                                        }) = result
                                        {
                                            pupil_ray.time = if let Some(point) =
                                                ray_plane_intersection(
                                                    pupil_ray,
                                                    Point3::new(0.0, 0.0, wall_position),
                                                    Vec3::Z,
                                                ) {
                                                (point - pupil_ray.origin).norm()
                                            } else {
                                                // tau = 0.0;s
                                                // flag 0
                                                // 100000.0
                                                0.0
                                            };
                                            rays.push((
                                                pupil_ray,
                                                XYZColor::from(SingleWavelength::new(
                                                    lambda,
                                                    tau.into(),
                                                )),
                                            ));
                                            // let t = (sensor_pos - pupil_ray.origin.z()) / pupil_ray.direction.z();
                                            // let point_at_film = pupil_ray.point_at_parameter(t);
                                            // let uv = (
                                            //     (((point_at_film.x() / sensor_size) + 1.0) / 2.0) % 1.0,
                                            //     (((point_at_film.y() / sensor_size) + 1.0) / 2.0) % 1.0,
                                            // );
                                            // film.write_at(
                                            //     (uv.0 * window_width as f32) as usize,
                                            //     (uv.1 * window_height as f32) as usize,
                                            //     film.at(
                                            //         (uv.0 * window_width as f32) as usize,
                                            //         (uv.1 * window_height as f32) as usize,
                                            //     ) + XYZColor::from(SingleWavelength::new(lambda, tau.into())),
                                            // );
                                        }
                                    }
                                    rays
                                })
                                .collect();
                        collected_rays.extend(rays.drain(..));
                    }
                    Mode::PinLight => {}
                }
            }

            RayGenerationMode::FromScene { forced_flat } => {
                let mut rays: Vec<(Ray, XYZColor)> = (0..opt.threads)
                    .into_par_iter()
                    .flat_map(|t| {
                        let mut rays = Vec::new();
                        let mut sampler = RandomSampler::new();
                        for _ in 0..(samples_per_iteration / opt.threads) {
                            let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                            // ray is generated according to texture scale.
                            let ray = match mode {
                                // diffuse emitter texture
                                Mode::Texture => {
                                    // 4 possible quadrants.
                                    let (rx, ry) = (
                                        sampler.draw_1d().x * 2.0 - 1.0,
                                        if forced_flat {
                                            0.0
                                        } else {
                                            sampler.draw_1d().x * 2.0 - 1.0
                                        },
                                    );

                                    let (r, mut phi) = (
                                        sampler.draw_1d().x
                                            * lens_assembly.lenses[0].housing_radius,
                                        sampler.draw_1d().x * TAU,
                                    );

                                    if forced_flat {
                                        phi = if sampler.draw_1d().x > 0.5 { PI } else { 0.0 };
                                    }

                                    let point_on_lens =
                                        Point3::new(r * phi.cos(), r * phi.sin(), 0.0);
                                    let point_on_texture = Point3::new(
                                        texture_scale * rx,
                                        texture_scale * ry,
                                        wall_position,
                                    );
                                    let v = (point_on_lens - point_on_texture).normalized();

                                    Ray::new(point_on_texture, v)
                                }
                                // parallel light
                                Mode::SpotLight => {
                                    // 4 quadrants.

                                    let (r, mut phi) = (
                                        sampler.draw_1d().x * texture_scale,
                                        sampler.draw_1d().x * TAU,
                                    );
                                    if forced_flat {
                                        phi = if sampler.draw_1d().x > 0.5 { PI } else { 0.0 };
                                    }

                                    let (px, py) = (r * phi.cos(), r * phi.sin());

                                    Ray::new(
                                        Point3::new(px + film_lateral_offset, py, wall_position),
                                        -Vec3::Z,
                                    )
                                }
                                Mode::PinLight => {
                                    // 4 quadrants.

                                    let (r, mut phi) = (
                                        sampler.draw_1d().x
                                            * lens_assembly.lenses[0].housing_radius,
                                        sampler.draw_1d().x * TAU,
                                    );

                                    if forced_flat {
                                        phi = if sampler.draw_1d().x > 0.5 { PI } else { 0.0 };
                                    }

                                    let (px, py) = (0.0, 0.0);
                                    let pin_point = Point3::new(px, py, wall_position);
                                    // solve x^2 + y^2 = lens_radius^2 for x
                                    // x = sqrt(r^2 - y^2)
                                    // adjusted_x = sqrt(r^2 - y^2) - r
                                    let l0r = lens_assembly.lenses[0].radius;
                                    let hr = lens_assembly.lenses[0].housing_radius;
                                    let depth = (l0r * l0r - hr * hr).sqrt() - l0r;
                                    // let depth = 0.0;
                                    let point_on_lens =
                                        Point3::new(r * phi.cos(), r * phi.sin(), depth);
                                    let direction = point_on_lens - pin_point;

                                    // let (dx, dy) = (r * phi.cos(), r * phi.sin());

                                    Ray::new(pin_point, direction)
                                }
                            };
                            // println!("{:?}", ray);

                            // do actual tracing through lens for film sample

                            let result = lens_assembly.trace_reverse_w_callback(
                                lens_zoom,
                                &Input { ray, lambda },
                                1.04,
                                |e| (bladed_aperture(aperture_radius, 6, e), false),
                                |ray, tau| {
                                    rays.push((
                                        ray,
                                        XYZColor::from(SingleWavelength::new(lambda, tau.into())),
                                    ))
                                },
                            );
                            if let Some(Output {
                                ray: mut sensor_ray,
                                mut tau,
                            }) = result
                            {
                                sensor_ray.time = if let Some(point) = ray_plane_intersection(
                                    sensor_ray,
                                    Point3::new(0.0, 0.0, -sensor_pos),
                                    Vec3::Z,
                                ) {
                                    if point.x().abs() < sensor_size / 2.0
                                        && point.y().abs() < sensor_size / 2.0
                                    {
                                        (point - sensor_ray.origin).norm()
                                    } else {
                                        // tau = 0.0;
                                        // flag 0
                                        // 0.0
                                        10000.0
                                    }
                                } else {
                                    // tau = 0.0;
                                    // flag 0
                                    // 0.0
                                    10000.0
                                };
                                rays.push((
                                    sensor_ray,
                                    XYZColor::from(SingleWavelength::new(lambda, tau.into())),
                                ));
                                // let t = (sensor_pos - pupil_ray.origin.z()) / pupil_ray.direction.z();
                                // let point_at_film = pupil_ray.point_at_parameter(t);
                                // let uv = (
                                //     (((point_at_film.x() / sensor_size) + 1.0) / 2.0) % 1.0,
                                //     (((point_at_film.y() / sensor_size) + 1.0) / 2.0) % 1.0,
                                // );
                                // film.write_at(
                                //     (uv.0 * window_width as f32) as usize,
                                //     (uv.1 * window_height as f32) as usize,
                                //     film.at(
                                //         (uv.0 * window_width as f32) as usize,
                                //         (uv.1 * window_height as f32) as usize,
                                //     ) + XYZColor::from(SingleWavelength::new(lambda, tau.into())),
                                // );
                            }
                        }
                        rays
                    })
                    .collect();
                collected_rays.extend(rays.drain(..));
            }
        }

        efficiency = (efficiency_heat) * efficiency
            + (1.0 - efficiency_heat) * (successes as f32 / attempts as f32);

        draw_rays(&mut film, &collected_rays, draw_mode, |r, color| {
            // take ray and convert to line2d with uv taken into account. i.e. clipping/projecting lines
            match projection_mode {
                ProjectionMode::Orthogonal {
                    scale,
                    normal,
                    origin,
                    up,
                } => {
                    // let space_scale = r.direction.norm() / (r.direction * normal).abs();
                    let space_scale = 1.0;
                    let w = normal;
                    let u = w.cross(up).normalized();
                    let v = w.cross(u);

                    // starting point
                    let mut uv_origin = aspect_aware_orthogonal_projection(
                        r.origin, origin, scale, u, v, aspect_wh,
                    );
                    if uv_origin.0 < 0.0
                        || uv_origin.1 < 0.0
                        || uv_origin.1 >= 1.0
                        || uv_origin.1 >= 1.0
                    {
                        // starting point outside the box
                        // thus intersect ray with boxes and accept the closest valid point.
                        let destination = {
                            // compute ray plane intersection on the 4 planes that mark the bounding box.
                            // accept the furthest valid point.
                            let mut closest_valid_intersection: Point3 =
                                r.point_at_parameter(100000000.0);
                            let cloned = closest_valid_intersection.clone();
                            let condition = |a: &Point3, b: Point3| {
                                let buv = aspect_aware_orthogonal_projection(
                                    b, origin, scale, u, v, aspect_wh,
                                );
                                (b - r.origin).norm_squared() < (*a - r.origin).norm_squared()
                                    && buv.0 < 1.0
                                    && buv.0 >= 0.0
                                    && buv.1 < 1.0
                                    && buv.1 >= 0.0
                            };

                            replace_if(
                                &mut closest_valid_intersection,
                                ray_plane_intersection(r, origin + u * scale * aspect_wh, u)
                                    .unwrap_or(cloned),
                                condition,
                            );
                            replace_if(
                                &mut closest_valid_intersection,
                                ray_plane_intersection(r, origin + v * scale, v).unwrap_or(cloned),
                                condition,
                            );
                            replace_if(
                                &mut closest_valid_intersection,
                                ray_plane_intersection(r, origin - u * scale * aspect_wh, u)
                                    .unwrap_or(cloned),
                                condition,
                            );
                            replace_if(
                                &mut closest_valid_intersection,
                                ray_plane_intersection(r, origin - v * scale, v).unwrap_or(cloned),
                                condition,
                            );
                            closest_valid_intersection
                        };
                        uv_origin = aspect_aware_orthogonal_projection(
                            destination,
                            origin,
                            scale,
                            u,
                            v,
                            aspect_wh,
                        );
                    }

                    // construct

                    let destination = if r.time != 0.0 {
                        r.point_at_parameter(r.time)
                    } else {
                        // compute ray plane intersection on the 4 planes that mark the bounding box.
                        // accept the furthest valid point.
                        let mut furthest_valid_intersection: Point3 =
                            r.point_at_parameter(100000000.0);
                        let cloned = furthest_valid_intersection.clone();
                        let condition = |a: &Point3, b: Point3| {
                            let buv = aspect_aware_orthogonal_projection(
                                b, origin, scale, u, v, aspect_wh,
                            );
                            (b - r.origin).norm_squared() > (*a - r.origin).norm_squared()
                                && buv.0 < 1.0
                                && buv.0 >= 0.0
                                && buv.1 < 1.0
                                && buv.1 >= 0.0
                        };

                        replace_if(
                            &mut furthest_valid_intersection,
                            ray_plane_intersection(r, origin + u * scale * aspect_wh, u)
                                .unwrap_or(cloned),
                            condition,
                        );
                        replace_if(
                            &mut furthest_valid_intersection,
                            ray_plane_intersection(r, origin + v * scale, v).unwrap_or(cloned),
                            condition,
                        );
                        replace_if(
                            &mut furthest_valid_intersection,
                            ray_plane_intersection(r, origin - u * scale * aspect_wh, u)
                                .unwrap_or(cloned),
                            condition,
                        );
                        replace_if(
                            &mut furthest_valid_intersection,
                            ray_plane_intersection(r, origin - v * scale, v).unwrap_or(cloned),
                            condition,
                        );
                        furthest_valid_intersection
                    };
                    let uv_endpoint = aspect_aware_orthogonal_projection(
                        destination,
                        origin,
                        scale,
                        u,
                        v,
                        aspect_wh,
                    );

                    ((uv_origin, uv_endpoint), color * space_scale)
                }
                ProjectionMode::Projective { origin } => (((0.0, 0.0), (1.0, 1.0)), color),
            }
        });

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
