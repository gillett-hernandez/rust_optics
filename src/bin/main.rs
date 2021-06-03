use std::f32::{
    consts::{SQRT_2, TAU},
    EPSILON,
};

#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use packed_simd::f32x4;
use rand::prelude::*;
use rayon::prelude::*;

use ::math::{random_cosine_direction, SingleWavelength, XYZColor};
use film::Film;
use lens_sampler::RadialSampler;
use optics::*;
use parse::*;

use ::math::spectral::BOUNDED_VISIBLE_RANGE;
use tonemap::{sRGB, Tonemapper};

fn main() {
    const WINDOW_WIDTH: usize = 800;
    const WINDOW_HEIGHT: usize = 800;

    rayon::ThreadPoolBuilder::new()
        .num_threads(22 as usize)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "Lens",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions {
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, XYZColor::BLACK);
    let mut window_pixels = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    let width = film.width;
    let height = film.height;

    let frame_dt = 6944.0 / 1000000.0;
    // let spec = "35.0 20.0 bk7 1.5 54.0 15.0
    // -35.0 1.73 air        15.0
    // 100000 3.00  iris    10.0
    // 1035.0 7.0 bk7 1.5 54.0 15.0
    // -35.0 20 air        15.0"; // simple 2
    // let spec = "42.97		9.8			LAK9	1.6910	54.8	19.2
    // -115.33		2.1			LLF7	1.5486  45.4	19.2
    // 306.84		4.16		air			           	19.2
    // 100000		4.0			iris			      	15
    // -59.06		1.87		SF7		1.6398  34.6   	17.3
    // 40.93		10.64		air		       			17.3
    // 183.92		7.05		LAK9	1.6910  54.8   	16.5
    // -48.91		79.831		air						16.5"; // brendel tressar
    // let spec = "52.9     5.8  abbe   1.517  62 15
    // -41.4    1.5  abbe   1.576  54 15
    // 436.2    23.3 air              15
    // 100000.0 23.3 iris             10
    // 104.8    2.2  abbe   1.517  62 9
    // 36.8     0.7  air              9
    // 45.5     3.6  abbe   1.576  54 9
    // -149.5   50   air              9"; // petzval
    // let spec = "164.12		10.99				SF5			1.673	32.2	54
    // 559.28		0.23				air							54
    // 100.12		11.45				BAF10		1.67	47.1    51
    // 213.54		0.23				air							51
    // 58.04		22.95				LAK9		1.691	54.7	41
    // 2551		2.58				SF5			1.673	32.2	41
    // 32.39		15.66				air							27
    // 10000		15.00				iris						25.5
    // -40.42		2.74				SF15		1.699	30.1	25
    // 192.98		27.92				SK16		1.62	60.3	36
    // -55.53		0.23				air							36
    // 192.98		7.98				LAK9		1.691	54.7	35
    // -225.28		0.23				air							35
    // 175.1		8.48				LAK9		1.691	54.7	35
    // -203.54		55.742				air							35"; // double gauss angenioux
    // let spec = "65.22    9.60  N-SSK8 1.5 50 24.0
    // -62.03   4.20  N-SF10 1.5 50 24.0
    // -1240.67 5.00  air           24.0
    // 100000  105.00  iris          20.0"; // lensbaby
    let spec = " 70.97  15.0 abbe 1.523 58.6  23
                         -56.79   4.5 abbe 1.617 38.5  23
                         100000.0 24.0 air             23
                         100000.0 25.3 iris            18
                         119.91   3.8 abbe 1.649 33.8  15
                         40.87    0.9 air              15
                         46.87    7.4 abbe 1.697 56.1  15
                         -282.05 56.5 air              15"; // petzval kodak
                                                            // let spec = "33.072		2.366				C3			1.518	59.0	8.9
                                                            // -53.387		0.077				air                 8.9
                                                            // 27.825		2.657				C3			1.518	59.0	8.4
                                                            // -35.934		1.025				LAF7		1.749	35.0	8.3
                                                            // 40.900		22.084				air							7.8
                                                            // 10000		1.794				FD110		1.785	25.7	4.7
                                                            // -16.775		0.641				TAFD5		1.835	43.0	4.6
                                                            // 27.153		8.607				air							4.5
                                                            // 10000       1.0                   iris                      4.8
                                                            // -120.75		1.035				CF6			1.517	52.2	4.8
                                                            // -12.105		4.705				air							4.8
                                                            // -9.386		0.641				TAF1		1.773	49.6	4.0
                                                            // -24.331		18.960				air							4.1"; // kreitzer telephoto
                                                            // let spec = "96.15 7.00 abbe 1.64 58.1             50.0
                                                            // 53.85 17.38 air                       50.0
                                                            // 117.48 5.59 abbe 1.58904 53.2         45.0
                                                            // 69.93 52.45 air                       45.0
                                                            // 106.64 15.73 abbe 1.76684 46.2        25.0
                                                            // -188.11 4.90 air                      25.0
                                                            // -192.31 15.38 abbe 1.76684 27.5       25.0
                                                            // -140.91 9.58 air                      25.0
                                                            // 100000 10.0 iris                      25.0
                                                            // -65.04 16.22 abbe 1.7552 27.5         25.0
                                                            // 188.11 2.52 air                       25.0
                                                            // -323.43 7.00 abbe 1.713 53.9          25.0
                                                            // -65.39 0.18 air                       25.0
                                                            // -8741.25 6.64 abbe 1.6583 57.3        30.0
                                                            // -117.55 131.19 air                    30.0"; // wideangle 2
    let (lenses, _last_ior, _last_vno) = parse_lenses_from(spec);
    let lens_assembly = LensAssembly::new(&lenses);

    let scene = get_scene("textures.toml").unwrap();

    let mut textures: Vec<TexStack> = Vec::new();
    for tex in scene.textures {
        textures.push(parse_texture_stack(tex.clone()));
    }

    let original_aperture_radius = lens_assembly.aperture_radius();
    let mut aperture_radius = original_aperture_radius / 10.0; // start with small aperture
    let mut heat_bias = 0.01;
    let mut heat_cap = 10.0;
    let mut lens_zoom = 0.0;
    let mut film_position = -lens_assembly.total_thickness_at(lens_zoom);
    let mut wall_position = 240.0;
    let mut sensor_size = 35.0;
    let mut samples_per_iteration = 1usize;
    let mut total_samples = 0;
    let mut focal_distance_suggestion = None;
    let mut focal_distance_vec: Vec<f32> = Vec::new();
    let mut variance: f32 = 0.0;
    let mut stddev: f32 = 0.0;

    // while window.is_open() && !window.is_key_down(Key::Escape) {
    //     let keys = window.get_keys_pressed(KeyRepeat::Yes);
    //     for key in keys.unwrap_or(vec![]) {
    //         match key {
    //             Key::A => {
    //                 aperture_radius /= 1.1;
    //                 println!(
    //                     "registered input A, aperture_size = {}, sensor_size = {}",
    //                     aperture_radius, sensor_size
    //                 );
    //             }
    //             Key::B => {
    //                 aperture_radius *= 1.1;
    //                 println!(
    //                     "registered input B, aperture_size = {}, sensor_size = {}",
    //                     aperture_radius, sensor_size
    //                 );
    //             }
    //             _ => {}
    //         }
    //     }
    //     buffer
    //         .buffer
    //         .par_iter_mut()
    //         .enumerate()
    //         .for_each(|(i, pixel)| {
    //             let px = i % width;
    //             let py = i / width;
    //             let u = px as f32 / width as f32;
    //             let v = py as f32 / height as f32;
    //             let dummy_ray = Ray::new(
    //                 Point3::new(sensor_size * (u - 0.5), sensor_size * (v - 0.5), 0.0),
    //                 Vec3::Z,
    //             );
    //             let rejected = bladed_aperture(aperture_radius, 6, dummy_ray);
    //             if !rejected {
    //                 *pixel = rgb_to_u32(255, 255, 255);
    //             } else {
    //                 *pixel = 0u32;
    //             }
    //         });

    //     window
    //         .update_with_buffer(&buffer.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
    //         .unwrap();
    // }

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;

    let direction_cache_radius_bins = 1024;
    let direction_cache_wavelength_bins = 1024;

    let mut direction_cache = RadialSampler::new(
        SQRT_2 * sensor_size / 2.0, // diagonal.
        direction_cache_radius_bins,
        direction_cache_wavelength_bins,
        wavelength_bounds,
        film_position,
        &lens_assembly,
        lens_zoom,
        |aperture_radius, ray| bladed_aperture(aperture_radius, 6, ray),
        heat_bias,
        sensor_size,
    );

    let mut last_pressed_hotkey = Key::A;
    let mut wavelength_sweep: f32 = 0.0;
    let mut wavelength_sweep_speed = 0.001;
    let mut texture_scale = 10.0;
    let mut efficiency = 0.0;
    let efficiency_heat = 0.99;
    let mut mode = Mode::PinLight;
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
                        "{:?}, {:?}",
                        film_position,
                        lens_assembly.total_thickness_at(lens_zoom)
                    );
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
                    film_position += 1.0 * config_direction;
                    println!(
                        "{:?}, {:?}",
                        film_position,
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
                .update_with_buffer(&window_pixels.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
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
                Mode::PinLight => Mode::Direction,
                Mode::Direction => Mode::Texture,
            };
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
                film_position,
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
            let origin = Point3::new(0.0, 0.0, film_position);
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

        // let lambda = wavelength_bounds.span() * random::<f32>() + wavelength_bounds.lower;

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
                    film_position,
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
                        let t = (wall_position - pupil_ray.origin.z()) / pupil_ray.direction.z();
                        let point_at_10 = pupil_ray.point_at_parameter(t);
                        let uv = (
                            (point_at_10.x().abs() / texture_scale) % 1.0,
                            (point_at_10.y().abs() / texture_scale) % 1.0,
                        );

                        match mode {
                            // // texture based
                            Mode::Texture => {
                                let m = textures[0].eval_at(lambda, uv);
                                let energy = tau * m * 3.0;
                                *pixel +=
                                    XYZColor::from(SingleWavelength::new(lambda, energy.into()));
                            }
                            // // spot light based
                            Mode::PinLight => {
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
                            Mode::Direction => {
                                *pixel = XYZColor::new(
                                    (1.0 + v.x()) * (1.0 + v.w()),
                                    (1.0 + v.y()) * (1.0 + v.w()),
                                    (1.0 + v.z()) * (1.0 + v.w()),
                                );
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
            .update_with_buffer(&window_pixels.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
