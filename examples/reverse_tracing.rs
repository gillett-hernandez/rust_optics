#![feature(portable_simd)]

use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::{Arc, RwLock};
use std::thread;
use std::{f32::consts::TAU, fs::File, io::Read};

use ::math::spectral::BOUNDED_VISIBLE_RANGE;
// use crate::math::Sample2D;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
// use packed_simd::f32x4;
// use rand::prelude::*;
use eframe::egui;
// use egui::prelude::*;
use crossbeam::channel::{unbounded, Receiver, Sender};
use optics::aperture::{Aperture, ApertureEnum, CircularAperture, SimpleBladedAperture};
use rayon::prelude::*;

use crate::dev::parsing::*;
use crate::vec2d::Vec2D;
use crate::math::{SingleWavelength, XYZColor};
// use lens_sampler::RadialSampler;
use optics::misc::{draw_line, project, Cycle, DrawMode, SceneMode, ViewMode};
use optics::*;

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Command {
    ChangeFloat(f32),
    ChangeInt(i32),
    Advance,
}

impl Command {
    pub fn as_int(self) -> Option<i32> {
        if let Self::ChangeInt(value) = self {
            Some(value)
        } else {
            None
        }
    }
    pub fn as_float(self) -> Option<f32> {
        if let Self::ChangeFloat(value) = self {
            Some(value)
        } else {
            None
        }
    }
}

impl From<f32> for Command {
    fn from(v: f32) -> Self {
        Command::ChangeFloat(v)
    }
}
impl From<i32> for Command {
    fn from(v: i32) -> Self {
        Command::ChangeInt(v)
    }
}

#[derive(Clone, Debug)]
pub struct SimulationState {
    maybe_sender: Option<Sender<(String, Command)>>,

    pub aperture_radius: f32,
    pub max_aperture_radius: f32,
    pub sensor_size: f32,
    pub max_sensor_size: f32,

    pub film_position: f32,
    pub min_film_position: f32,

    pub aperture: ApertureEnum,
    pub scene_mode: SceneMode,
    pub view_mode: ViewMode,

    pub heat_bias: f32,
    pub heat_cap: f32,
    pub samples: usize,

    pub lens_zoom: f32,
    pub paused: bool,
    //     "wavelength_sweep", // toggle

    //     "clear", // button
    //     "clear_film", // button

    //     "printout", // each output item needs to be bound to some variable and outputted directly
    //     "exit", // button
    // send control data from controller to mirror/puppet
    // receive reporting data from puppet to display in gui
    maybe_receiver: Option<Receiver<(String, Command)>>,
    // reporting data
    pub efficiency: f32,
    pub total_samples: usize,

    pub dirty: bool,
    // pub
}

impl SimulationState {
    pub fn data_update(&mut self, message: (String, Command)) {
        if self.maybe_sender.is_none() {
            // in puppet
            match message {
                m if m.0.starts_with("aperture_radius") => {
                    m.1.as_float().map(|float| self.aperture_radius = float);
                    self.dirty = true;
                }
                m if m.0.starts_with("sensor_size") => {
                    m.1.as_float().map(|float| self.sensor_size = float);
                    self.dirty = true;
                }
                m if m.0.starts_with("film_position") => {
                    m.1.as_float().map(|float| self.film_position = float);
                    self.dirty = true;
                }
                (target, Command::Advance) if target.starts_with("view_mode") => {
                    self.view_mode = self.view_mode.cycle();
                    println!("view mode is now {:?}", self.view_mode);
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("view_mode") => {
                    assert!(target.find('.') == Some("view_mode".len()));
                    let tail = &target["view_mode".len() + 1..];
                    print!("got view mode update command, (tail = {})", tail);
                    match &mut self.view_mode {
                        ViewMode::SpotOnFilm(x, y) => match tail {
                            "x" => {
                                println!("updating view mode position on data end, new x = {}", v);
                                *x = v;
                                self.dirty = true;
                            }
                            "y" => {
                                println!("updating view mode position on data end, new y = {}", v);
                                *y = v;
                                self.dirty = true;
                            }
                            _ => {
                                println!("but failed to match to subtarget");
                            }
                        },

                        ViewMode::XRay { bounds } => match tail {
                            "bounds.x_center" => {
                                let old_center = bounds.x.lower + bounds.x.span() / 2.0;
                                let adjustment = v - old_center;
                                bounds.x.lower += adjustment;
                                bounds.x.upper += adjustment;
                                self.dirty = true;
                            }
                            "bounds.x_span" => {
                                let old_span = bounds.x.span();
                                let adjustment = v - old_span;
                                // shrink or grow by `adjustment`
                                bounds.x.lower -= adjustment / 2.0;
                                bounds.x.upper += adjustment / 2.0;
                                self.dirty = true;
                            }
                            "bounds.y_center" => {
                                let old_center = bounds.y.lower + bounds.y.span() / 2.0;
                                let adjustment = v - old_center;
                                bounds.y.lower += adjustment;
                                bounds.y.upper += adjustment;
                                self.dirty = true;
                            }
                            "bounds.y_span" => {
                                let old_span = bounds.y.span();
                                let adjustment = v - old_span;
                                // shrink or grow by `adjustment`
                                bounds.y.lower -= adjustment / 2.0;
                                bounds.y.upper += adjustment / 2.0;
                                self.dirty = true;
                            }
                            _ => {
                                println!("but failed to match subtarget");
                            }
                        },
                        _ => {
                            println!();
                        }
                    }
                }
                (target, Command::Advance) if target.starts_with("scene_mode") => {
                    self.scene_mode = self.scene_mode.cycle();
                    println!("scene mode is now {:?}", self.scene_mode);
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("scene_mode") => {
                    // because length "12345" gives 5 but the index 5 is one further than the last index, we don't need to add one to skip the dot.
                    assert!(target.find('.') == Some("scene_mode".len()));
                    let tail = &target["scene_mode".len() + 1..];
                    match &mut self.scene_mode {
                        SceneMode::TexturedWall {
                            distance,
                            texture_scale,
                        } => {
                            if tail.starts_with("distance") {
                                *distance = v;
                            } else if tail.starts_with("texture_scale") {
                                *texture_scale = v;
                            }
                        }
                        SceneMode::SpotLight { pos, size, span } => match tail {
                            "pos.x" => {
                                pos.0[0] = v;
                            }
                            "pos.y" => {
                                pos.0[1] = v;
                            }
                            "pos.z" => {
                                pos.0[2] = v;
                            }
                            "size" => {
                                if v < 0.0 {
                                    println!("attempted to change size to some nonsensical value, ignoring.\nsize should be above 0");
                                    return;
                                }
                                *size = v;
                            }
                            "span" => {
                                if v < 0.0 || v >= 1.0 {
                                    println!("attempted to change span to some nonsensical value, ignoring.\nspan should be between 0 and 1, where near 1 cooresponds to a very focused spotlight.");
                                    return;
                                }
                                *span = v;
                                println!(
                                    "span is now {}, which corresponds to a max angle of {}",
                                    span,
                                    span.acos()
                                );
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                (target, command) => {
                    println!(
                        "received mutate command without a matching target, {}, {:?}",
                        target, command
                    );
                }
            }
        }
    }
}

impl eframe::App for SimulationState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let sender = self.maybe_sender.as_ref().unwrap();

            ui.label("aperture radius, mm");
            let response = ui.add(
                egui::DragValue::new(&mut self.aperture_radius)
                    .clamp_range(RangeInclusive::new(0.0, self.max_aperture_radius))
                    .speed(0.1 / self.max_aperture_radius),
            );
            if response.changed() {
                sender
                    .try_send((
                        "aperture_radius".into(),
                        Command::ChangeFloat(self.aperture_radius),
                    ))
                    .unwrap()
            }

            ui.label("sensor radius, mm");
            let response = ui.add(
                egui::DragValue::new(&mut self.sensor_size)
                    .clamp_range(RangeInclusive::new(1.0, self.max_sensor_size))
                    .speed(0.1 / self.max_sensor_size),
            );
            if response.changed() {
                sender
                    .try_send(("sensor_size".into(), Command::ChangeFloat(self.sensor_size)))
                    .unwrap()
            }

            ui.label("film position, mm");
            let response = ui.add(
                egui::DragValue::new(&mut self.film_position)
                    .clamp_range(RangeInclusive::new(-f32::INFINITY, self.min_film_position))
                    .speed(0.1),
            );
            if response.changed() {
                sender
                    .try_send((
                        "film_position".into(),
                        Command::ChangeFloat(self.film_position),
                    ))
                    .unwrap()
            }

            let response = ui.add(egui::Button::new("change scene"));
            if response.clicked() {
                self.scene_mode = self.scene_mode.cycle();
                sender
                    .try_send((String::from("scene_mode"), Command::Advance))
                    .unwrap();
            }

            ui.label(format!("scene mode is {:?}", self.scene_mode));
            match &mut self.scene_mode {
                SceneMode::TexturedWall {
                    distance,
                    texture_scale,
                } => {
                    ui.label("distance, mm");
                    let response = ui.add(
                        egui::DragValue::new(distance)
                            .clamp_range(RangeInclusive::new(0.0, f64::INFINITY)),
                    );
                    if response.changed() {
                        sender
                            .try_send((
                                "scene_mode.distance".into(),
                                Command::ChangeFloat(*distance),
                            ))
                            .unwrap()
                    }
                    ui.label("texture_scale");
                    let response = ui.add(
                        egui::DragValue::new(texture_scale)
                            .clamp_range(RangeInclusive::new(0.0, f64::INFINITY)),
                    );
                    if response.changed() {
                        sender
                            .try_send((
                                "scene_mode.texture_scale".into(),
                                Command::ChangeFloat(*texture_scale),
                            ))
                            .unwrap()
                    }
                }
                SceneMode::SpotLight { pos, size, span } => {
                    let [mut x, mut y, mut z, _]: [f32; 4] = pos.0.into();

                    let mut any_changed = false;

                    ui.label("pos.x");
                    let response = ui.add(egui::DragValue::new(&mut x).speed(0.01));
                    any_changed |= response.changed();

                    ui.label("pos.y");
                    let response = ui.add(egui::DragValue::new(&mut y).speed(0.01));
                    any_changed |= response.changed();

                    ui.label("pos.z");
                    let response = ui.add(egui::DragValue::new(&mut z).speed(0.01));
                    any_changed |= response.changed();

                    if any_changed {
                        sender
                            .try_send(("scene_mode.pos.x".into(), Command::ChangeFloat(x)))
                            .unwrap();
                        sender
                            .try_send(("scene_mode.pos.y".into(), Command::ChangeFloat(y)))
                            .unwrap();
                        sender
                            .try_send(("scene_mode.pos.z".into(), Command::ChangeFloat(z)))
                            .unwrap();
                    }
                    pos.0 = f32x4::from_array([x, y, z, 0.0]);

                    ui.label("size");
                    let response = ui.add(
                        egui::DragValue::new(size)
                            .clamp_range(RangeInclusive::new(0.0, f64::INFINITY)),
                    );
                    if response.changed() {
                        sender
                            .try_send(("scene_mode.size".into(), Command::ChangeFloat(*size)))
                            .unwrap()
                    }
                    ui.label("span");
                    let response = ui.add(egui::Slider::new(span, 0.0..=1.0));
                    if response.changed() {
                        sender
                            .try_send(("scene_mode.span".into(), Command::ChangeFloat(*span)))
                            .unwrap();
                        println!("sent span update");
                    }
                }
                SceneMode::PinLight => {}
            }

            let response = ui.add(egui::Button::new("change view mode"));
            if response.clicked() {
                sender
                    .try_send((String::from("view_mode"), Command::Advance))
                    .unwrap();
                self.view_mode = self.view_mode.cycle();
            }

            ui.label(format!("view mode is {:?}", self.view_mode));
            match &mut self.view_mode {
                ViewMode::SpotOnFilm(x, y) => {
                    let mut any_changed = false;

                    ui.label("x");
                    let response = ui.add(egui::DragValue::new(x).speed(0.01));
                    any_changed |= response.changed();

                    ui.label("y");
                    let response = ui.add(egui::DragValue::new(y).speed(0.01));
                    any_changed |= response.changed();

                    if any_changed {
                        sender
                            .try_send(("view_mode.x".into(), Command::ChangeFloat(*x)))
                            .unwrap();
                        sender
                            .try_send(("view_mode.y".into(), Command::ChangeFloat(*y)))
                            .unwrap();
                    }
                }
                ViewMode::XRay { bounds } => {
                    let mut any_changed = false;

                    let mut x = bounds.x.lower + bounds.x.span() / 2.0;
                    let mut x_span = bounds.x.span();
                    let mut y = bounds.y.lower + bounds.y.span() / 2.0;
                    let mut y_span = bounds.y.span();

                    ui.label("x center");
                    let response = ui.add(egui::DragValue::new(&mut x));
                    any_changed |= response.changed();

                    ui.label("x span");
                    let response = ui.add(egui::DragValue::new(&mut x_span));
                    any_changed |= response.changed();

                    ui.label("y center");
                    let response = ui.add(egui::DragValue::new(&mut y));
                    any_changed |= response.changed();

                    ui.label("y span");
                    let response = ui.add(egui::DragValue::new(&mut y_span));
                    any_changed |= response.changed();

                    if any_changed {
                        sender
                            .try_send(("view_mode.bounds.x_center".into(), Command::ChangeFloat(x)))
                            .unwrap();
                        sender
                            .try_send(("view_mode.bounds.y_center".into(), Command::ChangeFloat(y)))
                            .unwrap();
                        sender
                            .try_send((
                                "view_mode.bounds.x_span".into(),
                                Command::ChangeFloat(x_span),
                            ))
                            .unwrap();
                        sender
                            .try_send((
                                "view_mode.bounds.y_span".into(),
                                Command::ChangeFloat(y_span),
                            ))
                            .unwrap();
                        let old_center = bounds.x.lower + bounds.x.span() / 2.0;
                        let adjustment = x - old_center;
                        bounds.x.lower += adjustment;
                        bounds.x.upper += adjustment;

                        let old_center = bounds.y.lower + bounds.y.span() / 2.0;
                        let adjustment = y - old_center;
                        bounds.y.lower += adjustment;
                        bounds.y.upper += adjustment;

                        let old_span = bounds.x.span();
                        let adjustment = x_span - old_span;
                        // shrink or grow by `adjustment`
                        bounds.x.lower -= adjustment / 2.0;
                        bounds.x.upper += adjustment / 2.0;

                        let old_span = bounds.y.span();
                        let adjustment = y_span - old_span;
                        // shrink or grow by `adjustment`
                        bounds.y.lower -= adjustment / 2.0;
                        bounds.y.upper += adjustment / 2.0;
                    }
                }
                _ => {}
            }

            let response = ui.add(egui::Button::new("clear film"));
            if response.clicked() {
                sender
                    .try_send((String::from("clear"), Command::Advance))
                    .unwrap();
            }

            let receiver = self.maybe_receiver.as_ref().unwrap();
            for (target, command) in receiver.try_iter() {
                match target.as_str() {
                    "efficiency" => {
                        self.efficiency = command.as_float().unwrap();
                    }
                    "total_samples" => {
                        self.total_samples = command.as_int().unwrap() as usize;
                    }
                    _ => {
                        println!(
                            "reporting data was sent to a target that doesn't exist, {}",
                            target
                        );
                    }
                }
            }
            // ui.add(egui::tex)
            ui.label(format!("efficiency: {:?}", self.efficiency.to_string()));
            ui.label(format!("total_samples: {}", self.total_samples.to_string()));
        });
    }
}

fn run_simulation(
    opt: Opt,
    mut local_simulation_state: SimulationState,
    lens_assembly: LensAssembly,
    receiver: Receiver<(String, Command)>,
    sender: Sender<(String, Command)>,
) {
    use dev::tonemap::{sRGB, Tonemapper};

    println!("{:?}", opt);
    let window_width = opt.width;
    let window_height = opt.height;

    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "reverse tracing",
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

    let mut film = Vec2D::new(window_width, window_height, XYZColor::BLACK);
    let mut window_pixels = Vec2D::new(window_width, window_height, 0u32);

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    let width = film.width;

    let frame_dt = 6944.0 / 1000000.0;

    let scene = get_scene("textures.toml").unwrap();

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
    let mut textures: Vec<TexStack> = Vec::new();
    for tex in scene.textures {
        textures.push(parse_texture_stack(tex.clone(), wavelength_bounds));
    }

    let wall_texture = &textures[0];

    let mut samples_per_iteration = 100usize;
    let mut total_samples = 0;
    // let mut focal_distance_suggestion = None;
    // let mut focal_distance_vec: Vec<f32> = Vec::new();
    // let mut variance: f32 = 0.0;
    // let mut stddev: f32 = 0.0;

    let mut efficiency = 0.0;
    let efficiency_heat = 0.99;

    let mut paused = local_simulation_state.paused;
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

    'outer: while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut clear_film = false;

        for message in receiver.try_iter() {
            if message.0.as_str() == "clear" {
                clear_film = true;
            }
            local_simulation_state.data_update(message);
            if local_simulation_state.dirty {
                local_simulation_state.dirty = false;
                clear_film = true;
            }
            paused = local_simulation_state.paused;
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

        let srgb_tonemapper = sRGB::new(&film, 1.0);

        // autofocus:
        // {
        //     let n = 25;
        //     let origin = Point3::new(0.0, 0.0, local_simulation_state.film_position);
        //     let direction = Point3::new(
        //         0.0,
        //         lens_assembly.lenses.last().unwrap().housing_radius,
        //         0.0,
        //     ) - origin;
        //     let maximum_angle = -(direction.y() / direction.z()).atan();

        //     focal_distance_vec.clear();
        //     for i in 0..n {
        //         // choose angle to shoot ray from (0.0, 0.0, wall_position)
        //         let angle = ((i as f32 + 0.5) / n as f32) * maximum_angle;
        //         let ray = Ray::new(origin, Vec3::new(0.0, angle.sin(), angle.cos()));
        //         // println!("{:?}", ray);
        //         for w in 0..10 {
        //             let lambda =
        //                 wavelength_bounds.lower + (w as f32 / 10.0) * wavelength_bounds.span();
        //             let result = lens_assembly.trace_forward(
        //                 local_simulation_state.lens_zoom,
        //                 &Input::new( ray, lambda ),
        //                 1.0,
        //                 |e| {
        //                     (
        //                         bladed_aperture(local_simulation_state.aperture_radius, 6, e),
        //                         false,
        //                     )
        //                 },
        //             );
        //             if let Some(Output { ray: pupil_ray, .. }) = result {
        //                 let dt = (-pupil_ray.origin.y()) / pupil_ray.direction.y();
        //                 let point = pupil_ray.point_at_parameter(dt);
        //                 // println!("{:?}", point);

        //                 if point.z().is_finite() {
        //                     focal_distance_vec.push(point.z());
        //                 }
        //             }
        //         }
        //     }
        //     if focal_distance_vec.len() > 0 {
        //         let avg: f32 =
        //             focal_distance_vec.iter().sum::<f32>() / focal_distance_vec.len() as f32;
        //         focal_distance_suggestion = Some(avg);
        //         variance = focal_distance_vec
        //             .iter()
        //             .map(|e| (avg - *e).powf(2.0))
        //             .sum::<f32>()
        //             / focal_distance_vec.len() as f32;
        //         stddev = variance.sqrt();
        //     }
        // }

        total_samples += samples_per_iteration;

        let mut sampler = RandomSampler::new();

        let (mut successes, mut attempts) = (0, 0);

        for _ in 0..samples_per_iteration {
            let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
            let (ray, le) = match local_simulation_state.scene_mode {
                // diffuse emitter texture
                SceneMode::TexturedWall {
                    distance: wall_position,
                    texture_scale,
                } => {
                    // ray is generated according to texture scale.
                    // 4 possible quadrants.
                    let sample = sampler.draw_2d();
                    let (rx, ry) = (sample.x - 0.5, sample.y - 0.5);
                    // let (r, phi) = (
                    //     sampler.draw_1d().x.sqrt() * lens_assembly.lenses[0].housing_radius,
                    //     sampler.draw_1d().x * TAU,
                    // );

                    // let point_on_lens = Point3::new(r * phi.cos(), r * phi.sin(), 0.0);
                    let point_on_lens = sample_point_on_lens(
                        lens_assembly.lenses[0].radius,
                        lens_assembly.lenses[0].housing_radius,
                        sampler.draw_2d(),
                    );
                    let point_on_texture =
                        Point3::new(texture_scale * rx, texture_scale * ry, wall_position);
                    let v = (point_on_lens - point_on_texture).normalized();

                    (
                        Ray::new(point_on_texture, v),
                        wall_texture.eval_at(lambda, (sample.x, sample.y)),
                    )
                }
                // parallel light
                SceneMode::SpotLight { pos, span, size } => {
                    // 4 quadrants.

                    let (r, phi) = (sampler.draw_1d().x.sqrt() * size, sampler.draw_1d().x * TAU);

                    let (px, py) = (pos.x() + r * phi.cos(), pos.y() + r * phi.sin());

                    let ray_origin = Point3::new(px, py, pos.z());

                    // let (r, phi) = (sampler.draw_1d().x.sqrt() * span, sampler.draw_1d().x * TAU);

                    // let (px, py) = (r * phi.cos(), r * phi.sin());
                    // // TODO: use span and size to determine angle
                    // let dir = Vec3::new(px, py, -1.0).normalized();
                    // span is essentially the lower limit of the cosine of the angle i'm willing to sample.
                    let max_angle = span.acos();
                    let angle = sampler.draw_1d().x.sqrt() * max_angle;

                    let other_angle = sampler.draw_1d().x * TAU;
                    let dir = Vec3::new(
                        angle.sin() * other_angle.cos(),
                        angle.sin() * other_angle.sin(),
                        -angle.cos(),
                    );

                    // sample lens
                    // let point_on_lens = Point3::new(r * phi.cos(), r * phi.sin(), 0.0);
                    // let dir = (point_on_lens - ray_origin).normalized();

                    // // filter based on span
                    // if dir.z().abs() < span {
                    //     continue;
                    // }

                    (Ray::new(ray_origin, dir), 1.0)
                }
                SceneMode::PinLight => {
                    // 4 quadrants.

                    let (r, phi) = (sampler.draw_1d().x.sqrt(), sampler.draw_1d().x * TAU);

                    let (dx, dy) = (r * phi.cos(), r * phi.sin());
                    let (px, py) = (0.0, 0.0);

                    // TODO: add parameter to control pinlight stuff.
                    (
                        Ray::new(
                            Point3::new(px, py, 10.0),
                            Vec3::new(dx, dy, -10.0).normalized(),
                        ),
                        1.0,
                    )
                }
            };
            // println!("{:?}", ray);

            attempts += 1;
            // do actual tracing through lens for film sample

            match local_simulation_state.view_mode {
                ViewMode::Film | ViewMode::SpotOnFilm(_, _) => {
                    let result = lens_assembly.trace_reverse(
                        local_simulation_state.lens_zoom,
                        Input::new(ray, lambda / 1000.0),
                        1.04,
                        |e| {
                            (
                                local_simulation_state
                                    .aperture
                                    .intersects(local_simulation_state.aperture_radius, e),
                                false,
                            )
                        },
                        drop,
                    );
                    if let Some(Output {
                        ray: pupil_ray,
                        tau,
                    }) = result
                    {
                        successes += 1;
                        let t = (local_simulation_state.film_position - pupil_ray.origin.z())
                            / pupil_ray.direction.z();
                        let point_at_film = pupil_ray.point_at_parameter(t);
                        let uv = (
                            ((point_at_film.x() / local_simulation_state.sensor_size) + 1.0) / 2.0,
                            ((point_at_film.y() / local_simulation_state.sensor_size) + 1.0) / 2.0,
                        );
                        if uv.0 < 1.0 && uv.1 < 1.0 && uv.0 > 0.0 && uv.1 > 0.0 {
                            film.write_at(
                                (uv.0 * window_width as f32) as usize,
                                (uv.1 * window_height as f32) as usize,
                                film.at(
                                    (uv.0 * window_width as f32) as usize,
                                    (uv.1 * window_height as f32) as usize,
                                ) + XYZColor::from(SingleWavelength::new(
                                    lambda,
                                    (le * tau).into(),
                                )),
                            );
                        }
                    }
                }
                ViewMode::XRay { bounds } => {
                    let swizzle_project =
                        |pt| project(pt, Vec3::X, |v| simd_swizzle!(v, [2, 1, 0, 3]));
                    let invert = |pt: Point3| Point3(-pt.0);

                    let mut segments = Vec::new();
                    let result = lens_assembly.trace_reverse(
                        local_simulation_state.lens_zoom,
                        Input::new(ray, lambda / 1000.0),
                        1.04,
                        |e| {
                            (
                                local_simulation_state
                                    .aperture
                                    .intersects(local_simulation_state.aperture_radius, e),
                                false,
                            )
                        },
                        |(a, b, tau)| {
                            segments.push((a, b, tau));
                        },
                    );
                    if let Some(Output {
                        ray: pupil_ray,
                        tau,
                    }) = result
                    {
                        let t = (local_simulation_state.film_position - pupil_ray.origin.z())
                            / pupil_ray.direction.z();
                        if t <= 0.0 {
                            continue;
                        }
                        let point_at_film = pupil_ray.point_at_parameter(t);
                        let uv = (
                            ((point_at_film.x() / local_simulation_state.sensor_size) + 1.0) / 2.0,
                            ((point_at_film.y() / local_simulation_state.sensor_size) + 1.0) / 2.0,
                        );

                        if uv.0 < 1.0 && uv.1 < 1.0 && uv.0 > 0.0 && uv.1 > 0.0 {
                            // println!("point at film {:?}", point_at_film);
                            successes += 1;
                            for (a, b, tau) in segments {
                                draw_line(
                                    &mut film,
                                    bounds,
                                    swizzle_project(invert(a)),
                                    swizzle_project(invert(b)),
                                    lambda,
                                    tau,
                                    DrawMode::XiaolinWu,
                                );
                            }
                            draw_line(
                                &mut film,
                                bounds,
                                swizzle_project(pupil_ray.origin),
                                swizzle_project(point_at_film),
                                lambda,
                                tau,
                                DrawMode::XiaolinWu,
                            );
                        }
                    }
                }
            }
        }
        if attempts > 0 {
            efficiency = (efficiency_heat) * efficiency
                + (1.0 - efficiency_heat) * (successes as f32 / attempts as f32);
            sender
                .try_send((String::from("efficiency"), efficiency.into()))
                .unwrap();
        }
        sender
            .try_send((String::from("total_samples"), (total_samples as i32).into()))
            .unwrap();

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

fn main() {
    let opt = Opt::from_args();

    // let ui = egui();
    let options = eframe::NativeOptions {
        // initial_window_size: Some(egui::vec2(500.0, 900.0)),
        ..Default::default()
    };

    let mut camera_file = File::open(format!("data/cameras/{}.txt", opt.lens)).unwrap();
    let mut camera_spec = String::new();
    camera_file.read_to_string(&mut camera_spec).unwrap();

    let (lenses, _last_ior, _last_vno) = parse_lenses_from(&camera_spec);
    let lens_assembly = LensAssembly::new(&lenses);
    let original_aperture_radius = lens_assembly.aperture_radius();

    let lens_zoom = 0.0;

    let local_simulation_state = SimulationState {
        heat_bias: 0.01,
        heat_cap: 10.0,
        aperture_radius: original_aperture_radius / 3.0,
        max_aperture_radius: original_aperture_radius,
        sensor_size: 35.0,
        max_sensor_size: 35.0,
        aperture: ApertureEnum::CircularAperture(CircularAperture::default()),
        // aperture: ApertureEnum::SimpleBladedAperture(SimpleBladedAperture::new(6, 0.5)),
        scene_mode: SceneMode::PinLight,
        view_mode: ViewMode::XRay {
            bounds: Bounds2D::new((-400.0, 200.0).into(), (-200.0, 200.0).into()),
        },
        paused: false,
        samples: 1,
        film_position: -lens_assembly.total_thickness_at(lens_zoom),
        min_film_position: lens_assembly.lenses.last().unwrap().thickness_short
            - lens_assembly.total_thickness_at(lens_zoom),
        lens_zoom: 0.0,
        maybe_sender: None,
        maybe_receiver: None,
        efficiency: 0.0,
        total_samples: 0,
        dirty: false,
    };

    let (controller_sender, controller_receiver) = unbounded();
    let (reporting_sender, reporting_receiver) = unbounded();

    let simulation_state_egui = SimulationState {
        maybe_sender: Some(controller_sender),
        maybe_receiver: Some(reporting_receiver),
        ..local_simulation_state.clone()
    };
    let _ = thread::spawn(move || {
        run_simulation(
            opt,
            local_simulation_state,
            lens_assembly,
            controller_receiver,
            reporting_sender,
        )
    });

    let _ = eframe::run_native(
        "Reverse Tracer Control Panel",
        options,
        Box::new(|_cc| Box::new(simulation_state_egui)),
    );
}
