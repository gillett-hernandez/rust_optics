use std::collections::HashMap;
use std::f32::consts::SQRT_2;
use std::f32::EPSILON;
use std::ops::RangeInclusive;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use std::thread;
use std::{f32::consts::TAU, fs::File, io::Read};

use ::math::{f32x4, random_cosine_direction, PI};
// use crate::math::Sample2D;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
// use packed_simd::f32x4;
// use rand::prelude::*;
use eframe::egui;
// use egui::prelude::*;
use crossbeam::channel::{unbounded, Receiver, Sender};
use optics::aperture::{Aperture, ApertureEnum, CircularAperture, SimpleBladedAperture};
use optics::lens_sampler::RadialSampler;
use rayon::prelude::*;

use crate::math::{SingleWavelength, XYZColor};
use subcrate::{film::Film, parsing::*};
// use lens_sampler::RadialSampler;
use optics::misc::{Cycle, SceneMode, ViewMode};
use optics::*;

use crate::math::spectral::BOUNDED_VISIBLE_RANGE;

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

#[derive(Clone)]
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
    maybe_receiver: Option<Receiver<(String, Command)>>,
    // reporting data
    pub efficiency: f32,
    pub total_samples: usize,

    // dummy only:
    pub dirty: bool,

    pub halt: Arc<AtomicBool>,
}

impl SimulationState {
    pub fn data_update(&mut self, message: (String, Command)) {
        if self.maybe_sender.is_none() {
            // in puppet
            match message {
                (target, Command::ChangeFloat(v)) if target.starts_with("aperture_radius") => {
                    println!("changed aperture_radius = {}", v);
                    self.aperture_radius = v;
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("sensor_size") => {
                    println!("changed sensor_size = {}", v);
                    self.sensor_size = v;
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("film_position") => {
                    println!("changed film_position = {}", v);
                    self.film_position = v;
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("heat") => {
                    println!("changed solver heat = {}", v);
                    self.heat_bias = v;
                }
                (target, Command::Advance) if target.starts_with("view_mode") => {
                    self.view_mode = self.view_mode.cycle();
                    self.dirty = true;
                    println!("scene mode is now {:?}", self.view_mode);
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
                        _ => {
                            println!();
                        }
                    }
                }
                (target, Command::Advance) if target.starts_with("scene_mode") => {
                    self.scene_mode = self.scene_mode.cycle();
                    println!("scene mode is now {:?}", self.scene_mode);
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("scene_mode") => {
                    assert!(target.find('.') == Some("scene_mode".len()));
                    let tail = &target["scene_mode".len() + 1..];
                    print!("got scene mode update command, (tail = {})", tail);
                    match &mut self.scene_mode {
                        SceneMode::TexturedWall {
                            distance,
                            texture_scale,
                        } => {
                            if tail.starts_with("distance") {
                                println!("distance = {}", v);
                                *distance = v;
                                self.dirty = true;
                            } else if tail.starts_with("texture_scale") {
                                println!("texture_scale = {}", v);
                                *texture_scale = v;
                                self.dirty = true;
                            }
                        }
                        SceneMode::SpotLight { pos, size, span } => match tail {
                            "pos.x" => {
                                pos.0 = pos.0.replace(0, v);
                                println!("pos = {:?}", pos.0);
                                self.dirty = true;
                            }
                            "pos.y" => {
                                pos.0 = pos.0.replace(1, v);
                                println!("pos = {:?}", pos.0);
                                self.dirty = true;
                            }
                            "pos.z" => {
                                pos.0 = pos.0.replace(2, v);
                                println!("pos = {:?}", pos.0);
                                self.dirty = true;
                            }
                            "size" => {
                                println!();
                                if v < 0.0 {
                                    println!("attempted to change size to some nonsensical value, ignoring.\nsize should be above 0");
                                    return;
                                }
                                *size = v;
                                self.dirty = true;
                            }
                            "span" => {
                                println!();
                                if v < 0.0 || v >= 1.0 {
                                    println!("attempted to change span to some nonsensical value, ignoring.\nspan should be between 0 and 1, where near 1 cooresponds to a very focused spotlight.");
                                    return;
                                }
                                *span = v;
                                self.dirty = true;
                            }
                            _ => {
                                println!("but failed to match to subtarget");
                            }
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
                    .speed(0.1),
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

            ui.label("solver heat");
            let response = ui.add(
                egui::DragValue::new(&mut self.heat_bias)
                    .clamp_range(RangeInclusive::new(0.0, self.heat_cap))
                    .speed(0.1),
            );
            if response.changed() {
                sender
                    .try_send(("heat".into(), Command::ChangeFloat(self.heat_bias)))
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
                    pos.0 = f32x4::new(x, y, z, 0.0);

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
                            .unwrap()
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
                ViewMode::Film => {}
            }

            let response = ui.add(egui::Button::new("clear film"));
            if response.clicked() {
                sender
                    .try_send((String::from("clear film"), Command::Advance))
                    .unwrap();
            }

            let response = ui.add(egui::Button::new("clear direction cache"));
            if response.clicked() {
                sender
                    .try_send((String::from("clear direction cache"), Command::Advance))
                    .unwrap();
            }
            let response = ui.add(egui::Button::new("clear both"));
            if response.clicked() {
                sender
                    .try_send((String::from("clear both"), Command::Advance))
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

            let response = ui.add(egui::Button::new("halt"));
            if response.clicked() {
                self.halt.store(true, std::sync::atomic::Ordering::Relaxed);
                // this is definitely not the best way but i'm not sure how else to automatically close the egui window when the other window
                panic!();
            }
            if self.halt.load(std::sync::atomic::Ordering::Relaxed) {
                panic!();
            }
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
    use subcrate::tonemap::{sRGB, Tonemapper};

    println!("{:?}", opt);
    let width = opt.width;
    let height = opt.height;

    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .unwrap();

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
    let mut window = Window::new(
        "forward tracing",
        width,
        height,
        WindowOptions {
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut film = Film::new(width, height, XYZColor::BLACK);
    let mut window_pixels = Film::new(width, height, 0u32);

    // Limit to max ~144 fps update rate
    let width = film.width;

    let frame_dt = 6944.0 / 1000000.0;

    let scene = get_scene("textures.toml").unwrap();

    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

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
    let mut lens_zoom = 0.0;
    let mut wall_position = 5000.0;
    let mut texture_scale = 1.0;

    let mut samples_per_iteration = 5usize;
    let mut total_samples = 0;
    let mut focal_distance_suggestion = None;
    let mut focal_distance_vec: Vec<f32> = Vec::new();
    let mut variance: f32 = 0.0;
    let mut stddev: f32 = 0.0;

    let direction_cache_radius_bins = 512;
    let direction_cache_wavelength_bins = 512;

    let mut direction_cache = RadialSampler::new(
        SQRT_2 * local_simulation_state.sensor_size / 2.0, // diagonal.
        direction_cache_radius_bins,
        direction_cache_wavelength_bins,
        wavelength_bounds,
        local_simulation_state.film_position,
        &lens_assembly,
        lens_zoom,
        &local_simulation_state.aperture,
        local_simulation_state.heat_bias,
        local_simulation_state.sensor_size,
    );

    let mut wavelength_sweep: f32 = 0.0;
    let mut wavelength_sweep_speed = 0.001;
    let mut efficiency = 0.0;
    let efficiency_heat = 0.99;
    let mut paused = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if local_simulation_state
            .halt
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            break;
        }
        let mut clear_film = false;
        let mut clear_direction_cache = false;

        for message in receiver.try_iter() {
            if message.0.starts_with("clear film") || message.0.starts_with("clear both") {
                clear_film = true;
            }
            if message.0.starts_with("clear direction cache") || message.0.starts_with("clear both")
            {
                clear_direction_cache = true;
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
                .update_with_buffer(&window_pixels.buffer, width, height)
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
                SQRT_2 * local_simulation_state.sensor_size / 2.0, // diagonal.
                direction_cache_radius_bins,
                direction_cache_wavelength_bins,
                wavelength_bounds,
                local_simulation_state.film_position,
                &lens_assembly,
                lens_zoom,
                &local_simulation_state.aperture,
                local_simulation_state.heat_bias,
                local_simulation_state.sensor_size,
            );

            // autofocus:
            {
                let n = 25;
                let origin = Point3::new(0.0, 0.0, local_simulation_state.film_position);
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
                        let result = lens_assembly.trace_forward(
                            lens_zoom,
                            Input::new(ray, lambda / 1000.0),
                            1.0,
                            |e| {
                                (
                                    local_simulation_state
                                        .aperture
                                        .intersects(local_simulation_state.aperture_radius, e),
                                    false,
                                )
                            },
                        );
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
                    println!(
                        "focal distance suggestion: {}. stddev = {}",
                        focal_distance_suggestion.unwrap(),
                        stddev
                    );
                }
            }
        }

        let srgb_tonemapper = sRGB::new(&film, 1.0);

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
                let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                match local_simulation_state.view_mode {
                    ViewMode::Film => {
                        let central_point = Point3::new(
                            ((px as f32 + 0.5) / width as f32 - 0.5)
                                * local_simulation_state.sensor_size,
                            ((py as f32 + 0.5) / height as f32 - 0.5)
                                * local_simulation_state.sensor_size,
                            local_simulation_state.film_position,
                        );
                        for _ in 0..samples_per_iteration {
                            let v;
                            let s0 = sampler.draw_2d();
                            let [mut x, mut y, z, _]: [f32; 4] = central_point.0.into();
                            x += (s0.x - 0.5) / width as f32 * local_simulation_state.sensor_size;
                            y += (s0.y - 0.5) / height as f32 * local_simulation_state.sensor_size;

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
                            let result = lens_assembly.trace_forward(
                                lens_zoom,
                                Input::new(ray, lambda / 1000.0),
                                1.0,
                                |e| {
                                    (
                                        local_simulation_state
                                            .aperture
                                            .intersects(local_simulation_state.aperture_radius, e),
                                        false,
                                    )
                                },
                            );
                            if let Some(Output {
                                ray: pupil_ray,
                                tau,
                            }) = result
                            {
                                successes += 1;

                                match local_simulation_state.scene_mode {
                                    // // texture based
                                    // ignore because texture scale is used across multiple of these entries
                                    SceneMode::TexturedWall {
                                        distance,
                                        texture_scale,
                                    } => {
                                        let t = (distance - pupil_ray.origin.z())
                                            / pupil_ray.direction.z();
                                        let point_at_wall = pupil_ray.point_at_parameter(t);
                                        let uv = (
                                            (point_at_wall.x().abs() / texture_scale),
                                            (point_at_wall.y().abs() / texture_scale),
                                        );
                                        if (0.0..1.0).contains(&uv.0) && (0.0..1.0).contains(&uv.1)
                                        {
                                            let m = textures[0].eval_at(lambda, uv);
                                            let energy = tau * m * 3.0;
                                            *pixel += XYZColor::from(SingleWavelength::new(
                                                lambda,
                                                energy.into(),
                                            ));
                                        }
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
                                        let m = if (uv.0 - 0.5).powi(2) + (uv.1 - 0.5).powi(2)
                                            < 0.001
                                        {
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
                                        *pixel += XYZColor::from(SingleWavelength::new(
                                            lambda,
                                            energy.into(),
                                        ));
                                    }

                                    SceneMode::SpotLight { pos, size, span } => {
                                        let t = (pos.z() - pupil_ray.origin.z())
                                            / pupil_ray.direction.z();
                                        let point_at_light_z = pupil_ray.point_at_parameter(t);
                                        let m = if (point_at_light_z.x() - pos.x()).powi(2)
                                            + (point_at_light_z.y() - pos.y()).powi(2)
                                            < size
                                        {
                                            // if position matches
                                            if pupil_ray.direction.z().abs() > span {
                                                // if direction matches
                                                1.0
                                            } else {
                                                0.0
                                            }
                                        } else {
                                            0.0
                                        };
                                        let energy = tau * m * 3.0;
                                        *pixel += XYZColor::from(SingleWavelength::new(
                                            lambda,
                                            energy.into(),
                                        ));
                                    }
                                };
                            }
                        }
                    }
                    ViewMode::SpotOnFilm(x, y) => {
                        let central_point = Point3::new(x, y, local_simulation_state.film_position);

                        // figure out which mapping to use for pixels.
                        // for now, just cosine weghted hemisphere

                        let sample = sampler.draw_2d();
                        let (mut u, mut v) = (
                            (px as f32 + sample.x) / width as f32,
                            (py as f32 + sample.y) / height as f32,
                        );

                        // remap u and v such that forward directions are in the center of the screen

                        // in random_cosine_direction, u controls the angle and v controls the "altitude"
                        u -= 0.5;
                        v -= 0.5;
                        let radial_distance = (u.hypot(v) / SQRT_2 / 2.0).clamp(0.0, 1.0 - EPSILON);
                        let angle = ((u.atan2(v) + PI) / TAU).clamp(0.0, 1.0 - EPSILON);
                        let dir = random_cosine_direction(Sample2D::new(angle, radial_distance));
                        // TODO: add a way to visualize whether the current pixel would have been sampled by the direction cache
                        // direction_cache.cache.at_uv(uv)

                        let ray = Ray::new(central_point, dir);
                        attempts += 1;
                        let result = lens_assembly.trace_forward(
                            lens_zoom,
                            Input::new(ray, lambda / 1000.0),
                            1.0,
                            |e| {
                                (
                                    local_simulation_state
                                        .aperture
                                        .intersects(local_simulation_state.aperture_radius, e),
                                    false,
                                )
                            },
                        );
                        if let Some(Output {
                            ray: pupil_ray,
                            tau,
                        }) = result
                        {
                            successes += 1;

                            match local_simulation_state.scene_mode {
                                SceneMode::PinLight => {
                                    // using this as a debug scene, just to view which directions actually get through the lens

                                    // this is super jank but it'll get something visual
                                    let mut would_have_sampled = false;
                                    for _ in 0..100 {
                                        if direction_cache.sample(
                                            lambda,
                                            central_point,
                                            sampler.draw_2d(),
                                            sampler.draw_1d(),
                                        ) * dir
                                            > 0.99
                                        {
                                            // aligned enough
                                            would_have_sampled = true;
                                        }
                                    }
                                    if would_have_sampled {
                                        *pixel += XYZColor::from(SingleWavelength::new(
                                            620.0,
                                            1.0.into(),
                                        ));
                                    }
                                    *pixel +=
                                        XYZColor::from(SingleWavelength::new(lambda, tau.into()));
                                }
                                SceneMode::TexturedWall {
                                    distance,
                                    texture_scale,
                                } => {
                                    let t =
                                        (distance - pupil_ray.origin.z()) / pupil_ray.direction.z();
                                    let point_at_wall = pupil_ray.point_at_parameter(t);
                                    let uv = (
                                        (point_at_wall.x().abs() / texture_scale),
                                        (point_at_wall.y().abs() / texture_scale),
                                    );
                                    if (0.0..1.0).contains(&uv.0) && (0.0..1.0).contains(&uv.1) {
                                        let m = textures[0].eval_at(lambda, uv);
                                        let energy = tau * m * 3.0;
                                        *pixel += XYZColor::from(SingleWavelength::new(
                                            lambda,
                                            energy.into(),
                                        ));
                                    }
                                }
                                SceneMode::SpotLight { pos, size, span } => {
                                    let t =
                                        (pos.z() - pupil_ray.origin.z()) / pupil_ray.direction.z();
                                    let point_at_light_z = pupil_ray.point_at_parameter(t);
                                    let m = if (point_at_light_z.x() - pos.x()).powi(2)
                                        + (point_at_light_z.y() - pos.y()).powi(2)
                                        < size
                                    {
                                        // if position matches
                                        if pupil_ray.direction.z().abs() > span {
                                            // if direction matches
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    } else {
                                        0.0
                                    };
                                    let energy = tau * m * 3.0;
                                    *pixel += XYZColor::from(SingleWavelength::new(
                                        lambda,
                                        energy.into(),
                                    ));
                                }
                            }
                        }
                    }
                }
                (successes, attempts)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

        if b > 0 {
            efficiency =
                (efficiency_heat) * efficiency + (1.0 - efficiency_heat) * (a as f32 / b as f32);
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
            .update_with_buffer(&window_pixels.buffer, width, height)
            .unwrap();
    }
    local_simulation_state
        .halt
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

fn main() {
    let opt = Opt::from_args();

    // let ui = egui();
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(500.0, 900.0)),
        ..Default::default()
    };

    let mut camera_file = File::open(format!("data/cameras/{}.txt", opt.lens)).unwrap();
    let mut camera_spec = String::new();
    camera_file.read_to_string(&mut camera_spec).unwrap();

    let (lenses, _last_ior, _last_vno) = parse_lenses_from(&camera_spec);
    let lens_assembly = LensAssembly::new(&lenses);
    let original_aperture_radius = lens_assembly.aperture_radius();

    let lens_zoom = 0.0;
    let halt = Arc::new(AtomicBool::new(false));

    let local_simulation_state = SimulationState {
        heat_bias: 0.01,
        heat_cap: 10.0,
        aperture_radius: original_aperture_radius / 3.0,
        max_aperture_radius: original_aperture_radius,
        sensor_size: 35.0,
        max_sensor_size: 35.0,
        // aperture: ApertureEnum::CircularAperture(CircularAperture::default()),
        aperture: ApertureEnum::SimpleBladedAperture(SimpleBladedAperture::new(6, 1.3)),
        scene_mode: SceneMode::PinLight,
        view_mode: ViewMode::Film.cycle(),
        paused: false,
        samples: 1,
        film_position: -lens_assembly.total_thickness_at(lens_zoom),
        min_film_position: lens_assembly.lenses.last().unwrap().thickness_short
            - lens_assembly.total_thickness_at(lens_zoom),
        lens_zoom: 0.0,
        maybe_sender: None,
        maybe_receiver: None,
        dirty: false,
        efficiency: 0.0,
        total_samples: 0,
        halt,
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

    eframe::run_native(
        "Forward Tracer Control Panel",
        options,
        Box::new(|_cc| Box::new(simulation_state_egui)),
    );
}
