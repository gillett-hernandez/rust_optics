use std::f32::EPSILON;
use std::f32::consts::{FRAC_PI_2, PI, SQRT_2};
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
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
use crossbeam::channel::{Receiver, Sender, unbounded};
use optics::aperture::{Aperture, ApertureEnum, SimpleBladedAperture};
use optics::dev::parsing::*;
use optics::lens_sampler::RadialSampler;
use optics::math::*;
use optics::poly::{PolyAssembly, PolyLens};
use optics::vec2d::Vec2D;
use rayon::prelude::*;
// use lens_sampler::RadialSampler;
use optics::misc::{Cycle, DrawMode, SceneMode, ViewMode, draw_line, project};
use optics::*;

use structopt::StructOpt;

// The library is generic over the SIMD backend; this binary monomorphizes on
// the AVX2+FMA backend. These aliases shadow the generic re-exports from
// `optics::math::*` (and `SceneMode` from `optics::misc`) so the body below can
// use the bare type names.
type Backend = thermite::backend::x86_v3::X86V3;
type Vec3 = optics::math::Vec3<Backend>;
type Point3 = optics::math::Point3<Backend>;
type Ray = optics::math::Ray<Backend>;
type F32x4 = optics::math::F32x4<Backend>;
type XYZColor = optics::math::XYZColor<Backend>;

/// Surrounding-medium (world-side) IOR. Used by every trace *and* the poly model
/// build so poly mode and traced mode agree; previously the Film path used 1.04
/// while the poly model / XRay used 1.0, which shifted focus/scale between modes.
const ATMOSPHERE_IOR: f32 = 1.0;

/// Which way rays are traced through the assembly.
///   - `FromFilm`  : forward tracing, sensor -> scene (the radial sampler applies).
///   - `FromScene` : reverse tracing, scene -> sensor.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TraceDirection {
    FromFilm,
    FromScene,
}

impl TraceDirection {
    pub fn toggle(self) -> Self {
        match self {
            TraceDirection::FromFilm => TraceDirection::FromScene,
            TraceDirection::FromScene => TraceDirection::FromFilm,
        }
    }
}

/// How the focal-distance sweep probes the assembly.
///   - `FromFilm`     : rays diverge from the on-axis film point; report where they
///                      reconverge out in the world (the object-side focus distance).
///   - `FromInfinity` : an object at infinity; report the rear focal plane (the
///                      image-side point where collimated input focuses, i.e. where
///                      the sensor should sit to focus at infinity), per wavelength.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FocalMode {
    FromFilm,
    FromInfinity,
}

impl Cycle for FocalMode {
    fn cycle(self) -> Self {
        match self {
            FocalMode::FromFilm => FocalMode::FromInfinity,
            FocalMode::FromInfinity => FocalMode::FromFilm,
        }
    }
}

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
    pub use_sampler: bool,
    /// When set, the Film view evaluates the polynomial lens model
    /// ([`PolyLens::map_forward`]) instead of `trace_forward`, and the XRay view
    /// overlays the poly-predicted exit ray (in red) on the real traced path.
    pub use_poly: bool,
    /// In poly mode, whether to apply barrel vignetting + Fresnel falloff (the
    /// realistic look) or skip them (fast, flat-bright, aperture-stop only).
    pub poly_vignetting: bool,
    pub trace_direction: TraceDirection,
    pub focal_mode: FocalMode,
    // set by the egui side to request a focal-distance sweep on the next frame
    pub recompute_focal: bool,
    //     "wavelength_sweep", // toggle
    maybe_receiver: Option<Receiver<(String, Command)>>,

    // reporting data
    pub efficiency: f32,
    pub total_samples: usize,
    // last focal-distance sweep result, displayed on the egui side
    pub focal_distance: Option<f32>,
    pub focal_stddev: f32,

    // dummy only:
    pub dirty: bool,

    pub halt: Arc<AtomicBool>,
}

impl SimulationState {
    pub fn toggle_visualize_cache(&mut self) {
        match &mut self.view_mode {
            ViewMode::Film { visualize_cache }
            | ViewMode::SpotOnFilm {
                visualize_cache, ..
            } => *visualize_cache = !*visualize_cache,
            // ViewMode::SpotOnFilm {  visualize_cache, .. } => *visualize_cache = !*visualize_cache,
            ViewMode::XRay { .. } => {}
        }
    }
    pub fn data_update(&mut self, message: (String, Command)) {
        if self.maybe_sender.is_none() {
            // in puppet
            match message {
                (target, Command::ChangeFloat(v)) if target.starts_with("aperture_radius") => {
                    self.aperture_radius = v;
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("sensor_size") => {
                    self.sensor_size = v;
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("film_position") => {
                    self.film_position = v;
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("heat") => {
                    self.heat_bias = v;
                }
                (target, Command::Advance) if target.starts_with("view_mode") => {
                    self.view_mode = self.view_mode.cycle();
                    self.dirty = true;
                }
                (target, Command::Advance) if target.starts_with("visualize_cache") => {
                    println!("received toggle cache visualization command");
                    self.toggle_visualize_cache();
                    self.dirty = true;
                }
                (target, Command::ChangeFloat(v)) if target.starts_with("view_mode") => {
                    assert!(target.find('.') == Some("view_mode".len()));
                    let tail = &target["view_mode".len() + 1..];
                    match &mut self.view_mode {
                        ViewMode::SpotOnFilm {
                            point: (x, y),
                            visualize_cache,
                        } => match tail {
                            "x" => {
                                *x = v;
                                self.dirty = true;
                            }
                            "y" => {
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
                    self.dirty = true;
                }
                (target, Command::Advance) if target.starts_with("toggle sampler") => {
                    self.use_sampler = !self.use_sampler;
                    println!("use sampler is now {:?}", self.use_sampler);
                }
                (target, Command::Advance) if target.starts_with("toggle poly vignetting") => {
                    self.poly_vignetting = !self.poly_vignetting;
                    println!("poly vignetting is now {:?}", self.poly_vignetting);
                    self.dirty = true;
                }
                (target, Command::Advance) if target.starts_with("toggle poly") => {
                    self.use_poly = !self.use_poly;
                    println!("poly mode is now {:?}", self.use_poly);
                    // switching evaluators changes the Film image; start fresh
                    self.dirty = true;
                }
                (target, Command::Advance) if target.starts_with("trace_direction") => {
                    self.trace_direction = self.trace_direction.toggle();
                    println!("trace direction is now {:?}", self.trace_direction);
                    // changing direction invalidates the accumulated film
                    self.dirty = true;
                }
                (target, Command::Advance) if target.starts_with("focal_mode") => {
                    self.focal_mode = self.focal_mode.cycle();
                    println!("focal mode is now {:?}", self.focal_mode);
                }
                (target, Command::Advance) if target.starts_with("focal_distance") => {
                    // heavy sweep runs on the simulation thread, not here
                    self.recompute_focal = true;
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
                        SceneMode::SpotLight {
                            pos,
                            size,
                            max_angle,
                        } => match tail {
                            "pos.x" => {
                                pos.0 = v;
                                println!("pos = {:?}", pos);
                                self.dirty = true;
                            }
                            "pos.y" => {
                                pos.1 = v;
                                println!("pos = {:?}", pos);
                                self.dirty = true;
                            }
                            "pos.z" => {
                                pos.2 = v;
                                println!("pos = {:?}", pos);
                                self.dirty = true;
                            }
                            "size" => {
                                println!();
                                if v < 0.0 {
                                    println!(
                                        "attempted to change size to some nonsensical value, ignoring.\nsize should be above 0"
                                    );
                                    return;
                                }
                                *size = v;
                                self.dirty = true;
                            }
                            "max_angle" => {
                                println!();
                                if v < 0.0 || v >= FRAC_PI_2 {
                                    println!(
                                        "attempted to change max_angle to some nonsensical value, ignoring.\nmax_angle should be between 0 and PI/2, where near 0 cooresponds to a very focused spotlight."
                                    );
                                    return;
                                }
                                *max_angle = v;
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
                    .speed(1.0 / self.max_aperture_radius),
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
                    .speed(1.0),
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
                SceneMode::SpotLight {
                    pos,
                    size,
                    max_angle: span,
                } => {
                    let (x, y, z) = pos;

                    let mut any_changed = false;

                    ui.label("pos.x");
                    let response = ui.add(egui::DragValue::new(x).speed(0.01));
                    any_changed |= response.changed();

                    ui.label("pos.y");
                    let response = ui.add(egui::DragValue::new(y).speed(0.01));
                    any_changed |= response.changed();

                    ui.label("pos.z");
                    let response = ui.add(egui::DragValue::new(z).speed(0.01));
                    any_changed |= response.changed();

                    if any_changed {
                        sender
                            .try_send(("scene_mode.pos.x".into(), Command::ChangeFloat(*x)))
                            .unwrap();
                        sender
                            .try_send(("scene_mode.pos.y".into(), Command::ChangeFloat(*y)))
                            .unwrap();
                        sender
                            .try_send(("scene_mode.pos.z".into(), Command::ChangeFloat(*z)))
                            .unwrap();
                    }
                    *pos = (*x, *y, *z);

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
                    ui.label("max angle");
                    let response = ui.add(egui::Slider::new(span, 0.0..=1.57));
                    if response.changed() {
                        sender
                            .try_send(("scene_mode.max_angle".into(), Command::ChangeFloat(*span)))
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
                ViewMode::SpotOnFilm {
                    point: (x, y),
                    visualize_cache,
                } => {
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
                    ui.label("visualize cache");
                    let response = ui.add(egui::Checkbox::new(visualize_cache, "toggle"));

                    if response.changed() {
                        sender
                            .try_send(("visualize_cache".into(), Command::Advance))
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
                ViewMode::Film { visualize_cache } => {
                    ui.label("visualize cache");
                    let response = ui.add(egui::Checkbox::new(visualize_cache, "toggle"));

                    if response.changed() {
                        sender
                            .try_send(("visualize_cache".into(), Command::Advance))
                            .unwrap();
                    }
                }
                _ => {}
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
            let response = ui.add(egui::Button::new("toggle sampler"));
            if response.clicked() {
                sender
                    .try_send((String::from("toggle sampler"), Command::Advance))
                    .unwrap();
                self.use_sampler = !self.use_sampler;
            }

            let response = ui.add(egui::Button::new("toggle poly mode"));
            if response.clicked() {
                sender
                    .try_send((String::from("toggle poly"), Command::Advance))
                    .unwrap();
                self.use_poly = !self.use_poly;
            }

            let response = ui.add(egui::Button::new("toggle poly vignetting"));
            if response.clicked() {
                sender
                    .try_send((String::from("toggle poly vignetting"), Command::Advance))
                    .unwrap();
                self.poly_vignetting = !self.poly_vignetting;
            }

            let response = ui.add(egui::Button::new("toggle trace direction"));
            if response.clicked() {
                sender
                    .try_send((String::from("trace_direction"), Command::Advance))
                    .unwrap();
                self.trace_direction = self.trace_direction.toggle();
            }

            let response = ui.add(egui::Button::new("cycle focal mode"));
            if response.clicked() {
                sender
                    .try_send((String::from("focal_mode"), Command::Advance))
                    .unwrap();
                self.focal_mode = self.focal_mode.cycle();
            }

            let response = ui.add(egui::Button::new("compute focal distance"));
            if response.clicked() {
                sender
                    .try_send((String::from("focal_distance"), Command::Advance))
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
                    "focal_distance" => {
                        self.focal_distance = Some(command.as_float().unwrap());
                    }
                    "focal_stddev" => {
                        self.focal_stddev = command.as_float().unwrap();
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
            ui.label(format!("trace direction: {:?}", self.trace_direction));
            ui.label(format!(
                "poly mode: {} (vignetting: {})",
                self.use_poly, self.poly_vignetting
            ));
            ui.label(format!("focal mode: {:?}", self.focal_mode));
            match self.focal_distance {
                Some(d) => ui.label(format!(
                    "focal distance: {:.3} mm (stddev {:.3})",
                    d, self.focal_stddev
                )),
                None => ui.label("focal distance: (press \"compute focal distance\")"),
            };
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

/// Computes a suggested focal distance (world z) and its spread (stddev), per [`FocalMode`].
/// Returns `None` if nothing usable is found.
fn compute_focal_distance(
    lens_assembly: &LensAssembly,
    state: &SimulationState,
    lens_zoom: f32,
    wavelength_bounds: ::math::bounds::Bounds1D,
    mode: FocalMode,
) -> Option<(f32, f32)> {
    let mut focal_distance_vec: Vec<f32> = Vec::new();
    match mode {
        FocalMode::FromFilm => {
            // Rays diverge from the on-axis film point, fanned toward the rear element,
            // and trace forward (film -> world); record where each crosses the axis: the
            // object-side conjugate of the film point.
            let n = 25;
            let aperture_reject = |e: Ray| {
                (
                    state.aperture.is_rejected(state.aperture_radius, e.origin),
                    false,
                )
            };
            let origin = Point3::new(0.0, 0.0, state.film_position);
            let toward_edge = Point3::new(
                0.0,
                lens_assembly.lenses.last().unwrap().housing_radius,
                0.0,
            ) - origin;
            let maximum_angle = -(toward_edge.y() / toward_edge.z()).atan();
            for i in 0..n {
                let angle = ((i as f32 + 0.5) / n as f32) * maximum_angle;
                let ray = Ray::new(origin, Vec3::new(0.0, angle.sin(), angle.cos()));
                for w in 0..100 {
                    let lambda =
                        wavelength_bounds.lower + (w as f32 / 100.0) * wavelength_bounds.span();
                    if let Some(Output { ray: pupil_ray, .. }) = lens_assembly.trace_forward(
                        lens_zoom,
                        Input::new(ray, lambda / 1000.0),
                        1.0,
                        aperture_reject,
                        drop,
                    ) {
                        let dt = (-pupil_ray.origin.y()) / pupil_ray.direction.y();
                        let point = pupil_ray.point_at_parameter(dt);
                        if point.z().is_finite() {
                            focal_distance_vec.push(point.z());
                        }
                    }
                }
            }
        }
        FocalMode::FromInfinity => {
            // Object at infinity images at the rear focal plane. We locate it per
            // wavelength via the forward collimation root-find; the spread across
            // wavelengths is longitudinal chromatic aberration. (`rear_focal_plane_reverse`
            // gives an equivalent result by reverse-tracing literal parallel rays.)
            for w in 0..100 {
                let lambda_um = (wavelength_bounds.lower
                    + (w as f32 / 100.0) * wavelength_bounds.span())
                    / 1000.0;
                if let Some(z) =
                    lens_assembly.rear_focal_plane_forward::<Backend>(lens_zoom, lambda_um)
                {
                    focal_distance_vec.push(z);
                }
            }
        }
    }
    if focal_distance_vec.is_empty() {
        return None;
    }
    let avg: f32 = focal_distance_vec.iter().sum::<f32>() / focal_distance_vec.len() as f32;
    let variance = focal_distance_vec
        .iter()
        .map(|e| (avg - *e).powf(2.0))
        .sum::<f32>()
        / focal_distance_vec.len() as f32;
    Some((avg, variance.sqrt()))
}

fn run_simulation(
    opt: Opt,
    mut local_simulation_state: SimulationState,
    lens_assembly: LensAssembly,
    receiver: Receiver<(String, Command)>,
    sender: Sender<(String, Command)>,
) {
    use optics::dev::tonemap::{Tonemapper, sRGB};

    println!("{:?}", opt);
    let width = opt.width;
    let height = opt.height;

    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .unwrap();

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
    let mut window = Window::new(
        "lens analysis",
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

    let mut film = Vec2D::new(width, height, XYZColor::black());
    let mut window_pixels = Vec2D::new(width, height, 0u32);

    // Limit to max ~144 fps update rate
    let width = film.width;

    let frame_dt = 6944.0 / 1000000.0;

    let scene = get_scene("textures.toml").unwrap();

    window.set_target_fps(144);
    let mut textures: Vec<TexStack> = Vec::new();
    for tex in scene.textures {
        textures.push(parse_texture_stack(tex.clone(), wavelength_bounds));
    }

    let original_aperture_radius = lens_assembly.aperture_radius();
    let mut lens_zoom = 0.0;
    let mut wall_position = 5000.0;
    let mut texture_scale = 1.0;

    let samples_per_iteration = 1usize;
    let mut total_samples = 0;

    let direction_cache_radius_bins = 512;
    let direction_cache_wavelength_bins = 512;

    let mut direction_cache = RadialSampler::new::<Backend, _>(
        SQRT_2 * local_simulation_state.sensor_size, // diagonal.
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

    // Polynomial lens model, evaluated as a drop-in for `trace_forward` when poly
    // mode is on. `None` if the assembly has a surface type the poly builder does
    // not support yet (aspheric/cylindrical) — poly mode then silently falls back
    // to tracing. Rebuilt alongside the direction cache.
    const POLY_DEGREE: usize = 3;
    const POLY_WAVELENGTH_BINS: usize = 64;
    let mut poly_lens: Option<PolyLens> =
        PolyAssembly::new(&lens_assembly, lens_zoom, ATMOSPHERE_IOR, POLY_DEGREE, wavelength_bounds, POLY_WAVELENGTH_BINS)
            .ok()
            .map(PolyLens::new);

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
                .for_each(|e| *e = XYZColor::black())
        }
        if clear_direction_cache {
            direction_cache = RadialSampler::new::<Backend, _>(
                SQRT_2 * local_simulation_state.sensor_size, // diagonal.
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
            poly_lens = PolyAssembly::new(
                &lens_assembly,
                lens_zoom,
                ATMOSPHERE_IOR,
                POLY_DEGREE,
                wavelength_bounds,
                POLY_WAVELENGTH_BINS,
            )
            .ok()
            .map(PolyLens::new);
            println!("cleared direction cache");
        }

        // A focal-distance sweep probes the assembly with a fan of rays (geometry per
        // the selected FocalMode) and reports where they cross the optical axis. Run it
        // when the direction cache was just rebuilt, or when the user requested it.
        let recompute_focal = std::mem::replace(&mut local_simulation_state.recompute_focal, false);
        if clear_direction_cache || recompute_focal {
            match compute_focal_distance(
                &lens_assembly,
                &local_simulation_state,
                lens_zoom,
                wavelength_bounds,
                local_simulation_state.focal_mode,
            ) {
                Some((avg, sd)) => {
                    println!(
                        "[{:?}] focal distance suggestion: {}. stddev = {}",
                        local_simulation_state.focal_mode, avg, sd
                    );
                    let _ = sender
                        .try_send((String::from("focal_distance"), Command::ChangeFloat(avg)));
                    let _ =
                        sender.try_send((String::from("focal_stddev"), Command::ChangeFloat(sd)));
                }
                None => println!("focal distance sweep found no rays through the lens"),
            }
        }

        let srgb_tonemapper = sRGB::new(&film, 1.0);

        total_samples += samples_per_iteration;
        let (mut a, mut b) = (0, 0);

        if local_simulation_state.trace_direction == TraceDirection::FromScene {
            // ---- reverse tracing: scene -> film -------------------------------
            // Rays are generated out in the scene (per scene mode) and traced back
            // through the lens toward the sensor. The radial sampler does not apply
            // in this direction, so the "toggle sampler" control is inert here.
            let wall_texture = &textures[0];
            let mut sampler = RandomSampler::new();
            for _ in 0..local_simulation_state.samples {
                let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                let (ray, le) = match local_simulation_state.scene_mode {
                    SceneMode::TexturedWall {
                        distance: wall_position,
                        texture_scale,
                    } => {
                        let sample = sampler.draw_2d();
                        let (rx, ry) = (sample.x - 0.5, sample.y - 0.5);
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
                    SceneMode::SpotLight {
                        pos,
                        max_angle,
                        size,
                    } => {
                        let (r, phi) =
                            (sampler.draw_1d().x.sqrt() * size, sampler.draw_1d().x * TAU);
                        let (px, py) = (pos.0 + r * phi.cos(), pos.1 + r * phi.sin());
                        let ray_origin = Point3::new(px, py, pos.2);
                        // span is the lower limit of the cosine of the angle to sample

                        let angle = sampler.draw_1d().x.sqrt() * max_angle;
                        let other_angle = sampler.draw_1d().x * TAU;
                        let dir = Vec3::new(
                            angle.sin() * other_angle.cos(),
                            angle.sin() * other_angle.sin(),
                            -angle.cos(),
                        );
                        (Ray::new(ray_origin, dir), 1.0)
                    }
                    SceneMode::PinLight => {
                        let (r, phi) = (sampler.draw_1d().x.sqrt(), sampler.draw_1d().x * TAU);
                        let (dx, dy) = (r * phi.cos(), r * phi.sin());
                        (
                            Ray::new(
                                Point3::new(0.0, 0.0, 10.0),
                                Vec3::new(dx, dy, -10.0).normalized(),
                            ),
                            1.0,
                        )
                    }
                };

                b += 1;
                match local_simulation_state.view_mode {
                    ViewMode::Film { visualize_cache }
                    | ViewMode::SpotOnFilm {
                        visualize_cache, ..
                    } => {
                        let result = lens_assembly.trace_reverse(
                            lens_zoom,
                            Input::new(ray, lambda / 1000.0),
                            ATMOSPHERE_IOR,
                            |e| {
                                (
                                    local_simulation_state.aperture.is_rejected(
                                        local_simulation_state.aperture_radius,
                                        e.origin,
                                    ),
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
                            a += 1;
                            let t = (local_simulation_state.film_position - pupil_ray.origin.z())
                                / pupil_ray.direction.z();
                            let point_at_film = pupil_ray.point_at_parameter(t);
                            let uv = (
                                ((point_at_film.x() / local_simulation_state.sensor_size) + 1.0)
                                    / 2.0,
                                ((point_at_film.y() / local_simulation_state.sensor_size) + 1.0)
                                    / 2.0,
                            );
                            if uv.0 < 1.0 && uv.1 < 1.0 && uv.0 > 0.0 && uv.1 > 0.0 {
                                let (fx, fy) = (
                                    (uv.0 * width as f32) as usize,
                                    (uv.1 * height as f32) as usize,
                                );
                                film.write_at(
                                    fx,
                                    fy,
                                    film.at(fx, fy)
                                        + XYZColor::from(SingleWavelength::new(
                                            lambda,
                                            (le * tau).into(),
                                        )),
                                );
                            }
                        }
                    }
                    ViewMode::XRay { bounds } => {
                        // project onto the x=0 plane, then swap x<->z so depth maps to screen-x
                        let swizzle_project = |pt| {
                            project(pt, Vec3::x_axis(), |v: F32x4| {
                                F32x4::new([
                                    v.extract::<2>(),
                                    v.extract::<1>(),
                                    v.extract::<0>(),
                                    v.extract::<3>(),
                                ])
                            })
                        };

                        let mut segments = Vec::new();
                        let result = lens_assembly.trace_reverse(
                            lens_zoom,
                            Input::new(ray, lambda / 1000.0),
                            ATMOSPHERE_IOR,
                            |e| {
                                (
                                    local_simulation_state.aperture.is_rejected(
                                        local_simulation_state.aperture_radius,
                                        e.origin,
                                    ),
                                    false,
                                )
                            },
                            |(p0, p1, tau)| {
                                segments.push((p0, p1, tau));
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
                                ((point_at_film.x() / local_simulation_state.sensor_size) + 1.0)
                                    / 2.0,
                                ((point_at_film.y() / local_simulation_state.sensor_size) + 1.0)
                                    / 2.0,
                            );
                            if uv.0 < 1.0 && uv.1 < 1.0 && uv.0 > 0.0 && uv.1 > 0.0 {
                                a += 1;
                                for (seg_a, seg_b, seg_tau) in segments {
                                    draw_line(
                                        &mut film,
                                        bounds,
                                        swizzle_project(seg_a),
                                        swizzle_project(seg_b),
                                        lambda,
                                        seg_tau,
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
        } else {
            match local_simulation_state.view_mode {
                ViewMode::Film { visualize_cache } => {
                    let pair = film
                        .buffer
                        .par_iter_mut()
                        .enumerate()
                        .map(|(i, pixel)| {
                            let mut sampler = RandomSampler::new();
                            let px = i % width;
                            let py = i / width;

                            let (mut successes, mut attempts) = (0, 0);
                            let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
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
                                let [mut x, mut y, z, _] = central_point.as_array();
                                // jitter point within pixel
                                x += (s0.x - 0.5) / width as f32
                                    * local_simulation_state.sensor_size;
                                y += (s0.y - 0.5) / height as f32
                                    * local_simulation_state.sensor_size;

                                let point = Point3::new(x, y, z);
                                if local_simulation_state.use_sampler {
                                    // using radial sampler

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
                                // Either evaluate the polynomial lens model (poly mode)
                                // or trace through the lens. The poly map always honors
                                // the aperture stop; with `poly_vignetting` it also
                                // applies barrel housing vignetting + per-surface Fresnel
                                // falloff (matching the trace), otherwise it is flat-
                                // bright (tau = 1) and only stop-clipped. Falls back to
                                // tracing if poly is unsupported.
                                let result = match (
                                    local_simulation_state.use_poly,
                                    poly_lens.as_ref(),
                                ) {
                                    (true, Some(pl)) => pl
                                        .map_forward_clipped::<Backend, _>(
                                            ray,
                                            lambda,
                                            &local_simulation_state.aperture,
                                            local_simulation_state.aperture_radius,
                                            local_simulation_state.poly_vignetting,
                                        )
                                        .map(|(r, tau)| Output::new(r, tau)),
                                    _ => lens_assembly.trace_forward(
                                        lens_zoom,
                                        Input::new(ray, lambda / 1000.0),
                                        ATMOSPHERE_IOR,
                                        |e| {
                                            (
                                                local_simulation_state.aperture.is_rejected(
                                                    local_simulation_state.aperture_radius,
                                                    e.origin,
                                                ),
                                                false,
                                            )
                                        },
                                        drop,
                                    ),
                                };
                                if visualize_cache {
                                    // directly read energy
                                    let [x, y, _, _] = point.as_array();

                                    let film_radius = y.hypot(x);

                                    let u = film_radius / (SQRT_2 * direction_cache.sensor_size);
                                    let v = ((lambda - direction_cache.wavelength_bounds.lower)
                                        / direction_cache.wavelength_bounds.span())
                                    .clamp(0.0, 1.0 - EPSILON);
                                    debug_assert!(u < 1.0 && v < 1.0, "{}, {}", u, v);
                                    let d_x_idx = (u * direction_cache.radius_bins as f32) as usize;
                                    let d_y_idx =
                                        (v * direction_cache.wavelength_bins as f32) as usize;
                                    let angles00 = direction_cache.cache.at(d_x_idx, d_y_idx);
                                    *pixel += XYZColor::from(SingleWavelength::new(
                                        550.0,
                                        angles00.angle_spread,
                                    ));
                                    continue;
                                }
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
                                            if (0.0..1.0).contains(&uv.0)
                                                && (0.0..1.0).contains(&uv.1)
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

                                        SceneMode::SpotLight {
                                            pos,
                                            size,
                                            max_angle: span,
                                        } => {
                                            let t = (pos.2 - pupil_ray.origin.z())
                                                / pupil_ray.direction.z();
                                            let point_at_light_z = pupil_ray.point_at_parameter(t);
                                            let m = if (point_at_light_z.x() - pos.0).powi(2)
                                                + (point_at_light_z.y() - pos.1).powi(2)
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

                            (successes, attempts)
                        })
                        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
                    a += pair.0;
                    b += pair.1;
                }

                ViewMode::SpotOnFilm {
                    point: (x, y),
                    visualize_cache,
                } => {
                    let pair = film
                        .buffer
                        .par_iter_mut()
                        .enumerate()
                        .map(|(i, pixel)| {
                            let mut sampler = RandomSampler::new();
                            let px = i % width;
                            let py = i / width;

                            let (mut successes, mut attempts) = (0, 0);
                            let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                            let central_point =
                                Point3::new(x, y, local_simulation_state.film_position);

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
                            let radial_distance =
                                (u.hypot(v) / (SQRT_2 / 2.0)).clamp(0.0, 1.0 - EPSILON);
                            let angle = ((u.atan2(v) + PI) / TAU).clamp(0.0, 1.0 - EPSILON);
                            let dir =
                                random_cosine_direction(Sample2D::new(angle, radial_distance));
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
                                        local_simulation_state.aperture.is_rejected(
                                            local_simulation_state.aperture_radius,
                                            e.origin,
                                        ),
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

                                match local_simulation_state.scene_mode {
                                    SceneMode::PinLight => {
                                        // using this as a debug scene, just to view which directions actually get through the lens

                                        // this is super jank but it'll get something visual
                                        let mut would_have_sampled = false;
                                        for _ in 0..5 {
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
                                        *pixel += XYZColor::from(SingleWavelength::new(
                                            lambda,
                                            tau.into(),
                                        ));
                                    }
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
                                    SceneMode::SpotLight {
                                        pos,
                                        size,
                                        max_angle: span,
                                    } => {
                                        let t = (pos.2 - pupil_ray.origin.z())
                                            / pupil_ray.direction.z();
                                        let point_at_light_z = pupil_ray.point_at_parameter(t);
                                        let m = if (point_at_light_z.x() - pos.0).powi(2)
                                            + (point_at_light_z.y() - pos.1).powi(2)
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
                            } else {
                                // didn't make it through

                                let mut would_have_sampled = false;
                                for _ in 0..5 {
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
                                    *pixel +=
                                        XYZColor::from(SingleWavelength::new(450.0, 1.0.into()));
                                }
                            }
                            (successes, attempts)
                        })
                        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
                    a += pair.0;
                    b += pair.1;
                }
                ViewMode::XRay { bounds } => {
                    let mut sampler = RandomSampler::new();
                    // project onto the x=0 plane, then swap x<->z so depth maps to screen-x
                    let swizzle_project = |pt| {
                        project(pt, Vec3::x_axis(), |v: F32x4| {
                            F32x4::new([
                                v.extract::<2>(),
                                v.extract::<1>(),
                                v.extract::<0>(),
                                v.extract::<3>(),
                            ])
                            // v.swizzle_const(GenericArray::<u32, 4>::new(2, 1, 0, 3))
                            // v.swizzle_const(GenericArray::<u32, 4>::new(2, 1, 0, 3))
                        })
                    };

                    let mut segments = Vec::new();
                    for _ in 0..samples_per_iteration {
                        b += 1;
                        let (u, v) = {
                            let sample = sampler.draw_2d();
                            (sample.x - 0.5, sample.y - 0.5)
                        };

                        let origin = Point3::new(
                            u * local_simulation_state.sensor_size,
                            v * local_simulation_state.sensor_size,
                            local_simulation_state.film_position,
                        );

                        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                        let direction = if local_simulation_state.use_sampler {
                            direction_cache.sample(
                                lambda,
                                origin,
                                sampler.draw_2d(),
                                sampler.draw_1d(),
                            )
                        } else {
                            random_cosine_direction(sampler.draw_2d())
                        };
                        let ray = Ray::new(origin, direction);

                        segments.clear();

                        let result = lens_assembly.trace_forward(
                            lens_zoom,
                            Input::new(ray, lambda / 1000.0),
                            1.0,
                            |e| {
                                (
                                    local_simulation_state.aperture.is_rejected(
                                        local_simulation_state.aperture_radius,
                                        e.origin,
                                    ),
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
                            a += 1;
                            // println!("path {:?}", segments);
                            for (a, b, tau) in segments.iter().skip(1) {
                                draw_line(
                                    &mut film,
                                    bounds,
                                    swizzle_project(*a),
                                    swizzle_project(*b),
                                    lambda,
                                    *tau,
                                    DrawMode::XiaolinWu,
                                );
                            }
                            draw_line(
                                &mut film,
                                bounds,
                                swizzle_project(pupil_ray.origin),
                                swizzle_project(pupil_ray.point_at_parameter(1000.0)),
                                lambda,
                                tau,
                                DrawMode::XiaolinWu,
                            );
                        }

                        // Poly-mode overlay: the polynomial model only knows the
                        // endpoints, so it draws the lens as a black box — a straight
                        // red chord from the film point to the front pupil, plus the
                        // predicted exit ray. Runs even when the real ray is vignetted,
                        // so where the model diverges from the trace is visible.
                        if local_simulation_state.use_poly {
                            if let Some(pl) = poly_lens.as_ref() {
                                const POLY_OVERLAY_LAMBDA: f32 = 650.0; // red
                                let pe = pl.map_forward::<Backend>(ray, lambda);
                                draw_line(
                                    &mut film,
                                    bounds,
                                    swizzle_project(ray.origin),
                                    swizzle_project(pe.origin),
                                    POLY_OVERLAY_LAMBDA,
                                    1.0,
                                    DrawMode::XiaolinWu,
                                );
                                draw_line(
                                    &mut film,
                                    bounds,
                                    swizzle_project(pe.origin),
                                    swizzle_project(pe.point_at_parameter(1000.0)),
                                    POLY_OVERLAY_LAMBDA,
                                    1.0,
                                    DrawMode::XiaolinWu,
                                );
                            }
                        }
                    }
                }
            }
        }

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
                let [r, g, b, _] = mapped;
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
        view_mode: ViewMode::XRay {
            bounds: Bounds2D::new((-400.0, 200.0).into(), (-200.0, 200.0).into()),
        },
        paused: false,
        samples: 100,
        film_position: -lens_assembly.total_thickness_at(lens_zoom),
        min_film_position: lens_assembly.lenses.last().unwrap().thickness_short
            - lens_assembly.total_thickness_at(lens_zoom),
        lens_zoom: 0.0,
        maybe_sender: None,
        maybe_receiver: None,
        use_sampler: true,
        use_poly: false,
        poly_vignetting: true,
        trace_direction: TraceDirection::FromFilm,
        focal_mode: FocalMode::FromFilm,
        recompute_focal: false,
        dirty: false,
        efficiency: 0.0,
        total_samples: 0,
        focal_distance: None,
        focal_stddev: 0.0,
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

    let _ = eframe::run_native(
        "Lens Analysis Control Panel",
        options,
        Box::new(|_cc| Box::new(simulation_state_egui)),
    );
}
