use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use std::{f32::consts::TAU, fs::File, io::Read};

use egui::{Color32, ColorImage, Image, TextureHandle};
// use crate::math::Sample2D;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
// use packed_simd::f32x4;
// use rand::prelude::*;
use eframe::egui;
use egui_extras::RetainedImage;
// use egui::prelude::*;
use crossbeam::channel::{unbounded, Receiver, Sender};
use rayon::prelude::*;

use crate::math::{SingleWavelength, XYZColor};
use subcrate::{film::Film, parsing::*};
// use lens_sampler::RadialSampler;
use optics::*;

use crate::math::spectral::BOUNDED_VISIBLE_RANGE;
use crate::{SceneMode, ViewMode};

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

    pub heat_bias: f32,
    pub heat_cap: f32,
    pub scene_mode: SceneMode,
    pub view_mode: ViewMode,
    pub paused: bool,
    pub samples: usize,

    pub aperture_radius: f32,
    pub max_aperture_radius: f32,
    pub sensor_size: f32,
    pub max_sensor_size: f32,
    pub wall_position: f32,
    pub film_position: f32,
    pub lens_zoom: f32,
    pub texture_scale: f32,
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
    // pub
}

impl SimulationState {
    pub fn data_update(&mut self, message: (String, Command)) {
        if self.maybe_sender.is_none() {
            // in puppet
            match message.0.as_str() {
                "aperture_radius" => {
                    message
                        .1
                        .as_float()
                        .map(|float| self.aperture_radius = float);
                }
                "sensor_size" => {
                    message.1.as_float().map(|float| self.sensor_size = float);
                }
                target => {
                    println!(
                        "received mutate command without a matching target, {}",
                        target
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

            let mut percentile = (100.0 * self.aperture_radius / self.max_aperture_radius) as i32;
            let response = ui.add(egui::Slider::new(&mut percentile, 0..=100));
            if response.changed() {
                self.aperture_radius = self.max_aperture_radius * (percentile as f32) / 100.0;
                sender
                    .try_send((String::from("aperture_radius"), self.aperture_radius.into()))
                    .unwrap()
            }

            let mut percentile = (100.0 * self.sensor_size / self.max_sensor_size) as i32;
            let response = ui.add(egui::Slider::new(&mut percentile, 0..=100));
            if response.changed() {
                self.sensor_size = self.max_sensor_size * (percentile as f32) / 100.0;
                sender
                    .try_send((String::from("sensor_size"), self.sensor_size.into()))
                    .unwrap()
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
    use subcrate::tonemap::{sRGB, Tonemapper};

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

    let mut film = Film::new(window_width, window_height, XYZColor::BLACK);
    let mut window_pixels = Film::new(window_width, window_height, 0u32);

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

    let mut samples_per_iteration = 1000usize;
    let mut total_samples = 0;
    let mut focal_distance_suggestion = None;
    let mut focal_distance_vec: Vec<f32> = Vec::new();
    let mut variance: f32 = 0.0;
    let mut stddev: f32 = 0.0;

    let mut wavelength_sweep: f32 = 0.0;
    let mut wavelength_sweep_speed = 0.001;
    let mut efficiency = 0.0;
    let efficiency_heat = 0.99;
    let mut scene_mode = local_simulation_state.scene_mode;
    let mut texture_scale = local_simulation_state.texture_scale;
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
        let mut clear_direction_cache = false;

        for message in receiver.try_iter() {
            if message.0.as_str() == "clear" {
                clear_film = true;
            }
            local_simulation_state.data_update(message);
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
        if clear_direction_cache {
            unimplemented!("direction cache is currently unimplemented for reverse tracing.");
        }

        let srgb_tonemapper = sRGB::new(&film, 1.0);

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
                        local_simulation_state.lens_zoom,
                        &Input { ray, lambda },
                        1.0,
                        |e| {
                            (
                                bladed_aperture(local_simulation_state.aperture_radius, 6, e),
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
            }
        }

        total_samples += samples_per_iteration;

        let mut sampler = RandomSampler::new();

        let (mut successes, mut attempts) = (0, 0);

        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);

        for _ in 0..samples_per_iteration {
            let ray = match local_simulation_state.scene_mode {
                // diffuse emitter texture
                SceneMode::TexturedWall => {
                    // ray is generated according to texture scale.
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
                    let point_on_texture = Point3::new(
                        texture_scale * rx,
                        texture_scale * ry,
                        local_simulation_state.wall_position,
                    );
                    let v = (point_on_lens - point_on_texture).normalized();

                    Ray::new(point_on_texture, v)
                }
                // parallel light
                SceneMode::SpotLight { pos, span, size } => {
                    // 4 quadrants.

                    let (r, phi) = (
                        sampler.draw_1d().x.sqrt() * local_simulation_state.texture_scale,
                        sampler.draw_1d().x * TAU,
                    );

                    let (px, py) = (pos.x() + r * phi.cos(), pos.y() + r * phi.sin());
                    // TODO: use span and size to determine angle

                    Ray::new(
                        Point3::new(px, py, local_simulation_state.wall_position),
                        -Vec3::Z,
                    )
                }
                SceneMode::PinLight => {
                    // 4 quadrants.

                    let (r, phi) = (
                        sampler.draw_1d().x.sqrt() * local_simulation_state.texture_scale,
                        sampler.draw_1d().x * TAU,
                    );

                    let (dx, dy) = (r * phi.cos(), r * phi.sin());
                    let (px, py) = (0.0, 0.0);

                    Ray::new(
                        Point3::new(px, py, local_simulation_state.wall_position),
                        Vec3::new(dx, dy, -1.0).normalized(),
                    )
                }
            };
            // println!("{:?}", ray);

            attempts += 1;
            // do actual tracing through lens for film sample
            let result = lens_assembly.trace_reverse(
                local_simulation_state.lens_zoom,
                &Input { ray, lambda },
                1.04,
                |e| {
                    (
                        bladed_aperture(local_simulation_state.aperture_radius, 6, e),
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
                let t = (local_simulation_state.film_position - pupil_ray.origin.z())
                    / pupil_ray.direction.z();
                let point_at_film = pupil_ray.point_at_parameter(t);
                let uv = (
                    (((point_at_film.x() / local_simulation_state.sensor_size) + 1.0) / 2.0) % 1.0,
                    (((point_at_film.y() / local_simulation_state.sensor_size) + 1.0) / 2.0) % 1.0,
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
        sender
            .try_send((String::from("efficiency"), efficiency.into()))
            .unwrap();
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

    let local_simulation_state = SimulationState {
        heat_bias: 0.01,
        heat_cap: 10.0,
        aperture_radius: original_aperture_radius / 3.0,
        max_aperture_radius: original_aperture_radius,
        sensor_size: 35.0,
        max_sensor_size: 35.0,
        scene_mode: SceneMode::PinLight,
        view_mode: ViewMode::Film,
        paused: false,
        samples: 1,
        wall_position: 500.0,
        film_position: -lens_assembly.total_thickness_at(lens_zoom),
        lens_zoom: 0.0,
        texture_scale: 1.0,
        maybe_sender: None,
        maybe_receiver: None,
        efficiency: 0.0,
        total_samples: 0,
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
        "Show an image with eframe/egui",
        options,
        Box::new(|_cc| Box::new(simulation_state_egui)),
    );
}
