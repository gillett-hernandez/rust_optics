#![feature(portable_simd)]
extern crate minifb;

use ::math::prelude::*;
use rayon::prelude::*;

pub mod aperture;
#[cfg(feature = "dev")]
pub mod dev;
pub mod vec2d;
pub mod lens;
pub mod lens_sampler;
pub mod math;
pub mod misc;
pub mod spectrum;

use crate::aperture::*;
pub use crate::math::{Input, Output, PlaneRay, SphereRay};

pub use lens::*;

pub extern crate nalgebra as na;
pub use na::{Matrix3, Vector3};

#[cfg(feature = "dev")]
use std::collections::HashMap;

use std::f32::{
    consts::{SQRT_2, TAU},
    EPSILON,
};

fn simulate_phase1(assembly: LensAssembly, inputs: &Vec<Input<Ray>>) -> Vec<Option<Output<Ray>>> {
    let mut outputs: Vec<Option<Output<Ray>>> = Vec::new();
    let aperture_radius = 10.0;
    let mut failed = 0;
    let aperture = SimpleBladedAperture::new(6, 0.5);
    for input in inputs {
        let output = assembly.trace_forward(
            0.0,
            *input,
            1.04,
            |ray| (aperture.intersects(aperture_radius, ray), false),
            drop,
        );
        if output.is_none() {
            failed += 1;
        }
        outputs.push(output);
    }
    println!(
        "simulated {} rays, {} rays failed to exit the lens assembly",
        inputs.len(),
        failed
    );
    outputs
}

fn simulate_phase2(assembly: LensAssembly, inputs: &Vec<Input<Ray>>) -> Vec<Option<Output<Ray>>> {
    let mut outputs: Vec<Option<Output<Ray>>> = Vec::new();
    let aperture_radius = 10.0;
    let aperture = SimpleBladedAperture::new(6, 0.5);

    let mut failed = 0;
    for input in inputs {
        let output = assembly.trace_reverse(
            0.0,
            *input,
            1.04,
            |ray| (aperture.intersects(aperture_radius, ray), false),
            drop,
        );
        if output.is_none() {
            failed += 1;
        }
        outputs.push(output);
    }
    println!(
        "simulated {} rays, {} rays failed to exit the lens assembly",
        inputs.len(),
        failed
    );
    outputs
}

pub fn parse_lenses_from(spec: &str) -> (Vec<LensInterface>, f32, f32) {
    // returns a vec of lens elements, the last ior and last vno
    let lines = spec.lines();
    let mut lenses: Vec<LensInterface> = Vec::new();
    let (mut last_ior, mut last_vno) = (1.0, 0.0);
    for line in lines {
        if line.is_empty() || line.starts_with('#') || line == "\n" {
            continue;
        }
        let lens = LensInterface::parse_from(line, last_ior, last_vno).unwrap();
        last_ior = lens.ior;
        last_vno = lens.vno;
        // println!("successfully parsed lens {:?}", lens);
        lenses.push(lens);
    }
    (lenses, last_ior, last_vno)
}

#[cfg(test)]
mod test {
    use super::*;
    //     #[test]
    //     fn test() {
    //         let (lenses, last_ior, last_vno) = parse_lenses_from(spec);
    //         let assembly = LensAssembly::new(&lenses);

    //         let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 20));
    //         // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
    //         const ZOOM: f32 = 0.0;

    //         let dist = assembly.total_thickness_at(ZOOM);
    //         println!("total lens thickness is {}", dist);
    //         let plane_ray_sampler = |x: f32, y: f32, sampler: &mut Box<dyn Sampler>| {
    //             let Sample2D { x: u, y: v } = sampler.draw_2d();
    //             let theta = 2.0 * 3.1415926535 * u;
    //             let (sin, cos) = theta.sin_cos();
    //             let R = lenses.last().unwrap().housing_radius;
    //             let x = 35.0 * (x - 0.5);
    //             let y = 35.0 * (y - 0.5);
    //             let mut plane_ray = PlaneRay::new(
    //                 x,
    //                 y,
    //                 R / dist * cos * v.sqrt() - x / dist,
    //                 R / dist * sin * v.sqrt() - y / dist,
    //             );
    //             plane_ray
    //         };
    //         let wavelength_sampler = |sampler: &mut Box<dyn Sampler>| sampler.draw_1d().x * 0.3 + 0.4;

    //         let mut inputs: Vec<Input<PlaneRay>> = Vec::new();
    //         let w = 1000;
    //         let h = 1000;
    //         for y in 0..h {
    //             for x in 0..w {
    //                 let ray = plane_ray_sampler(x as f32 / w as f32, y as f32 / h as f32, &mut sampler);
    //                 let lambda = wavelength_sampler(&mut sampler);
    //                 inputs.push(Input::new( ray, lambda ));
    //             }
    //         }

    //         let outputs1 = simulate_phase1(&lenses, &inputs);
    //         let outputs2 = simulate_phase2(&lenses, &inputs);

    //         use std::io::BufWriter;
    //         let mut file1 = BufWriter::new(File::create("output1.txt").unwrap());
    //         let mut file2 = BufWriter::new(File::create("output2.txt").unwrap());
    //         let lens0: LensInterface = lenses[0];
    //         let mut count = 10;
    //         for (input, output) in inputs.iter().zip(outputs1.iter()) {
    //             if count > 0 {
    //                 print!("input {:?}, {} ", input.ray, input.lambda,);
    //                 if let Some(out) = output {
    //                     let ray = sphere_to_camera_space(out.ray, -lens0.radius, lens0.radius);
    //                     println!(
    //                         ": output {:?}, {:?}, {}",
    //                         ray.origin, ray.direction, out.tau
    //                     );
    //                 } else {
    //                     println!("");
    //                 }
    //                 count -= 1;
    //             }
    //             file1
    //                 .write(
    //                     format!(
    //                         "{} {} {} {} {}",
    //                         input.ray.x(),
    //                         input.ray.y(),
    //                         input.ray.dx(),
    //                         input.ray.dy(),
    //                         input.lambda
    //                     )
    //                     .as_bytes(),
    //                 )
    //                 .unwrap();
    //             if let Some(out) = output {
    //                 // let ray = sphere_to_camera_space(out.ray, -lens0.radius, lens0.radius);
    //                 file1
    //                     .write(
    //                         format!(
    //                             " {} {} {} {} {}\n",
    //                             out.ray.x(),
    //                             out.ray.y(),
    //                             out.ray.dx(),
    //                             out.ray.dy(),
    //                             out.tau
    //                         )
    //                         .as_bytes(),
    //                     )
    //                     .unwrap();
    //             } else {
    //                 file1.write(b" !\n").unwrap();
    //             }
    //         }
    //         let mut count = 10;
    //         let aperture_pos = assembly.aperture_position(ZOOM);
    //         for (input, output) in inputs.iter().zip(outputs2.iter()) {
    //             if count > 0 {
    //                 print!("input {:?}, {} ", input.ray, input.lambda,);
    //                 if let Some(out) = output {
    //                     let ray = plane_to_camera_space(out.ray, aperture_pos);
    //                     println!(
    //                         ": output {:?}, {:?}, {}",
    //                         ray.origin, ray.direction, out.tau
    //                     );
    //                 } else {
    //                     println!("");
    //                 }
    //                 count -= 1;
    //             }
    //             file2
    //                 .write(
    //                     format!(
    //                         "{} {} {} {} {}",
    //                         input.ray.x(),
    //                         input.ray.y(),
    //                         input.ray.dx(),
    //                         input.ray.dy(),
    //                         input.lambda
    //                     )
    //                     .as_bytes(),
    //                 )
    //                 .unwrap();
    //             if let Some(out) = output {
    //                 file2
    //                     .write(
    //                         format!(
    //                             " {} {} {} {} {}\n",
    //                             out.ray.x(),
    //                             out.ray.y(),
    //                             out.ray.dx(),
    //                             out.ray.dy(),
    //                             out.tau
    //                         )
    //                         .as_bytes(),
    //                     )
    //                     .unwrap();
    //             } else {
    //                 file2.write(b" !\n").unwrap();
    //             }
    //         }
    // }
}
