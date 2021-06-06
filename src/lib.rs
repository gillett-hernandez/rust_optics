#![allow(unused_imports, dead_code, unused_variables)]
// #![feature(slice_fill)]
extern crate minifb;
#[macro_use]
extern crate packed_simd;

use rayon::prelude::*;

pub mod film;
pub mod lens;
pub mod lens_sampler;
pub mod math;
pub mod parse;
pub mod spectrum;
pub mod tonemap;

pub use film::Film;

pub use crate::math::{Input, Output, PlaneRay, SphereRay};
use ::math::*;
pub use lens::*;

use packed_simd::f32x4;
use rand::prelude::*;

pub extern crate nalgebra as na;
pub use na::{Matrix3, Vector3};

use std::fs::File;
use std::io::prelude::*;

use std::f32::{
    consts::{SQRT_2, TAU},
    EPSILON,
};


pub fn circular_aperture(aperture_radius: f32, ray: Ray) -> bool {
    ray.origin.x().hypot(ray.origin.y()) > aperture_radius
}

pub fn bladed_aperture(aperture_radius: f32, blades: usize, ray: Ray) -> bool {
    match blades {
        6 => {
            let phi = std::f32::consts::PI / 3.0;
            let top = Vec3::new(phi.cos(), phi.sin(), 0.0);
            let bottom = Vec3::new(phi.cos(), -phi.sin(), 0.0);
            let mut point = Vec3::from(ray.origin);
            point.0 = point.0.replace(2, 0.0);
            // point = point.normalized();
            let cos_top = point * top;
            let cos_bottom = point * bottom;
            let cos_apex = point.x();
            let minimum = ((1.0 + cos_top.abs().powf(0.5)) / cos_top.abs())
                .min((1.0 + cos_bottom.abs().powf(0.5)) / cos_bottom.abs())
                .min((1.0 + cos_apex.abs().powf(0.5)) / cos_apex.abs());
            point.x().hypot(point.y()) > minimum * aperture_radius
        }
        _ => circular_aperture(aperture_radius, ray),
    }
}

fn simulate_phase1(assembly: LensAssembly, inputs: &Vec<Input<Ray>>) -> Vec<Option<Output<Ray>>> {
    let mut outputs: Vec<Option<Output<Ray>>> = Vec::new();
    let aperture_radius = 10.0;
    let mut failed = 0;
    for input in inputs {
        let output = assembly.trace_forward(0.0, input, 1.04, |e| {
            (bladed_aperture(aperture_radius, 6, e), false)
        });
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
    let mut failed = 0;
    for input in inputs {
        let output = assembly.trace_reverse(0.0, input, 1.04, |e| {
            (bladed_aperture(aperture_radius, 6, e), false)
        });
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
    let lines = spec.lines();
    let mut lenses: Vec<LensInterface> = Vec::new();
    let (mut last_ior, mut last_vno) = (1.0, 0.0);
    for line in lines {
        if line.starts_with("#") || line == "" || line == "\n" {
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
    //         //     let lines = "# whatever
    //         // 65.22    9.60  N-SSK8 1.5 50 24.0
    //         // -62.03   4.20  N-SF10 1.5 50 24.0
    //         // -1240.67 5.00  air           24.0
    //         // 100000  105.00  iris          20.0"
    //         //         .lines();
    //         let spec = "164.12		10.99				SF5			1.673	32.2	54
    // 559.28		0.23				air							54
    // 100.12		11.45				BAF10		1.67	47.1    51
    // 213.54		0.23				air							51
    // 58.04		22.95				LAK9		1.691	54.7	41
    // 2551		2.58				SF5			1.673	32.2	41
    // 32.39		15.66				air							27
    // 10000		15.00				IRIS						25.5
    // -40.42		2.74				SF15		1.699	30.1	25
    // 192.98		27.92				SK16		1.62	60.3	36
    // -55.53		0.23				air							36
    // 192.98		7.98				LAK9		1.691	54.7	35
    // -225.28		0.23				air							35
    // 175.1		8.48				LAK9		1.691	54.7	35
    // -203.54		55.742				air							35";
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
    //                 inputs.push(Input { ray, lambda });
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
