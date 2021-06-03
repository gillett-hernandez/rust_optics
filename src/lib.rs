#![allow(unused_imports, dead_code, unused_variables)]
// #![feature(slice_fill)]
extern crate minifb;

#[macro_use]
extern crate packed_simd;

pub mod lens;
pub mod math;
pub mod spectrum;
pub mod trace;

pub use lens::*;
use ::math::*;
pub use crate::math::{Input, Output, PlaneRay, SphereRay};
pub use trace::*;

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


enum Mode {
    Texture,
    PinLight,
    Direction,
}

fn circular_aperture(aperture_radius: f32, ray: Ray) -> bool {
    ray.origin.x().hypot(ray.origin.y()) > aperture_radius
}

fn bladed_aperture(aperture_radius: f32, blades: usize, ray: Ray) -> bool {
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

// the following function only works and applies to lens with radial symmetry
fn recalculate_and_cache_directions<F>(
    radius_cap: f32,
    radius_bins: usize,
    wavelength_bins: usize,
    wavelength_bounds: Bounds1D,
    film_position: f32,
    lens_assembly: &LensAssembly,
    lens_zoom: f32,
    aperture_callback: F,
    solver_heat: f32,
) -> Film<f32x4>
where
    F: Send + Sync + Fn(f32, Ray) -> bool,
{
    // create film of vecs.
    let mut film = Film::new(radius_bins, wavelength_bins, f32x4::splat(0.0));
    let aperture_radius = lens_assembly.aperture_radius();
    film.buffer.par_iter_mut().enumerate().for_each(|(i, v)| {
        let radius_bin = i % radius_bins;
        let wavelength_bin = i / radius_bins;
        let lambda = wavelength_bin as f32 * wavelength_bounds.span() / wavelength_bins as f32
            + wavelength_bounds.lower;
        let radius = radius_cap * radius_bin as f32 / radius_bins as f32;
        // find direction (with fixed y = 0) for sampling aperture and outer pupil, and find corresponding sampling "radius"

        // switch flag to change from random to stratified.
        let ray_origin = Point3::new(radius, 0.0, film_position);
        let mut direction;
        // let mut state = 0;
        loop {
            // directions range from straight forward (0 degrees) to almost critical (90 degrees, tangent)
            if true {
                // random sampling along axis until direction is found.
                let s = Sample1D::new_random_sample();
                let angle = s.x * std::f32::consts::FRAC_PI_2 * 0.97;
                direction = Vec3::new(-angle.sin(), 0.0, angle.cos());
            } else {
                // stratified sampling along axis until direction is found.
                // state += 1;
                panic!();
            }
            let ray = Ray::new(ray_origin, direction);
            let result = lens_assembly.trace_forward(lens_zoom, &Input { ray, lambda }, 1.0, |e| {
                (aperture_callback(aperture_radius, e), false)
            });
            if let Some(Output { .. }) = result {
                // found good direction, so break
                break;
            }
        }
        // expand around direction to find radius and correct centroid.
        // measured in radians.
        let mut min_angle: f32 = 0.0;
        let mut max_angle: f32 = 0.0;
        let mut radius = 0.0;
        let mut sum_angle = 0.0;
        let mut valid_angle_count = 0;

        // maybe rewrite this as tree search?
        'outer: loop {
            radius += solver_heat;
            let mut ct = 0;
            for mult in vec![-1.0, 1.0] {
                let old_angle = (-direction.x() / direction.z()).atan();
                let new_angle = old_angle + radius * mult;
                let new_direction = Vec3::new(-new_angle.sin(), 0.0, new_angle.cos());
                let ray = Ray::new(ray_origin, new_direction);
                let result =
                    lens_assembly.trace_forward(lens_zoom, &Input { ray, lambda }, 1.0, |e| {
                        (aperture_callback(aperture_radius, e), false)
                    });
                if let Some(Output { .. }) = result {
                    // found good direction. keep expanding.
                    max_angle = max_angle.max(new_angle);
                    min_angle = min_angle.min(new_angle);
                    sum_angle += new_angle;
                    valid_angle_count += 1;
                } else {
                    // found bad direction with this mult. keep expanding until both sides are bad.
                    ct += 1;
                    if ct == 2 {
                        // both sides are bad. break.
                        break 'outer;
                    }
                }
            }
        }
        let avg_angle = sum_angle / (valid_angle_count as f32);

        *v = f32x4::new(avg_angle, (max_angle - min_angle).abs() / 2.0, 0.0, 0.0);
    });
    film
}

fn simulate_phase1(
    lenses: &Vec<LensElement>,
    inputs: &Vec<Input<PlaneRay>>,
) -> Vec<Option<Output<SphereRay>>> {
    let mut outputs: Vec<Option<Output<SphereRay>>> = Vec::new();
    let mut failed = 0;
    for input in inputs {
        let output = evaluate(lenses, 0.0, input, 0, 1.04);
        if output.is_err() {
            failed += 1;
        }
        outputs.push(output.ok());
    }
    println!(
        "simulated {} rays, {} rays failed to exit the lens assembly",
        inputs.len(),
        failed
    );
    outputs
}

fn simulate_phase2(
    lenses: &Vec<LensElement>,
    inputs: &Vec<Input<PlaneRay>>,
) -> Vec<Option<Output<PlaneRay>>> {
    let mut outputs: Vec<Option<Output<PlaneRay>>> = Vec::new();
    let mut failed = 0;
    for input in inputs {
        let output = evaluate_aperture(lenses, 0.0, input, 0, 1.04);
        if output.is_err() {
            failed += 1;
        }
        outputs.push(output.ok());
    }
    println!(
        "simulated {} rays, {} rays failed to exit the lens assembly",
        inputs.len(),
        failed
    );
    outputs
}

pub fn parse_lenses_from(spec: &str) -> (Vec<LensElement>, f32, f32) {
    let lines = spec.lines();
    let mut lenses: Vec<LensElement> = Vec::new();
    let (mut last_ior, mut last_vno) = (1.0, 0.0);
    for line in lines {
        if line.starts_with("#") {
            continue;
        }
        let lens = LensElement::parse_from(line, last_ior, last_vno).unwrap();
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
    #[test]
    fn test() {
        //     let lines = "# whatever
        // 65.22    9.60  N-SSK8 1.5 50 24.0
        // -62.03   4.20  N-SF10 1.5 50 24.0
        // -1240.67 5.00  air           24.0
        // 100000  105.00  iris          20.0"
        //         .lines();
        let spec = "164.12		10.99				SF5			1.673	32.2	54
559.28		0.23				air							54
100.12		11.45				BAF10		1.67	47.1    51
213.54		0.23				air							51
58.04		22.95				LAK9		1.691	54.7	41
2551		2.58				SF5			1.673	32.2	41
32.39		15.66				air							27
10000		15.00				IRIS						25.5
-40.42		2.74				SF15		1.699	30.1	25
192.98		27.92				SK16		1.62	60.3	36
-55.53		0.23				air							36
192.98		7.98				LAK9		1.691	54.7	35
-225.28		0.23				air							35
175.1		8.48				LAK9		1.691	54.7	35
-203.54		55.742				air							35";
        let (lenses, last_ior, last_vno) = parse_lenses_from(spec);

        let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 20));
        // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        const ZOOM: f32 = 0.0;

        let dist = LensElement::total_thickness_at(lenses.as_slice(), ZOOM);
        println!("total lens thickness is {}", dist);
        let plane_ray_sampler = |x: f32, y: f32, sampler: &mut Box<dyn Sampler>| {
            let Sample2D { x: u, y: v } = sampler.draw_2d();
            let theta = 2.0 * 3.1415926535 * u;
            let (sin, cos) = theta.sin_cos();
            let R = lenses.last().unwrap().housing_radius;
            let x = 35.0 * (x - 0.5);
            let y = 35.0 * (y - 0.5);
            let mut plane_ray = PlaneRay::new(
                x,
                y,
                R / dist * cos * v.sqrt() - x / dist,
                R / dist * sin * v.sqrt() - y / dist,
            );
            plane_ray
        };
        let wavelength_sampler = |sampler: &mut Box<dyn Sampler>| sampler.draw_1d().x * 0.3 + 0.4;

        let mut inputs: Vec<Input<PlaneRay>> = Vec::new();
        let w = 1000;
        let h = 1000;
        for y in 0..h {
            for x in 0..w {
                let ray = plane_ray_sampler(x as f32 / w as f32, y as f32 / h as f32, &mut sampler);
                let lambda = wavelength_sampler(&mut sampler);
                inputs.push(Input { ray, lambda });
            }
        }

        let outputs1 = simulate_phase1(&lenses, &inputs);
        let outputs2 = simulate_phase2(&lenses, &inputs);

        use std::io::BufWriter;
        let mut file1 = BufWriter::new(File::create("output1.txt")?);
        let mut file2 = BufWriter::new(File::create("output2.txt")?);
        let lens0: LensElement = lenses[0];
        let mut count = 10;
        for (input, output) in inputs.iter().zip(outputs1.iter()) {
            if count > 0 {
                print!("input {:?}, {} ", input.ray, input.lambda,);
                if let Some(out) = output {
                    let ray = sphere_to_camera_space(out.ray, -lens0.radius, lens0.radius);
                    println!(
                        ": output {:?}, {:?}, {}",
                        ray.origin, ray.direction, out.tau
                    );
                } else {
                    println!("");
                }
                count -= 1;
            }
            file1.write(
                format!(
                    "{} {} {} {} {}",
                    input.ray.x(),
                    input.ray.y(),
                    input.ray.dx(),
                    input.ray.dy(),
                    input.lambda
                )
                .as_bytes(),
            )?;
            if let Some(out) = output {
                // let ray = sphere_to_camera_space(out.ray, -lens0.radius, lens0.radius);
                file1.write(
                    format!(
                        " {} {} {} {} {}\n",
                        out.ray.x(),
                        out.ray.y(),
                        out.ray.dx(),
                        out.ray.dy(),
                        out.tau
                    )
                    .as_bytes(),
                )?;
            } else {
                file1.write(b" !\n")?;
            }
        }
        let mut count = 10;
        let aperture_pos = LensElement::aperture_position(lenses.as_slice(), ZOOM);
        for (input, output) in inputs.iter().zip(outputs2.iter()) {
            if count > 0 {
                print!("input {:?}, {} ", input.ray, input.lambda,);
                if let Some(out) = output {
                    let ray = plane_to_camera_space(out.ray, aperture_pos);
                    println!(
                        ": output {:?}, {:?}, {}",
                        ray.origin, ray.direction, out.tau
                    );
                } else {
                    println!("");
                }
                count -= 1;
            }
            file2.write(
                format!(
                    "{} {} {} {} {}",
                    input.ray.x(),
                    input.ray.y(),
                    input.ray.dx(),
                    input.ray.dy(),
                    input.lambda
                )
                .as_bytes(),
            )?;
            if let Some(out) = output {
                file2.write(
                    format!(
                        " {} {} {} {} {}\n",
                        out.ray.x(),
                        out.ray.y(),
                        out.ray.dx(),
                        out.ray.dy(),
                        out.tau
                    )
                    .as_bytes(),
                )?;
            } else {
                file2.write(b" !\n")?;
            }
        }

        Ok(())
    }
}
