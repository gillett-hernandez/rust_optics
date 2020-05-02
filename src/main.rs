#![allow(unused_imports, dead_code, unused_variables)]
mod lens;
mod math;
mod spectrum;
mod trace;

use lens::*;
use math::*;
use trace::*;

use packed_simd::f32x4;
use rand::prelude::*;

pub extern crate nalgebra as na;
pub use na::{Matrix3, Vector3};

use std::fs::File;
use std::io::prelude::*;

#[allow(unused_mut)]
fn simulate(
    lenses: &Vec<LensElement>,
    mut sampler: &mut Box<dyn Sampler>,
    ray_sampler: fn(&mut Box<dyn Sampler>) -> PlaneRay,
    wavelength_sampler: fn(&mut Box<dyn Sampler>) -> f32,
    iterations: usize,
) -> (Vec<Input>, Vec<Option<Output>>) {
    let mut inputs: Vec<Input> = Vec::new();
    let mut outputs: Vec<Option<Output>> = Vec::new();
    let mut failed = 0;
    for _ in 0..iterations {
        let input = Input {
            ray: ray_sampler(sampler),
            lambda: wavelength_sampler(sampler),
        };
        let output = evaluate(lenses, 0.0, input, 0);
        inputs.push(input);
        if output.is_err() {
            failed += 1;
        }
        outputs.push(output.ok());
    }
    println!(
        "simulated {} rays, {} rays failed to exit the lens assembly",
        iterations, failed
    );
    (inputs, outputs)
}
#[allow(unused_mut)]
fn main() -> std::io::Result<()> {
    //     let lines = "# whatever
    // 65.22    9.60  N-SSK8 1.5 50 24.0
    // -62.03   4.20  N-SF10 1.5 50 24.0
    // -1240.67 5.00  air           24.0
    // 100000  105.00  iris          20.0"
    //         .lines();
    let lines = "164.12		10.99				SF5			1.673	32.2	54
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
-203.54		55.742				air							35"
        .lines();
    let mut lenses: Vec<LensElement> = Vec::new();
    let (mut last_ior, mut last_vno) = (1.0, 0.0);
    for line in lines {
        if line.starts_with("#") {
            continue;
        }
        let lens = LensElement::parse_from(line, last_ior, last_vno).unwrap();
        last_ior = lens.ior;
        last_vno = lens.vno;
        println!("successfully parsed lens {:?}", lens);
        lenses.push(lens);
    }

    let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 20));
    // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
    let plane_ray_sampler = |mut sampler: &mut Box<dyn Sampler>| {
        let Sample2D { x: x1, y: y1 } = sampler.draw_2d();
        let Sample2D { x: x2, y: y2 } = sampler.draw_2d();
        PlaneRay::new(
            35.0 * (1.0 - x1),
            35.0 * (1.0 - x1),
            x2 * 2.0 - 1.0,
            y2 * 2.0 - 1.0,
        )
    };
    let wavelength_sampler = |mut sampler: &mut Box<dyn Sampler>| sampler.draw_1d().x * 0.3 + 0.4;
    let (inputs, outputs) = simulate(
        &lenses,
        &mut sampler,
        ray_sampler,
        wavelength_sampler,
        1000000,
    );

    use std::io::BufWriter;
    let mut file = BufWriter::new(File::create("output.txt")?);
    // for (input, output) in inputs.iter().zip(outputs.iter()) {
    //     file.write(
    //         format!(
    //             "{} {} {} {} {}",
    //             input.ray.origin.x(),
    //             input.ray.origin.y(),
    //             input.ray.direction.x(),
    //             input.ray.direction.y(),
    //             input.lambda
    //         )
    //         .as_bytes(),
    //     );
    //     if let Some(output) = output {
    //         file.write(
    //             format!(
    //                 " {} {} {} {} {}\n",
    //                 output.ray.origin.x(),
    //                 output.ray.origin.y(),
    //                 output.ray.direction.x(),
    //                 output.ray.direction.y(),
    //                 output.tau
    //             )
    //             .as_bytes(),
    //         );
    //     } else {
    //         file.write(b" !\n");
    //     }
    // }
    let lens0: LensElement = lenses[0];
    let mut count = 10;
    for (input, output) in inputs.iter().zip(outputs.iter()) {
        if count > 0 {
            print!(
                "input {:?}, {:?}, {} ",
                input.ray.origin, input.ray.direction, input.lambda,
            );
            if let Some(out) = output {
                let normal_ray = sphere_to_cs(out.ray, -lens0.radius, lens0.radius);
                let (exterior_point, normal) =
                    (normal_ray.origin, normal_ray.direction.normalized());
                let ex = Vec3::new(normal.z(), 0.0, -normal.x()).normalized();
                // normal is in world space, but out origin and direction are not.
                let frame = TangentFrame::from_tangent_and_normal(ex, normal);
                let o_dir = frame.to_world(&out.ray.direction);
                println!(": output {:?}, {:?}, {}", exterior_point, o_dir, out.tau);
            } else {
                println!("");
            }
            count -= 1;
        }
        file.write(
            format!(
                "{} {} {} {} {}",
                input.ray.origin.x(),
                input.ray.origin.y(),
                input.ray.direction.x(),
                input.ray.direction.y(),
                input.lambda
            )
            .as_bytes(),
        )?;
        if let Some(output) = output {
            let normal_ray = sphere_to_cs(output.ray, -lens0.radius, lens0.radius);
            let (exterior_point, normal) = (normal_ray.origin, normal_ray.direction.normalized());
            let ex = Vec3::new(normal.z(), 0.0, -normal.x()).normalized();
            // normal is in world space, but output origin and direction are not.
            let frame = TangentFrame::from_tangent_and_normal(ex, normal);
            let o_dir = frame.to_world(&output.ray.direction);
            file.write(
                format!(
                    " {} {} {} {} {} {} {}\n",
                    exterior_point.x(),
                    exterior_point.y(),
                    exterior_point.z(),
                    o_dir.x(),
                    o_dir.y(),
                    o_dir.z(),
                    output.tau
                )
                .as_bytes(),
            )?;
        } else {
            file.write(b" !\n")?;
        }
    }

    Ok(())
}
