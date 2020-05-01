mod lens;
mod math;
mod spectrum;
mod trace;

use lens::*;
use math::*;
use packed_simd::f32x4;
use rand::prelude::*;
use trace::*;

pub extern crate nalgebra as na;
pub use na::{Matrix3, Vector3};

fn main() {
    println!("testing usage of new Vector3 as defined in nalgebra");
    let v1: Vector3<f32> = Vector3::new(1.0, 1.0, 1.0);
    let v2: Vector3<f32> = Vector3::new(1.0, 1.0, 1.0);
    println!("{:?}", v1.dot(&v2));

    let lines = "# whatever
65.22    9.60  N-SSK8 1.5 50 24.0
-62.03   4.20  N-SF10 1.5 50 24.0
-1240.67 5.00  air           24.0
100000  105.00  iris          20.0"
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
    for _ in 0..100 {
        let input: Input = Input {
            ray: Ray::new(
                Point3::ZERO + Vec3::new(random_between(-1.0, 1.0), random_between(-1.0, 1.0), 0.0),
                Vec3::new(random_between(-1.0, 1.0), random_between(-1.0, 1.0), 3.0).normalized(),
            ),
            lambda: 0.450,
        };
        let res = evaluate(&lenses, 0.0, input, 0);
        match res {
            Ok(output) => {
                println!(
                    "ray on outer pupil, transmittance: {:?}, {}",
                    output.ray, output.tau
                );
            }
            Err(error) => {
                print!("{} ", error);
            }
        }
    }
}
