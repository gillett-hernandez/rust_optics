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

    // format is:
    // lens := radius thickness_short(/thickness_mid(/thickness_long)?)? (anamorphic)? (mtl_name|'air'|'iris') ior vno housing_radius ('#!aspheric='aspheric_correction)?
    // radius := float
    // thickness_short := float
    // thickness_mid := float
    // thickness_long := float
    // anamorphic := 'cx_'
    // mtl_name := word
    // ior := float
    // vno := float
    // housing_radius := float
    // aspheric_correction := (float','){3}float

    let data = "# whatever
65.22    9.60  N-SSK8 1.5 50 24.0
-62.03   4.20  N-SF10 1.5 50 24.0
-1240.67 5.00  air           24.0
100000  105.00  iris          20.0";
    let mut lenses: Vec<LensElement> = Vec::new();
    lenses.push(LensElement {
        radius: 65.22,
        thickness_short: 9.6,
        thickness_mid: 9.6,
        thickness_long: 9.6,
        anamorphic: false,
        lens_type: LensType::Solid,
        ior: 1.5,
        vno: 50.0,
        housing_radius: 24.0,
        aspheric: 0,
        correction: f32x4_ZERO,
    });
    lenses.push(LensElement {
        radius: -62.03,
        thickness_short: 4.2,
        thickness_mid: 4.2,
        thickness_long: 4.2,
        anamorphic: false,
        lens_type: LensType::Solid,
        ior: 1.5,
        vno: 50.0,
        housing_radius: 24.0,
        aspheric: 0,
        correction: f32x4_ZERO,
    });
    lenses.push(LensElement {
        radius: -1240.67,
        thickness_short: 5.00,
        thickness_mid: 5.00,
        thickness_long: 5.00,
        anamorphic: false,
        lens_type: LensType::Air,
        ior: 1.0,
        vno: 0.0,
        housing_radius: 24.0,
        aspheric: 0,
        correction: f32x4_ZERO,
    });
    lenses.push(LensElement {
        radius: 10000.0,
        thickness_short: 105.0,
        thickness_mid: 105.0,
        thickness_long: 105.0,
        anamorphic: false,
        lens_type: LensType::Aperture,
        ior: 1.0,
        vno: 0.0,
        housing_radius: 20.0,
        aspheric: 0,
        correction: f32x4_ZERO,
    });
    for _ in 0..100 {
        let input: Input = Input {
            ray: Ray::new(
                Point3::ZERO + Vec3::new(random::<f32>() - 0.5, random::<f32>() - 0.5, 0.0),
                Vec3::new(random::<f32>() - 0.5, random::<f32>() - 0.5, 3.0).normalized(),
            ),
            lambda: 450.0,
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
                println!("errored with code {}", error);
            }
        }
    }
}
