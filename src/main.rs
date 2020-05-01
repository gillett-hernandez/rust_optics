mod math;
use math::*;
use packed_simd::f32x4;
use rand::prelude::*;

pub extern crate nalgebra as na;
pub use na::{Matrix3, Vector3};

const f32x4_ZERO: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);

fn main() {
    println!("testing usage of new Vector3 as defined in nalgebra");
    let v1: Vector3<f32> = Vector3::new(1.0, 1.0, 1.0);
    let v2: Vector3<f32> = Vector3::new(1.0, 1.0, 1.0);
    println!("{:?}", v1.dot(&v2));

    println!("testing usage of old Vec3");
    let av1 = Vec3::new(1.0, 1.0, 1.0);
    let av2 = Vec3::new(1.0, 1.0, 1.0);
    println!("{:?}", av1 * av2);

    println!("testing construction of input");
    let input: Input = Input {
        ray: Ray::new(
            Point3::new(random::<f32>() / 10.0, random::<f32>() / 10.0, 0.0),
            Vec3::Z,
        ),
        lambda: 450.0,
    };
    println!("{:?}", input.slice());
    println!("testing trace spherical with given input");
    let result = trace_spherical(input.ray, 0.9, 1.0, 0.9);
    match result {
        Ok((ray, normal)) => {
            println!("{:?}, {:?}", ray, normal);
        }
        Err(error) => {
            println!("error occurred with code {}", error);
        }
    };

    println!("testing evaluate aspherical with given input");
    let result = evaluate_aspherical(input.ray.origin, 0.9, 1, f32x4_ZERO);
    println!("{}", result);
    println!("testing evaluate aspherical derivative with given input");
    let result = evaluate_aspherical_derivative(input.ray.origin, 0.9, 1, f32x4_ZERO);
    println!("{}", result);
}
