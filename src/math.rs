pub use packed_simd::f32x4;
use rand::prelude::*;
use std::f32::INFINITY;
use std::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, RngCore};

#[allow(non_upper_case_globals)]
pub const f32x4_ZERO: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);

pub use math::{
    Point3, RandomSampler, Ray, Sample1D, Sample2D, Sample3D, Sampler, StratifiedSampler,
    TangentFrame, Vec3,
};

#[derive(Copy, Clone, Debug)]
pub struct Input<T> {
    pub ray: T,
    pub lambda: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Output<T> {
    pub ray: T,
    pub tau: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct PlaneRay(pub f32x4);

impl PlaneRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self {
            0: f32x4::new(x, y, dx, dy),
        }
    }
    pub fn x(&self) -> f32 {
        self.0.extract(0)
    }
    pub fn y(&self) -> f32 {
        self.0.extract(1)
    }
    pub fn dx(&self) -> f32 {
        self.0.extract(2)
    }
    pub fn dy(&self) -> f32 {
        self.0.extract(3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SphereRay(pub f32x4);

impl SphereRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self {
            0: f32x4::new(x, y, dx, dy),
        }
    }
    pub fn x(&self) -> f32 {
        self.0.extract(0)
    }
    pub fn y(&self) -> f32 {
        self.0.extract(1)
    }
    pub fn dx(&self) -> f32 {
        self.0.extract(2)
    }
    pub fn dy(&self) -> f32 {
        self.0.extract(3)
    }
}

impl From<SphereRay> for PlaneRay {
    fn from(other: SphereRay) -> Self {
        Self { 0: other.0 }
    }
}

impl From<PlaneRay> for SphereRay {
    fn from(other: PlaneRay) -> Self {
        Self { 0: other.0 }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    fn function(x: f32) -> f32 {
        x * x - x + 1.0
    }
    #[test]
    fn test_random_sampler_1d() {
        let mut sampler = Box::new(RandomSampler::new());
        let mut s = 0.0;
        for i in 0..1000000 {
            let sample = sampler.draw_1d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            s += function(sample.x);
        }
        println!("{}", s / 1000000.0);
    }
    #[test]
    fn test_stratified_sampler_1d() {
        let mut sampler = Box::new(StratifiedSampler::new(10, 10, 10));
        let mut s = 0.0;
        for i in 0..1000000 {
            let sample = sampler.draw_1d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            s += function(sample.x);
        }
        println!("{}", s / 1000000.0);
    }
    #[test]
    fn test_stratified_sampler_2d() {
        let mut sampler = Box::new(StratifiedSampler::new(10, 10, 10));

        for i in 0..1000000 {
            let sample = sampler.draw_2d();
            assert!(0.0 <= sample.x && sample.x <= 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y <= 1.0, "{}", sample.y);
        }
    }
    #[test]
    fn test_stratified_sampler_3d() {
        let mut sampler = Box::new(StratifiedSampler::new(10, 10, 10));

        for i in 0..1000000 {
            let sample = sampler.draw_3d();
            assert!(0.0 <= sample.x && sample.x <= 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y <= 1.0, "{}", sample.y);
            assert!(0.0 <= sample.z && sample.z <= 1.0, "{}", sample.z);
        }
    }
}
