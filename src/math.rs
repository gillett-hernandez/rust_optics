pub(crate) use math::prelude::*;
pub(crate) use std::simd::{
    cmp::{SimdPartialEq, SimdPartialOrd},
    f32x4,
    num::{SimdFloat, SimdInt},
    simd_swizzle, StdFloat,
};

#[derive(Copy, Clone, Debug)]
pub struct Input<T> {
    pub ray: T,
    pub lambda: f32,
}

impl<T> Input<T> {
    pub fn new(ray: T, lambda: f32) -> Self {
        Self { ray, lambda }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Output<T> {
    pub ray: T,
    // transmittance
    pub tau: f32,
}

impl<T> Output<T> {
    pub fn new(ray: T, tau: f32) -> Self {
        Self { ray, tau }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PlaneRay(pub f32x4);

impl PlaneRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self(f32x4::from_array([x, y, dx, dy]))
    }
    pub fn x(&self) -> f32 {
        self.0[0]
    }
    pub fn y(&self) -> f32 {
        self.0[1]
    }
    pub fn dx(&self) -> f32 {
        self.0[2]
    }
    pub fn dy(&self) -> f32 {
        self.0[3]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SphereRay(pub f32x4);

impl SphereRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self(f32x4::from_array([x, y, dx, dy]))
    }
    pub fn x(&self) -> f32 {
        self.0[0]
    }
    pub fn y(&self) -> f32 {
        self.0[1]
    }
    pub fn dx(&self) -> f32 {
        self.0[2]
    }
    pub fn dy(&self) -> f32 {
        self.0[3]
    }
}

// impl From<SphereRay> for PlaneRay {
//     fn from(other: SphereRay) -> Self {
//         // should probably not just blindly convert
//         Self( other.0 )
//     }
// }

// impl From<PlaneRay> for SphereRay {
//     fn from(other: PlaneRay) -> Self {
//         // should probably not just blindly convert
//         Self ( other.0 )
//     }
// }

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
        for _ in 0..1000000 {
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
        for _ in 0..1000000 {
            let sample = sampler.draw_1d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            s += function(sample.x);
        }
        println!("{}", s / 1000000.0);
    }
    #[test]
    fn test_stratified_sampler_2d() {
        let mut sampler = Box::new(StratifiedSampler::new(10, 10, 10));

        for _ in 0..1000000 {
            let sample = sampler.draw_2d();
            assert!(0.0 <= sample.x && sample.x <= 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y <= 1.0, "{}", sample.y);
        }
    }
    #[test]
    fn test_stratified_sampler_3d() {
        let mut sampler = Box::new(StratifiedSampler::new(10, 10, 10));

        for _ in 0..1000000 {
            let sample = sampler.draw_3d();
            assert!(0.0 <= sample.x && sample.x <= 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y <= 1.0, "{}", sample.y);
            assert!(0.0 <= sample.z && sample.z <= 1.0, "{}", sample.z);
        }
    }
}
