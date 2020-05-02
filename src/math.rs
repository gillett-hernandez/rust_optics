pub use packed_simd::f32x4;
use rand::prelude::*;
use std::f32::INFINITY;
use std::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, RngCore};

#[allow(non_upper_case_globals)]
pub const f32x4_ZERO: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);

#[derive(Copy, Clone, Debug)]
pub struct Vec3(pub f32x4);

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        // Vec3 { x, y, z, w: 0.0 }
        Vec3(f32x4::new(x, y, z, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> Vec3 {
        Vec3(v)
    }
    pub const ZERO: Vec3 = Vec3::from_raw(f32x4::splat(0.0));
    pub const MASK: f32x4 = f32x4::new(1.0, 1.0, 1.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
}

impl Vec3 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    #[inline(always)]
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
}

impl Mul for Vec3 {
    type Output = f32;
    fn mul(self, other: Vec3) -> f32 {
        // self.x * other.x + self.y * other.y + self.z * other.z
        (self.0 * other.0).sum()
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f32) -> Vec3 {
        Vec3::from_raw(self.0 * other)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::from_raw(self * other.0)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f32) -> Vec3 {
        Vec3::from_raw(self.0 / other)
    }
}

// impl Div for Vec3 {
//     type Output = Vec3;
//     fn div(self, other: Vec3) -> Vec3 {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         Vec3::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for Vec3 {
//     type Output = Vec3;
//     fn add(self, other: f32) -> Vec3 {
//         Vec3::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for Vec3 {
//     type Output = Vec3;
//     fn sub(self, other: f32) -> Vec3 {
//         Vec3::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::from_raw(self.0 + other.0)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::from_raw(-self.0)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        self + (-other)
    }
}

impl From<f32> for Vec3 {
    fn from(s: f32) -> Vec3 {
        Vec3::from_raw(f32x4::splat(s) * Vec3::MASK)
    }
}

impl From<Vec3> for f32x4 {
    fn from(v: Vec3) -> f32x4 {
        v.0
    }
}

impl Vec3 {
    pub fn cross(&self, other: Vec3) -> Self {
        let (x1, y1, z1) = (self.x(), self.y(), self.z());
        let (x2, y2, z2) = (other.x(), other.y(), other.z());
        Vec3::new(y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - x2 * y1)
    }

    pub fn norm_squared(&self) -> f32 {
        (self.0 * self.0 * Vec3::MASK).sum()
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        Vec3::from_raw(self.0 / norm)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Point3(pub f32x4);

impl Point3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3(f32x4::new(x, y, z, 1.0))
    }
    pub const fn from_raw(v: f32x4) -> Point3 {
        Point3(v)
    }
    pub const ZERO: Point3 = Point3::from_raw(f32x4::splat(0.0));
}

impl Point3 {
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
    pub fn normalize(mut self) -> Self {
        unsafe {
            self.0 = self.0 / self.0.extract_unchecked(3);
        }
        self
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, other: Vec3) -> Point3 {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        Point3::from_raw(self.0 + other.0)
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;
    fn sub(self, other: Vec3) -> Point3 {
        // Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Point3::from_raw(self.0 - other.0)
    }
}

// // don't implement adding or subtracting floats from Point3, because that's equivalent to adding or subtracting a Vector with components f,f,f and why would you want to do that.

impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, other: Point3) -> Vec3 {
        // Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Vec3::from_raw((self.0 - other.0) * f32x4::new(1.0, 1.0, 1.0, 0.0))
    }
}

// #define INTENSITY_EPS 1e-5
impl From<Point3> for Vec3 {
    fn from(p: Point3) -> Self {
        // Vec3::new(p.x, p.y, p.z)
        Vec3::from_raw(p.0.replace(3, 0.0))
    }
}

impl From<Vec3> for Point3 {
    fn from(v: Vec3) -> Point3 {
        // Point3::from_raw(v.0.replace(3, 1.0))
        Point3::from_raw(v.0).normalize()
    }
}
#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: f32,
    pub tmax: f32,
}

impl Ray {
    pub const fn new(origin: Point3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction,
            time: 0.0,
            tmax: INFINITY,
        }
    }

    pub const fn new_with_time(origin: Point3, direction: Vec3, time: f32) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax: INFINITY,
        }
    }
    pub const fn new_with_time_and_tmax(
        origin: Point3,
        direction: Vec3,
        time: f32,
        tmax: f32,
    ) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax,
        }
    }
    pub fn with_tmax(mut self, tmax: f32) -> Self {
        self.tmax = tmax;
        self
    }
    pub fn at_time(mut self, time: f32) -> Self {
        self.origin = self.point_at_parameter(time);
        self.time = time;
        self
    }
    pub fn point_at_parameter(self, time: f32) -> Point3 {
        self.origin + self.direction * time
    }
}

// also known as an orthonormal basis.
pub struct TangentFrame {
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub normal: Vec3,
}

impl TangentFrame {
    pub fn new(tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Self {
        assert!(
            (tangent * bitangent).abs() < 0.000001,
            "tbit:{:?} * {:?} was != 0",
            tangent,
            bitangent
        );
        assert!(
            (tangent * normal).abs() < 0.000001,
            "tn: {:?} * {:?} was != 0",
            tangent,
            normal
        );
        assert!(
            (bitangent * normal).abs() < 0.000001,
            "bitn:{:?} * {:?} was != 0",
            bitangent,
            normal
        );
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: bitangent.normalized(),
            normal: normal.normalized(),
        }
    }
    pub fn from_tangent_and_normal(tangent: Vec3, normal: Vec3) -> Self {
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: tangent.normalized().cross(normal.normalized()).normalized(),
            normal: normal.normalized(),
        }
    }

    pub fn from_normal(normal: Vec3) -> Self {
        // let n2 = Vec3::from_raw(normal.0 * normal.0);
        // let (x, y, z) = (normal.x(), normal.y(), normal.z());
        let [x, y, z, _]: [f32; 4] = normal.0.into();
        let sign = (1.0 as f32).copysign(z);
        let a = -1.0 / (sign + z);
        let b = x * y * a;
        TangentFrame {
            tangent: Vec3::new(1.0 + sign * x * x * a, sign * b, -sign * x),
            bitangent: Vec3::new(b, sign + y * y * a, -y),
            normal,
        }
    }

    #[inline(always)]
    pub fn to_world(&self, v: &Vec3) -> Vec3 {
        self.tangent * v.x() + self.bitangent * v.y() + self.normal * v.z()
    }

    #[inline(always)]
    pub fn to_local(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
            self.tangent * (*v),
            self.bitangent * (*v),
            self.normal * (*v),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Input {
    pub ray: Ray,
    pub lambda: f32,
}

impl Input {
    pub fn slice(&self) -> [f32; 5] {
        [
            self.ray.origin.x(),
            self.ray.origin.y(),
            self.ray.direction.x(),
            self.ray.direction.y(),
            self.lambda,
        ]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Output {
    pub ray: Ray,
    pub tau: f32,
}

impl Output {
    pub fn slice(&self) -> [f32; 5] {
        [
            self.ray.origin.x(),
            self.ray.origin.y(),
            self.ray.direction.x(),
            self.ray.direction.y(),
            self.tau,
        ]
    }
}

pub struct PlaneRay {
    pub x: f32,
    pub y: f32,
    pub u: f32,
    pub v: f32,
}

pub struct SphereRay {
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
}

#[derive(Debug)]
pub struct Sample1D {
    pub x: f32,
}

impl Sample1D {
    pub fn new_random_sample() -> Self {
        Sample1D { x: random() }
    }
}

#[derive(Debug)]
pub struct Sample2D {
    pub x: f32,
    pub y: f32,
}

impl Sample2D {
    pub fn new_random_sample() -> Self {
        Sample2D {
            x: random(),
            y: random(),
        }
    }
}
#[derive(Debug)]
pub struct Sample3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Sample3D {
    pub fn new_random_sample() -> Self {
        Sample3D {
            x: random(),
            y: random(),
            z: random(),
        }
    }
}

pub trait Sampler {
    fn draw_1d(&mut self) -> Sample1D;
    fn draw_2d(&mut self) -> Sample2D;
    fn draw_3d(&mut self) -> Sample3D;
}

pub struct RandomSampler {}

impl RandomSampler {
    pub const fn new() -> RandomSampler {
        RandomSampler {}
    }
}

impl Sampler for RandomSampler {
    fn draw_1d(&mut self) -> Sample1D {
        Sample1D::new_random_sample()
    }
    fn draw_2d(&mut self) -> Sample2D {
        Sample2D::new_random_sample()
    }
    fn draw_3d(&mut self) -> Sample3D {
        Sample3D::new_random_sample()
    }
}
pub struct StratifiedSampler {
    pub dims: [usize; 3],
    pub indices: [usize; 3],
    pub first: Vec<usize>,
    pub second: Vec<usize>,
    pub third: Vec<usize>,
    rng: Box<dyn RngCore>,
}

impl StratifiedSampler {
    pub fn new(xdim: usize, ydim: usize, zdim: usize) -> Self {
        StratifiedSampler {
            dims: [xdim, ydim, zdim],
            indices: [0, 0, 0],
            first: (0..xdim).into_iter().collect(),
            second: (0..(xdim * ydim)).into_iter().collect(),
            third: (0..(xdim * ydim * zdim)).into_iter().collect(),
            rng: Box::new(thread_rng()),
        }
    }
}

impl Sampler for StratifiedSampler {
    fn draw_1d(&mut self) -> Sample1D {
        if self.indices[0] == 0 {
            // shuffle, then draw.
            self.first.shuffle(&mut self.rng);
            // println!("shuffled strata order for draw_1d");
            // print!(".");
        }
        let idx = self.first[self.indices[0]];
        let (width, depth, height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[0] += 1;
        if self.indices[0] >= width {
            self.indices[0] = 0;
        }
        // convert idx to the "pixel" based on dims
        let mut sample = Sample1D::new_random_sample();
        let x = idx;
        sample.x = sample.x / (width as f32) + (x as f32) / (width as f32);
        sample
    }
    fn draw_2d(&mut self) -> Sample2D {
        if self.indices[1] == 0 {
            // shuffle, then draw.
            self.second.shuffle(&mut self.rng);
            // println!("shuffled strata order for draw_2d");
            // print!("*");
        }
        let idx = self.second[self.indices[1]];
        let (width, depth, height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[1] += 1;
        if self.indices[1] >= width * height {
            self.indices[1] = 0;
        }
        // convert idx to the "pixel" based on dims
        let (x, y) = (idx % width, idx / width);
        let mut sample = Sample2D::new_random_sample();
        sample.x = sample.x / (width as f32) + (x as f32) / (width as f32);
        sample.y = sample.y / (depth as f32) + (y as f32) / (depth as f32);
        sample
    }
    fn draw_3d(&mut self) -> Sample3D {
        if self.indices[2] == 0 {
            // shuffle, then draw.
            self.third.shuffle(&mut self.rng);
            println!("shuffled strata order for draw_3d");
        }
        let (width, depth, height) = (self.dims[0], self.dims[1], self.dims[2]);
        let idx = self.third[self.indices[2]];
        self.indices[2] += 1;
        if self.indices[2] >= width * depth * height {
            self.indices[2] = 0;
        }
        // idx = x + width * y + width * depth * z
        // convert idx to the "pixel" based on dims
        let z = idx / (depth * width);
        // z coordinate is how many slices high the sample is
        let y = (idx / width) % depth;
        // y coordinate is how far into a slice a given "pixel" is
        let x = idx % width;
        // x coordinate is how far along width a given pixel is
        let mut sample = Sample3D::new_random_sample();
        sample.x = sample.x / (width as f32) + (x as f32) / (width as f32);
        sample.y = sample.y / (depth as f32) + (y as f32) / (depth as f32);
        sample.z = sample.z / (height as f32) + (z as f32) / (height as f32);
        sample
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
