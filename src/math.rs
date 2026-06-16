pub use ::math::prelude::*;

// NB: `Simd`, `LinAlg3Register` and `Vector` are all re-exported by the prelude
// glob above. We deliberately do *not* `use` `Simd`/`LinAlg3Register` by name —
// a private import would shadow the public glob re-export and trip the
// `hidden_glob_reexports` lint — so the bound below is written fully-qualified.

/// Marker for any thermite SIMD backend usable as the geometry backend in this
/// crate: a [`Simd`](thermite::simd::Simd) whose 4-lane f32 register supports
/// the 3D linear-algebra ops (`dot3`/`cross3`) that `Vec3`, `Point3`, `Ray`,
/// `TangentFrame` and `XYZColor` rely on.
///
/// This crate stays fully generic over the backend: every geometric type is
/// `Type<S>` and every function is generic over `S: SimdBackend`. A concrete
/// backend (e.g. [`thermite::backend::x86_v3::X86V3`], the AVX2+FMA backend) is
/// chosen only at the edges — tests, benches, examples and downstream binaries.
/// Build those with `RUSTFLAGS="-C target-cpu=native"` so the AVX2 paths are
/// actually emitted.
///
/// The blanket impl means you never implement this by hand; it applies to every
/// qualifying backend automatically. As a supertrait the associated-type bound
/// is elaborated, so `S: SimdBackend` implies `S::f32x4: LinAlg3Register` at
/// every use site without restating the where-clause.
pub trait SimdBackend:
    thermite::simd::Simd<f32x4: thermite::register::LinAlg3Register>
{
}
impl<S> SimdBackend for S where
    S: thermite::simd::Simd<f32x4: thermite::register::LinAlg3Register>
{
}

/// 4-lane f32 register vector for backend `S`. Replaces the old
/// `std::simd::f32x4`; generic so callers pick the backend.
pub type F32x4<S> = Vector<<S as thermite::simd::Simd>::f32x4>;

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
pub struct PlaneRay(pub [f32; 4]);

impl PlaneRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self([x, y, dx, dy])
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
pub struct SphereRay(pub [f32; 4]);

impl SphereRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self([x, y, dx, dy])
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
