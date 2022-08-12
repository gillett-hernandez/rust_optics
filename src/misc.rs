pub use crate::math::{Input, Output, PlaneRay, SphereRay};
use math::*;

pub trait Cycle {
    fn cycle(self) -> Self;
}

#[derive(Copy, Clone, Debug)]
pub enum SceneMode {
    // diffuse emitter texture
    TexturedWall { distance: f32, texture_scale: f32 },
    // small diffuse lights
    PinLight,

    // spot light shining with a specific angle
    SpotLight { pos: Vec3, size: f32, span: f32 },
}

impl Cycle for SceneMode {
    fn cycle(self) -> Self {
        match self {
            SceneMode::TexturedWall { .. } => SceneMode::PinLight,
            SceneMode::PinLight => SceneMode::SpotLight {
                pos: Vec3::ZERO + 100.0 * Vec3::Z,
                size: 0.1,
                span: 0.99,
            },
            // defaults to 5000mm == 5meters away
            SceneMode::SpotLight { .. } => SceneMode::TexturedWall {
                distance: 5000.0,
                texture_scale: 1.0,
            },
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ViewMode {
    Film,
    SpotOnFilm(f32, f32),
}
impl Cycle for ViewMode {
    fn cycle(self) -> Self {
        match self {
            ViewMode::Film => ViewMode::SpotOnFilm(0.0, 0.0),
            ViewMode::SpotOnFilm(_, _) => ViewMode::Film,
        }
    }
}
