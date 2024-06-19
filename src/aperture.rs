use math::prelude::*;

pub trait Aperture {
    fn intersects(&self, aperture_radius: f32, ray: Ray) -> bool;
}

#[derive(Default, Copy, Clone, Debug)]
pub struct CircularAperture {}

impl Aperture for CircularAperture {
    fn intersects(&self, aperture_radius: f32, ray: Ray) -> bool {
        ray.origin.x().hypot(ray.origin.y()) > aperture_radius
    }
}
#[derive(Copy, Clone, Debug)]
pub struct SimpleBladedAperture {
    blades: u8,
    p: f32,
}

impl SimpleBladedAperture {
    pub fn new(blades: u8, p: f32) -> Self {
        assert!(blades >= 3);
        assert!(p > 0.0);
        Self { blades, p }
    }
}

impl Aperture for SimpleBladedAperture {
    fn intersects(&self, aperture_radius: f32, ray: Ray) -> bool {
        match self.blades {
            3..=10 => {
                let mut p = ray.origin;
                p.0[2] = 0.0;
                let mut theta = p.y().atan2(p.x());
                theta %= std::f32::consts::TAU / self.blades as f32;
                let cos = theta.cos();

                let v = self.p / (self.p + cos);
                let dist = p.x().hypot(p.y());
                dist < v
            }
            _ => CircularAperture::intersects(&CircularAperture::default(), aperture_radius, ray),
        }
    }
}
macro_rules! generate_enum {
    ($($e:ident),+) => {
        #[derive(Clone, Debug)]
        pub enum ApertureEnum {
            $(
                $e($e),
            )+
        }

        impl Aperture for ApertureEnum {
            fn intersects(&self, aperture_radius: f32, ray: Ray) -> bool {
                match self {
                    $(
                        ApertureEnum::$e(inner) => inner.intersects(aperture_radius, ray),
                   )+
                }
            }
        }
    }
}

generate_enum! {CircularAperture, SimpleBladedAperture}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_bladed_aperture() {
        let bladed = SimpleBladedAperture::new(6, 0.5);
    }
}
