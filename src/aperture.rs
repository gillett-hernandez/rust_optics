use math::{Ray, Vec3};

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
        assert!((-2.0..2.4).contains(&p));
        Self { blades, p: 0.5 }
    }
}

impl Aperture for SimpleBladedAperture {
    fn intersects(&self, aperture_radius: f32, ray: Ray) -> bool {
        match self.blades {
            6 => {
                let phi = std::f32::consts::PI / 3.0;
                let top = Vec3::new(phi.cos(), phi.sin(), 0.0);
                let bottom = Vec3::new(phi.cos(), -phi.sin(), 0.0);
                let mut point = Vec3::from(ray.origin);
                point.0 = point.0.replace(2, 0.0);
                let normalized = point.normalized();
                let cos_top = normalized * top;
                let cos_bottom = normalized * bottom;
                let cos_apex = normalized.x();
                let minimum = ((1.0 + cos_top.abs().powf(self.p)) / cos_top.abs())
                    .min((1.0 + cos_bottom.abs().powf(self.p)) / cos_bottom.abs())
                    .min((1.0 + cos_apex.abs().powf(self.p)) / cos_apex.abs());
                point.x().hypot(point.y()) > minimum * aperture_radius
                // 1.0 > minimum * aperture_radius
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
