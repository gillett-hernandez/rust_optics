use std::simd::f32x4;

use math::prelude::*;

pub trait Aperture {
    /// returns whether the specific point would be / is rejected by the aperture
    fn is_rejected(&self, aperture_radius: f32, p: Point3) -> bool;
}

#[derive(Default, Copy, Clone, Debug)]
pub struct CircularAperture {}

impl Aperture for CircularAperture {
    fn is_rejected(&self, aperture_radius: f32, p: Point3) -> bool {
        p.x().hypot(p.y()) > aperture_radius
    }
}
#[derive(Copy, Clone, Debug)]
pub struct SimpleBladedAperture {
    pub blades: u8,
    pub p: f32,
    recip_p: f32,
}

impl SimpleBladedAperture {
    pub fn new(blades: u8, p: f32) -> Self {
        assert!(blades >= 3);
        assert!(p > 0.0);
        Self {
            blades,
            p,
            recip_p: p.recip(),
        }
    }
}

impl Aperture for SimpleBladedAperture {
    fn is_rejected(&self, aperture_radius: f32, mut p: Point3) -> bool {
        p.0[2] = 0.0;
        p.0 = p.0 / f32x4::splat(aperture_radius);
        p = p.normalize();
        let extent = std::f32::consts::TAU / self.blades as f32;
        match self.blades {
            3..=10 => {
                let mut theta = p.y().atan2(p.x());
                theta %= extent;
                theta -= extent / 2.0;
                #[cfg(test)]
                println!("{}", theta);
                let cos = theta.cos();

                // let v = self.p / (self.p + cos);
                // let 1/v = (1 + cos / self.p);
                let v = (1.0 + cos * self.recip_p).recip();
                let dist = p.x().hypot(p.y());
                println!("{}, {}", dist, v);
                dist < v
            }
            _ => CircularAperture::is_rejected(&CircularAperture::default(), aperture_radius, p),
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
            fn is_rejected(&self, aperture_radius: f32, p: Point3) -> bool {
                match self {
                    $(
                        ApertureEnum::$e(inner) => inner.is_rejected(aperture_radius, p),
                   )+
                }
            }
        }
    }
}

generate_enum! {CircularAperture, SimpleBladedAperture}

pub trait ApertureSample {
    /// samples the aperture given a random sample
    /// the sample is constrained to lie within the unit circle with a z value of zero,
    /// so scale it to the actual aperture size where appropriate
    fn sample(&self, sample: Sample2D) -> Point3;
}

impl ApertureSample for CircularAperture {
    fn sample(&self, sample: Sample2D) -> Point3 {
        random_in_unit_disk(sample).into()
    }
}

impl ApertureSample for SimpleBladedAperture {
    fn sample(&self, mut sample: Sample2D) -> Point3 {
        // rejection sampling
        // the subtractions and such might introduce bias
        // TODO: measure and address bias
        let p: Point3 = random_in_unit_disk(sample).into();
        if !self.is_rejected(1.0, p) {
            return p;
        }

        sample.x = 1.0 - sample.x;
        let p: Point3 = random_in_unit_disk(sample).into();
        if !self.is_rejected(1.0, p) {
            return p;
        }

        sample.y = 1.0 - sample.y;
        let p: Point3 = random_in_unit_disk(sample).into();
        if !self.is_rejected(1.0, p) {
            return p;
        }

        sample.x = 1.0 - sample.x;
        sample.y = 1.0 - sample.y;
        let p: Point3 = random_in_unit_disk(sample).into();
        if !self.is_rejected(1.0, p) {
            return p;
        }
        // give up
        Point3::ORIGIN
    }
}

impl ApertureSample for ApertureEnum {
    fn sample(&self, sample: Sample2D) -> Point3 {
        match self {
            ApertureEnum::CircularAperture(inner) => inner.sample(sample),
            ApertureEnum::SimpleBladedAperture(inner) => inner.sample(sample),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_bladed_aperture() {
        let bladed = SimpleBladedAperture::new(3, 0.5);

        assert!(bladed.is_rejected(1.0, Point3::new(0.49, 0.0, 0.0)));
        assert!(!bladed.is_rejected(1.0, Point3::new(0.51, 0.0, 0.0)));
    }
}
