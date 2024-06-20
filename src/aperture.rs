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
    pub max_radius: f32,
}

impl SimpleBladedAperture {
    pub fn new(blades: u8, p: f32) -> Self {
        assert!(blades >= 3);
        assert!(p > 0.0);
        let recip_p = p.recip();
        let cos = (std::f32::consts::PI / blades as f32).cos();
        Self {
            blades,
            p,
            recip_p,
            max_radius: (1.0 + cos * recip_p).recip(),
        }
    }
}

impl Aperture for SimpleBladedAperture {
    fn is_rejected(&self, aperture_radius: f32, mut p: Point3) -> bool {
        p.0[2] = 0.0;
        p.0 = p.0 / f32x4::splat(aperture_radius);
        p = p.normalize();
        let repeat_angle = std::f32::consts::TAU / self.blades as f32;
        match self.blades {
            3..=10 => {
                let mut theta = p.y().atan2(p.x());
                theta = theta.rem_euclid(repeat_angle);
                theta -= repeat_angle / 2.0;
                let cos = theta.cos();

                // let v = self.p / (self.p + cos);
                // let 1/v = (1 + cos / self.p);
                let v = (1.0 + cos * self.recip_p).recip();
                let dist = p.x().hypot(p.y());
                dist > v
            }
            _ => CircularAperture::default().is_rejected(aperture_radius, p),
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
    fn sample(&self, sample: Sample2D) -> Result<Point3, ()>;
}

impl ApertureSample for CircularAperture {
    fn sample(&self, sample: Sample2D) -> Result<Point3, ()> {
        Ok(random_in_unit_disk(sample).into())
    }
}

impl ApertureSample for SimpleBladedAperture {
    fn sample(&self, sample: Sample2D) -> Result<Point3, ()> {
        let p: Point3 = (random_in_unit_disk(sample) * self.max_radius).into(); // max radius is less than 1.0, multiplies down our sample space to improve efficiency.
        (!self.is_rejected(1.0, p)).then_some(p).ok_or(())
    }
}

impl ApertureSample for ApertureEnum {
    fn sample(&self, sample: Sample2D) -> Result<Point3, ()> {
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

        assert!(!bladed.is_rejected(1.0, Point3::new(0.49, 0.0, 0.0)));
        assert!(bladed.is_rejected(1.0, Point3::new(0.51, 0.0, 0.0)));
    }

    #[test]
    fn test_bladed_aperture_sample() {
        let bladed = SimpleBladedAperture::new(3, 0.5);

        let mut total_attempts = 0;

        for _ in 0..10000 {
            loop {
                let s = Sample2D::new_random_sample();
                let maybe_point = bladed.sample(s);

                total_attempts += 1;
                if let Ok(point) = maybe_point {
                    print!("({}, {}),", point.x(), point.y());
                    break;
                }
            }
        }
        println!();
        println!("iterations per sample: {}", total_attempts as f32 / 10000.0);
    }
}
