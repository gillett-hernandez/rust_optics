use ::math::bounds::Bounds1D;

use crate::aperture::Aperture;
use crate::math::*;
use crate::vec2d::Vec2D;
use crate::*;

#[derive(Debug, Default, Copy, Clone)]
pub struct CacheCell {
    pub angle: f32,        // chief-ray angle from +Z, radians
    pub angle_spread: f32, // min enclosing circle half-angle of the aperture view, radians
    pub eccentricity: f32, // ellipse eccentricity of the aperture view, 0 = circular
}

impl CacheCell {
    pub fn new(angle: f32, angle_spread: f32, eccentricity: f32) -> Self {
        CacheCell {
            angle,
            angle_spread,
            eccentricity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RadialSampler {
    pub cache: Vec2D<CacheCell>,
    pub sensor_size: f32,
    pub wavelength_bounds: Bounds1D,
    pub wavelength_bins: usize,
    pub radius_bins: usize,
}

impl RadialSampler {
    // the following function only works and applies to lens with radial symmetry.
    // this is most lenses, but there are a few that are anamorphic
    pub fn new<S, A>(
        radius_cap: f32,
        radius_bins: usize,
        wavelength_bins: usize,
        wavelength_bounds: Bounds1D,
        film_position: f32,
        lens_assembly: &LensAssembly,
        lens_zoom: f32,
        aperture: &A,
        solver_heat: f32,
        sensor_size: f32,
    ) -> Self
    where
        S: SimdBackend,
        A: Send + Sync + Aperture,
    {
        // create film of [f32; 4]s
        let mut film = Vec2D::new(radius_bins, wavelength_bins, CacheCell::default());
        let aperture_radius = lens_assembly.aperture_radius();
        film.buffer.par_iter_mut().enumerate().for_each(|(i, v)| {
            let radius_bin = i % radius_bins;
            let wavelength_bin = i / radius_bins;

            let lambda =
                wavelength_bounds.sample((0.5 + wavelength_bin as f32) / wavelength_bins as f32);
            let radius = radius_cap * radius_bin as f32 / radius_bins as f32;
            // find direction (with fixed y = 0) for sampling aperture and outer pupil, and find corresponding sampling "radius"

            // The cache is indexed by film radius only, so the stored cone must cover
            // the aperture's image for *every* azimuth of the film point — i.e. the
            // union of the (possibly bladed) aperture over all blade orientations.
            // The union of a polygon over all rotations about its center is exactly
            // its circumscribed disk, so we build the cone against a *circular*
            // aperture at the circumscribed radius. The boundary is then the image of
            // a circle (a smooth ellipse), which yields smooth, deterministic cone
            // parameters. Rays inside the cone but outside the real aperture are
            // rejected at trace time, so the blade shape and rim effects survive.
            let bound = aperture.bounding_radius(aperture_radius);
            let passes = |origin: Point3<S>, dir: Vec3<S>| -> bool {
                lens_assembly
                    .trace_forward(
                        lens_zoom,
                        Input::new(Ray::new(origin, dir), lambda / 1000.0),
                        1.0,
                        |e| (e.origin.x().hypot(e.origin.y()) > bound, false),
                        drop,
                    )
                    .is_some()
            };

            let ray_origin: Point3<S> = Point3::new(radius, 0.0, film_position);
            let mut direction: Vec3<S> = Vec3::z_axis();
            let mut found_valid = false;
            const GRAZING_ANGLE_MARGIN: f32 = 0.7;
            const DIRECTION_SEARCH_SAMPLES: usize = 1000;
            let max_search_angle = std::f32::consts::FRAC_PI_2 * GRAZING_ANGLE_MARGIN;
            // Deterministic uniform sweep over [-max, max) of meridional angles (0 =
            // straight ahead, ±max ≈ grazing) for any direction interior to the cone.
            // Its exact value doesn't matter — the bisection below recenters — so a
            // fixed sweep (vs. the old RNG search) makes the whole cache reproducible.
            for i in 0..DIRECTION_SEARCH_SAMPLES {
                let t = (i as f32 + 0.5) / DIRECTION_SEARCH_SAMPLES as f32;
                let angle = (2.0 * t - 1.0) * max_search_angle;
                direction = Vec3::new(-angle.sin(), 0.0, angle.cos());

                if passes(ray_origin, direction) {
                    found_valid = true;
                    break;
                }
            }
            if !found_valid {
                // no valid direction found for this bin, store zero-extent sentinel
                *v = CacheCell::default();
                return;
            }
            // The circular-aperture image is convex, so along any line the pass/fail
            // boundary is a single transition we can locate by bisection. `probe`
            // tests the direction at meridional angle `m`, sagittal angle `s`; the
            // chief direction is interior, so probes through it pass at the center.
            let chief_angle = (-direction.x() / direction.z()).atan();
            let hi0 = std::f32::consts::FRAC_PI_2 * GRAZING_ANGLE_MARGIN;
            let probe = |m: f32, s: f32| -> bool {
                let dir = Vec3::<S>::new(-m.tan(), s.tan(), 1.0).normalized();
                passes(ray_origin, dir)
            };
            // bisect the offset t in [0, hi0] where `f(t)` goes true (interior) -> false
            let bisect = |f: &dyn Fn(f32) -> bool| -> f32 {
                if f(hi0) {
                    return hi0; // boundary beyond the bracket; clamp
                }
                let (mut lo, mut hi) = (0.0f32, hi0);
                while hi - lo > solver_heat {
                    let mid = 0.5 * (lo + hi);
                    if f(mid) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo
            };

            // Meridional extent (both sides of the chief), then recenter. The chief is
            // a *random* valid direction, so we must measure the sagittal extent at the
            // recovered meridional center — not at the chief — otherwise it samples an
            // arbitrary chord of the ellipse and the eccentricity becomes noisy.
            let mer_plus = bisect(&|t| probe(chief_angle + t, 0.0));
            let mer_minus = bisect(&|t| probe(chief_angle - t, 0.0));
            let center_angle = chief_angle + 0.5 * (mer_plus - mer_minus);
            let a_m = 0.5 * (mer_plus + mer_minus); // meridional semi-axis
            let a_s = bisect(&|t| probe(center_angle, t)); // sagittal semi-axis, at center
            // minimum enclosing circle of the axis-aligned ellipse
            let cone_radius = a_m.max(a_s);
            // ellipse eccentricity (0 = circular), for diagnostics
            let (major, minor) = (a_m.max(a_s), a_m.min(a_s));
            let eccentricity = if major > 0.0 {
                (1.0 - (minor / major).powi(2)).max(0.0).sqrt()
            } else {
                0.0
            };

            *v = CacheCell::new(center_angle, cone_radius, eccentricity);
        });
        RadialSampler {
            cache: film,
            sensor_size,
            wavelength_bounds,
            wavelength_bins,

            radius_bins,
        }
    }

    pub fn sample<S: SimdBackend>(
        &self,
        lambda: f32,
        point: Point3<S>,
        s2d: Sample2D,
        s1: Sample1D,
    ) -> Vec3<S> {
        let [x, y, _, _] = point.as_array();

        let rotation_angle = y.atan2(x);

        let film_radius = y.hypot(x);

        let u = film_radius / (SQRT_2 * self.sensor_size);
        let v = ((lambda - self.wavelength_bounds.lower) / self.wavelength_bounds.span())
            .clamp(0.0, 1.0 - EPSILON);
        debug_assert!(u < 1.0 && v < 1.0, "{}, {}", u, v);
        let d_x_idx = (u * self.radius_bins as f32) as usize;
        let d_y_idx = (v * self.wavelength_bins as f32) as usize;
        let angles00 = self.cache.at(d_x_idx, d_y_idx);
        let angles01 = if d_y_idx + 1 < self.wavelength_bins {
            self.cache.at(d_x_idx, d_y_idx + 1)
        } else {
            angles00
        };
        let angles10 = if d_x_idx + 1 < self.radius_bins {
            self.cache.at(d_x_idx + 1, d_y_idx)
        } else {
            angles00
        };
        let angles11 = if d_x_idx + 1 < self.radius_bins && d_y_idx + 1 < self.wavelength_bins {
            self.cache.at(d_x_idx + 1, d_y_idx + 1)
        } else {
            angles00
        };
        let du = u * self.radius_bins as f32 - d_x_idx as f32;
        let dv = v * self.wavelength_bins as f32 - d_y_idx as f32;

        // bilinear interpolation; only lanes 0 (phi) and 1 (dphi) are meaningful
        let (w00, w10, w01, w11) = (
            (1.0 - du) * (1.0 - dv),
            du * (1.0 - dv),
            (1.0 - du) * dv,
            du * dv,
        );
        let phi = w00 * angles00.angle
            + w10 * angles10.angle
            + w01 * angles01.angle
            + w11 * angles11.angle;
        let dphi = w00 * angles00.angle_spread
            + w10 * angles10.angle_spread
            + w01 * angles01.angle_spread
            + w11 * angles11.angle_spread;

        // direction is pointing towards the center somewhat and assumes direction.y() == 0.0
        // thus rotate to match actual central point of ray.

        let dx = -phi.sin();
        let direction = Vec3::new(
            dx * rotation_angle.cos(),
            dx * rotation_angle.sin(),
            phi.cos(),
        );
        debug_assert!(phi.is_finite(), "{}", phi);
        debug_assert!(rotation_angle.is_finite());
        debug_assert!(dx.is_finite(), "{}", dx);
        debug_assert!(direction.is_finite());
        // slightly inflate cone to avoid missing edge rays due to FP rounding
        const CONE_INFLATION: f32 = 1.01;
        let radius = dphi * CONE_INFLATION;
        // direction should be a valid unit vector by construction
        let frame = TangentFrame::from_normal(direction);
        let phi = s1.x * TAU;
        let r = s2d.x.sqrt() * radius;
        debug_assert!(!r.is_nan());
        let unnormalized_v = Vec3::z_axis() + Vec3::new(r * phi.cos(), r * phi.sin(), 0.0);
        debug_assert!(unnormalized_v.is_finite());
        // transforming a normalized vector should yield another normalized vector, as long as all the frame components are orthonormal.
        frame.to_world(&unnormalized_v.normalized())
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Read};

    use ::math::spectral::BOUNDED_VISIBLE_RANGE;

    use super::*;

    type Backend = thermite::backend::x86_v3::X86V3;
    type Vec3 = crate::math::Vec3<Backend>;
    type Point3 = crate::math::Point3<Backend>;
    type Ray = crate::math::Ray<Backend>;

    #[test]
    fn test_radial_sampler() {
        let mut camera_file = File::open(&"data/cameras/petzval_kodak.txt").unwrap();
        let mut camera_spec = String::new();
        camera_file.read_to_string(&mut camera_spec).unwrap();
        let (interfaces, _, _) = parse_lenses_from(&camera_spec);
        let assembly = LensAssembly::new(interfaces.as_slice()).as_debug();
        let aperture_radius = assembly.aperture_radius();
        let aperture = SimpleBladedAperture::new(6, 0.5);

        assembly.trace_reverse(
            0.0,
            Input::new(
                Ray::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)),
                0.550,
            ),
            1.001,
            |e| (aperture.is_rejected(aperture_radius, e.origin), false),
            drop,
        );

        let film_position = assembly.total_thickness_at(0.0);
        let radial_sampler = RadialSampler::new::<Backend, _>(
            35.0,
            100,
            100,
            BOUNDED_VISIBLE_RANGE,
            -film_position,
            &assembly,
            0.0,
            &aperture,
            0.1,
            35.0,
        );

        // verify that sample returns finite, normalized directions
        for _ in 0..100 {
            let s2d = Sample2D::new_random_sample();
            let s1d = Sample1D::new_random_sample();
            let lambda = BOUNDED_VISIBLE_RANGE.sample(s1d.x);
            let point = Point3::new(s2d.x * 10.0 - 5.0, s2d.y * 10.0 - 5.0, -film_position);
            let dir = radial_sampler.sample(
                lambda,
                point,
                Sample2D::new_random_sample(),
                Sample1D::new_random_sample(),
            );
            assert!(
                dir.0.is_finite().all(),
                "sample returned non-finite direction: {:?}",
                dir
            );
            let len = dir.norm();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "sample returned non-unit direction with length {}",
                len
            );
        }
    }
}
