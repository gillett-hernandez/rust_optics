use ::math::bounds::Bounds1D;

use crate::aperture::Aperture;
use crate::math::*;
use crate::vec2d::Vec2D;
use crate::*;

#[derive(Debug, Clone)]
pub struct RadialSampler {
    pub cache: Vec2D<[f32; 2]>,
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
        let mut film = Vec2D::new(radius_bins, wavelength_bins, [0.0f32; 2]);
        let aperture_radius = lens_assembly.aperture_radius();
        film.buffer.par_iter_mut().enumerate().for_each(|(i, v)| {
            let radius_bin = i % radius_bins;
            let wavelength_bin = i / radius_bins;

            let lambda =
                wavelength_bounds.sample((0.5 + wavelength_bin as f32) / wavelength_bins as f32);
            let radius = radius_cap * radius_bin as f32 / radius_bins as f32;
            // find direction (with fixed y = 0) for sampling aperture and outer pupil, and find corresponding sampling "radius"

            let ray_origin: Point3<S> = Point3::new(radius, 0.0, film_position);
            let mut direction: Vec3<S> = Vec3::z_axis();
            let mut found_valid = false;
            const MAX_DIRECTION_SEARCH_ATTEMPTS: usize = 1000;
            const GRAZING_ANGLE_MARGIN: f32 = 0.97;
            let mut sampler = StratifiedSampler::new(1000, 1, 1);
            for _ in 0..MAX_DIRECTION_SEARCH_ATTEMPTS {
                // directions range from straight forward (0 degrees) to almost critical (90 degrees, tangent)
                let s = sampler.draw_1d();
                let angle = s.x * std::f32::consts::FRAC_PI_2 * GRAZING_ANGLE_MARGIN;
                direction = Vec3::new(-angle.sin(), 0.0, angle.cos());

                let ray = Ray::new(ray_origin, direction);
                let result = lens_assembly.trace_forward(
                    lens_zoom,
                    Input::new(ray, lambda / 1000.0),
                    1.0,
                    |e| (aperture.is_rejected(aperture_radius, e.origin), false),
                    drop,
                );
                if let Some(Output { .. }) = result {
                    found_valid = true;
                    break;
                }
            }
            if !found_valid {
                // no valid direction found for this bin, store zero-extent sentinel
                *v = [0.0; 2];
                return;
            }
            // expand around direction to find radius and correct centroid.
            // measured in radians.
            let mut min_angle: f32 = 0.0;
            let mut max_angle: f32 = 0.0;
            let mut search_offset = 0.0;
            let base_angle = (-direction.x() / direction.z()).atan();

            const MAX_EXPANSION_ITERS: usize = 1000;
            'outer: for _ in 0..MAX_EXPANSION_ITERS {
                search_offset += solver_heat;
                let mut ct = 0;
                for mult in [-1.0, 1.0] {
                    let new_angle = base_angle + search_offset * mult;
                    let new_direction = Vec3::new(-new_angle.sin(), 0.0, new_angle.cos());
                    let ray = Ray::new(ray_origin, new_direction);
                    let result = lens_assembly.trace_forward(
                        lens_zoom,
                        Input::new(ray, lambda / 1000.0),
                        1.0,
                        |e| (aperture.is_rejected(aperture_radius, e.origin), false),
                        drop,
                    );
                    if result.is_some() {
                        // found good direction. keep expanding.
                        max_angle = max_angle.max(new_angle);
                        min_angle = min_angle.min(new_angle);
                    } else {
                        // found bad direction with this mult. keep expanding until both sides are bad.
                        ct += 1;
                        if ct == 2 {
                            // both sides are bad. break.
                            break 'outer;
                        }
                    }
                }
            }
            let avg_angle = if min_angle != 0.0 || max_angle != 0.0 {
                (min_angle + max_angle) / 2.0
            } else {
                0.0
            };

            *v = [avg_angle, (max_angle - min_angle).abs() / 2.0];
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

        let u = film_radius / (SQRT_2 * self.sensor_size / 2.0);
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
        let phi = w00 * angles00[0] + w10 * angles10[0] + w01 * angles01[0] + w11 * angles11[0];
        let dphi = w00 * angles00[1] + w10 * angles10[1] + w01 * angles01[1] + w11 * angles11[1];

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
