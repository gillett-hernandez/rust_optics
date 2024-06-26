use ::math::bounds::Bounds1D;

use crate::aperture::Aperture;
use crate::math::*;
use crate::vec2d::Vec2D;
use crate::*;

#[derive(Debug, Clone)]
pub struct RadialSampler {
    pub cache: Vec2D<f32x4>,
    pub sensor_size: f32,
    pub wavelength_bounds: Bounds1D,
    pub wavelength_bins: usize,
    pub radius_bins: usize,
}

impl RadialSampler {
    // the following function only works and applies to lens with radial symmetry
    pub fn new<A>(
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
        A: Send + Sync + Aperture,
    {
        // create film of f32x4s
        let mut film = Vec2D::new(radius_bins, wavelength_bins, f32x4::splat(0.0));
        let aperture_radius = lens_assembly.aperture_radius();
        film.buffer.par_iter_mut().enumerate().for_each(|(i, v)| {
            let radius_bin = i % radius_bins;
            let wavelength_bin = i / radius_bins;

            let lambda =
                wavelength_bounds.sample((0.5 + wavelength_bin as f32) / wavelength_bins as f32);
            let radius = radius_cap * radius_bin as f32 / radius_bins as f32;
            // find direction (with fixed y = 0) for sampling aperture and outer pupil, and find corresponding sampling "radius"

            // switch flag to change from random to stratified.
            let ray_origin = Point3::new(radius, 0.0, film_position);
            let mut direction = Vec3::Z;
            // let mut state = 0;
            for _ in 0..1000 {
                // directions range from straight forward (0 degrees) to almost critical (90 degrees, tangent)

                // random sampling along axis until direction is found.
                let s = Sample1D::new_random_sample();
                let angle = s.x * std::f32::consts::FRAC_PI_2 * 0.97;
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
                    // found good direction, so break
                    break;
                }
            }
            // expand around direction to find radius and correct centroid.
            // measured in radians.
            let mut min_angle: f32 = 0.0;
            let mut max_angle: f32 = 0.0;
            let mut radius = 0.0;
            let mut sum_angle = 0.0;
            let mut valid_angle_count = 0;

            // maybe rewrite this as tree search?
            'outer: loop {
                radius += solver_heat;
                let mut ct = 0;
                for mult in [-1.0, 1.0] {
                    let old_angle = (-direction.x() / direction.z()).atan();
                    let new_angle = old_angle + radius * mult;
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
                        sum_angle += new_angle;
                        valid_angle_count += 1;
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
            let avg_angle = if valid_angle_count > 0 {
                sum_angle / (valid_angle_count as f32)
            } else {
                0.0
            };

            *v = f32x4::from_array([avg_angle, (max_angle - min_angle).abs() / 2.0, 0.0, 0.0]);
        });
        RadialSampler {
            cache: film,
            sensor_size,
            wavelength_bounds,
            wavelength_bins,

            radius_bins,
        }
    }

    pub fn sample(&self, lambda: f32, point: Point3, s0: Sample2D, s1: Sample1D) -> Vec3 {
        let [x, y, _, _]: [f32; 4] = point.0.into();

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
        // do bilinear interpolation?
        let (du, dv) = (
            u - d_x_idx as f32 / self.radius_bins as f32,
            v - d_y_idx as f32 / self.wavelength_bins as f32,
        );

        // let (phi, dphi) = (angles00[0], angles00[1]);

        // direct lookup through uv
        // let [phi, dphi, _, _]: [f32; 4] = direction_cache_film.at_uv((u, v)).into();

        debug_assert!(du.is_finite(), "{}", du);
        debug_assert!(dv.is_finite(), "{}", dv);
        debug_assert!(angles00[0].is_finite(), "{:?}", angles00);
        debug_assert!(angles01[0].is_finite(), "{:?}", angles01);
        debug_assert!(angles10[0].is_finite(), "{:?}", angles10);
        debug_assert!(angles11[0].is_finite(), "{:?}", angles11);
        debug_assert!(angles00[1].is_finite(), "{:?}", angles00);
        debug_assert!(angles01[1].is_finite(), "{:?}", angles01);
        debug_assert!(angles10[1].is_finite(), "{:?}", angles10);
        debug_assert!(angles11[1].is_finite(), "{:?}", angles11);
        // bilinear interpolation
        let (phi, dphi) = (
            (1.0 - du) * (1.0 - dv) * angles00[0]
                + du * (1.0 - dv) * angles01[0]
                + dv * (1.0 - du) * angles10[0]
                + du * dv * angles11[0],
            (1.0 - du) * (1.0 - dv) * angles00[1]
                + du * (1.0 - dv) * angles01[1]
                + dv * (1.0 - du) * angles10[1]
                + du * dv * angles11[1],
        );

        // direction is pointing towards the center somewhat and assumes direction.y() == 0.0
        // thus rotate to match actual central point of ray.

        let dx = -phi.sin();
        let direction = Vec3(f32x4::from_array([
            dx * rotation_angle.cos(),
            dx * rotation_angle.sin(),
            phi.cos(),
            0.0,
        ]));
        debug_assert!(phi.is_finite(), "{}", phi);
        debug_assert!(rotation_angle.is_finite());
        debug_assert!(dx.is_finite(), "{}", dx);
        debug_assert!(direction.0.is_finite().all());
        let radius = dphi * 1.01;

        // choose direction somehow

        let s2d = s0;
        let frame = TangentFrame::from_normal(Vec3(direction.0 * Vec3::MASK));
        let phi = s1.x * TAU;
        let r = s2d.x.sqrt() * radius;
        debug_assert!(!r.is_nan());
        let unnormalized_v = Vec3::Z + Vec3::new(r * phi.cos(), r * phi.sin(), 0.0);
        debug_assert!(unnormalized_v.0.is_finite().all());
        // transforming a normalized vector should yield another normalized vector, as long as all the frame components are orthonormal.
        frame.to_world(&unnormalized_v.normalized())
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Read};

    use ::math::spectral::BOUNDED_VISIBLE_RANGE;

    use super::*;
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
        let sampler = RadialSampler::new(
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
    }
}
