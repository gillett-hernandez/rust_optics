use crate::lens::*;
use crate::math::*;
use crate::spectrum::*;
use rand::prelude::*;

const INTENSITY_EPS: f32 = 0.0001;

pub fn trace_spherical(
    ray: Ray,
    r: f32,
    center: f32,
    housing_radius: f32,
) -> Result<(Ray, Vec3), i16> {
    let scv = Vec3::from(ray.origin - Vec3::Z * center);
    let a = ray.direction * ray.direction;
    let b = 2.0 * ray.direction * scv;
    let c = scv * scv - r * r;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        Err(4)
    } else {
        let mut error = 0;
        let t;
        let a2 = 2.0 * a;
        let t0 = (-b - discriminant.sqrt()) / a2;
        let t1 = (-b + discriminant.sqrt()) / a2;
        if t0 < -1.0e-4 {
            t = t1;
        } else {
            t = t0.min(t1);
        }
        if t < -1.0e-4 {
            Err(16)
        } else {
            let ray = ray.at_time(t);
            let (rx, ry) = (ray.origin.x(), ray.origin.y());
            error |= (rx * rx + ry * ry > housing_radius * housing_radius) as i16;
            let normal = Vec3::new(rx, ry, ray.origin.z() - center) / r;
            if error == 0 {
                Ok((ray, normal.normalized()))
            } else {
                Err(error)
            }
        }
    }
}

pub fn evaluate_aspherical(pos: Point3, r: f32, k: i32, correction: f32x4) -> f32 {
    let h = (pos.x() * pos.x() + pos.y() * pos.y()).sqrt();
    let hr = h / r;
    let h2 = h * h;
    let h4 = h2 * h2;
    let h6 = h4 * h2;
    let h8 = h4 * h4;
    let h10 = h8 * h2;
    let corv = f32x4::new(h4, h6, h8, h10);
    h * hr / (1.0 + (1.0 - (1.0 + k as f32) * hr * hr).max(0.0).sqrt()) + (correction * corv).sum()
}

pub fn evaluate_aspherical_derivative(pos: Point3, r: f32, k: i32, correction: f32x4) -> f32 {
    let h = (pos.x() * pos.x() + pos.y() * pos.y()).sqrt();
    let hr = h / r;
    let h2 = h * h;
    let h3 = h2 * h;

    let h4 = h2 * h2;
    let h5 = h3 * h2;
    let h6 = h4 * h2;
    let h7 = h4 * h3;
    let h9 = h6 * h3;
    let corv = f32x4::new(4.0 * h3, 6.0 * h5, 8.0 * h7, 10.0 * h9);
    let hr2 = hr * hr;
    let subexpr = (1.0 - (1.0 + k as f32) * hr2).max(0.0).sqrt();
    2.0 * hr / (1.0 + subexpr)
        + hr2 * hr * (k as f32 + 1.0) / (subexpr * (subexpr + 1.0).powf(2.0))
        + (correction * corv).sum()
}

pub fn trace_aspherical(
    mut ray: Ray,
    r: f32,
    center: f32,
    k: i32,
    mut correction: f32x4,
    housing_radius: f32,
) -> Result<(Ray, Vec3), i32> {
    let mut t = 0.0;
    let result = trace_spherical(ray, r, center, housing_radius)?;
    ray = result.0;
    let normal = result.1;
    let mut rad = r;
    if (center + r - ray.origin.z()).abs() > (center - r - ray.origin.z()).abs() {
        rad = -r;
        correction = -correction;
    }

    let mut position_error;
    // repeatedly trace the ray forwads and backwards until the position error is less than some constant.
    for _ in 0..100 {
        position_error =
            rad + center - ray.origin.z() - evaluate_aspherical(ray.origin, rad, k, correction);
        let terr = position_error / ray.direction.z();
        t += terr;
        ray = ray.at_time(terr);
        if position_error.abs() < 1.0e-4 {
            break;
        }
    }
    let dz = evaluate_aspherical_derivative(ray.origin, rad, k, correction)
        * if normal.z() < 0.0 { -1.0 } else { 1.0 };
    let sqr = ray.origin.0 * ray.origin.0;
    let new_r = (sqr.extract(0) + sqr.extract(1)).sqrt();
    let normal = Vec3::new(
        ray.origin.x() / new_r * dz,
        ray.origin.y() / new_r * dz,
        normal.z() / normal.z().abs(),
    )
    .normalized();

    Ok((ray.at_time(t), normal))
}

pub fn trace_cylindrical(
    mut ray: Ray,
    r: f32,
    center: f32,
    housing_radius: f32,
) -> Result<(Ray, Vec3), i32> {
    let scv = Vec3::new(ray.origin.x(), 0.0, ray.origin.z() - center);
    let a = ray.direction * ray.direction;
    let b = 2.0 * ray.direction * scv;
    let c = scv * scv - r * r;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Err(4);
    }
    let t;
    if r > 0.0 {
        t = (-b - discriminant.sqrt()) / (2.0 * a);
    } else {
        t = (-b + discriminant.sqrt()) / (2.0 * a);
    }
    ray = ray.at_time(t);
    let sqr = ray.origin.0 * ray.origin.0;
    if sqr.extract(0) + sqr.extract(1) > housing_radius * housing_radius {
        return Err(8);
    }
    let normal = Vec3::new(ray.origin.x(), 0.0, ray.origin.z() - center) / r;
    Ok((ray, normal))
}

pub fn fresnel(n1: f32, n2: f32, cosr: f32, cost: f32) -> f32 {
    if cost <= 0.0 {
        1.0
    } else {
        let n2cost = n2 * cost;
        let n1cosr = n1 * cosr;
        let n1cost = n1 * cost;
        let n2cosr = n2 * cosr;
        let rs = (n1cosr - n2cost) / (n1cosr + n2cost);
        let rp = (n1cost - n2cosr) / (n1cost + n2cosr);
        ((rs * rs + rp * rp) / 2.0).min(1.0)
    }
}

pub fn refract(n1: f32, n2: f32, normal: Vec3, dir: Vec3) -> (Vec3, f32) {
    if n1 == n2 {
        (dir, 1.0)
    } else {
        let eta = n1 / n2;
        let norm = dir.norm();
        let cos1 = -(normal * dir) / norm;
        let cos2_2 = 1.0 - eta * eta * (1.0 - cos1 * cos1);
        if cos2_2 < 0.0 {
            (dir, 0.0)
        } else {
            let cos2 = cos2_2.sqrt();
            (
                dir * eta / norm + (eta * cos1 - cos2) * normal,
                1.0 - fresnel(n1, n2, cos1, cos2),
            )
        }
    }
}

pub fn plane_to_camera_space(ray_in: PlaneRay, plane_pos: f32) -> Ray {
    let [x, y, dx, dy]: [f32; 4] = ray_in.0.into();
    Ray::new(
        Point3::new(x, y, plane_pos),
        Vec3::new(dx, dy, 1.0).normalized(),
    )
}

pub fn camera_space_to_plane(ray_in: Ray, plane_pos: f32) -> PlaneRay {
    let [x, y, z, _]: [f32; 4] = ray_in.origin.0.into();
    let [dx, dy, dz, _]: [f32; 4] = ray_in.direction.0.into();
    let t = (plane_pos - z) / dz;

    PlaneRay::new(x + t * dx, y + t * dy, dx / dz.abs(), dy / dz.abs())
}

pub fn sphere_to_camera_space(ray_in: SphereRay, sphere_center: f32, sphere_radius: f32) -> Ray {
    let [x, y, dx, dy]: [f32; 4] = ray_in.0.into();
    let normal = Vec3::new(
        x / sphere_radius,
        y / sphere_radius,
        (sphere_radius * sphere_radius - x * x - y * y)
            .max(0.0)
            .sqrt()
            / sphere_radius.abs(),
    );
    let temp_direction = Vec3::new(dx, dy, (1.0 - dx * dx - dy * dy).max(0.0).sqrt());
    let ex = Vec3::new(normal.z(), 0.0, -normal.x()).normalized();
    let frame = TangentFrame::from_tangent_and_normal(ex, normal);

    Ray::new(
        Point3::new(x, y, normal.z() * sphere_radius + sphere_center),
        frame.to_world(&temp_direction).normalized(),
    )
}

pub fn camera_space_to_sphere(ray_in: Ray, sphere_center: f32, sphere_radius: f32) -> SphereRay {
    let [x, y, z, _]: [f32; 4] = ray_in.origin.0.into();
    let normal = Vec3::new(x, y, (z - sphere_center).abs()) / sphere_radius;
    let temp_direction = ray_in.direction.normalized();
    let ex = Vec3::new(normal.z(), 0.0, -normal.x());
    let frame = TangentFrame::from_tangent_and_normal(ex, normal);
    SphereRay {
        0: shuffle!(
            ray_in.origin.0,
            frame.to_local(&temp_direction).0,
            [0, 1, 4, 5]
        ),
    }
}

// traces rays from the sensor to the outer pupil
pub fn evaluate(
    lenses: &Vec<LensElement>,
    zoom: f32,
    input: &Input<PlaneRay>,
    aspheric: i16,
) -> Result<Output<SphereRay>, i32> {
    assert!(lenses.len() > 0);
    let mut error = 0;
    let mut n1 = spectrum_eta_from_abbe_num(
        lenses.last().unwrap().ior,
        lenses.last().unwrap().vno,
        input.lambda,
    );
    let mut ray: Ray;
    let mut intensity = 1.0;
    ray = plane_to_camera_space(input.ray, 0.0);
    let mut distsum = 0.0;
    for (k, lens) in lenses.iter().rev().enumerate() {
        let r = -lens.radius;
        let dist = lens.thickness_at(zoom);
        distsum += dist;
        let res: (Ray, Vec3);
        if lens.anamorphic {
            res = trace_cylindrical(ray, r, distsum + r, lens.housing_radius)?;
        } else if aspheric > 0 {
            res = trace_aspherical(
                ray,
                r,
                distsum + r,
                lens.aspheric,
                lens.correction,
                lens.housing_radius,
            )?;
        } else {
            res = trace_spherical(ray, r, distsum + r, lens.housing_radius)?;
        }
        ray = res.0;
        let normal = res.1;
        let n2 = if k > 0 {
            spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda)
        } else {
            1.0
        };
        // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
        let res = refract(n1, n2, normal, ray.direction);
        ray.direction = res.0;
        intensity *= res.1;
        if intensity < INTENSITY_EPS {
            error |= 8;
        }
        if error > 0 {
            return Err(error);
        }
        // not sure why this normalize is here.
        ray.direction = ray.direction.normalized();
        n1 = n2;
    }
    Ok(Output {
        ray: camera_space_to_sphere(ray, distsum - lenses[0].radius, lenses[0].radius),
        tau: intensity,
    })
}

// evaluate scene to sensor
pub fn evaluate_reverse(
    lenses: &Vec<LensElement>,
    zoom: f32,
    input: &Input<SphereRay>,
    aspheric: i16,
) -> Result<Output<PlaneRay>, i32> {
    assert!(lenses.len() > 0);
    let mut error = 0;
    let mut n1 = 1.0;
    let mut ray: Ray;
    let mut intensity = 1.0;
    ray = sphere_to_camera_space(input.ray, 0.0, lenses[0].radius);
    let mut distsum = 0.0;
    ray.direction = -ray.direction;
    for (_k, lens) in lenses.iter().enumerate() {
        let r = lens.radius;
        let dist = lens.thickness_at(zoom);
        let res: (Ray, Vec3);
        if lens.anamorphic {
            res = trace_cylindrical(ray, r, distsum + r, lens.housing_radius)?;
        } else if aspheric > 0 {
            res = trace_aspherical(
                ray,
                r,
                distsum + r,
                lens.aspheric,
                lens.correction,
                lens.housing_radius,
            )?;
        } else {
            res = trace_spherical(ray, r, distsum + r, lens.housing_radius)?;
        }
        ray = res.0;
        let normal = res.1;
        let n2 = spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda);
        // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
        let res = refract(n1, n2, normal, ray.direction);
        ray.direction = res.0;
        intensity *= res.1;
        if intensity < INTENSITY_EPS {
            error |= 8;
        }
        if error > 0 {
            return Err(error);
        }
        // not sure why this normalize is here.
        ray.direction = ray.direction.normalized();
        distsum += dist;
        n1 = n2;
    }
    Ok(Output {
        ray: camera_space_to_plane(ray, distsum),
        tau: intensity,
    })
}

// traces rays from the sensor to aperture
pub fn evaluate_aperture(
    lenses: &Vec<LensElement>,
    zoom: f32,
    input: &Input<PlaneRay>,
    aspheric: i16,
) -> Result<Output<PlaneRay>, i32> {
    assert!(lenses.len() > 0);
    let mut error = 0;
    let mut n1 = spectrum_eta_from_abbe_num(
        lenses.last().unwrap().ior,
        lenses.last().unwrap().vno,
        input.lambda,
    );
    let mut ray: Ray;
    let mut intensity = 1.0;
    ray = plane_to_camera_space(input.ray, 0.0);
    let mut distsum = 0.0;
    for (k, lens) in lenses.iter().rev().enumerate() {
        let r = -lens.radius;
        let dist = lens.thickness_at(zoom);
        distsum += dist;
        if lens.lens_type == LensType::Aperture {
            break;
        }
        let res: (Ray, Vec3);
        if lens.anamorphic {
            res = trace_cylindrical(ray, r, distsum + r, lens.housing_radius)?;
        } else if aspheric > 0 {
            res = trace_aspherical(
                ray,
                r,
                distsum + r,
                lens.aspheric,
                lens.correction,
                lens.housing_radius,
            )?;
        } else {
            res = trace_spherical(ray, r, distsum + r, lens.housing_radius)?;
        }
        ray = res.0;
        let normal = res.1;
        let n2 = if k > 0 {
            spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda)
        } else {
            1.0
        };
        // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
        let res = refract(n1, n2, normal, ray.direction);
        ray.direction = res.0;
        intensity *= res.1;
        if intensity < INTENSITY_EPS {
            error |= 8;
        }
        if error > 0 {
            return Err(error);
        }
        // not sure why this normalize is here.
        ray.direction = ray.direction.normalized();
        n1 = n2;
    }
    Ok(Output {
        ray: camera_space_to_plane(ray, distsum),
        tau: intensity,
    })
}

// evaluate scene to aperture
pub fn evaluate_aperture_reverse(
    lenses: &Vec<LensElement>,
    zoom: f32,
    input: &Input<SphereRay>,
    aspheric: i16,
) -> Result<Output<PlaneRay>, i32> {
    assert!(lenses.len() > 0);
    let mut error = 0;
    let mut n1 = 1.0;
    let mut ray: Ray;
    let mut intensity = 1.0;
    ray = sphere_to_camera_space(input.ray, 0.0, lenses[0].radius);
    let mut distsum = 0.0;
    ray.direction = -ray.direction;
    for (_k, lens) in lenses.iter().enumerate() {
        let r = lens.radius;
        let dist = lens.thickness_at(zoom);
        let normal;
        let res: (Ray, Vec3);
        if lens.anamorphic {
            res = trace_cylindrical(ray, r, distsum + r, lens.housing_radius)?;
        } else if aspheric > 0 {
            res = trace_aspherical(
                ray,
                r,
                distsum + r,
                lens.aspheric,
                lens.correction,
                lens.housing_radius,
            )?;
        } else {
            res = trace_spherical(ray, r, distsum + r, lens.housing_radius)?;
        }
        ray = res.0;
        normal = res.1;
        let n2 = spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda);
        // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
        let res = refract(n1, n2, normal, ray.direction);
        ray.direction = res.0;
        intensity *= res.1;
        if intensity < INTENSITY_EPS {
            error |= 8;
        }
        if error > 0 {
            return Err(error);
        }
        // not sure why this normalize is here.
        ray.direction = ray.direction.normalized();
        distsum += dist;
        if lens.lens_type == LensType::Aperture {
            break;
        }
        n1 = n2;
    }
    Ok(Output {
        ray: camera_space_to_plane(ray, distsum),
        tau: intensity,
    })
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_old_vec3() {
        println!("testing usage of old Vec3");
        let av1 = Vec3::new(1.0, 1.0, 1.0);
        let av2 = Vec3::new(1.0, 1.0, 1.0);
        println!("{:?}", av1 * av2);
    }

    fn basic_incoming_ray() -> Ray {
        Ray::new(Point3::new(0.1, 0.1, 0.0), -Vec3::Z)
    }
    fn random_incoming_ray() -> Ray {
        Ray::new(
            Point3::new(0.1, 0.1, 0.0),
            Vec3::new(random::<f32>() - 0.5, random::<f32>() - 0.5, -10.0).normalized(),
        )
    }

    fn basic_plane_input() -> Input<PlaneRay> {
        Input {
            ray: PlaneRay::new(
                35.0 * (random::<f32>() - 0.5),
                35.0 * (random::<f32>() - 0.5),
                random::<f32>() / 10.0,
                random::<f32>() / 10.0,
            ),
            lambda: 0.45,
        }
    }

    fn basic_sphere_input() -> Input<SphereRay> {
        let incoming = random_incoming_ray();
        Input {
            ray: camera_space_to_sphere(incoming, -10.0, 10.0),
            lambda: 0.45,
        }
    }

    #[test]
    fn test_plane_input() {
        println!("testing construction of input");
        let input = basic_plane_input();
        println!("{:?}", input);
    }
    #[test]
    fn test_sphere_input() {
        println!("testing construction of input");
        let input = basic_sphere_input();
        println!("{:?}", input);
    }
    #[test]
    fn test_trace_spherical() {
        let incoming = basic_incoming_ray();
        println!("testing trace spherical with given input {:?}", incoming);
        let result = trace_spherical(incoming, 0.9, -1.0, 0.9);
        match result {
            Ok((new_ray, normal)) => {
                println!("{:?}, {:?}", new_ray, normal);
            }
            Err(error) => {
                println!("error occurred with code {}", error);
            }
        };
    }
    #[test]
    fn test_evaluate_aspherical() {
        let incoming_ray = basic_incoming_ray();
        println!("testing evaluate aspherical with given incoming_ray");
        let result = evaluate_aspherical(incoming_ray.origin, 0.9, 1, f32x4_ZERO);
        println!("{}", result);
    }
    #[test]
    fn test_evaluate_aspherical_derivative() {
        let incoming_ray = basic_incoming_ray();
        println!("testing evaluate aspherical_derivative with given incoming_ray");
        let result = evaluate_aspherical_derivative(incoming_ray.origin, 0.9, 1, f32x4_ZERO);
        println!("{}", result);
    }
    #[test]
    fn test_trace_aspherical() {
        let incoming = basic_incoming_ray();

        println!("testing trace aspherical with given input");
        let result = trace_aspherical(incoming, 0.9, 1.0, 1, f32x4_ZERO, 0.9);
        match result {
            Ok((ray, normal)) => {
                println!("{:?}, {:?}", ray, normal);
            }
            Err(error) => {
                println!("error occurred with code {}", error);
            }
        };
    }
    #[test]
    fn test_trace_cylindrical() {
        let incoming = basic_incoming_ray();
        println!("testing trace cylindrical with given input");
        let trace_result = trace_cylindrical(incoming, 0.9, 1.0, 0.9);
        match trace_result {
            Ok((ray, normal)) => {
                println!("{:?}, {:?}", ray, normal);
            }
            Err(error) => {
                println!("error occurred with code {}", error);
            }
        };
    }
    #[test]
    fn test_plane_space() {
        let incoming = basic_incoming_ray();
        println!("{:?}", incoming);

        println!("testing camera space to plane space and back with given ray");
        let plane = camera_space_to_plane(incoming, 0.0);
        println!("{:?}", plane);

        let new_ray = plane_to_camera_space(plane, 0.0);
        println!("{:?}", new_ray);

        assert!((incoming.origin - new_ray.origin).norm() < 0.00001);
        // assert!((incoming.direction - new_ray.direction).norm() < 0.00001);
    }
    #[test]
    fn test_sphere_space() {
        let incoming = basic_sphere_input();
        println!("{:?}", incoming);

        let new_ray = sphere_to_camera_space(incoming.ray, 0.0, 1.0);
        println!("{:?}", new_ray);
        let sphere = camera_space_to_sphere(new_ray, 0.0, 1.0);
        println!("{:?}", sphere);
        assert!((sphere.0 - incoming.ray.0).abs().sum() < 0.000001);
    }

    #[test]
    fn test_refract_and_fresnel() {
        // basic input is a vector near the origin, with z component 0, pointing nearly straight downward (negative Z-ward)
        let input = basic_incoming_ray();
        println!("{:?}", input);
        let mut trace_result = trace_spherical(input, 40.0, -42.0, 30.0).unwrap();

        let normal = trace_result.1;
        let cos_r = normal * input.direction;

        let result = refract(1.0, 1.45, normal, input.direction);
        println!("{:?}", result);

        trace_result.0.direction = result.0;

        let cos_i = normal * trace_result.0.direction;

        println!("{:?}, {:?}, {}, {}", input, trace_result, cos_r, cos_i);

        println!("testing fresnel with given input");
        let result = fresnel(1.0, 1.45, cos_i, cos_r);
        println!("{}", result);
    }

    fn construct_lenses() -> Vec<LensElement> {
        let lines = "164.12		10.99				SF5			1.673	32.2	54
559.28		0.23				air							54
100.12		11.45				BAF10		1.67	47.1    51
213.54		0.23				air							51
58.04		22.95				LAK9		1.691	54.7	41
2551		2.58				SF5			1.673	32.2	41
32.39		15.66				air							27
10000		15.00				IRIS						25.5
-40.42		2.74				SF15		1.699	30.1	25
192.98		27.92				SK16		1.62	60.3	36
-55.53		0.23				air							36
192.98		7.98				LAK9		1.691	54.7	35
-225.28		0.23				air							35
175.1		8.48				LAK9		1.691	54.7	35
-203.54		55.742				air							35"
            .lines();
        let mut lenses: Vec<LensElement> = Vec::new();
        let (mut last_ior, mut last_vno) = (1.0, 0.0);
        for line in lines {
            if line.starts_with("#") {
                continue;
            }
            let lens = LensElement::parse_from(line, last_ior, last_vno).unwrap();
            last_ior = lens.ior;
            last_vno = lens.vno;
            println!("successfully parsed lens {:?}", lens);
            lenses.push(lens);
        }
        lenses
    }
    #[allow(unused_mut)]
    #[test]
    fn test_evaluate() {
        let lenses = construct_lenses();

        let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 20));
        // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let plane_ray_sampler = |mut sampler: &mut Box<dyn Sampler>| {
            let Sample2D { x: x1, y: y1 } = sampler.draw_2d();
            let Sample2D { x: x2, y: y2 } = sampler.draw_2d();
            PlaneRay::new(
                35.0 * (1.0 - x1),
                35.0 * (1.0 - y1),
                (x2 * 2.0 - 1.0) / 5.0,
                (y2 * 2.0 - 1.0) / 5.0,
            )
        };
        let wavelength_sampler =
            |mut sampler: &mut Box<dyn Sampler>| sampler.draw_1d().x * 0.3 + 0.4;
        let mut succeeded = false;
        for _ in 0..100 {
            let input = Input {
                ray: plane_ray_sampler(&mut sampler),
                lambda: wavelength_sampler(&mut sampler),
            };

            let maybe_output = evaluate(&lenses, 0.0, input, 0);
            println!("{:?}", input);
            if let Ok(output) = maybe_output {
                println!("{:?}", output);
                succeeded = true;
                break;
            }
        }
        assert!(succeeded);
    }
    #[test]
    fn test_evaluate_aperture() {
        let lenses = construct_lenses();

        let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 20));
        // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let plane_ray_sampler = |mut sampler: &mut Box<dyn Sampler>| {
            let Sample2D { x: x1, y: y1 } = sampler.draw_2d();
            let Sample2D { x: x2, y: y2 } = sampler.draw_2d();
            PlaneRay::new(
                35.0 * (1.0 - x1),
                35.0 * (1.0 - x1),
                (x2 * 2.0 - 1.0) / 5.0,
                (y2 * 2.0 - 1.0) / 5.0,
            )
        };
        let wavelength_sampler =
            |mut sampler: &mut Box<dyn Sampler>| sampler.draw_1d().x * 0.3 + 0.4;
        let mut succeeded = false;
        for _ in 0..100 {
            let input = Input {
                ray: plane_ray_sampler(&mut sampler),
                lambda: wavelength_sampler(&mut sampler),
            };

            let maybe_output = evaluate_aperture(&lenses, 0.0, input, 0);
            println!("{:?}", input);
            if let Ok(output) = maybe_output {
                println!("{:?}", output);
                succeeded = true;
                break;
            }
        }
        assert!(succeeded);
    }

    #[test]
    fn test_evaluate_reverse() {
        let lenses = construct_lenses();

        // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 20));
        // // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        // let ray_sampler = |mut sampler: &mut Box<dyn Sampler>| {
        //     let Sample2D { x: x1, y: y1 } = sampler.draw_2d();
        //     let Sample2D { x: x2, y: y2 } = sampler.draw_2d();
        //     Ray::new(
        //         Point3::ZERO + Vec3::new((2.0 * x1 - 1.0) / 2.0, (2.0 * y1 - 1.0) / 2.0, 0.0),
        //         Vec3::new((x1 * 2.0 - 1.0) / 2.0, (y2 * 2.0 - 1.0) / 2.0, 7.0).normalized(),
        //     )
        // };
        // let wavelength_sampler =
        //     |mut sampler: &mut Box<dyn Sampler>| sampler.draw_1d().x * 0.3 + 0.4;
        let mut succeeded = false;
        for _ in 0..100 {
            let input = basic_sphere_input();

            let maybe_output = evaluate_reverse(&lenses, 0.0, input, 0);
            println!("{:?}", input);
            match maybe_output {
                Ok(output) => {
                    println!("{:?}", output);
                    succeeded = true;
                    break;
                }
                Err(e) => {
                    println!("{}", e);
                }
            }
        }
        assert!(succeeded);
    }
    #[test]
    fn test_evaluate_reverse_aperture() {
        let lenses = construct_lenses();

        // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 20));
        // // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        // let ray_sampler = |mut sampler: &mut Box<dyn Sampler>| {
        //     let Sample2D { x: x1, y: y1 } = sampler.draw_2d();
        //     let Sample2D { x: x2, y: y2 } = sampler.draw_2d();
        //     Ray::new(
        //         Point3::ZERO + Vec3::new((2.0 * x1 - 1.0) / 2.0, (2.0 * y1 - 1.0) / 2.0, 0.0),
        //         Vec3::new((x1 * 2.0 - 1.0) / 2.0, (y2 * 2.0 - 1.0) / 2.0, 7.0).normalized(),
        //     )
        // };
        // let wavelength_sampler =
        //     |mut sampler: &mut Box<dyn Sampler>| sampler.draw_1d().x * 0.3 + 0.4;
        let mut succeeded = false;
        for _ in 0..100 {
            // let input = basic_sphere_input();
            let input = Input {
                ray: SphereRay::new(0.0, 0.0, 0.0, 0.0),
                lambda: 0.45,
            };

            let maybe_output = evaluate_aperture_reverse(&lenses, 0.0, input, 0);
            println!("{:?}", input);
            match maybe_output {
                Ok(output) => {
                    println!("{:?}", output);
                    succeeded = true;
                    break;
                }
                Err(e) => {
                    println!("{}", e);
                }
            }
        }
        assert!(succeeded);
    }
}
