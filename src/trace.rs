use crate::lens::*;
use crate::math::*;
use crate::spectrum::*;

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
    let subexpr = (1.0 - (1.0 + k as f32) * hr * hr).max(0.0).sqrt();
    2.0 * hr / (1.0 + subexpr)
        + hr * hr * hr * (k as f32 + 1.0) / (subexpr * (subexpr + 1.0).powf(2.0))
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

pub fn plane_to_cs(ray_in: Ray, plane_pos: f32) -> Ray {
    Ray::new(
        Point3::from_raw(ray_in.origin.0.replace(2, plane_pos)),
        Vec3::from_raw(ray_in.direction.0.replace(2, 1.0)).normalized(),
    )
}

pub fn cs_to_plane(ray_in: Ray, plane_pos: f32) -> Ray {
    let [x, y, z, _]: [f32; 4] = ray_in.origin.0.into();
    let [dx, dy, dz, _]: [f32; 4] = ray_in.direction.0.into();
    let t = (plane_pos - z) / dz;
    // TODO: double check that the z members of origin and direction should or should not be 0.0
    Ray::new(
        Point3::new(x + t * dx, y + t * dy, 0.0),
        Vec3::new(dx / dz.abs(), dy / dz.abs(), 0.0),
    )
}

// static inline void csToPlane(const float *inpos, const float *indir, float *outpos, float *outdir, const float planepos)
// {
//   //intersection with plane at z = planepos
//   const double t = (planepos - inpos[2]) / indir[2];

//   outpos[0] = inpos[0] + t * indir[0];
//   outpos[1] = inpos[1] + t * indir[1];

//   outdir[0] = indir[0] / fabsf(indir[2]);
//   outdir[1] = indir[1] / fabsf(indir[2]);
// }

pub fn sphere_to_cs(ray_in: Ray, sphere_center: f32, sphere_radius: f32) -> Ray {
    let [x, y, _, _]: [f32; 4] = ray_in.origin.0.into();
    let [dx, dy, _, _]: [f32; 4] = ray_in.direction.0.into();
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
    // todo: confirm if this should be frame.to_local or frame.to_world
    Ray::new(
        Point3::new(x, y, normal.z() * sphere_radius + sphere_center),
        frame.to_world(&temp_direction).normalized(),
    )
}

pub fn cs_to_sphere(ray_in: Ray, sphere_center: f32, sphere_radius: f32) -> Ray {
    let [x, y, z, _]: [f32; 4] = ray_in.origin.0.into();
    let normal = Vec3::new(x, y, (z - sphere_center).abs()) / sphere_radius;
    let temp_direction = ray_in.direction.normalized();
    let ex = Vec3::new(normal.z(), 0.0, -normal.x());
    let frame = TangentFrame::from_tangent_and_normal(ex, normal);
    // TODO: determine if these `replace`s are correct or not. in the original c code, they were mutable parameters and the z components were unchanged.
    Ray::new(
        ray_in.origin,
        // Point3::from_raw(ray_in.origin.0.replace(2, 0.0)),
        frame.to_local(&temp_direction),
        // Vec3::from_raw(frame.to_local(&temp_direction).0.replace(2, 0.0)),
    )
}

pub fn evaluate(
    lenses: Vec<LensElement>,
    zoom: f32,
    input: Input,
    aspheric: i16,
) -> Result<Output, i32> {
    assert!(lenses.len() > 0);
    let mut error = 0;
    let mut n1 = spectrum_eta_from_abbe_num(
        lenses.last().unwrap().ior,
        lenses.last().unwrap().vno,
        input.lambda,
    );
    let mut ray: Ray;
    let mut intensity = 1.0;
    ray = plane_to_cs(input.ray, 0.0);
    let mut distsum = 0.0;
    for (k, lens) in lenses.iter().rev().enumerate() {
        let r = -lens.radius;
        let dist = lens.get_thickness(zoom);
        distsum += dist;
        let mut normal = Vec3::ZERO;
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
    ray = cs_to_sphere(ray, distsum - lenses[0].radius, lenses[0].radius);
    Ok(Output {
        ray,
        tau: intensity,
    })
}

// // evaluate scene to sensor:
// static inline int evaluate_reverse(const lens_element_t *lenses, const int lenses_cnt, const float zoom, const float *in, float *out, int aspheric)
// {
//   int error = 0;
//   float n1 = 1.0f;
//   float pos[3], dir[3];
//   float intensity = 1.0f;

//   sphereToCs(in, in + 2, pos, dir, 0, lenses[0].radius);

//   for(int i = 0; i < 2; i++) dir[i] = -dir[i];

//   float distsum = 0;

//   for(int k=0;k<lenses_cnt;k++)
//   {
//     const float R = lenses[k].radius;
//     float t = 0.0f;
//     const float dist = lens_get_thickness(lenses+k, zoom);

//     //normal at intersection
//     float n[3] = {0.0};

//     if(lenses[k].anamorphic)
//       error |= cylindrical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);
//     else if(aspheric)
//       error |= aspherical(pos, dir, &t, R, distsum + R, lenses[k].aspheric, lenses[k].aspheric_correction_coefficients, lenses[k].housing_radius, n);
//     else
//       error |= spherical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);

//     // index of refraction and ratio current/next:
//     const float n2 = spectrum_eta_from_abbe_um(lenses[k].ior, lenses[k].vno, in[4]);
//     intensity *= refract(n1, n2, n, dir);
//     if(intensity < INTENSITY_EPS) error |= 8;

//     if(error)
//       return error;

//     // and renormalise:
//     raytrace_normalise(dir);

//     distsum += dist;
//     n1 = n2;
//   }
//   // return [x,y,dx,dy,lambda]
//   csToPlane(pos, dir, out, out + 2, distsum);
//   out[4] = intensity;
//   return error;
// }

// static inline int evaluate_aperture(const lens_element_t *lenses, const int lenses_cnt, const float zoom, const float *in, float *out, int aspheric)
// {
//   int error = 0;
//   float n1 = spectrum_eta_from_abbe_um(lenses[lenses_cnt-1].ior, lenses[lenses_cnt-1].vno, in[4]);
//   float pos[3], dir[3];
//   float intensity = 1.0f;

//   planeToCs(in, in + 2, pos, dir, 0);

//   float distsum = 0;

//   for(int k=lenses_cnt-1;k>=0;k--)
//   {
//     // propagate the ray reverse to the plane of intersection optical axis/lens element:
//     const float R = -lenses[k].radius; // negative, evaluate() is the adjoint case
//     float t = 0.0f;
//     const float dist = lens_get_thickness(lenses+k, zoom);
//     distsum += dist;

//     // stop after moving to aperture.
//     if(!strcasecmp(lenses[k].material, "iris")) break;

//     //normal at intersection
//     float n[3] = {0.0f};

//     if(lenses[k].anamorphic)
//       error |= cylindrical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);
//     else if(aspheric)
//       error |= aspherical(pos, dir, &t, R, distsum + R, lenses[k].aspheric, lenses[k].aspheric_correction_coefficients, lenses[k].housing_radius, n);
//     else
//       error |= spherical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);

//     // index of refraction and ratio current/next:
//     const float n2 = k ? spectrum_eta_from_abbe_um(lenses[k-1].ior, lenses[k-1].vno, in[4]) : 1.0f; // outside the lens there is vacuum
//     intensity *= refract(n1, n2, n, dir);
//     if(intensity < INTENSITY_EPS) error |= 8;
//     if(error)
//       return error;

//     // mark this ray as theoretically dead:
//     //if(dir[2] <= 0.0f) return error |= 2;
//     // and renormalise:
//     raytrace_normalise(dir);

//     n1 = n2;
//   }
//   // return [x,y,dx,dy,lambda]
//   csToPlane(pos, dir, out, out + 2, distsum);
//   out[4] = intensity;
//   return error;
// }

// // evaluate scene to sensor:
// static inline int evaluate_aperture_reverse(const lens_element_t *lenses, const int lenses_cnt, const float zoom, const float *in, float *out, int aspheric)
// {
//   int error = 0;
//   float n1 = 1.0f;
//   float pos[3], dir[3];
//   float intensity = 1.0f;

//   sphereToCs(in, in + 2, pos, dir, 0, lenses[0].radius);
//   for(int i = 0; i < 2; i++) dir[i] = -dir[i];

//   float distsum = 0;
//   for(int k=0;k<lenses_cnt;k++)
//   {
//     const float R = lenses[k].radius;
//     float t = 0.0f;
//     const float dist = lens_get_thickness(lenses+k, zoom);

//     //normal at intersection
//     float n[3];

//     if(lenses[k].anamorphic)
//       error |= cylindrical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);
//     else if(aspheric)
//       error |= aspherical(pos, dir, &t, R, distsum + R, lenses[k].aspheric, lenses[k].aspheric_correction_coefficients, lenses[k].housing_radius, n);
//     else
//       error |= spherical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);

//     // index of refraction and ratio current/next:
//     const float n2 = spectrum_eta_from_abbe_um(lenses[k].ior, lenses[k].vno, in[4]);
//     intensity *= refract(n1, n2, n, dir);
//     if(intensity < INTENSITY_EPS) error |= 8;
//     if(error)
//       return error;

//     // and renormalise:
//     raytrace_normalise(dir);

//     // move to next interface:
//     distsum += dist;

//     // stop after processing aperture but before moving to next element
//     if(k < lenses_cnt-1 && !strcasecmp(lenses[k+1].material, "iris")) break;

//     n1 = n2;
//   }
//   // return [x,y,dx,dy,lambda]
//   csToPlane(pos, dir, out, out + 2, distsum);
//   out[4] = intensity;
//   return error;
// }

// // evaluate scene to sensor:
// evaluate_aperture_reverse(const lens_element_t *lenses, const int lenses_cnt, const float zoom, const float *in, float *out, int aspheric)
// {
//   int error = 0;
//   float n1 = 1.0f;
//   float pos[3], dir[3];
//   float intensity = 1.0f;

//   sphereToCs(in, in + 2, pos, dir, 0, lenses[0].radius);
//   for(int i = 0; i < 2; i++) dir[i] = -dir[i];

//   float distsum = 0;
//   for(int k=0;k<lenses_cnt;k++)
//   {
//     const float R = lenses[k].radius;
//     float t = 0.0f;
//     const float dist = lens_get_thickness(lenses+k, zoom);

//     //normal at intersection
//     float n[3];

//     if(lenses[k].anamorphic)
//       error |= cylindrical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);
//     else if(aspheric)
//       error |= aspherical(pos, dir, &t, R, distsum + R, lenses[k].aspheric, lenses[k].aspheric_correction_coefficients, lenses[k].housing_radius, n);
//     else
//       error |= spherical(pos, dir, &t, R, distsum + R, lenses[k].housing_radius, n);

//     // index of refraction and ratio current/next:
//     const float n2 = spectrum_eta_from_abbe_um(lenses[k].ior, lenses[k].vno, in[4]);
//     intensity *= refract(n1, n2, n, dir);
//     if(intensity < INTENSITY_EPS) error |= 8;
//     if(error)
//       return error;

//     // and renormalise:
//     raytrace_normalise(dir);

//     // move to next interface:
//     distsum += dist;

//     // stop after processing aperture but before moving to next element
//     if(k < lenses_cnt-1 && !strcasecmp(lenses[k+1].material, "iris")) break;

//     n1 = n2;
//   }
//   // return [x,y,dx,dy,lambda]
//   csToPlane(pos, dir, out, out + 2, distsum);
//   out[4] = intensity;
//   return error;
// }
