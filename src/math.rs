// // for raytrace_dot etc
// #include "spectrum.h"
// #include "lenssystem.h"
// #include <stdio.h>
// #include <strings.h>

use packed_simd::f32x4;
use std::f32::INFINITY;
use std::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

#[derive(Copy, Clone, Debug)]
pub struct Vec3(pub f32x4);

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        // Vec3 { x, y, z, w: 0.0 }
        Vec3(f32x4::new(x, y, z, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> Vec3 {
        Vec3(v)
    }
    pub const ZERO: Vec3 = Vec3::from_raw(f32x4::splat(0.0));
    pub const MASK: f32x4 = f32x4::new(1.0, 1.0, 1.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
}

impl Vec3 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    #[inline(always)]
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
}

impl Mul for Vec3 {
    type Output = f32;
    fn mul(self, other: Vec3) -> f32 {
        // self.x * other.x + self.y * other.y + self.z * other.z
        (self.0 * other.0).sum()
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f32) -> Vec3 {
        Vec3::from_raw(self.0 * other)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::from_raw(self * other.0)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f32) -> Vec3 {
        Vec3::from_raw(self.0 / other)
    }
}

// impl Div for Vec3 {
//     type Output = Vec3;
//     fn div(self, other: Vec3) -> Vec3 {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         Vec3::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for Vec3 {
//     type Output = Vec3;
//     fn add(self, other: f32) -> Vec3 {
//         Vec3::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for Vec3 {
//     type Output = Vec3;
//     fn sub(self, other: f32) -> Vec3 {
//         Vec3::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::from_raw(self.0 + other.0)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::from_raw(-self.0)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        self + (-other)
    }
}

impl From<f32> for Vec3 {
    fn from(s: f32) -> Vec3 {
        Vec3::from_raw(f32x4::splat(s) * Vec3::MASK)
    }
}

impl From<Vec3> for f32x4 {
    fn from(v: Vec3) -> f32x4 {
        v.0
    }
}

impl Vec3 {
    pub fn cross(&self, other: Vec3) -> Self {
        let (x1, y1, z1) = (self.x(), self.y(), self.z());
        let (x2, y2, z2) = (other.x(), other.y(), other.z());
        Vec3::new(y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - x2 * y1)
    }

    pub fn norm_squared(&self) -> f32 {
        (self.0 * self.0 * Vec3::MASK).sum()
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        Vec3::from_raw(self.0 / norm)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Point3(pub f32x4);

impl Point3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3(f32x4::new(x, y, z, 1.0))
    }
    pub const fn from_raw(v: f32x4) -> Point3 {
        Point3(v)
    }
    pub const ZERO: Point3 = Point3::from_raw(f32x4::splat(0.0));
}

impl Point3 {
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
    pub fn normalize(mut self) -> Self {
        unsafe {
            self.0 = self.0 / self.0.extract_unchecked(3);
        }
        self
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, other: Vec3) -> Point3 {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        Point3::from_raw(self.0 + other.0)
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;
    fn sub(self, other: Vec3) -> Point3 {
        // Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Point3::from_raw(self.0 - other.0)
    }
}

// // don't implement adding or subtracting floats from Point3, because that's equivalent to adding or subtracting a Vector with components f,f,f and why would you want to do that.

impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, other: Point3) -> Vec3 {
        // Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Vec3::from_raw((self.0 - other.0) * f32x4::new(1.0, 1.0, 1.0, 0.0))
    }
}

// #define INTENSITY_EPS 1e-5
impl From<Point3> for Vec3 {
    fn from(p: Point3) -> Self {
        // Vec3::new(p.x, p.y, p.z)
        Vec3::from_raw(p.0.replace(3, 0.0))
    }
}

impl From<Vec3> for Point3 {
    fn from(v: Vec3) -> Point3 {
        // Point3::from_raw(v.0.replace(3, 1.0))
        Point3::from_raw(v.0).normalize()
    }
}
#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: f32,
    pub tmax: f32,
}

impl Ray {
    pub const fn new(origin: Point3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction,
            time: 0.0,
            tmax: INFINITY,
        }
    }

    pub const fn new_with_time(origin: Point3, direction: Vec3, time: f32) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax: INFINITY,
        }
    }
    pub const fn new_with_time_and_tmax(
        origin: Point3,
        direction: Vec3,
        time: f32,
        tmax: f32,
    ) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax,
        }
    }
    pub fn with_tmax(mut self, tmax: f32) -> Self {
        self.tmax = tmax;
        self
    }
    // pub fn at_time(mut self, time: f32) -> Self {
    //     // self.origin =
    // }
    pub fn point_at_parameter(self, time: f32) -> Point3 {
        self.origin + self.direction * time
    }
}

pub struct Input {
    pub ray: Ray,
    pub lambda: f32,
}

impl Input {
    pub fn slice(&self) -> [f32; 5] {
        [
            self.ray.origin.x(),
            self.ray.origin.y(),
            self.ray.direction.x(),
            self.ray.direction.y(),
            self.lambda,
        ]
    }
}

pub struct Output {
    pub ray: Ray,
    pub tau: f32,
}

impl Output {
    pub fn slice(&self) -> [f32; 5] {
        [
            self.ray.origin.x(),
            self.ray.origin.y(),
            self.ray.direction.x(),
            self.ray.direction.y(),
            self.tau,
        ]
    }
}

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
            let new_ray = Ray::new_with_time(ray.point_at_parameter(t), ray.direction, t);
            let (rx, ry) = (ray.origin.x(), ray.origin.y());
            error |= (rx * rx + ry * ry > housing_radius * housing_radius) as i16;
            let normal = Vec3::new(rx, ry, ray.origin.z() - center) / r;
            if error == 0 {
                Ok((new_ray, normal.normalized()))
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
    ray = result.ray;
    let normal = result.normal;
    let mut rad = r;
    if (center + r - ray.origin.z()).abs() > (center - r - ray.origin.z()).abs() {
        rad = -r;
        correction = -correction;
    }

    let mut position_error = 1.0e7;
}

// static inline int aspherical(float *pos, float *dir, float *dist, const float R, const float center, const int k, const float *correction, const float housing_rad, float *normal)
// {
//   //first intersect sphere, then do correction iteratively
//   float t = 0;
//   int error = spherical(pos, dir, &t, R, center, housing_rad, normal);

//   float rad = R;
//   float corr[4] = {correction[0], correction[1], correction[2], correction[3]};
//   if(fabs(center+R-pos[2]) > fabs(center-R-pos[2]))
//   {
//     rad = -R;
//     for(int i = 0; i < 4; i++) corr[i] = -corr[i];
//   }

//   float position_error = 1e7;

//   for(int i = 0; i < 100; i++)
//   {
//     position_error = rad+center-pos[2]-evaluate_aspherical(pos, rad, k, correction);
//     float tErr = position_error/dir[2];
//     t += tErr;
//     propagate(pos, dir, tErr);
//     if(fabs(position_error) < 1e-4) break;
//   }

//   float dz = evaluate_aspherical_derivative(pos, rad, k, correction);
//   const float r = sqrt(pos[0]*pos[0]+pos[1]*pos[1]);

//   if(normal[2] < 0) dz = -dz;
//   normal[0] = pos[0]/r*dz;
//   normal[1] = pos[1]/r*dz;
//   normal[2] /= fabs(normal[2]);

//   raytrace_normalise(normal);

//   *dist = t;
//   return error;
// }

// static inline int cylindrical(float *pos, float *dir, float *dist, float R, float center, float housing_rad, float *normal)
// {
//   const float scv[3] = {pos[0], 0, pos[2] - center};
//   const float a = raytrace_dot(dir, dir);
//   const float b = 2 * raytrace_dot(dir, scv);
//   const float c = raytrace_dot(scv, scv) - R*R;
//   const float discr = b*b-4*a*c;
//   if(discr < 0.0f) return 4;
//   int error = 0;
//   float t = 0.0f;
//   if(R > 0.0f)
//     t = (-b-sqrtf(discr))/(2*a);
//   else if(R < 0.0f)
//     t = (-b+sqrtf(discr))/(2*a);

//   propagate(pos, dir, t);
//   error |= (int)(pos[0]*pos[0] + pos[1]*pos[1] > housing_rad*housing_rad)<<4;

//   normal[0] = pos[0]/R;
//   normal[1] = 0.0f;
//   normal[2] = (pos[2] - center)/R;

//   *dist = t;
//   return error;
// }

// static inline float fresnel(const float n1, const float n2, const float cosr, const float cost)
// {
//   if(cost <= 0.0f) return 1.0f; // total inner reflection
//   // fresnel for unpolarized light:
//   const float Rs = (n1*cosr - n2*cost)/(n1*cosr + n2*cost);
//   const float Rp = (n1*cost - n2*cosr)/(n1*cost + n2*cosr);
//   return fminf(1.0f, (Rs*Rs + Rp*Rp)*.5f);
// }

// static inline float refract(const float n1, const float n2, const float *n, float *dir)
// {
//   if(n1 == n2)
//     return 1;
//   const float eta = n1/n2;

//   const float norm = sqrtf(raytrace_dot(dir,dir));
//   const float cos1 = - raytrace_dot(n, dir)/norm;
//   const float cos2_2 = 1.0f-eta*eta*(1.0f-cos1*cos1);
//   // total (inner) reflection?
//   if(cos2_2 < 0.0f)
//     return 0;
//   const float cos2 = sqrtf(cos2_2);

//   for(int i=0;i<3;i++) dir[i] = dir[i]*eta/norm + (eta*cos1-cos2)*n[i];

//   return 1.0f-fresnel(n1, n2, cos1, cos2);
// }

// static inline void planeToCs(const float *inpos, const float *indir, float *outpos, float *outdir, const float planepos)
// {
//   outpos[0] = inpos[0];
//   outpos[1] = inpos[1];
//   outpos[2] = planepos;

//   outdir[0] = indir[0];
//   outdir[1] = indir[1];
//   outdir[2] = 1;

//   raytrace_normalise(outdir);
// }

// static inline void csToPlane(const float *inpos, const float *indir, float *outpos, float *outdir, const float planepos)
// {
//   //intersection with plane at z = planepos
//   const double t = (planepos - inpos[2]) / indir[2];

//   outpos[0] = inpos[0] + t * indir[0];
//   outpos[1] = inpos[1] + t * indir[1];

//   outdir[0] = indir[0] / fabsf(indir[2]);
//   outdir[1] = indir[1] / fabsf(indir[2]);
// }

// static inline void sphereToCs(const float *inpos, const float *indir, float *outpos, float *outdir, const float sphereCenter, const float sphereRad)
// {
//   const float normal[3] =
//   {
//     inpos[0]/sphereRad,
//     inpos[1]/sphereRad,
//     sqrtf(max(0, sphereRad*sphereRad-inpos[0]*inpos[0]-inpos[1]*inpos[1]))/fabsf(sphereRad)
//   };
//   const float tempDir[3] = {indir[0], indir[1], sqrtf(max(0.0, 1.0f-indir[0]*indir[0]-indir[1]*indir[1]))};

//   float ex[3] = {normal[2], 0, -normal[0]};
//   raytrace_normalise(ex);
//   float ey[3];
//   raytrace_cross(ey, normal, ex);

//   outdir[0] = tempDir[0] * ex[0] + tempDir[1] * ey[0] + tempDir[2] * normal[0];
//   outdir[1] = tempDir[0] * ex[1] + tempDir[1] * ey[1] + tempDir[2] * normal[1];
//   outdir[2] = tempDir[0] * ex[2] + tempDir[1] * ey[2] + tempDir[2] * normal[2];
//   outpos[0] = inpos[0];
//   outpos[1] = inpos[1];
//   outpos[2] = normal[2] * sphereRad + sphereCenter;
// }

// static inline void csToSphere(const float *inpos, const float *indir, float *outpos, float *outdir, const float sphereCenter, const float sphereRad)
// {
//   const float normal[3] =
//   {
//     inpos[0]/sphereRad,
//     inpos[1]/sphereRad,
//     fabsf((inpos[2]-sphereCenter)/sphereRad)
//   };
//   float tempDir[3] = {indir[0], indir[1], indir[2]};
//   raytrace_normalise(tempDir);

//   float ex[3] = {normal[2], 0, -normal[0]};
//   raytrace_normalise(ex);
//   float ey[3];
//   raytrace_cross(ey, normal, ex);
//   outdir[0] = raytrace_dot(tempDir, ex);
//   outdir[1] = raytrace_dot(tempDir, ey);
//   outpos[0] = inpos[0];
//   outpos[1] = inpos[1];
// }

// // evalute sensor to outer pupil acounting for fresnel:
// static inline int evaluate(const lens_element_t *lenses, const int lenses_cnt, const float zoom, const float *in, float *out, int aspheric)
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
//     const float R = -lenses[k].lens_radius; // negative, evaluate() is the adjoint case
//     float t = 0.0f;
//     const float dist = lens_get_thickness(lenses+k, zoom);
//     distsum += dist;

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
//     if(error) return error;

//     raytrace_normalise(dir);

//     n1 = n2;
//   }
//   // return [x,y,dx,dy,lambda]
//   csToSphere(pos, dir, out, out + 2, distsum-fabs(lenses[0].lens_radius), lenses[0].lens_radius);
//   out[4] = intensity;
//   return error;
// }

// // evaluate scene to sensor:
// static inline int evaluate_reverse(const lens_element_t *lenses, const int lenses_cnt, const float zoom, const float *in, float *out, int aspheric)
// {
//   int error = 0;
//   float n1 = 1.0f;
//   float pos[3], dir[3];
//   float intensity = 1.0f;

//   sphereToCs(in, in + 2, pos, dir, 0, lenses[0].lens_radius);

//   for(int i = 0; i < 2; i++) dir[i] = -dir[i];

//   float distsum = 0;

//   for(int k=0;k<lenses_cnt;k++)
//   {
//     const float R = lenses[k].lens_radius;
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
//     const float R = -lenses[k].lens_radius; // negative, evaluate() is the adjoint case
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

//   sphereToCs(in, in + 2, pos, dir, 0, lenses[0].lens_radius);
//   for(int i = 0; i < 2; i++) dir[i] = -dir[i];

//   float distsum = 0;
//   for(int k=0;k<lenses_cnt;k++)
//   {
//     const float R = lenses[k].lens_radius;
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

//   sphereToCs(in, in + 2, pos, dir, 0, lenses[0].lens_radius);
//   for(int i = 0; i < 2; i++) dir[i] = -dir[i];

//   float distsum = 0;
//   for(int k=0;k<lenses_cnt;k++)
//   {
//     const float R = lenses[k].lens_radius;
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
