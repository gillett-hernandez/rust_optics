use packed_simd::f32x4;

use std::cmp::PartialEq;

use crate::{Point3, Ray, TangentFrame, Vec3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LensType {
    Solid,
    Air,
    Aperture,
}

#[derive(Copy, Clone, Debug)]
pub struct LensInterface {
    pub radius: f32,
    pub thickness_short: f32,
    pub thickness_mid: f32,
    pub thickness_long: f32,
    pub anamorphic: bool,
    pub lens_type: LensType,
    pub ior: f32, // index of refraction
    pub vno: f32, // abbe number
    pub housing_radius: f32,
    pub aspheric: i32,
    pub correction: f32x4,
}

impl LensInterface {
    pub fn thickness_at(self, mut zoom: f32) -> f32 {
        if zoom < 0.5 {
            zoom *= 2.0;
            self.thickness_short * (1.0 - zoom) + self.thickness_mid * zoom
        } else {
            zoom -= 0.5;
            zoom *= 2.0;
            self.thickness_mid * (1.0 - zoom) + self.thickness_long * zoom
        }
    }

    pub fn parse_from(string: &str, default_ior: f32, default_vno: f32) -> Result<Self, &str> {
        // format is:
        // lens := radius thickness_short(/thickness_mid(/thickness_long)?)? (anamorphic)? (mtl_name|'air'|'iris') ior vno housing_radius ('#!aspheric='aspheric_correction)?
        // radius := float
        // thickness_short := float
        // thickness_mid := float
        // thickness_long := float
        // anamorphic := 'cx_'
        // mtl_name := word
        // ior := float
        // vno := float
        // housing_radius := float
        // aspheric_correction := (float','){3}float

        if string.starts_with("#") {
            return Err("line started with comment");
        }
        println!("{}", string);
        let mut tokens = string.split_ascii_whitespace();
        let radius = tokens
            .next()
            .ok_or("ran out of tokens at radius")?
            .parse::<f32>()
            .map_err(|_e| "err parsing float at radius")?;
        let thickness_token: &str = tokens
            .next()
            .ok_or("ran out of tokens at thickness token")?;
        let mut thickness_iterator = thickness_token.split("/");
        let thickness_short = thickness_iterator
            .next()
            .unwrap()
            .parse::<f32>()
            .map_err(|_e| "err parsing float at thickness short")?;
        let thickness_mid = match thickness_iterator.next() {
            Some(token) => token
                .parse::<f32>()
                .map_err(|_e| "err parsing float at thickness mid")?,
            None => thickness_short,
        };
        let thickness_long = match thickness_iterator.next() {
            Some(token) => token
                .parse::<f32>()
                .map_err(|_e| "err parsing float at thickness long")?,
            None => thickness_short,
        };
        let maybe_anamorphic_or_lens = tokens.next().ok_or("ran out of tokens at anamorphic")?;
        let anamorphic = maybe_anamorphic_or_lens == "cx_";
        let next_token = if !anamorphic {
            maybe_anamorphic_or_lens
        } else {
            tokens.next().ok_or("ran out of tokens at lens type")?
        };
        let lens_type = match next_token {
            "air" => LensType::Air,
            "iris" => LensType::Aperture,
            _ => LensType::Solid,
        };
        let (ior, vno, housing_radius);
        let (a, b) = (tokens.next(), tokens.next());
        match (a, b) {
            (Some(token1), Some(token2)) => {
                ior = token1
                    .parse::<f32>()
                    .map_err(|_e| "err parsing float at ior")?;
                vno = token2
                    .parse::<f32>()
                    .map_err(|_e| "err parsing float at vno")?;
                housing_radius = tokens
                    .next()
                    .ok_or("ran out of tokens at housing radius branch 1")?
                    .parse::<f32>()
                    .map_err(|_e| "err parsing float at housing radius branch 1")?;
                let _aspheric = tokens.next();
            }
            (Some(token1), None) => {
                // this must be the situation where there is a housing radius but no aspheric correction.
                ior = match lens_type {
                    LensType::Solid => default_ior,
                    _ => 1.0,
                };
                vno = match lens_type {
                    LensType::Solid => default_vno,
                    _ => 0.0,
                };
                housing_radius = token1
                    .parse::<f32>()
                    .map_err(|_e| "error parsing float at housing radius branch 2")?;
            }
            (None, None) => {
                return Err("ran_out_of_tokens");
            }
            (None, Some(_)) => {
                return Err("what the fuck");
            }
        }

        Ok(LensInterface {
            radius,
            thickness_short,
            thickness_mid,
            thickness_long,
            anamorphic,
            lens_type,
            ior,
            vno,
            housing_radius,
            aspheric: 0,
            correction: f32x4::splat(0.0),
        })
    }
}

pub struct LensAssembly {
    pub lenses: Vec<LensInterface>,
    pub aperture_index: usize,
}

impl LensAssembly {
    pub fn new(lenses: &[LensInterface]) -> Self {
        // returns last index if slice does not contain an aperture
        let mut i = 0;
        for elem in lenses {
            if elem.lens_type == LensType::Aperture {
                break;
            }
            i += 1;
        }
        LensAssembly {
            lenses: lenses.into(),
            aperture_index: i,
        }
    }
    pub fn aperture_radius(&self) -> f32 {
        let aperture_index = self.aperture_index;
        self.lenses[aperture_index].housing_radius
    }
    pub fn aperture_position(&self, zoom: f32) -> f32 {
        // returns the end if there is no aperture
        let mut pos = 0.0;
        for elem in self.lenses.iter() {
            if elem.lens_type == LensType::Aperture {
                break;
            }
            pos += elem.thickness_at(zoom);
        }
        pos
    }
    pub fn total_thickness_at(&self, zoom: f32) -> f32 {
        let mut pos = 0.0;
        for elem in self.lenses.iter() {
            pos += elem.thickness_at(zoom);
        }
        pos
    }

    // traces rays from the sensor to the outer pupil
    pub fn trace_forward<F>(
        &self,
        zoom: f32,
        input: &Input<Ray>,
        atmosphere_ior: f32,
        aperture_hook: F,
    ) -> Option<Output<Ray>>
    where
        F: Fn(Ray) -> (bool, bool),
    {
        assert!(self.lenses.len() > 0);
        let mut error = 0;
        let mut n1 = spectrum_eta_from_abbe_num(
            self.lenses.last().unwrap().ior,
            self.lenses.last().unwrap().vno,
            input.lambda,
        );
        let mut ray = input.ray;
        let mut intensity = 1.0;
        let total_thickness = self.total_thickness_at(zoom);
        let mut position = -total_thickness;
        let t = (position - ray.origin.z()) / (ray.direction.z());
        // compute jacobian
        // let mut jacobian = f32x4::splat(1.0);
        ray.origin = ray.point_at_parameter(t);
        for (k, lens) in self.lenses.iter().rev().enumerate() {
            let r = -lens.radius;
            let thickness = lens.thickness_at(zoom);
            position += thickness;
            if lens.lens_type == LensType::Aperture {
                match aperture_hook(ray) {
                    (false, true) => {
                        // not blocked by aperture, but still should return early
                        return Some(Output {
                            ray,
                            tau: intensity,
                        });
                    }
                    (false, false) => {}
                    (true, _) => {
                        // blocked by aperture (and so no need to trace more) or should return early
                        return None;
                    }
                }
            }
            let res: (Ray, Vec3);
            if lens.anamorphic {
                res = trace_cylindrical(ray, r, position + r, lens.housing_radius).ok()?;
            } else if lens.aspheric > 0 {
                res = trace_aspherical(
                    ray,
                    r,
                    position + r,
                    lens.aspheric,
                    lens.correction,
                    lens.housing_radius,
                )
                .ok()?;
            } else {
                res = trace_spherical(ray, r, position + r, lens.housing_radius).ok()?;
            }
            ray = res.0;
            let normal = res.1;
            let n2 = if k > 0 {
                spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda)
            } else {
                atmosphere_ior
            };
            // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
            let res = refract(n1, n2, normal, ray.direction);
            ray.direction = res.0;
            intensity *= res.1;
            if intensity < INTENSITY_EPS {
                error |= 8;
            }
            if error > 0 {
                return None;
            }
            // not sure why this normalize is here.
            ray.direction = ray.direction.normalized();
            n1 = n2;
        }
        Some(Output {
            ray,
            tau: intensity,
        })
    }

    // evaluate scene to sensor. input ray must be facing away from the camera.
    pub fn trace_reverse<F>(
        &self,
        zoom: f32,
        input: &Input<Ray>,
        atmosphere_ior: f32,
        aperture_hook: F,
    ) -> Option<Output<Ray>>
    where
        F: Fn(Ray) -> (bool, bool),
    {
        assert!(self.lenses.len() > 0);
        let mut error = 0;
        let mut n1 = atmosphere_ior;
        let mut ray = input.ray;
        let mut intensity = 1.0;
        let mut distsum = 0.0;

        let t = (-ray.origin.z()) / (ray.direction.z());
        ray.origin = ray.point_at_parameter(t);
        ray.direction = -ray.direction;
        ray.origin = Point3::from(-Vec3::from(ray.origin));
        for (_k, lens) in self.lenses.iter().enumerate() {
            let r = lens.radius;

            let dist = lens.thickness_at(zoom);

            distsum += dist;
            if lens.lens_type == LensType::Aperture {
                match aperture_hook(ray) {
                    (false, true) => {
                        // not blocked by aperture, but still should return early
                        return Some(Output {
                            ray,
                            tau: intensity,
                        });
                    }
                    (false, false) => {}
                    (true, _) => {
                        // blocked by aperture (and so no need to trace more) or should return early
                        return None;
                    }
                }
            }
            let res: (Ray, Vec3);
            if lens.anamorphic {
                res = trace_cylindrical(ray, r, distsum - r, lens.housing_radius).ok()?;
            } else if lens.aspheric > 0 {
                res = trace_aspherical(
                    ray,
                    r,
                    distsum - r,
                    lens.aspheric,
                    lens.correction,
                    lens.housing_radius,
                )
                .ok()?;
            } else {
                res = trace_spherical(ray, r, distsum - r, lens.housing_radius).ok()?;
            }
            ray = res.0;
            let normal = res.1;

            let n2 = spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda);
            // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
            let res = refract(n1, n2, normal, ray.direction);
            ray.direction = res.0;

            println!("new ray {:?}", ray);
            intensity *= res.1;

            if intensity < INTENSITY_EPS {
                error |= 8;
            }
            if error > 0 {
                return None;
            }
            n1 = n2;
        }
        Some(Output {
            ray,
            tau: intensity,
        })
    }
}

pub fn spectrum_cauchy_from_abbe_num(nd: f32, vd: f32) -> (f32, f32) {
    if vd == 0.0 {
        (nd, 0.0)
    } else {
        const LC: f32 = 0.6563;
        const LF: f32 = 0.4861;
        const LD: f32 = 0.587561;
        const LC2: f32 = LC * LC;
        const LF2: f32 = LF * LF;
        const C: f32 = LC2 * LF2 / (LC2 - LF2);
        let b = (nd - 1.0) / vd * C;
        (nd - b / (LD * LD), b)
    }
}

pub fn spectrum_eta_from_abbe_num(nd: f32, vd: f32, lambda: f32) -> f32 {
    let (a, b) = spectrum_cauchy_from_abbe_num(nd, vd);
    a + b / (lambda * lambda)
}

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

#[derive(Copy, Clone, Debug)]
pub struct PlaneRay(pub f32x4);

impl PlaneRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self {
            0: f32x4::new(x, y, dx, dy),
        }
    }
    pub fn x(&self) -> f32 {
        self.0.extract(0)
    }
    pub fn y(&self) -> f32 {
        self.0.extract(1)
    }
    pub fn dx(&self) -> f32 {
        self.0.extract(2)
    }
    pub fn dy(&self) -> f32 {
        self.0.extract(3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SphereRay(pub f32x4);

impl SphereRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self {
            0: f32x4::new(x, y, dx, dy),
        }
    }
    pub fn x(&self) -> f32 {
        self.0.extract(0)
    }
    pub fn y(&self) -> f32 {
        self.0.extract(1)
    }
    pub fn dx(&self) -> f32 {
        self.0.extract(2)
    }
    pub fn dy(&self) -> f32 {
        self.0.extract(3)
    }
}

impl From<SphereRay> for PlaneRay {
    fn from(other: SphereRay) -> Self {
        Self { 0: other.0 }
    }
}

impl From<PlaneRay> for SphereRay {
    fn from(other: PlaneRay) -> Self {
        Self { 0: other.0 }
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

pub fn parse_lenses_from(spec: &str) -> (Vec<LensInterface>, f32, f32) {
    let lines = spec.lines();
    let mut lenses: Vec<LensInterface> = Vec::new();
    let (mut last_ior, mut last_vno) = (1.0, 0.0);
    for line in lines {
        if line.starts_with("#") {
            continue;
        }
        let lens = LensInterface::parse_from(line, last_ior, last_vno).unwrap();
        last_ior = lens.ior;
        last_vno = lens.vno;
        // println!("successfully parsed lens {:?}", lens);
        lenses.push(lens);
    }
    (lenses, last_ior, last_vno)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_parse() {
        let test_string = "65.22 9.60  N-SSK8 1.5 50 24.0";
        let lens = LensInterface::parse_from(test_string, 1.0, 0.0);
        println!("{:?}", lens);
    }

    #[test]
    fn test_multi_lenses() {
        let spec = "164.12		10.99				SF5			1.673	32.2	54
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
        -203.54		55.742				air							35";
        let (lenses, _last_ior, _last_vno) = parse_lenses_from(spec);
        let lens_assembly = LensAssembly::new(&lenses);
        let output = lens_assembly.trace_forward(
            0.0,
            &Input {
                ray: Ray::new(Point3::new(0.0, 0.0, -1000.0), Vec3::Z),
                lambda: 500.0,
            },
            1.0,
            |_| (false, true),
        );

        println!("{:?}", output);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_parse() {
        let test_string = "65.22 9.60  N-SSK8 1.5 50 24.0";
        let lens = LensElement::parse_from(test_string, 1.0, 0.0);
        println!("{:?}", lens);
    }
}
