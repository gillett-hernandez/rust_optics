use optics::math::*;
use optics::{LensAssembly, parse_lenses_from};

type Backend = thermite::backend::x86_v3::X86V3;
type Vec3 = optics::math::Vec3<Backend>;
type Point3 = optics::math::Point3<Backend>;
type Ray = optics::math::Ray<Backend>;

/// Output ray y-slope for a small-angle forward ray launched from the on-axis point
/// (0,0,film_z). Zero slope == collimated == film_z is the rear focal plane.
fn exit_slope(assembly: &LensAssembly, film_z: f32, lambda: f32) -> Option<f32> {
    let angle = 0.003_f32;
    let ray = Ray::new(Point3::new(0.0, 0.0, film_z), Vec3::new(0.0, angle.sin(), angle.cos()));
    assembly
        .trace_forward(0.0, Input::new(ray, lambda), 1.0, |_e: Ray| (false, false), drop)
        .map(|o| o.ray.direction.y())
}

/// Find the rear focal plane (world z) by locating where the exit slope crosses zero.
fn rear_focal_plane(assembly: &LensAssembly, total: f32, lambda: f32) -> Option<f32> {
    // scan from just behind the rear vertex toward (and past) the film for a sign change
    let z_hi = -0.3 * total; // closer to the lens
    let z_lo = -1.8 * total; // well behind the film
    let steps = 200;
    let mut prev: Option<(f32, f32)> = None;
    let (mut a, mut b) = (f32::NAN, f32::NAN);
    for i in 0..=steps {
        let z = z_hi + (z_lo - z_hi) * i as f32 / steps as f32;
        if let Some(s) = exit_slope(assembly, z, lambda) {
            if let Some((pz, ps)) = prev {
                if ps == 0.0 || (ps < 0.0) != (s < 0.0) {
                    a = pz;
                    b = z;
                    break;
                }
            }
            prev = Some((z, s));
        }
    }
    if a.is_nan() {
        return None;
    }
    // bisect
    let mut a = a;
    let mut b = b;
    let mut fa = exit_slope(assembly, a, lambda)?;
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        let fm = exit_slope(assembly, m, lambda)?;
        if (fa < 0.0) != (fm < 0.0) {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    Some(0.5 * (a + b))
}

#[test]
fn forward_rear_focal_plane() {
    let spec = std::fs::read_to_string("data/cameras/petzval_kodak.txt").unwrap();
    let (interfaces, _, _) = parse_lenses_from(&spec);
    let assembly = LensAssembly::new(&interfaces);
    let total = assembly.total_thickness_at(0.0);
    println!("film_position = {}", -total);
    for &nm in &[450.0_f32, 550.0, 650.0] {
        let f = rear_focal_plane(&assembly, total, nm / 1000.0);
        println!("lambda={nm}nm -> rear focal plane z = {:?}", f);
    }
}
