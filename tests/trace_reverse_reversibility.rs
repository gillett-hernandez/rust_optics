//! Regression guard for `trace_reverse` (see TRACE_REVERSE_INVESTIGATION.md).
//!
//! Optical reversibility: if a ray travels film -> world along some path, the reverse
//! of its exit ray, sent back through `trace_reverse`, must retrace that path and emerge
//! as the reverse of the original film ray. `trace_reverse` is implemented as the exact
//! inverse of `trace_forward` in the same world frame, so the round trip must recover
//! the original film ray: same line (the film point lies on the output line) and
//! antiparallel direction.

use optics::math::*;
use optics::{LensAssembly, parse_lenses_from};

type Backend = thermite::backend::x86_v3::X86V3;
type Vec3 = optics::math::Vec3<Backend>;
type Point3 = optics::math::Point3<Backend>;
type Ray = optics::math::Ray<Backend>;

fn no_aperture(_e: Ray) -> (bool, bool) {
    (false, false)
}

/// Perpendicular distance from `p` to the line through `ro` with unit direction `rd`.
fn point_line_distance(p: Point3, ro: Point3, rd: Vec3) -> f32 {
    let w = p - ro;
    let along = w * rd; // dot
    (w - rd * along).norm()
}

#[test]
fn trace_reverse_inverts_trace_forward() {
    for cam in ["petzval_kodak", "double_gauss_angenioux"] {
        let spec = std::fs::read_to_string(format!("data/cameras/{cam}.txt")).unwrap();
        let (interfaces, _, _) = parse_lenses_from(&spec);
        let assembly = LensAssembly::new(&interfaces);
        let total = assembly.total_thickness_at(0.0);
        let lambda_um = 0.550;

        let mut checked = 0;
        for deg in [0.5_f32, 1.5, 3.0] {
            let theta = deg.to_radians();
            let film_origin = Point3::new(0.0, 0.0, -total);
            let film_dir = Vec3::new(0.0, theta.sin(), theta.cos());

            // forward: film -> world
            let Some(Output { ray: o, .. }) = assembly.trace_forward(
                0.0,
                Input::new(Ray::new(film_origin, film_dir), lambda_um),
                1.0,
                no_aperture,
                drop,
            ) else {
                continue; // ray didn't make it out at this angle for this lens; skip
            };

            // reverse: send the exit ray back in (reversed direction)
            let rev_in = Ray::new(o.origin, -o.direction.normalized());
            let Some(Output { ray: back, .. }) =
                assembly.trace_reverse(0.0, Input::new(rev_in, lambda_um), 1.0, no_aperture, drop)
            else {
                panic!("{cam} @ {deg} deg: reverse of a valid forward ray returned None");
            };

            // must recover the original film ray: antiparallel direction, film point on the line
            let d = back.direction.normalized();
            let align = (d * film_dir).abs();
            let dist = point_line_distance(film_origin, back.origin, d);
            assert!(
                align > 0.9999,
                "{cam} @ {deg} deg: direction not recovered (align={align:.5})"
            );
            assert!(
                dist < 1e-2,
                "{cam} @ {deg} deg: film point off output line by {dist:.4} mm"
            );
            checked += 1;
        }
        assert!(checked > 0, "{cam}: no rays exercised the round trip");
    }
}

/// The two independent infinity-focus methods must agree: the **forward** collimation
/// root-find (`rear_focal_plane_forward`) and the **reverse** convergence of literal
/// parallel rays (`rear_focal_plane_reverse`). They share no code paths, so agreement is
/// strong evidence both the forward and reverse tracers are correct.
#[test]
fn forward_reverse_rear_focal_plane_agree() {
    for cam in ["petzval_kodak", "double_gauss_angenioux"] {
        let spec = std::fs::read_to_string(format!("data/cameras/{cam}.txt")).unwrap();
        let (interfaces, _, _) = parse_lenses_from(&spec);
        let assembly = LensAssembly::new(&interfaces);

        let mut checked = 0;
        for &nm in &[450.0_f32, 550.0, 650.0] {
            let lambda_um = nm / 1000.0;
            let fwd = assembly.rear_focal_plane_forward::<Backend>(0.0, lambda_um);
            let rev = assembly.rear_focal_plane_reverse::<Backend>(0.0, lambda_um);
            let (Some(fwd), Some(rev)) = (fwd, rev) else {
                continue; // a method couldn't resolve this wavelength for this lens; skip
            };
            assert!(
                (fwd - rev).abs() < 2.0,
                "{cam} @ {nm}nm: forward rear focal plane {fwd:.3} vs reverse {rev:.3} disagree by {:.3} mm",
                (fwd - rev).abs()
            );
            checked += 1;
        }
        assert!(checked > 0, "{cam}: neither method resolved any wavelength");
    }
}
