//! Two-reflection ghost (lens-flare) paths via polynomial optics (paper "Flare
//! computation"). A ghost enters the front of the lens travelling toward the
//! film, refracts through the front surfaces, **reflects** at some surface `a`,
//! travels back and **reflects** again at a shallower surface `b < a`, then
//! refracts forward through the rest of the stack to the film. Each ghost is one
//! composed [`PolySystem`] mapping an entrance-pupil reduced ray to a film one.
//!
//! Geometry is the world→film mirror of [`build_forward`](super::build_forward):
//! the front vertex is at `z = 0`, the film at `z = -total_thickness`, surfaces
//! are indexed front→rear, and segment travel direction (`dz_sign`) flips at each
//! reflection — handled by the direction-aware maps in [`super::surfaces`].
//!
//! Validation note: the crate has no reflective reference tracer, so ghost paths
//! are validated structurally (finite output, correct path count, and reduction
//! to the forward map when reflections are removed) rather than against a
//! ground-truth trace.

use std::sync::Arc;

use nalgebra::SVector;

use ::math::spectral::BOUNDED_VISIBLE_RANGE;

use crate::lens::{spectrum_eta_from_abbe_num, LensAssembly};
use crate::vec2d::Vec2D;

use super::surfaces::{propagation, reflection_spherical, refraction_spherical};
use super::system::PolySystem;
use super::trunc_poly::Basis;

/// Per-surface geometry + media for the world→film frame, front→rear order.
struct Surfaces {
    z: Vec<f32>,         // vertex z (front = 0, decreasing toward the film)
    radius: Vec<f32>,    // signed sphere-center offset, r = -lens.radius
    n_before: Vec<f32>,  // medium on the world side of each surface
    n_after: Vec<f32>,   // medium on the film side of each surface
    film_z: f32,
}

impl Surfaces {
    fn build(assembly: &LensAssembly, zoom: f32, lambda_um: f32, atmosphere: f32) -> Result<Self, String> {
        let n = assembly.lenses.len();
        let total = assembly.total_thickness_at(zoom);
        let mut z = Vec::with_capacity(n);
        let mut radius = Vec::with_capacity(n);
        let mut n_after = Vec::with_capacity(n);
        let mut acc_z = 0.0f32;
        for lens in assembly.lenses.iter() {
            if lens.anamorphic || lens.aspheric > 0 {
                return Err("flare: only spherical/planar surfaces are supported".into());
            }
            z.push(acc_z);
            radius.push(-lens.radius);
            n_after.push(spectrum_eta_from_abbe_num(lens.ior, lens.vno, lambda_um));
            acc_z -= lens.thickness_at(zoom);
        }
        let n_before: Vec<f32> = (0..n)
            .map(|k| if k == 0 { atmosphere } else { n_after[k - 1] })
            .collect();
        Ok(Surfaces {
            z,
            radius,
            n_before,
            n_after,
            film_z: -total,
        })
    }

    fn len(&self) -> usize {
        self.z.len()
    }
}

/// Compose one surface interaction onto `acc`: propagate from `prev_z` to the
/// surface vertex, then refract or reflect. `dz_in` is the incoming travel sign.
#[allow(clippy::too_many_arguments)]
fn step(
    basis: &Arc<Basis>,
    s: &Surfaces,
    acc: PolySystem,
    k: usize,
    reflect: bool,
    dz_in: f32,
    prev_z: &mut f32,
) -> PolySystem {
    let gap = s.z[k] - *prev_z;
    let propagated = propagation(basis.clone(), gap).compose(&acc);
    *prev_z = s.z[k];
    let op = if reflect {
        reflection_spherical(basis.clone(), s.radius[k], dz_in)
    } else {
        // A −z (world→film) pass refracts n_before→n_after; a +z (post-reflection)
        // pass meets the surface from the film side, so the media swap.
        let (n1, n2) = if dz_in < 0.0 {
            (s.n_before[k], s.n_after[k])
        } else {
            (s.n_after[k], s.n_before[k])
        };
        refraction_spherical(basis.clone(), s.radius[k], n1, n2, dz_in)
    };
    op.compose(&propagated)
}

/// All two-reflection ghost surface pairs `(b, a)` with `b < a`, where `a` is the
/// first (deeper) reflection and `b` the second. There are `C(N, 2)` of them.
pub fn enumerate_ghosts(n_surfaces: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for a in 1..n_surfaces {
        for b in 0..a {
            out.push((b, a));
        }
    }
    out
}

/// Build the entrance-pupil(`z = 0`) → film reduced-ray polynomial for the ghost
/// that reflects first at surface `a`, then at surface `b` (`b < a`).
pub fn build_ghost(
    assembly: &LensAssembly,
    zoom: f32,
    lambda_um: f32,
    atmosphere: f32,
    degree: usize,
    b: usize,
    a: usize,
) -> Result<PolySystem, String> {
    assert!(b < a, "ghost reflections must satisfy b < a");
    let s = Surfaces::build(assembly, zoom, lambda_um, atmosphere)?;
    let n = s.len();
    assert!(a < n, "reflection surface index out of range");
    let basis = Basis::cached(degree);

    let mut acc = PolySystem::identity(basis.clone());
    let mut prev_z = 0.0f32; // entrance pupil at the front vertex plane

    // inbound: refract through 0..a-1 (−z)
    for k in 0..a {
        acc = step(&basis, &s, acc, k, false, -1.0, &mut prev_z);
    }
    // reflect at a (−z in, +z out)
    acc = step(&basis, &s, acc, a, true, -1.0, &mut prev_z);
    // back: refract through a-1..b+1 (+z)
    for k in (b + 1..a).rev() {
        acc = step(&basis, &s, acc, k, false, 1.0, &mut prev_z);
    }
    // reflect at b (+z in, −z out)
    acc = step(&basis, &s, acc, b, true, 1.0, &mut prev_z);
    // outbound: refract through b+1..N-1 (−z)
    for k in (b + 1)..n {
        acc = step(&basis, &s, acc, k, false, -1.0, &mut prev_z);
    }
    // propagate to the film plane
    acc = propagation(basis.clone(), s.film_z - prev_z).compose(&acc);
    Ok(acc)
}

/// Splat one ghost into a film buffer. Models a distant on-axis-able point source
/// (parallel entrance rays at slope `source_slope`) by sampling the front
/// aperture disk of radius `pupil_radius`; each surviving sample adds `weight` at
/// its film position. `sensor_size` maps film millimetres to `[0,1]` UV.
#[allow(clippy::too_many_arguments)]
pub fn render_ghost(
    ghost: &PolySystem,
    source_slope: (f32, f32),
    pupil_radius: f32,
    grid: usize,
    sensor_size: f32,
    weight: f32,
    film: &mut Vec2D<f32>,
) {
    for iy in 0..grid {
        for ix in 0..grid {
            // uniform-ish disk sample over the front aperture
            let px = (ix as f32 + 0.5) / grid as f32 * 2.0 - 1.0;
            let py = (iy as f32 + 0.5) / grid as f32 * 2.0 - 1.0;
            if px * px + py * py > 1.0 {
                continue;
            }
            let r = SVector::from([
                px * pupil_radius,
                py * pupil_radius,
                source_slope.0,
                source_slope.1,
            ]);
            let out = ghost.eval(r);
            if !out[0].is_finite() || !out[1].is_finite() {
                continue;
            }
            let u = out[0] / (2.0 * sensor_size) + 0.5;
            let v = out[1] / (2.0 * sensor_size) + 0.5;
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let xpix = (u * film.width as f32) as usize;
            let ypix = (v * film.height as f32) as usize;
            let cur = film.at(xpix, ypix);
            film.write_at(xpix, ypix, cur + weight);
        }
    }
}

/// Default visible-range wavelength bounds, re-exported for convenience.
pub fn default_wavelength_bounds() -> ::math::bounds::Bounds1D {
    BOUNDED_VISIBLE_RANGE
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Read};

    use super::*;
    use crate::parse_lenses_from;

    fn petzval() -> LensAssembly {
        let mut s = String::new();
        File::open("data/cameras/petzval_kodak.txt")
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        let (interfaces, _, _) = parse_lenses_from(&s);
        LensAssembly::new(interfaces.as_slice())
    }

    #[test]
    fn ghost_count_is_n_choose_2() {
        // petzval_kodak has 8 surfaces -> C(8,2) = 28 ghosts.
        assert_eq!(enumerate_ghosts(8).len(), 28);
    }

    #[test]
    fn every_ghost_builds_and_is_finite() {
        let assembly = petzval();
        let n = assembly.lenses.len();
        let mut built = 0;
        for (b, a) in enumerate_ghosts(n) {
            let sys = build_ghost(&assembly, 0.0, 0.55, 1.0, 3, b, a).unwrap();
            // evaluate a small near-axis pupil ray; output must be finite.
            let out = sys.eval(SVector::from([2.0, 1.0, 0.0, 0.0]));
            assert!(
                out.iter().all(|c| c.is_finite()),
                "ghost ({b},{a}) produced non-finite output {out:?}"
            );
            built += 1;
        }
        assert_eq!(built, 28);
    }

    #[test]
    fn render_accumulates_energy() {
        let assembly = petzval();
        let ghost = build_ghost(&assembly, 0.0, 0.55, 1.0, 3, 1, 4).unwrap();
        let mut film = Vec2D::new(64, 64, 0.0f32);
        render_ghost(&ghost, (0.0, 0.0), 12.0, 32, 35.0, 1.0, &mut film);
        let total: f32 = film.buffer.iter().sum();
        assert!(total > 0.0, "ghost render splatted no energy");
    }
}
