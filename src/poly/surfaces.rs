//! Branch-free reduced-ray maps for single optical elements, built over
//! [`TruncPoly`] so the output polynomials drop straight out of running the
//! analytic geometry on identity-seeded inputs.
//!
//! Convention (matches [`camera_space_to_plane`](crate::lens::camera_space_to_plane)):
//! a reduced ray is `[x, y, u, v]` where `u = Dx/Dz`, `v = Dy/Dz` are ray slopes,
//! given at a plane perpendicular to the optical axis. Each surface map takes its
//! input on the surface's *vertex* plane (local `z = 0`, sphere center at
//! `(0, 0, R)`) and returns its output on the same plane (paper Fig. 2). Slopes
//! make free-space propagation linear and reuse the crate's existing plane↔camera
//! conversions.

use std::sync::Arc;

use super::system::PolySystem;
use super::trunc_poly::{Basis, TruncPoly, NUM_VARS};

/// A 3-vector whose components are truncated polynomials.
#[derive(Clone)]
struct PVec3 {
    x: TruncPoly,
    y: TruncPoly,
    z: TruncPoly,
}

impl PVec3 {
    fn add(&self, o: &PVec3) -> PVec3 {
        PVec3 {
            x: self.x.add(&o.x),
            y: self.y.add(&o.y),
            z: self.z.add(&o.z),
        }
    }
    fn sub(&self, o: &PVec3) -> PVec3 {
        PVec3 {
            x: self.x.sub(&o.x),
            y: self.y.sub(&o.y),
            z: self.z.sub(&o.z),
        }
    }
    /// Multiply every component by a scalar polynomial.
    fn scale(&self, s: &TruncPoly) -> PVec3 {
        PVec3 {
            x: self.x.mul(s),
            y: self.y.mul(s),
            z: self.z.mul(s),
        }
    }
    /// Multiply every component by a plain scalar.
    fn scale_f32(&self, s: f32) -> PVec3 {
        PVec3 {
            x: self.x.scale(s),
            y: self.y.scale(s),
            z: self.z.scale(s),
        }
    }
    fn dot(&self, o: &PVec3) -> TruncPoly {
        self.x.mul(&o.x).add(&self.y.mul(&o.y)).add(&self.z.mul(&o.z))
    }
    fn norm(&self) -> TruncPoly {
        self.dot(self).sqrt()
    }
}

/// The four identity-seeded inputs `[x, y, u, v]`.
fn seed(basis: &Arc<Basis>) -> [TruncPoly; NUM_VARS] {
    std::array::from_fn(|v| TruncPoly::var(basis.clone(), v))
}

/// Free-space propagation over an axial distance `dz_gap` (slope convention is
/// exact and linear): `x' = x + u·dz, y' = y + v·dz`, slopes unchanged.
pub fn propagation(basis: Arc<Basis>, dz_gap: f32) -> PolySystem {
    let [x, y, u, v] = seed(&basis);
    PolySystem::new([
        x.add(&u.scale(dz_gap)),
        y.add(&v.scale(dz_gap)),
        u,
        v,
    ])
}

/// Reconstruct the local geometry shared by refraction and reflection: returns
/// the intersection point `I` with the sphere centered at `(0, 0, radius)`, the
/// (un-normalized) incident direction `D`, and the unit normal `N = (I − C)/radius`
/// (matching `trace_spherical`'s normal).
///
/// `dz_sign` is the sign of the ray's z-travel: `+1` for the usual forward
/// (film→world) pass, `-1` for a segment running backward after a reflection.
/// The slope convention `[x, y, u, v]` cannot itself carry this sign, so callers
/// (the ghost-path builder) supply it per segment.
fn intersect(basis: &Arc<Basis>, radius: f32, dz_sign: f32) -> (PVec3, PVec3, PVec3) {
    let [x, y, u, v] = seed(basis);
    let p = PVec3 {
        x: x.clone(),
        y: y.clone(),
        z: TruncPoly::constant(basis.clone(), 0.0),
    };
    // Direction with Dz = dz_sign: since the slope is u = Dx/Dz, the x/y
    // components are u·dz_sign, v·dz_sign — i.e. D = dz_sign·(u, v, 1). (For the
    // usual forward pass dz_sign = +1 this is just (u, v, 1).)
    let d = PVec3 {
        x: u.scale(dz_sign),
        y: v.scale(dz_sign),
        z: TruncPoly::constant(basis.clone(), dz_sign),
    };
    let center = PVec3 {
        x: TruncPoly::constant(basis.clone(), 0.0),
        y: TruncPoly::constant(basis.clone(), 0.0),
        z: TruncPoly::constant(basis.clone(), radius),
    };
    let s = p.sub(&center); // S = P - C
    let a = d.dot(&d);
    let b = d.dot(&s).scale(2.0);
    let c = s.dot(&s).sub(&TruncPoly::constant(basis.clone(), radius * radius));
    let disc = b.mul(&b).sub(&a.mul(&c).scale(4.0));
    let sqrt_disc = disc.sqrt();
    let two_a = a.scale(2.0);
    // The ray crosses the vertex plane on the sphere, so one root is ≈0 (the hit
    // we want); the other is ≈2R (far side). The near-zero root is
    // (-b + s·√disc)/2a with s = -dz_sign·sign(radius), chosen so the numerator
    // vanishes on-axis. `s` depends only on build-time constants, not ray vars.
    let s_sign = -dz_sign * radius.signum();
    let t = if s_sign > 0.0 {
        b.scale(-1.0).add(&sqrt_disc).div(&two_a)
    } else {
        b.scale(-1.0).sub(&sqrt_disc).div(&two_a)
    };
    let i = p.add(&d.scale(&t)); // intersection point
    let normal = i.sub(&center).scale(&TruncPoly::constant(basis.clone(), 1.0 / radius));
    (i, d, normal)
}

/// Project a ray leaving intersection `i` with direction `dout` back onto the
/// vertex plane (`z = 0`) and read off the output reduced ray `[x', y', u', v']`.
fn to_plane(i: &PVec3, dout: &PVec3) -> [TruncPoly; NUM_VARS] {
    let up = dout.x.div(&dout.z); // u' = Dx'/Dz'
    let vp = dout.y.div(&dout.z); // v' = Dy'/Dz'
    // travel from I to z = 0: x' = I.x - I.z·u', y' = I.y - I.z·v'.
    let xp = i.x.sub(&i.z.mul(&up));
    let yp = i.y.sub(&i.z.mul(&vp));
    [xp, yp, up, vp]
}

/// Refraction at a spherical interface of signed `radius` (sphere center at local
/// `z = radius`), from medium `n1` into `n2`. Vector Snell's law, mirroring
/// [`refract`](crate::lens::refract) without its TIR / `n1 == n2` guards (the
/// expansion is centered on-axis where neither triggers).
pub fn refraction_spherical(
    basis: Arc<Basis>,
    radius: f32,
    n1: f32,
    n2: f32,
    dz_sign: f32,
) -> PolySystem {
    let (i, d, normal) = intersect(&basis, radius, dz_sign);
    // Orient the geometric normal to face the incident ray. For the forward pass
    // (dz_sign = +1) this is `(I−C)/radius` unchanged; for a backward (post-
    // reflection) pass it flips, mirroring `trace_reverse`'s `-normal`.
    let normal = normal.scale_f32(dz_sign);
    let eta = n1 / n2;
    let norm = d.norm();
    // cos1 = -(N · D)/|D|
    let cos1 = normal.dot(&d).scale(-1.0).div(&norm);
    // cos2 = sqrt(1 - eta^2 (1 - cos1^2))
    let one = TruncPoly::constant(basis.clone(), 1.0);
    let cos2_2 = one
        .sub(&one.sub(&cos1.mul(&cos1)).scale(eta * eta));
    let cos2 = cos2_2.sqrt();
    // D' = D·eta/|D| + (eta·cos1 - cos2)·N
    let dout = d
        .scale(&norm.recip().scale(eta))
        .add(&normal.scale(&cos1.scale(eta).sub(&cos2)));
    PolySystem::new(to_plane(&i, &dout))
}

/// Mirror reflection at a spherical interface of signed `radius`:
/// `D' = D − 2(D·N)N` (used for ghost / flare paths).
pub fn reflection_spherical(basis: Arc<Basis>, radius: f32, dz_sign: f32) -> PolySystem {
    let (i, d, normal) = intersect(&basis, radius, dz_sign);
    let dn = d.dot(&normal);
    let dout = d.sub(&normal.scale(&dn.scale(2.0)));
    PolySystem::new(to_plane(&i, &dout))
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{SMatrix, SVector};

    #[test]
    fn propagation_linear_part_is_shear() {
        let basis = Basis::cached(2);
        let dz = 5.0;
        let sys = propagation(basis, dz);
        let lin = sys.linear_part();
        // [[I, dz*I],[0, I]] in [x,y,u,v] ordering.
        let mut expected = SMatrix::<f32, 4, 4>::identity();
        expected[(0, 2)] = dz;
        expected[(1, 3)] = dz;
        assert!((lin - expected).norm() < 1e-5, "{}", lin);
    }

    #[test]
    fn refraction_flat_surface_no_index_change_is_identity() {
        // Large radius (flat) with n1 == n2 leaves rays essentially unchanged.
        let basis = Basis::cached(3);
        let sys = refraction_spherical(basis, 1.0e5, 1.5, 1.5, 1.0);
        let r = SVector::from([0.3, -0.2, 0.02, 0.01]);
        let out = sys.eval(r);
        assert!((out - r).norm() < 1e-3, "{} vs {}", out, r);
    }

    #[test]
    fn refraction_is_reversible_across_travel_direction() {
        // Snell's law is reversible: refracting -z through a surface (n1->n2) then
        // +z back through it (n2->n1) returns the original reduced ray. This
        // exercises the dz_sign=+1 path used by the ghost builder.
        let basis = Basis::cached(3);
        let (radius, n1, n2) = (-60.0_f32, 1.0_f32, 1.6_f32);
        let fwd = refraction_spherical(basis.clone(), radius, n1, n2, -1.0);
        let bwd = refraction_spherical(basis.clone(), radius, n2, n1, 1.0);
        let round = bwd.compose(&fwd);
        // Paraxial magnitudes: a sign error would give O(1) error, while
        // truncation leaves only ~1e-4, cleanly isolating direction correctness.
        for r in [
            SVector::from([0.2, -0.1, 0.005, 0.008]),
            SVector::from([0.3, 0.15, -0.01, 0.006]),
        ] {
            assert!((round.eval(r) - r).norm() < 1e-3, "{} vs {}", round.eval(r), r);
        }
    }

    #[test]
    fn refraction_linear_part_matches_paraxial() {
        // Paraxial refraction transfer for slopes [x,u]:
        // x' = x, u' = (n1/n2) u + (n1 - n2)/(n2 R) x   (R = sphere-center z).
        let basis = Basis::cached(3);
        let (r, n1, n2) = (70.97_f32, 1.0_f32, 1.523_f32);
        // surface map uses center at z = radius; trace_forward uses r = -lens.radius.
        let radius = -r;
        let sys = refraction_spherical(basis, radius, n1, n2, 1.0);
        let lin = sys.linear_part();
        let power = (n1 - n2) / (n2 * radius);
        assert!((lin[(0, 0)] - 1.0).abs() < 1e-4); // x' from x
        assert!(lin[(0, 2)].abs() < 1e-4); // x' independent of u at the vertex plane
        assert!((lin[(2, 2)] - n1 / n2).abs() < 1e-4); // u' from u
        assert!((lin[(2, 0)] - power).abs() < 1e-4); // u' from x (refractive power)
    }
}
