//! Build a [`PolySystem`] for a whole [`LensAssembly`] by composing per-surface
//! propagation + refraction maps (paper Eq. 9–10), and evaluate it as a cheap
//! drop-in for `trace_forward`.
//!
//! Geometry mirrors [`LensAssembly::trace_forward`] exactly: the film plane is at
//! `z = -total_thickness`, the front vertex at `z = 0`, interfaces are walked
//! rear→front with `r = -lens.radius` and the same `(n1 → n2)` media pairing, so
//! the degree-1 system reproduces matrix optics and higher degrees the trace.

use nalgebra::SVector;
use rayon::prelude::*;

use ::math::bounds::Bounds1D;

use crate::lens::{
    camera_space_to_plane, plane_to_camera_space, spectrum_eta_from_abbe_num, LensAssembly,
};
use crate::math::{PlaneRay, Ray, SimdBackend};

use super::surfaces::{propagation, refraction_spherical};
use super::system::PolySystem;
use super::trunc_poly::Basis;

/// One optical interface as `trace_forward` processes it (rear→front, +z): a
/// `gap` to propagate from the previous plane to this vertex, the signed radius
/// `r`, and the incident/transmitted media `n1`/`n2`.
struct ForwardEvent {
    gap: f32,
    r: f32,
    n1: f32,
    n2: f32,
}

/// Capture the per-surface propagation gaps + refraction media exactly as
/// `trace_forward` (and `build_forward`) traverse them, so `build_reverse` can
/// invert them element-by-element rather than re-deriving the media convention.
fn forward_events(
    assembly: &LensAssembly,
    zoom: f32,
    lambda_um: f32,
    atmosphere_ior: f32,
) -> Result<Vec<ForwardEvent>, String> {
    let total = assembly.total_thickness_at(zoom);
    let mut n1 = spectrum_eta_from_abbe_num(
        assembly.lenses.last().unwrap().ior,
        assembly.lenses.last().unwrap().vno,
        lambda_um,
    );
    let mut position = -total; // film plane
    let mut prev_z = -total;
    let mut events = Vec::with_capacity(assembly.lenses.len());

    for (k, lens) in assembly.lenses.iter().rev().enumerate() {
        if lens.anamorphic {
            return Err("anamorphic (cylindrical) surfaces are not supported yet".into());
        }
        if lens.aspheric > 0 {
            return Err("aspherical surfaces are not supported yet".into());
        }
        let r = -lens.radius;
        position += lens.thickness_at(zoom);
        let vertex_z = position;
        let gap = vertex_z - prev_z;
        // Same media bookkeeping as trace_forward.
        let n2 = if k > 0 {
            spectrum_eta_from_abbe_num(lens.ior, lens.vno, lambda_um)
        } else {
            atmosphere_ior
        };
        events.push(ForwardEvent { gap, r, n1, n2 });
        n1 = n2;
        prev_z = vertex_z;
    }
    Ok(events)
}

/// Build the film→front reduced-ray polynomial for `assembly` at the given zoom,
/// wavelength (micrometres, as `Input::lambda`), surrounding medium, and degree.
///
/// Returns `Err` if the assembly contains a surface type not yet supported
/// (aspherical or cylindrical/anamorphic).
pub fn build_forward(
    assembly: &LensAssembly,
    zoom: f32,
    lambda_um: f32,
    atmosphere_ior: f32,
    degree: usize,
) -> Result<PolySystem, String> {
    let basis = Basis::cached(degree);
    let events = forward_events(assembly, zoom, lambda_um, atmosphere_ior)?;
    let mut acc = PolySystem::identity(basis.clone());
    // Apply each surface in order: propagate to the vertex (+z), then refract.
    // The aperture interface is a flat air gap (n1 == n2), so it needs no
    // special-casing.
    for e in &events {
        acc = propagation(basis.clone(), e.gap).compose(&acc);
        acc = refraction_spherical(basis.clone(), e.r, e.n1, e.n2, 1.0).compose(&acc);
    }
    Ok(acc)
}

/// Build the front→film reduced-ray polynomial: the world→sensor map, a drop-in
/// for [`LensAssembly::trace_reverse`]. The input is a world ray at the front
/// vertex plane (`z = 0`) travelling toward the film (`−z`); the output is at the
/// film plane (`z = −total_thickness`).
///
/// `trace_reverse` is exactly the element-wise inverse of `trace_forward`, so we
/// invert the same per-surface events: traverse them front→rear (reverse order),
/// inverting each refraction (swap media, travel `−z`) and each propagation
/// (negate the gap). The direction-aware [`refraction_spherical`] flips the
/// normal to face the incident ray, mirroring `trace_reverse`'s `-normal`.
pub fn build_reverse(
    assembly: &LensAssembly,
    zoom: f32,
    lambda_um: f32,
    atmosphere_ior: f32,
    degree: usize,
) -> Result<PolySystem, String> {
    let basis = Basis::cached(degree);
    let events = forward_events(assembly, zoom, lambda_um, atmosphere_ior)?;
    let mut acc = PolySystem::identity(basis.clone());
    // Inverse of `forward = ∘_k (refract_k ∘ prop_k)` is `∘_k (prop_k⁻¹ ∘ refract_k⁻¹)`
    // applied front→rear: refract⁻¹ = refraction with media swapped and dz_sign −1,
    // prop⁻¹ = propagation by the negated gap.
    for e in events.iter().rev() {
        acc = refraction_spherical(basis.clone(), e.r, e.n2, e.n1, -1.0).compose(&acc);
        acc = propagation(basis.clone(), -e.gap).compose(&acc);
    }
    Ok(acc)
}

/// Reduce a world-space ray to `[x, y, u, v]` at plane `z = plane_z`, using the
/// **signed** slope `u = Dx/Dz` (no `abs`), so the convention is consistent for
/// rays travelling in either z direction. Forward (`+z`) agrees with
/// [`camera_space_to_plane`]; reverse (`−z`) keeps the correct slope sign.
fn to_reduced<S: SimdBackend>(ray: Ray<S>, plane_z: f32) -> [f32; 4] {
    let [ox, oy, oz, _] = ray.origin.as_array();
    let [dx, dy, dz, _] = ray.direction.as_array();
    let t = (plane_z - oz) / dz;
    [ox + t * dx, oy + t * dy, dx / dz, dy / dz]
}

/// Inverse of [`to_reduced`]: reconstruct a ray at plane `z = plane_z` travelling
/// in the `dz_sign` z-direction. The direction is `dz_sign·(u, v, 1)` normalized,
/// so the recovered slope `Dx/Dz = u` regardless of sign.
fn from_reduced<S: SimdBackend>(reduced: [f32; 4], plane_z: f32, dz_sign: f32) -> Ray<S> {
    use crate::math::{Point3, Vec3};
    Ray::new(
        Point3::new(reduced[0], reduced[1], plane_z),
        Vec3::new(dz_sign * reduced[2], dz_sign * reduced[3], dz_sign).normalized(),
    )
}

/// Per-wavelength cache of film→front polynomial systems for one assembly/zoom,
/// mirroring the [`RadialSampler`](crate::lens_sampler::RadialSampler) cache
/// pattern. Build once, then [`PolyAssembly::map_forward`] cheaply.
#[derive(Clone)]
pub struct PolyAssembly {
    /// Film→front (sensor→world) systems, one per wavelength bin.
    pub systems: Vec<PolySystem>,
    /// Front→film (world→sensor) systems, one per wavelength bin.
    pub reverse_systems: Vec<PolySystem>,
    pub wavelength_bounds: Bounds1D, // nanometres, e.g. BOUNDED_VISIBLE_RANGE
    pub wavelength_bins: usize,
    pub degree: usize,
    /// Film plane z (`-total_thickness`) and front vertex z (`0`).
    pub film_position: f32,
    pub front_position: f32,
}

impl PolyAssembly {
    /// Build the per-wavelength cache. `wavelength_bounds` are in nanometres
    /// (matching `BOUNDED_VISIBLE_RANGE`); they are converted to micrometres for
    /// the dispersion model internally.
    pub fn new(
        assembly: &LensAssembly,
        zoom: f32,
        atmosphere_ior: f32,
        degree: usize,
        wavelength_bounds: Bounds1D,
        wavelength_bins: usize,
    ) -> Result<Self, String> {
        let total = assembly.total_thickness_at(zoom);
        let lambda_um = |bin: usize| {
            wavelength_bounds.sample((0.5 + bin as f32) / wavelength_bins as f32) / 1000.0
        };
        let systems: Result<Vec<PolySystem>, String> = (0..wavelength_bins)
            .into_par_iter()
            .map(|bin| build_forward(assembly, zoom, lambda_um(bin), atmosphere_ior, degree))
            .collect();
        let reverse_systems: Result<Vec<PolySystem>, String> = (0..wavelength_bins)
            .into_par_iter()
            .map(|bin| build_reverse(assembly, zoom, lambda_um(bin), atmosphere_ior, degree))
            .collect();
        Ok(PolyAssembly {
            systems: systems?,
            reverse_systems: reverse_systems?,
            wavelength_bounds,
            wavelength_bins,
            degree,
            film_position: -total,
            front_position: 0.0,
        })
    }

    fn bin_for(&self, lambda_nm: f32) -> usize {
        let v = ((lambda_nm - self.wavelength_bounds.lower) / self.wavelength_bounds.span())
            .clamp(0.0, 1.0 - f32::EPSILON);
        ((v * self.wavelength_bins as f32) as usize).min(self.wavelength_bins - 1)
    }

    fn system_for(&self, lambda_nm: f32) -> &PolySystem {
        &self.systems[self.bin_for(lambda_nm)]
    }

    /// Map a film-side world ray (travelling toward +z) to the front vertex plane,
    /// a cheap drop-in for `trace_forward`. `lambda_nm` selects the wavelength bin.
    pub fn map_forward<S: SimdBackend>(&self, ray: Ray<S>, lambda_nm: f32) -> Ray<S> {
        let reduced = camera_space_to_plane::<S>(ray, self.film_position);
        let r = SVector::from(reduced.0);
        let out = self.system_for(lambda_nm).eval(r);
        plane_to_camera_space::<S>(PlaneRay([out[0], out[1], out[2], out[3]]), self.front_position)
    }

    /// Map a world ray travelling toward the lens (`−z`) back to the film plane,
    /// a cheap drop-in for `trace_reverse`. The input ray may sit anywhere in the
    /// world; it is reduced to the front vertex plane first. The returned ray sits
    /// on the film plane, still travelling `−z`. `lambda_nm` selects the bin.
    pub fn map_reverse<S: SimdBackend>(&self, ray: Ray<S>, lambda_nm: f32) -> Ray<S> {
        let reduced = to_reduced::<S>(ray, self.front_position);
        let out = self.reverse_systems[self.bin_for(lambda_nm)].eval(SVector::from(reduced));
        from_reduced::<S>([out[0], out[1], out[2], out[3]], self.film_position, -1.0)
    }
}

/// Thin convenience wrapper exposing a single cached system as a `Ray → Ray`
/// camera transform.
pub struct PolyLens {
    pub assembly: PolyAssembly,
}

impl PolyLens {
    pub fn new(assembly: PolyAssembly) -> Self {
        PolyLens { assembly }
    }

    pub fn map_forward<S: SimdBackend>(&self, ray: Ray<S>, lambda_nm: f32) -> Ray<S> {
        self.assembly.map_forward::<S>(ray, lambda_nm)
    }

    pub fn map_reverse<S: SimdBackend>(&self, ray: Ray<S>, lambda_nm: f32) -> Ray<S> {
        self.assembly.map_reverse::<S>(ray, lambda_nm)
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Read};

    use super::*;
    use crate::lens::camera_space_to_plane as c2p;
    use crate::math::{Input, PlaneRay};
    use crate::parse_lenses_from;

    type Backend = thermite::backend::x86_v3::X86V3;
    type Ray = crate::math::Ray<Backend>;

    fn petzval() -> LensAssembly {
        let mut s = String::new();
        File::open("data/cameras/petzval_kodak.txt")
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        let (interfaces, _, _) = parse_lenses_from(&s);
        LensAssembly::new(interfaces.as_slice())
    }

    // Build a film-plane world ray from a reduced ray [x,y,u,v].
    fn film_ray(assembly: &LensAssembly, x: f32, y: f32, u: f32, v: f32) -> Ray {
        let film_z = -assembly.total_thickness_at(0.0);
        crate::lens::plane_to_camera_space::<Backend>(PlaneRay([x, y, u, v]), film_z)
    }

    // Reduced output of the real tracer, projected to the front vertex plane z=0.
    fn trace_ref(assembly: &LensAssembly, ray: Ray, lambda_um: f32) -> Option<[f32; 4]> {
        let out = assembly.trace_forward(
            0.0,
            Input::new(ray, lambda_um),
            1.0,
            |_| (false, false),
            drop,
        )?;
        Some(c2p::<Backend>(out.ray, 0.0).0)
    }

    #[test]
    fn degree1_matches_finite_difference_jacobian() {
        let assembly = petzval();
        let lambda = 0.55;
        let sys = build_forward(&assembly, 0.0, lambda, 1.0, 1).unwrap();
        let lin = sys.linear_part();

        // Finite-difference the real film->z0 reduced map about the axial ray.
        let eps = 1e-3;
        let base = trace_ref(&assembly, film_ray(&assembly, 0.0, 0.0, 0.0, 0.0), lambda).unwrap();
        for j in 0..4 {
            let mut p = [0.0f32; 4];
            p[j] = eps;
            let r = film_ray(&assembly, p[0], p[1], p[2], p[3]);
            let plus = trace_ref(&assembly, r, lambda).unwrap();
            for i in 0..4 {
                let fd = (plus[i] - base[i]) / eps;
                assert!(
                    (fd - lin[(i, j)]).abs() < 5e-3,
                    "J[{i},{j}]: fd={fd} poly={}",
                    lin[(i, j)]
                );
            }
        }
    }

    #[test]
    fn degree3_matches_trace_near_axis() {
        let assembly = petzval();
        let lambda = 0.55;
        let sys = build_forward(&assembly, 0.0, lambda, 1.0, 3).unwrap();
        let film_z = -assembly.total_thickness_at(0.0);

        // A small grid of near-axis rays that survive the trace.
        let mut tested = 0;
        for &x in &[0.0f32, 1.0, 2.0] {
            for &u in &[0.0f32, 0.02, 0.05] {
                for &v in &[0.0f32, 0.03] {
                    let ray = film_ray(&assembly, x, 0.0, u, v);
                    let Some(reference) = trace_ref(&assembly, ray, lambda) else {
                        continue;
                    };
                    let reduced_in = c2p::<Backend>(ray, film_z).0;
                    let poly = sys.eval(SVector::from(reduced_in));
                    let err = ((poly[0] - reference[0]).powi(2)
                        + (poly[1] - reference[1]).powi(2))
                    .sqrt();
                    let derr = ((poly[2] - reference[2]).powi(2)
                        + (poly[3] - reference[3]).powi(2))
                    .sqrt();
                    assert!(
                        err < 0.05 && derr < 1e-3,
                        "x={x} u={u} v={v}: pos_err={err} dir_err={derr}"
                    );
                    tested += 1;
                }
            }
        }
        assert!(tested >= 6, "too few rays survived: {tested}");
    }

    // Reference: world->film reduced map via the real `trace_reverse`. The input
    // reduced ray is at the front plane z=0; the world ray is pushed forward (+z)
    // so the tracer's first sphere intersection is well-conditioned.
    fn trace_rev_ref(assembly: &LensAssembly, reduced: [f32; 4], lambda: f32) -> Option<[f32; 4]> {
        let at_front = from_reduced::<Backend>(reduced, 0.0, -1.0);
        let ray: Ray = at_front.at_time(-20.0); // move origin to z = +20 along the same line
        let film_z = -assembly.total_thickness_at(0.0);
        let out = assembly.trace_reverse(
            0.0,
            Input::new(ray, lambda),
            1.0,
            |_| (false, false),
            drop,
        )?;
        Some(to_reduced::<Backend>(out.ray, film_z))
    }

    #[test]
    fn reverse_degree1_matches_finite_difference_jacobian() {
        let assembly = petzval();
        let lambda = 0.55;
        let sys = build_reverse(&assembly, 0.0, lambda, 1.0, 1).unwrap();
        let lin = sys.linear_part();

        let eps = 1e-3;
        let base = trace_rev_ref(&assembly, [0.0; 4], lambda).unwrap();
        for j in 0..4 {
            let mut p = [0.0f32; 4];
            p[j] = eps;
            let plus = trace_rev_ref(&assembly, p, lambda).unwrap();
            for i in 0..4 {
                let fd = (plus[i] - base[i]) / eps;
                assert!(
                    (fd - lin[(i, j)]).abs() < 5e-3,
                    "reverse J[{i},{j}]: fd={fd} poly={}",
                    lin[(i, j)]
                );
            }
        }
    }

    #[test]
    fn reverse_degree3_matches_trace_near_axis() {
        let assembly = petzval();
        let lambda = 0.55;
        let sys = build_reverse(&assembly, 0.0, lambda, 1.0, 3).unwrap();

        let mut tested = 0;
        for &x in &[0.0f32, 1.0, 2.0] {
            for &u in &[0.0f32, 0.01, 0.02] {
                for &v in &[0.0f32, 0.015] {
                    let reduced = [x, 0.0, u, v];
                    let Some(reference) = trace_rev_ref(&assembly, reduced, lambda) else {
                        continue;
                    };
                    let poly = sys.eval(SVector::from(reduced));
                    let err = ((poly[0] - reference[0]).powi(2)
                        + (poly[1] - reference[1]).powi(2))
                    .sqrt();
                    let derr = ((poly[2] - reference[2]).powi(2)
                        + (poly[3] - reference[3]).powi(2))
                    .sqrt();
                    assert!(
                        err < 0.05 && derr < 1e-3,
                        "x={x} u={u} v={v}: pos_err={err} dir_err={derr}"
                    );
                    tested += 1;
                }
            }
        }
        assert!(tested >= 6, "too few reverse rays survived: {tested}");
    }
}
