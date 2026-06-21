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

use crate::aperture::Aperture;
use crate::lens::{
    camera_space_to_plane, plane_to_camera_space, spectrum_eta_from_abbe_num, LensAssembly, LensType,
};
use crate::math::{PlaneRay, Point3, Ray, SimdBackend};

use super::surfaces::{propagation, refraction_spherical};
use super::system::PolySystem;
use super::trunc_poly::Basis;

/// One optical interface as `trace_forward` processes it (rear→front, +z): a
/// `gap` to propagate from the previous plane to this vertex, the signed radius
/// `r`, the incident/transmitted media `n1`/`n2`, the `housing_radius` (barrel
/// vignetting limit), and whether it is the aperture stop.
struct ForwardEvent {
    gap: f32,
    r: f32,
    n1: f32,
    n2: f32,
    housing_radius: f32,
    is_stop: bool,
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
        events.push(ForwardEvent {
            gap,
            r,
            n1,
            n2,
            housing_radius: lens.housing_radius,
            is_stop: lens.lens_type == LensType::Aperture,
        });
        n1 = n2;
        prev_z = vertex_z;
    }
    Ok(events)
}

/// A per-surface "tap": the reduced ray `[x, y, u, v]` of the *incident* ray at
/// one surface's vertex plane, plus the geometry/media needed to test that
/// surface for vignetting (housing radius or aperture stop) and to accumulate its
/// Fresnel transmittance. Stored per wavelength bin in [`PolyAssembly`].
#[derive(Clone)]
pub(crate) struct SurfaceTap {
    map: PolySystem,
    radius: f32,
    housing_radius: f32,
    n1: f32,
    n2: f32,
    is_stop: bool,
}

/// Fresnel transmittance (`1 − R`) at a spherical surface, from the incident
/// reduced ray `[x, y, u, v]` (slopes) and the surface's signed `radius`/media.
/// Returns 0 on total internal reflection. Mirrors `refract`/`fresnel` in
/// `lens.rs`; the incidence cosine uses `|D̂·N|`, so it is travel-direction-sign
/// independent (works for forward and reverse taps).
fn surface_transmittance(x: f32, y: f32, u: f32, v: f32, radius: f32, n1: f32, n2: f32) -> f32 {
    if n1 == n2 {
        return 1.0;
    }
    let inv = (1.0 + u * u + v * v).sqrt().recip();
    let (dx, dy, dz) = (u * inv, v * inv, inv);
    let rho2 = x * x + y * y;
    // unit surface normal at lateral (x,y): (x/R, y/R, -sqrt(R^2-rho^2)/R)
    let nz = -((radius * radius - rho2).max(0.0)).sqrt() / radius;
    let cos1 = (dx * (x / radius) + dy * (y / radius) + dz * nz).abs();
    let eta = n1 / n2;
    let cos2_2 = 1.0 - eta * eta * (1.0 - cos1 * cos1);
    if cos2_2 <= 0.0 {
        return 0.0; // total internal reflection
    }
    let cos2 = cos2_2.sqrt();
    (1.0 - crate::lens::fresnel(n1, n2, cos1, cos2)).max(0.0)
}

/// Build the film→world map plus a per-surface tap for each interface (incident
/// reduced ray at that surface), used for barrel vignetting + Fresnel falloff.
pub(crate) fn build_forward_taps(
    assembly: &LensAssembly,
    zoom: f32,
    lambda_um: f32,
    atmosphere_ior: f32,
    degree: usize,
) -> Result<(PolySystem, Vec<SurfaceTap>), String> {
    let basis = Basis::cached(degree);
    let events = forward_events(assembly, zoom, lambda_um, atmosphere_ior)?;
    let mut acc = PolySystem::identity(basis.clone());
    let mut taps = Vec::with_capacity(events.len());
    for e in &events {
        // propagate to the vertex; `acc` now maps film → incident ray at this surface
        acc = propagation(basis.clone(), e.gap).compose(&acc);
        let incident = acc.clone();
        acc = refraction_spherical(basis.clone(), e.r, e.n1, e.n2, 1.0).compose(&acc);
        taps.push(SurfaceTap {
            map: incident,
            radius: e.r,
            housing_radius: e.housing_radius,
            n1: e.n1,
            n2: e.n2,
            is_stop: e.is_stop,
        });
    }
    Ok((acc, taps))
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
    Ok(build_forward_taps(assembly, zoom, lambda_um, atmosphere_ior, degree)?.0)
}

/// Build the world→film map plus per-surface taps for the reverse direction.
/// `trace_reverse` is the element-wise inverse of `trace_forward`: traverse the
/// events front→rear, inverting each refraction (swap media, travel `−z`) and
/// each propagation (negate the gap). The incident tap is snapshotted before each
/// reverse refraction, with media swapped to the reverse travel direction.
pub(crate) fn build_reverse_taps(
    assembly: &LensAssembly,
    zoom: f32,
    lambda_um: f32,
    atmosphere_ior: f32,
    degree: usize,
) -> Result<(PolySystem, Vec<SurfaceTap>), String> {
    let basis = Basis::cached(degree);
    let events = forward_events(assembly, zoom, lambda_um, atmosphere_ior)?;
    let mut acc = PolySystem::identity(basis.clone());
    let mut taps = Vec::with_capacity(events.len());
    for e in events.iter().rev() {
        let incident = acc.clone(); // world → incident ray at this surface
        acc = refraction_spherical(basis.clone(), e.r, e.n2, e.n1, -1.0).compose(&acc);
        taps.push(SurfaceTap {
            map: incident,
            radius: e.r,
            housing_radius: e.housing_radius,
            n1: e.n2,
            n2: e.n1,
            is_stop: e.is_stop,
        });
        acc = propagation(basis.clone(), -e.gap).compose(&acc);
    }
    Ok((acc, taps))
}

/// Build the front→film reduced-ray polynomial: the world→sensor map, a drop-in
/// for [`LensAssembly::trace_reverse`]. The input is a world ray at the front
/// vertex plane (`z = 0`) travelling toward the film (`−z`); the output is at the
/// film plane (`z = −total_thickness`).
pub fn build_reverse(
    assembly: &LensAssembly,
    zoom: f32,
    lambda_um: f32,
    atmosphere_ior: f32,
    degree: usize,
) -> Result<PolySystem, String> {
    Ok(build_reverse_taps(assembly, zoom, lambda_um, atmosphere_ior, degree)?.0)
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
    /// Per-surface incident-ray taps for the forward direction (`[bin][surface]`),
    /// used to clip at each surface's housing / the aperture stop and to
    /// accumulate Fresnel transmittance.
    pub(crate) forward_taps: Vec<Vec<SurfaceTap>>,
    /// Per-surface incident-ray taps for the reverse direction.
    pub(crate) reverse_taps: Vec<Vec<SurfaceTap>>,
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
        let forward: Vec<(PolySystem, Vec<SurfaceTap>)> = (0..wavelength_bins)
            .into_par_iter()
            .map(|bin| build_forward_taps(assembly, zoom, lambda_um(bin), atmosphere_ior, degree))
            .collect::<Result<_, _>>()?;
        let reverse: Vec<(PolySystem, Vec<SurfaceTap>)> = (0..wavelength_bins)
            .into_par_iter()
            .map(|bin| build_reverse_taps(assembly, zoom, lambda_um(bin), atmosphere_ior, degree))
            .collect::<Result<_, _>>()?;
        let (systems, forward_taps): (Vec<_>, Vec<_>) = forward.into_iter().unzip();
        let (reverse_systems, reverse_taps): (Vec<_>, Vec<_>) = reverse.into_iter().unzip();
        Ok(PolyAssembly {
            systems,
            reverse_systems,
            forward_taps,
            reverse_taps,
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

    /// Like [`map_forward`](Self::map_forward), but applies aperture clipping and,
    /// when `vignetting` is set, the lens's barrel vignetting + Fresnel falloff.
    ///
    /// The aperture stop is *always* honored (so the aperture control stays
    /// meaningful): the stop tap is tested against [`Aperture::is_rejected`], so
    /// bladed/arbitrary stops work for free. With `vignetting = true` every other
    /// surface is also tested against its housing radius and its Fresnel
    /// transmittance is multiplied in; with `vignetting = false` neither applies
    /// and the returned transmittance is `1`. Returns `(exit ray, transmittance)`,
    /// or `None` if blocked. (See [`map_forward`](Self::map_forward) for the
    /// fully-unclipped geometric map.)
    pub fn map_forward_clipped<S: SimdBackend, A: Aperture>(
        &self,
        ray: Ray<S>,
        lambda_nm: f32,
        aperture: &A,
        aperture_radius: f32,
        vignetting: bool,
    ) -> Option<(Ray<S>, f32)> {
        let bin = self.bin_for(lambda_nm);
        let reduced = camera_space_to_plane::<S>(ray, self.film_position).0;
        let tau = self.clip_and_transmittance::<S, A>(
            &self.forward_taps[bin],
            &reduced,
            aperture,
            aperture_radius,
            vignetting,
        )?;
        let out = self.systems[bin].eval(SVector::from(reduced));
        Some((
            plane_to_camera_space::<S>(
                PlaneRay([out[0], out[1], out[2], out[3]]),
                self.front_position,
            ),
            tau,
        ))
    }

    /// Walk the per-surface taps. The aperture stop is always tested; with
    /// `vignetting` also reject on each surface's housing radius and accumulate
    /// Fresnel transmittance. `None` if any surface blocks the ray.
    fn clip_and_transmittance<S: SimdBackend, A: Aperture>(
        &self,
        taps: &[SurfaceTap],
        reduced: &[f32; 4],
        aperture: &A,
        aperture_radius: f32,
        vignetting: bool,
    ) -> Option<f32> {
        let mut tau = 1.0f32;
        for tap in taps {
            // Position is only needed for the stop, the barrel test, or Fresnel.
            if !tap.is_stop && !vignetting {
                continue;
            }
            let x = tap.map.polys[0].eval(reduced);
            let y = tap.map.polys[1].eval(reduced);
            if tap.is_stop {
                if aperture.is_rejected(aperture_radius, Point3::<S>::new(x, y, 0.0)) {
                    return None;
                }
            } else if x * x + y * y > tap.housing_radius * tap.housing_radius {
                return None; // barrel vignetting
            }
            if vignetting && tap.n1 != tap.n2 {
                let u = tap.map.polys[2].eval(reduced);
                let v = tap.map.polys[3].eval(reduced);
                let t = surface_transmittance(x, y, u, v, tap.radius, tap.n1, tap.n2);
                if t <= 0.0 {
                    return None; // total internal reflection
                }
                tau *= t;
            }
        }
        Some(tau)
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

    /// Reverse counterpart of [`map_forward_clipped`](Self::map_forward_clipped):
    /// vignettes + attenuates a world→film ray. Returns `(film ray, transmittance)`
    /// or `None` if blocked.
    pub fn map_reverse_clipped<S: SimdBackend, A: Aperture>(
        &self,
        ray: Ray<S>,
        lambda_nm: f32,
        aperture: &A,
        aperture_radius: f32,
        vignetting: bool,
    ) -> Option<(Ray<S>, f32)> {
        let bin = self.bin_for(lambda_nm);
        let reduced = to_reduced::<S>(ray, self.front_position);
        let tau = self.clip_and_transmittance::<S, A>(
            &self.reverse_taps[bin],
            &reduced,
            aperture,
            aperture_radius,
            vignetting,
        )?;
        let out = self.reverse_systems[bin].eval(SVector::from(reduced));
        Some((
            from_reduced::<S>([out[0], out[1], out[2], out[3]], self.film_position, -1.0),
            tau,
        ))
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

    pub fn map_forward_clipped<S: SimdBackend, A: Aperture>(
        &self,
        ray: Ray<S>,
        lambda_nm: f32,
        aperture: &A,
        aperture_radius: f32,
        vignetting: bool,
    ) -> Option<(Ray<S>, f32)> {
        self.assembly
            .map_forward_clipped::<S, A>(ray, lambda_nm, aperture, aperture_radius, vignetting)
    }

    pub fn map_reverse<S: SimdBackend>(&self, ray: Ray<S>, lambda_nm: f32) -> Ray<S> {
        self.assembly.map_reverse::<S>(ray, lambda_nm)
    }

    pub fn map_reverse_clipped<S: SimdBackend, A: Aperture>(
        &self,
        ray: Ray<S>,
        lambda_nm: f32,
        aperture: &A,
        aperture_radius: f32,
        vignetting: bool,
    ) -> Option<(Ray<S>, f32)> {
        self.assembly
            .map_reverse_clipped::<S, A>(ray, lambda_nm, aperture, aperture_radius, vignetting)
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Read};

    use super::*;
    use ::math::spectral::BOUNDED_VISIBLE_RANGE;
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

    #[test]
    fn clip_with_huge_aperture_passes_and_matches_unclipped() {
        use crate::aperture::CircularAperture;
        let assembly = petzval();
        let poly = PolyAssembly::new(&assembly, 0.0, 1.0, 3, BOUNDED_VISIBLE_RANGE, 16).unwrap();
        let circ = CircularAperture::default();
        for &(x, u) in &[(0.0f32, 0.0f32), (2.0, 0.01), (-3.0, -0.02)] {
            let ray = film_ray(&assembly, x, 0.0, u, 0.0);
            // huge aperture + huge implied housings: never vignettes; geometry must
            // match the unclipped map, and transmittance is in (0, 1].
            let (clipped, tau) = poly
                .map_forward_clipped::<Backend, _>(ray, 550.0, &circ, 1.0e6, true)
                .expect("huge aperture must never reject");
            let plain = poly.map_forward::<Backend>(ray, 550.0);
            assert!((clipped.origin - plain.origin).norm() < 1e-5);
            assert!((clipped.direction - plain.direction).norm() < 1e-5);
            assert!(tau > 0.0 && tau <= 1.0, "transmittance out of range: {tau}");
        }
    }

    #[test]
    fn clip_with_tiny_bladed_aperture_rejects_offaxis() {
        use crate::aperture::SimpleBladedAperture;
        let assembly = petzval();
        let poly = PolyAssembly::new(&assembly, 0.0, 1.0, 3, BOUNDED_VISIBLE_RANGE, 16).unwrap();
        let bladed = SimpleBladedAperture::new(6, 0.5);
        // A ray steeply off-axis at the film crosses the stop far off-center and
        // must be culled by a sub-millimetre aperture.
        let ray = film_ray(&assembly, 0.0, 0.0, 0.2, 0.0);
        assert!(
            poly.map_forward_clipped::<Backend, _>(ray, 550.0, &bladed, 0.05, true)
                .is_none(),
            "tiny bladed stop should reject the steep off-axis ray"
        );
        // An on-axis ray passes the stop dead-center and survives.
        let axial = film_ray(&assembly, 0.0, 0.0, 0.0, 0.0);
        assert!(
            poly.map_forward_clipped::<Backend, _>(axial, 550.0, &bladed, 0.05, true)
                .is_some(),
            "on-axis ray should pass the stop"
        );
    }

    #[test]
    fn clip_iris_position_matches_trace() {
        // The aperture-stop tap's position polys should predict the real iris
        // crossing (from trace_forward's per-surface step hook) near axis.
        let assembly = petzval();
        let n = assembly.lenses.len();
        let ai = assembly.aperture_index;
        let film_z = -assembly.total_thickness_at(0.0);
        let (_, taps) = build_forward_taps(&assembly, 0.0, 0.55, 1.0, 3).unwrap();
        // forward taps are in rear->front order; the stop is at the same index.
        let stop_tap = taps.iter().find(|t| t.is_stop).expect("petzval has a stop");

        let mut tested = 0;
        for &(x, u) in &[(0.0f32, 0.0f32), (1.0, 0.0), (0.0, 0.02), (2.0, -0.01)] {
            let ray = film_ray(&assembly, x, 0.0, u, 0.0);
            let mut ends: Vec<crate::math::Point3<Backend>> = Vec::new();
            let survived = assembly
                .trace_forward(
                    0.0,
                    Input::new(ray, 0.55),
                    1.0,
                    |_| (false, false),
                    |(_, b, _)| ends.push(b),
                )
                .is_some();
            if !survived {
                continue;
            }
            // step hook emits: [film-plane point, then one per surface, rear->front].
            let iris = ends[1 + (n - 1 - ai)];
            let reduced = c2p::<Backend>(ray, film_z).0;
            let px = stop_tap.map.polys[0].eval(&reduced);
            let py = stop_tap.map.polys[1].eval(&reduced);
            assert!(
                (px - iris.x()).abs() < 0.1 && (py - iris.y()).abs() < 0.1,
                "iris pos poly=({px},{py}) trace=({},{})",
                iris.x(),
                iris.y()
            );
            tested += 1;
        }
        assert!(tested >= 3, "too few rays reached the iris: {tested}");
    }

    #[test]
    fn transmittance_falls_off_off_axis() {
        // Fresnel transmittance should decrease as incidence angles steepen
        // (more off-axis), reproducing the trace's edge darkening.
        use crate::aperture::CircularAperture;
        let assembly = petzval();
        let poly = PolyAssembly::new(&assembly, 0.0, 1.0, 3, BOUNDED_VISIBLE_RANGE, 16).unwrap();
        let circ = CircularAperture::default();
        let axial = poly
            .map_forward_clipped::<Backend, _>(
                film_ray(&assembly, 0.0, 0.0, 0.0, 0.0),
                550.0,
                &circ,
                1.0e6,
                true,
            )
            .unwrap()
            .1;
        let off = poly
            .map_forward_clipped::<Backend, _>(
                film_ray(&assembly, 0.0, 0.0, 0.08, 0.0),
                550.0,
                &circ,
                1.0e6,
                true,
            )
            .expect("modest off-axis ray should survive vignetting")
            .1;
        assert!(axial > 0.0 && axial <= 1.0);
        assert!(off < axial, "off-axis tau {off} should be < axial tau {axial}");
    }

    #[test]
    fn unvignetted_keeps_tau_one_and_skips_barrel() {
        // With vignetting off: the aperture stop is still honored, but no barrel
        // clipping and no Fresnel — so tau == 1, and a ray that the barrel would
        // vignette survives (as long as it clears the stop).
        use crate::aperture::CircularAperture;
        let assembly = petzval();
        let poly = PolyAssembly::new(&assembly, 0.0, 1.0, 3, BOUNDED_VISIBLE_RANGE, 16).unwrap();
        let circ = CircularAperture::default();
        // slope 0.25 is barrel-vignetted with vignetting on (see other tests)...
        let ray = film_ray(&assembly, 0.0, 0.0, 0.25, 0.0);
        assert!(
            poly.map_forward_clipped::<Backend, _>(ray, 550.0, &circ, 1.0e6, true)
                .is_none(),
            "vignetting on should cull the steep ray at the barrel"
        );
        // ...but survives with vignetting off, at full transmittance.
        let (_, tau) = poly
            .map_forward_clipped::<Backend, _>(ray, 550.0, &circ, 1.0e6, false)
            .expect("vignetting off should not barrel-clip");
        assert_eq!(tau, 1.0);
        // The aperture stop is still enforced even with vignetting off.
        use crate::aperture::SimpleBladedAperture;
        let bladed = SimpleBladedAperture::new(6, 0.5);
        assert!(
            poly.map_forward_clipped::<Backend, _>(
                film_ray(&assembly, 0.0, 0.0, 0.2, 0.0),
                550.0,
                &bladed,
                0.05,
                false
            )
            .is_none(),
            "the stop must still cull even with vignetting off"
        );
    }
}
