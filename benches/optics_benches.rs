//! Benchmarks for the major functions of `rust_optics`.
//!
//! Groups, roughly in order of how hot they are in a render:
//!   - `assembly_trace`   : full sensor->pupil traversal (`trace_forward` / `trace_reverse`)
//!   - `surface_intersect`: per-surface intersection primitives
//!   - `optics_math`      : refraction / fresnel / dispersion
//!   - `coordinate_space` : plane/sphere/camera ray-space conversions
//!   - `aperture`         : rejection test + importance sampling
//!   - `radial_sampler`   : cache construction + per-sample lookup
//!
//! Post-migration these run against `rust_cg_math` 3.0.0 (thermite backend).
//! The library is generic over the backend; these benches monomorphize on
//! `x86_v3::X86V3` (see the `Backend` alias below). Build with
//! `RUSTFLAGS="-C target-cpu=native"` so the AVX2 paths are actually emitted.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use math::spectral::BOUNDED_VISIBLE_RANGE;
use optics::math::*;

use optics::aperture::ApertureSample;
use optics::lens::{
    camera_space_to_plane, camera_space_to_sphere, evaluate_aspherical,
    evaluate_aspherical_derivative, fresnel, plane_to_camera_space, refract, sample_point_on_lens,
    sphere_to_camera_space, spectrum_eta_from_abbe_num, trace_aspherical, trace_cylindrical,
    trace_spherical,
};
use optics::lens_sampler::RadialSampler;
// Input, PlaneRay, SphereRay come in via `optics::math::*` above.
use optics::{
    parse_lenses_from, Aperture, CircularAperture, LensAssembly, SimpleBladedAperture,
};

// The library is generic over the SIMD backend; the benches pick a concrete one
// (AVX2+FMA). These aliases shadow the generic re-exports from `optics::math::*`
// so the bench bodies can use the bare type names.
// type Backend = thermite::backend::x86_v3::X86V3;
type Backend = thermite::backend::scalar::Scalar;
type Vec3 = optics::math::Vec3<Backend>;
type Point3 = optics::math::Point3<Backend>;
type Ray = optics::math::Ray<Backend>;

/// A real, multi-element double-Gauss-ish lens so the assembly traces exercise
/// a representative number of surfaces.
const LENS_SPEC: &str = include_str!("../data/cameras/petzval_kodak.txt");

fn build_assembly() -> LensAssembly {
    let (lenses, _, _) = parse_lenses_from(LENS_SPEC);
    LensAssembly::new(&lenses)
}

/// A ray near the optical axis, pointed straight down the +Z axis, that survives
/// the surface-intersection primitives below (mirrors `basic_incoming_ray` in the tests).
fn axial_ray() -> Ray {
    Ray::new(Point3::new(0.1, 0.0, 10.0), -Vec3::z_axis())
}

fn bench_surface_intersect(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface_intersect");
    let ray = axial_ray();
    let correction = [0.0f32; 4];

    group.bench_function("trace_spherical", |b| {
        b.iter(|| trace_spherical(black_box(ray), black_box(0.9), black_box(-1.0), black_box(0.9)))
    });
    group.bench_function("trace_cylindrical", |b| {
        b.iter(|| {
            trace_cylindrical(black_box(ray), black_box(0.9), black_box(1.0), black_box(0.9))
        })
    });
    group.bench_function("trace_aspherical", |b| {
        b.iter(|| {
            trace_aspherical(
                black_box(ray),
                black_box(0.9),
                black_box(1.0),
                black_box(1),
                black_box(correction),
                black_box(0.9),
            )
        })
    });
    group.bench_function("evaluate_aspherical", |b| {
        b.iter(|| {
            evaluate_aspherical(black_box(ray.origin), black_box(0.9), black_box(1), black_box(correction))
        })
    });
    group.bench_function("evaluate_aspherical_derivative", |b| {
        b.iter(|| {
            evaluate_aspherical_derivative(
                black_box(ray.origin),
                black_box(0.9),
                black_box(1),
                black_box(correction),
            )
        })
    });
    group.finish();
}

fn bench_optics_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics_math");

    // recover a real surface normal to refract against
    let ray = axial_ray();
    let (_, normal) = trace_spherical(ray, 40.0, -42.0, 30.0).unwrap();
    let dir = ray.direction;

    group.bench_function("refract", |b| {
        b.iter(|| {
            refract(
                black_box(1.0),
                black_box(1.45),
                black_box(normal),
                black_box(dir),
            )
        })
    });
    group.bench_function("fresnel", |b| {
        b.iter(|| {
            fresnel(
                black_box(1.0),
                black_box(1.45),
                black_box(0.8),
                black_box(0.6),
            )
        })
    });
    group.bench_function("spectrum_eta_from_abbe_num", |b| {
        b.iter(|| {
            spectrum_eta_from_abbe_num(black_box(1.5), black_box(54.0), black_box(0.55))
        })
    });
    group.finish();
}

fn bench_coordinate_space(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinate_space");

    let plane_ray = PlaneRay::new(1.0, 2.0, 0.05, 0.05);
    let camera_ray: Ray = plane_to_camera_space(plane_ray, 1.0);
    let sphere_ray = camera_space_to_sphere(camera_ray, -100.0, 100.0);
    let sample = Sample2D::new(0.3, 0.7);

    group.bench_function("plane_to_camera_space", |b| {
        b.iter(|| plane_to_camera_space::<Backend>(black_box(plane_ray), black_box(1.0)))
    });
    group.bench_function("camera_space_to_plane", |b| {
        b.iter(|| camera_space_to_plane(black_box(camera_ray), black_box(1.0)))
    });
    group.bench_function("sphere_to_camera_space", |b| {
        b.iter(|| {
            sphere_to_camera_space::<Backend>(black_box(sphere_ray), black_box(-100.0), black_box(100.0))
        })
    });
    group.bench_function("camera_space_to_sphere", |b| {
        b.iter(|| camera_space_to_sphere(black_box(camera_ray), black_box(-100.0), black_box(100.0)))
    });
    group.bench_function("sample_point_on_lens", |b| {
        b.iter(|| sample_point_on_lens::<Backend>(black_box(35.0), black_box(15.0), black_box(sample)))
    });
    group.finish();
}

fn bench_aperture(c: &mut Criterion) {
    let mut group = c.benchmark_group("aperture");

    let circular = CircularAperture::default();
    let bladed = SimpleBladedAperture::new(6, 0.5);
    let p = Point3::new(4.0, 3.0, 0.0);

    group.bench_function("circular_is_rejected", |b| {
        b.iter(|| circular.is_rejected(black_box(10.0), black_box(p)))
    });
    group.bench_function("bladed_is_rejected", |b| {
        b.iter(|| bladed.is_rejected(black_box(10.0), black_box(p)))
    });
    group.bench_function("bladed_sample", |b| {
        b.iter(|| bladed.sample::<Backend>(black_box(Sample2D::new_random_sample())))
    });
    group.finish();
}

fn bench_assembly_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly_trace");

    let assembly = build_assembly();
    let aperture = SimpleBladedAperture::new(6, 0.5);
    let aperture_radius = assembly.aperture_radius();
    let film_position = assembly.total_thickness_at(0.0);

    // forward: from the sensor plane out through the front of the lens
    let forward_input = Input::new(
        Ray::new(Point3::new(0.0, 0.0, -film_position), Vec3::z_axis()),
        0.55,
    );
    group.bench_function("trace_forward", |b| {
        b.iter(|| {
            assembly.trace_forward(
                black_box(0.0),
                black_box(forward_input),
                black_box(1.04),
                |ray| (aperture.is_rejected(aperture_radius, ray.origin), false),
                drop,
            )
        })
    });

    // reverse: from a world-space ray back to the sensor
    let reverse_input = Input::new(
        Ray::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)),
        0.55,
    );
    group.bench_function("trace_reverse", |b| {
        b.iter(|| {
            assembly.trace_reverse(
                black_box(0.0),
                black_box(reverse_input),
                black_box(1.04),
                |ray| (aperture.is_rejected(aperture_radius, ray.origin), false),
                drop,
            )
        })
    });
    group.finish();
}

fn bench_radial_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("radial_sampler");
    // keep the bin counts modest so cache construction stays bench-friendly
    const RADIUS_BINS: usize = 64;
    const WAVELENGTH_BINS: usize = 64;

    let assembly = build_assembly();
    let aperture = SimpleBladedAperture::new(6, 0.5);
    let film_position = assembly.total_thickness_at(0.0);

    let build = || {
        RadialSampler::new::<Backend, _>(
            35.0,
            RADIUS_BINS,
            WAVELENGTH_BINS,
            BOUNDED_VISIBLE_RANGE,
            -film_position,
            &assembly,
            0.0,
            &aperture,
            0.1,
            35.0,
        )
    };

    // sample-time cost (the hot path during rendering)
    let sampler = build();
    group.bench_function("sample", |b| {
        b.iter(|| {
            let lambda = BOUNDED_VISIBLE_RANGE.sample(Sample1D::new_random_sample().x);
            let point = Point3::new(3.0, 2.0, -film_position);
            sampler.sample(
                black_box(lambda),
                black_box(point),
                black_box(Sample2D::new_random_sample()),
                black_box(Sample1D::new_random_sample()),
            )
        })
    });

    // one-off cache construction cost
    group.sample_size(20);
    group.bench_function("new_cache", |b| b.iter(build));
    group.finish();
}

criterion_group!(
    benches,
    bench_surface_intersect,
    bench_optics_math,
    bench_coordinate_space,
    bench_aperture,
    bench_assembly_trace,
    bench_radial_sampler,
);
criterion_main!(benches);
