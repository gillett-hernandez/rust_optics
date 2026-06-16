//! Graphs the importance-sampling *efficiency* of [`RadialSampler`] across the
//! film, for a fixed lens assembly and aperture.
//!
//! `RadialSampler` precomputes, per (film-radius, wavelength) bin, the angular
//! cone of ray directions that make it through the lens + aperture, then draws
//! sample directions uniformly within that (slightly inflated) cone. Because the
//! cone is an approximation, not every sampled ray actually survives a real
//! `trace_forward` — the surviving fraction is the sampler's efficiency.
//!
//! This binary sweeps the distance from the optical axis (x) and plots the
//! measured efficiency (y, 0..1), one curve per wavelength, to a PNG, and writes
//! the raw numbers to a CSV alongside it.
//!
//! Build with `RUSTFLAGS="-C target-cpu=native"` so the AVX2 paths are emitted.
//! Example:
//!   cargo run --release --example radial_efficiency -- --lens data/cameras/petzval_kodak.txt

use std::error::Error;
use std::f32::consts::SQRT_2;
use std::fs::File;
use std::io::{Read, Write};

use math::spectral::BOUNDED_VISIBLE_RANGE;
use plotters::prelude::*;
use structopt::StructOpt;

use optics::aperture::{Aperture, SimpleBladedAperture};
use optics::lens_sampler::RadialSampler;
use optics::math::*;
use optics::{LensAssembly, parse_lenses_from};

// The library is generic over the SIMD backend; this binary monomorphizes on the
// AVX2+FMA backend. These aliases shadow the generic re-exports from
// `optics::math::*` so the body can use the bare type names.
type Backend = thermite::backend::x86_v3::X86V3;
type Point3 = optics::math::Point3<Backend>;
type Ray = optics::math::Ray<Backend>;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    /// Path to the lens spec file.
    #[structopt(long)]
    pub lens: String,

    /// Aperture radius ("aperture size"). Defaults to the assembly's own aperture radius.
    #[structopt(long)]
    pub aperture_radius: Option<f32>,

    /// Number of aperture blades.
    #[structopt(long, default_value = "6")]
    pub blades: u8,

    /// Bladed-aperture roundness parameter (p > 0).
    #[structopt(long, default_value = "0.5")]
    pub p: f32,

    /// Comma-separated wavelengths in nm, one efficiency curve each.
    #[structopt(long, default_value = "450,550,650")]
    pub wavelengths: String,

    /// Number of annulus bins along the x axis (radius). Keep < radius-bins so the
    /// cache's per-cell construction lottery averages out.
    #[structopt(long, default_value = "64")]
    pub radius_steps: usize,

    /// Rays traced per (annulus, wavelength) measurement.
    #[structopt(long, default_value = "4000")]
    pub samples: usize,

    /// Radial bins for the sampler's direction cache.
    #[structopt(long, default_value = "256")]
    pub radius_bins: usize,

    /// Wavelength bins for the sampler's direction cache.
    #[structopt(long, default_value = "100")]
    pub wavelength_bins: usize,

    /// Physical sensor size in mm (sets the radius sweep extent).
    #[structopt(long, default_value = "35.0")]
    pub sensor_size: f32,

    #[structopt(long, default_value = "0.0")]
    pub film_position_offset: f32,

    /// Angular step used while building the direction cache.
    #[structopt(long, default_value = "0.01")]
    pub solver_heat: f32,

    /// Atmosphere IOR used at trace time. Matches the cache (1.0) by default.
    #[structopt(long, default_value = "1.0")]
    pub atmosphere_ior: f32,

    /// Output file prefix; writes `<prefix>.png` and `<prefix>.csv`.
    #[structopt(long, default_value = "radial_efficiency")]
    pub out: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    let wavelengths: Vec<f32> = opt
        .wavelengths
        .split(',')
        .map(|s| s.trim().parse::<f32>().expect("invalid wavelength"))
        .collect();
    assert!(!wavelengths.is_empty(), "need at least one wavelength");

    // --- fixed lens assembly + aperture ---------------------------------------
    let mut camera_spec = String::new();
    File::open(&opt.lens)
        .unwrap_or_else(|e| panic!("could not open lens file {}: {}", opt.lens, e))
        .read_to_string(&mut camera_spec)?;
    let (interfaces, _, _) = parse_lenses_from(&camera_spec);
    let assembly = LensAssembly::new(&interfaces);

    let aperture = SimpleBladedAperture::new(opt.blades, opt.p);
    let aperture_radius = opt
        .aperture_radius
        .unwrap_or_else(|| assembly.aperture_radius());
    let film_position = assembly.total_thickness_at(0.0) + opt.film_position_offset;
    let radius_cap = SQRT_2 * opt.sensor_size / 2.0;

    println!(
        "lens: {} | aperture radius: {:.3} | film position: {:.3} | radius cap: {:.3}",
        opt.lens, aperture_radius, film_position, radius_cap
    );

    // --- one direction cache for the fixed configuration ----------------------
    println!("building radial sampler cache...");
    let sampler = RadialSampler::new::<Backend, _>(
        radius_cap,
        opt.radius_bins,
        opt.wavelength_bins,
        BOUNDED_VISIBLE_RANGE,
        -film_position,
        &assembly,
        0.0,
        &aperture,
        opt.solver_heat,
        opt.sensor_size,
    );

    // --- measure efficiency ---------------------------------------------------
    // Each graph point is an annulus bin [r_lo, r_hi); we draw many random film
    // points within it (random radius + rotation angle) and, for each, importance-
    // sample a direction and trace it. Efficiency = survivors / samples. The cache
    // is finer than the graph (radius_bins >> radius_steps) so its per-cell
    // construction lottery averages out into a smooth, renderer-relevant curve:
    // the expected pass rate for a source at that field radius.
    println!(
        "measuring efficiency over {} annuli x {} wavelengths ({} samples each)...",
        opt.radius_steps,
        wavelengths.len(),
        opt.samples
    );
    let mut radii: Vec<f32> = Vec::with_capacity(opt.radius_steps);
    // curves[w] = Vec<(radius, efficiency)> for wavelengths[w]
    let mut curves: Vec<Vec<(f32, f32)>> =
        vec![Vec::with_capacity(opt.radius_steps); wavelengths.len()];

    for step in 0..opt.radius_steps {
        let r_lo = radius_cap * step as f32 / opt.radius_steps as f32;
        let r_hi = radius_cap * (step + 1) as f32 / opt.radius_steps as f32;
        let r_mid = 0.5 * (r_lo + r_hi);
        radii.push(r_mid);

        for (w, &lambda_nm) in wavelengths.iter().enumerate() {
            let mut successes = 0usize;
            for _ in 0..opt.samples {
                // random film point in this annulus (uniform in radius, any angle)
                let radius = r_lo + (r_hi - r_lo) * rand::random::<f32>();
                let theta = std::f32::consts::TAU * rand::random::<f32>();
                let point = Point3::new(radius * theta.cos(), radius * theta.sin(), -film_position);

                let direction = sampler.sample(
                    lambda_nm,
                    point,
                    Sample2D::new_random_sample(),
                    Sample1D::new_random_sample(),
                );
                let ray = Ray::new(point, direction);
                let result = assembly.trace_forward(
                    0.0,
                    Input::new(ray, lambda_nm / 1000.0),
                    opt.atmosphere_ior,
                    |e| (aperture.is_rejected(aperture_radius, e.origin), false),
                    drop,
                );
                if result.is_some() {
                    successes += 1;
                }
            }
            let efficiency = successes as f32 / opt.samples as f32;
            curves[w].push((r_mid, efficiency));
        }
    }

    // --- CSV ------------------------------------------------------------------
    let csv_path = format!("{}.csv", opt.out);
    {
        let mut csv = File::create(&csv_path)?;
        write!(csv, "radius")?;
        for wl in &wavelengths {
            write!(csv, ",eff_{}", wl)?;
        }
        writeln!(csv)?;
        for (i, &r) in radii.iter().enumerate() {
            write!(csv, "{}", r)?;
            for curve in &curves {
                write!(csv, ",{}", curve[i].1)?;
            }
            writeln!(csv)?;
        }
    }
    println!("wrote {}", csv_path);

    // --- PNG ------------------------------------------------------------------
    let png_path = format!("{}.png", opt.out);
    {
        let root = BitMapBackend::new(&png_path, (1000, 700)).into_drawing_area();
        root.fill(&WHITE)?;

        let title = format!(
            "RadialSampler efficiency — {} (aperture r={:.2}, {} blades)",
            opt.lens, aperture_radius, opt.blades
        );
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 22))
            .margin(15)
            .x_label_area_size(45)
            .y_label_area_size(55)
            .build_cartesian_2d(0f32..radius_cap, 0f32..1f32)?;

        chart
            .configure_mesh()
            .x_desc("distance from optical axis (mm)")
            .y_desc("efficiency (survivors / samples)")
            .draw()?;

        for (w, &lambda_nm) in wavelengths.iter().enumerate() {
            let color = Palette99::pick(w);
            let style = ShapeStyle {
                color: color.to_rgba(),
                filled: false,
                stroke_width: 2,
            };
            chart
                .draw_series(LineSeries::new(curves[w].iter().copied(), style.clone()))?
                .label(format!("{} nm", lambda_nm))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style.clone()));
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
    }
    println!("wrote {}", png_path);

    Ok(())
}
