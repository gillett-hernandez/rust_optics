//! Reproduces the paper's Fig. 6: reconstruction error of the polynomial lens
//! model vs. an exact `trace_forward`, as a function of off-axis distance, one
//! curve per polynomial degree (1/3/5).
//!
//! We sweep the film point off-axis (the "world position" analog in our forward
//! setup), trace each ray both ways to the front vertex plane, and plot the
//! position error (log scale). Higher degrees track the true trace far better
//! near the axis and converge toward the linear model in the far periphery —
//! the shape of paper Fig. 6.
//!
//! Build with `RUSTFLAGS="-C target-cpu=native"` so the AVX2 paths are emitted:
//!   cargo run --features poly --example poly_error -- --lens data/cameras/petzval_kodak.txt

use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};

use plotters::prelude::*;
use structopt::StructOpt;

use optics::lens::{camera_space_to_plane, plane_to_camera_space};
use optics::math::*;
use optics::na::SVector;
use optics::poly::build_forward;
use optics::{LensAssembly, parse_lenses_from};

type Backend = thermite::backend::x86_v3::X86V3;
type Ray = optics::math::Ray<Backend>;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    /// Path to the lens spec file.
    #[structopt(long)]
    pub lens: String,

    /// Wavelength in micrometres (matches `Input::lambda`).
    #[structopt(long, default_value = "0.55")]
    pub lambda: f32,

    /// Polynomial degrees to compare (comma-separated).
    #[structopt(long, default_value = "1,3,5")]
    pub degrees: String,

    /// Maximum off-axis film distance to sweep, in mm.
    #[structopt(long, default_value = "20.0")]
    pub max_radius: f32,

    /// Number of sweep samples.
    #[structopt(long, default_value = "200")]
    pub steps: usize,

    /// Output file prefix; writes `<prefix>.png` and `<prefix>.csv`.
    #[structopt(long, default_value = "poly_error")]
    pub out: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    let mut spec = String::new();
    File::open(&opt.lens)?.read_to_string(&mut spec)?;
    let (interfaces, _, _) = parse_lenses_from(&spec);
    let assembly = LensAssembly::new(interfaces.as_slice());
    let film_z = -assembly.total_thickness_at(0.0);

    let degrees: Vec<usize> = opt
        .degrees
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    // Build one system per degree.
    let systems: Vec<(usize, _)> = degrees
        .iter()
        .map(|&d| {
            (
                d,
                build_forward(&assembly, 0.0, opt.lambda, 1.0, d)
                    .expect("lens has an unsupported (aspheric/cylindrical) surface"),
            )
        })
        .collect();

    // Exact reference projected to the front vertex plane z = 0.
    let trace_ref = |x: f32| -> Option<[f32; 4]> {
        let ray: Ray = plane_to_camera_space::<Backend>(PlaneRay([x, 0.0, 0.0, 0.0]), film_z);
        let out = assembly.trace_forward(
            0.0,
            Input::new(ray, opt.lambda),
            1.0,
            |_| (false, false),
            drop,
        )?;
        Some(camera_space_to_plane::<Backend>(out.ray, 0.0).0)
    };

    // rows: (radius, [pos_err per degree])
    let mut rows: Vec<(f32, Vec<f32>)> = Vec::with_capacity(opt.steps);
    for i in 0..opt.steps {
        let x = opt.max_radius * i as f32 / (opt.steps - 1).max(1) as f32;
        let Some(reference) = trace_ref(x) else {
            continue;
        };
        let reduced_in = camera_space_to_plane::<Backend>(
            plane_to_camera_space::<Backend>(PlaneRay([x, 0.0, 0.0, 0.0]), film_z),
            film_z,
        )
        .0;
        let errs: Vec<f32> = systems
            .iter()
            .map(|(_, sys)| {
                let p = sys.eval(SVector::from(reduced_in));
                ((p[0] - reference[0]).powi(2) + (p[1] - reference[1]).powi(2)).sqrt()
            })
            .collect();
        rows.push((x, errs));
    }

    // CSV
    let csv_path = format!("{}.csv", opt.out);
    let mut csv = File::create(&csv_path)?;
    write!(csv, "radius_mm")?;
    for (d, _) in &systems {
        write!(csv, ",pos_err_deg{}", d)?;
    }
    writeln!(csv)?;
    for (x, errs) in &rows {
        write!(csv, "{}", x)?;
        for e in errs {
            write!(csv, ",{}", e)?;
        }
        writeln!(csv)?;
    }
    println!("wrote {csv_path}");

    // PNG: log10(position error) vs radius, one curve per degree.
    let png_path = format!("{}.png", opt.out);
    let root = BitMapBackend::new(&png_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_lo = -8.0f32;
    let y_hi = 2.0f32;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Polynomial position error vs off-axis distance ({})", opt.lens),
            ("sans-serif", 22),
        )
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..opt.max_radius, y_lo..y_hi)?;
    chart
        .configure_mesh()
        .x_desc("film off-axis distance [mm]")
        .y_desc("log10 position error [mm]")
        .draw()?;

    let palette = [RED, GREEN, BLUE, MAGENTA, CYAN];
    for (k, (d, _)) in systems.iter().enumerate() {
        let color = palette[k % palette.len()];
        chart
            .draw_series(LineSeries::new(
                rows.iter()
                    .map(|(x, errs)| (*x, errs[k].max(1e-9).log10())),
                color.stroke_width(2),
            ))?
            .label(format!("degree {}", d))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("wrote {png_path}");

    Ok(())
}
