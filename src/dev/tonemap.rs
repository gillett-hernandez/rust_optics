#![allow(unused, unused_imports)]

use crate::vec2d::Vec2D;
use crate::math::*;

// extern crate exr;
// use exr::prelude::rgba_image::*;
use nalgebra::{Matrix3, Vector3};

use std::time::Instant;

pub trait Tonemapper {
    fn map(&self, film: &Vec2D<XYZColor>, pixel: (usize, usize)) -> (f32x4, f32x4);
    // fn write_to_files(&self, film: &Film<XYZColor>, exr_filename: &str, png_filename: &str);
}

#[allow(non_camel_case_types)]
pub struct sRGB {
    pub factor: f32,
    pub exposure_adjustment: f32,
    // pub gamma_adjustment: f32,
}

impl sRGB {
    pub fn new(film: &Vec2D<XYZColor>, exposure_adjustment: f32) -> Self {
        let mut max_luminance = 0.0;
        let mut total_luminance = 0.0;
        for y in 0..film.height {
            for x in 0..film.width {
                let color = film.at(x, y);
                let lum = color.y();
                debug_assert!(!lum.is_nan(), "nan {:?} at ({},{})", color, x, y);
                if lum.is_nan() {
                    continue;
                }
                total_luminance += lum;
                if lum > max_luminance {
                    // println!("max lum {} at ({}, {})", max_luminance, x, y);
                    max_luminance = lum;
                }
            }
        }
        let avg_luminance = total_luminance / film.total_pixels() as f32;
        // println!(
        //     "computed tonemapping: max luminance {}, avg luminance {}, exposure is {}",
        //     max_luminance,
        //     avg_luminance,
        //     exposure_adjustment / max_luminance
        // );
        sRGB {
            factor: (1.0 / max_luminance).min(1000000.0),
            exposure_adjustment,
            // gamma_adjustment,
        }
    }
}

impl Tonemapper for sRGB {
    fn map(&self, film: &Vec2D<XYZColor>, pixel: (usize, usize)) -> (f32x4, f32x4) {
        let cie_xyz_color = film.at(pixel.0, pixel.1);
        let mut scaled_cie_xyz_color = cie_xyz_color * self.factor * self.exposure_adjustment;
        if !scaled_cie_xyz_color.0.is_finite().all() {
            scaled_cie_xyz_color = XYZColor::BLACK;
        }

        let xyz_to_rgb: Matrix3<f32> = Matrix3::new(
            3.24096994,
            -1.53738318,
            -0.49861076,
            -0.96924364,
            1.8759675,
            0.04155506,
            0.05563008,
            -0.20397696,
            1.05697151,
        );
        let [x, y, z, _]: [f32; 4] = scaled_cie_xyz_color.0.into();
        let intermediate = xyz_to_rgb * Vector3::new(x, y, z);

        let rgb_linear =
            f32x4::from_array([intermediate[0], intermediate[1], intermediate[2], 0.0]);
        let S313: f32x4 = f32x4::splat(0.0031308);
        let S323_25: f32x4 = f32x4::splat(323.0 / 25.0);
        let S5_12: f32x4 = f32x4::splat(5.0 / 12.0);
        let S211: f32x4 = f32x4::splat(211.0);
        let S11: f32x4 = f32x4::splat(11.0);
        let S200: f32x4 = f32x4::splat(200.0);
        let srgb = (rgb_linear.simd_lt(S313)).select(
            S323_25 * rgb_linear,
            (S211 * rgb_linear.powf(S5_12) - S11) / S200,
        );
        (srgb, rgb_linear / f32x4::splat(self.factor))
    }
    // fn write_to_files(&self, film: &Film<XYZColor>, exr_filename: &str, png_filename: &str) {
    //     let now = Instant::now();
    //     // generate a color for each pixel position
    //     let generate_pixels = |position: Vec2<usize>| {
    //         let (_mapped, linear) = self.map(&film, (position.x() as usize, position.y() as usize));
    //         let [r, g, b, _]: [f32; 4] = linear.into();
    //         Pixel::rgb(r, g, b)
    //     };

    //     let image_info = ImageInfo::rgb(
    //         (film.width, film.height), // pixel resolution
    //         SampleType::F16,           // convert the generated f32 values to f16 while writing
    //     );

    //     image_info
    //         .write_pixels_to_file(
    //             exr_filename,
    //             write_options::high(), // higher speed, but higher memory usage
    //             &generate_pixels,      // pass our pixel generator
    //         )
    //         .unwrap();

    //     let mut img: image::RgbImage =
    //         image::ImageBuffer::new(film.width as u32, film.height as u32);

    //     for (x, y, pixel) in img.enumerate_pixels_mut() {
    //         //apply tonemap here

    //         let (mapped, _linear) = self.map(&film, (x as usize, y as usize));

    //         let [r, g, b, _]: [f32; 4] = mapped.into();

    //         *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
    //     }
    //     println!("saving image...");
    //     img.save(png_filename).unwrap();

    //     println!(
    //         "took {}s to tonemap and output\n",
    //         (now.elapsed().as_millis() as f32) / 1000.0
    //     );
    // }
}
