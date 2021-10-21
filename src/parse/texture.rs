use crate::film::Film;
#[cfg(bin)]
use crate::parse::curves::{parse_curve, CurveData};
use crate::*;

use packed_simd::f32x4;
use serde::{Deserialize, Serialize};

use std::path::Path;

#[cfg(feature="parse")]
mod parse {
    #[derive(Serialize, Deserialize, Clone)]
    #[serde(tag = "type")]
    pub enum TextureData {
        Texture1 {
            curve: CurveData,
            filename: String,
        },
        Texture4 {
            curves: [CurveData; 4],
            filename: String,
        },
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct TextureStackData {
        pub name: String,
        pub texture_stack: Vec<TextureData>,
    }

    pub fn parse_rgba(filepath: &str) -> Film<f32x4> {
        println!("parsing rgba texture at {}", filepath);
        let path = Path::new(filepath);
        let img = image::open(path).unwrap();
        let rgba_image = img.into_rgba8();
        let (width, height) = rgba_image.dimensions();
        let mut new_film = Film::new(width as usize, height as usize, f32x4::splat(0.0));
        for (x, y, pixel) in rgba_image.enumerate_pixels() {
            let [r, g, b, a]: [u8; 4] = pixel.0.into();
            new_film.write_at(
                x as usize,
                y as usize,
                f32x4::new(
                    r as f32 / 256.0,
                    g as f32 / 256.0,
                    b as f32 / 256.0,
                    a as f32 / 256.0,
                ),
            );
        }
        new_film
    }

    pub fn parse_bitmap(filepath: &str) -> Film<f32> {
        println!("parsing greyscale texture at {}", filepath);
        let path = Path::new(filepath);
        let img = image::open(path).unwrap();
        let greyscale = img.into_luma8();
        let (width, height) = greyscale.dimensions();
        let mut new_film = Film::new(width as usize, height as usize, 0.0);
        for (x, y, pixel) in greyscale.enumerate_pixels() {
            let grey: [u8; 1] = pixel.0.into();
            new_film.write_at(x as usize, y as usize, grey[0] as f32 / 256.0);
        }
        new_film
    }

    pub fn select_channel(film: &Film<f32x4>, channel: usize) -> Film<f32> {
        assert!(channel < 4);

        Film {
            buffer: film.buffer.iter().map(|v| v.extract(channel)).collect(),
            width: film.width,
            height: film.height,
        }
    }

    fn convert_to_array<'a, 'b>(vec: Vec<CDF<'a, 'b>>) -> [CDF<'a, 'b>; 4] {
        let mut arr: [CDF<'a, 'b>; 4] = [
            CDF::default(),
            CDF::default(),
            CDF::default(),
            CDF::default(),
        ];
        arr[0] = vec[0].clone();
        arr[1] = vec[1].clone();
        arr[2] = vec[2].clone();
        arr[3] = vec[3].clone();
        arr
    }

    pub fn hsv_to_rgb(h: usize, s: f32, v: f32) -> (u8, u8, u8) {
        let c = v * s;
        let x = c * (1.0 - ((h as f32 / 60.0) % 2.0 - 1.0)).abs();
        let m = v - c;
        let convert = |(r, g, b), m| {
            (
                ((r + m) * 255.0) as u8,
                ((g + m) * 255.0) as u8,
                ((b + m) * 255.0) as u8,
            )
        };
        match h {
            0..=60 => convert((c, x, 0.0), m),
            61..=120 => convert((x, c, 0.0), m),
            121..=180 => convert((0.0, c, x), m),
            181..=240 => convert((0.0, x, c), m),
            241..=300 => convert((x, 0.0, c), m),
            301..=360 => convert((c, 0.0, x), m),
            _ => (0, 0, 0),
        }
    }

    pub fn triple_to_u32(triple: (u8, u8, u8)) -> u32 {
        let c = ((triple.0 as u32) << 16) + ((triple.1 as u32) << 8) + (triple.2 as u32);
        c
    }

    pub fn attempt_write(film: &mut Film<u32>, px: usize, py: usize, c: u32) {
        if py * film.width + px >= film.buffer.len() {
            return;
        }
        film.buffer[py * film.width + px] = c;
    }

    pub fn blit_circle(film: &mut Film<u32>, radius: f32, x: usize, y: usize, c: u32) {
        let approx_pixel_circumference = radius as f32 * std::f32::consts::TAU;

        let pixel_x_size = 1.0 / film.width as f32;
        let pixel_y_size = 1.0 / film.height as f32;
        for phi in 0..(approx_pixel_circumference as usize) {
            let (new_px, new_py) = (
                (x as f32 * pixel_x_size
                    + radius as f32
                        * pixel_x_size
                        * (phi as f32 * std::f32::consts::TAU / approx_pixel_circumference).cos())
                    / pixel_x_size,
                (y as f32 * pixel_y_size
                    + radius as f32
                        * pixel_y_size
                        * (phi as f32 * std::f32::consts::TAU / approx_pixel_circumference).sin())
                    / pixel_y_size,
            );
            attempt_write(film, new_px as usize, new_py as usize, c);
        }
    }

    #[derive(Clone)]
    pub struct Texture4<'a, 'b> {
        pub curves: [CDF<'a, 'b>; 4],
        pub texture: Film<f32x4>,
        pub interpolation_mode: InterpolationMode,
    }

    impl<'a, 'b> Texture4<'a, 'b> {
        // evaluate the 4 CDFs with the mixing ratios specified by the texture.
        // not clamped to 0 to 1, so that should be done by the callee
        pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
            // TODO: bilinear or bicubic texture interpolation/filtering
            let factors = self.texture.at_uv(uv);
            let eval = f32x4::new(
                self.curves[0].evaluate_power(lambda),
                self.curves[1].evaluate_power(lambda),
                self.curves[2].evaluate_power(lambda),
                self.curves[3].evaluate_power(lambda),
            );
            (factors * eval).sum()
        }
    }
    #[derive(Clone)]
    pub struct Texture1<'a, 'b> {
        pub curve: CDF<'a, 'b>,
        pub texture: Film<f32>,
        pub interpolation_mode: InterpolationMode,
    }

    impl<'a, 'b> Texture1<'a, 'b> {
        // evaluate the 4 CDFs with the mixing ratios specified by the texture.
        // not clamped to 0 to 1, so that should be done by the callee
        pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
            // TODO: bilinear or bicubic texture interpolation/filtering
            let factor = self.texture.at_uv(uv);
            let eval = self.curve.evaluate_power(lambda);
            factor * eval
        }
    }
    #[derive(Clone)]
    pub enum Texture<'a, 'b> {
        Texture1(Texture1<'a, 'b>),
        Texture4(Texture4<'a, 'b>),
    }

    impl<'a, 'b> Texture<'a, 'b> {
        pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
            // TODO: bilinear or bicubic texture interpolation/filtering
            match self {
                Texture::Texture1(tex) => tex.eval_at(lambda, uv),
                Texture::Texture4(tex) => tex.eval_at(lambda, uv),
            }
        }
    }

    #[derive(Clone)]
    pub struct TexStack<'a, 'b> {
        pub textures: Vec<Texture<'a, 'b>>,
    }
    impl<'a, 'b> TexStack<'a, 'b> {
        pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
            let mut s = 0.0;
            for tex in self.textures.iter() {
                s += tex.eval_at(lambda, uv);
            }
            s
        }
    }

    pub fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
        ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
    }

    pub fn parse_texture<'a, 'b>(texture: TextureData) -> Texture<'a, 'b> {
        match texture {
            TextureData::Texture1 { curve, filename } => {
                let cdf: CDF = parse_curve(curve).into();
                Texture::Texture1(Texture1 {
                    curve: cdf,
                    texture: parse_bitmap(&filename),
                    interpolation_mode: InterpolationMode::Nearest,
                })
            }
            TextureData::Texture4 { curves, filename } => {
                let cdfs: [CDF; 4] = convert_to_array(
                    curves
                        .iter()
                        .map(|curve| parse_curve(curve.clone()).into())
                        .collect(),
                );
                Texture::Texture4(Texture4 {
                    curves: cdfs,
                    texture: parse_rgba(&filename),
                    interpolation_mode: InterpolationMode::Nearest,
                })
            }
        }
    }

    pub fn parse_texture_stack<'a, 'b>(tex_stack: TextureStackData) -> TexStack<'a, 'b> {
        let mut textures = Vec::new();
        for v in tex_stack.texture_stack.iter() {
            textures.push(parse_texture(v.clone()));
        }
        TexStack { textures }
    }
}
