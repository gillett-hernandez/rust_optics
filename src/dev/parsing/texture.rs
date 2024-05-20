use crate::math::*;

use serde::{Deserialize, Serialize};

use std::path::Path;

use super::curves::{parse_curve, CurveData};
use crate::vec2d::Vec2D;

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

pub fn parse_rgba(filepath: &str) -> Vec2D<f32x4> {
    println!("parsing rgba texture at {}", filepath);
    let path = Path::new(filepath);
    let img = image::open(path).unwrap();
    let rgba_image = img.into_rgba8();
    let (width, height) = rgba_image.dimensions();
    let mut new_film = Vec2D::new(width as usize, height as usize, f32x4::splat(0.0));
    for (x, y, pixel) in rgba_image.enumerate_pixels() {
        let [r, g, b, a]: [u8; 4] = pixel.0.into();
        new_film.write_at(
            x as usize,
            y as usize,
            f32x4::from_array([
                r as f32 / 256.0,
                g as f32 / 256.0,
                b as f32 / 256.0,
                a as f32 / 256.0,
            ]),
        );
    }
    new_film
}

pub fn parse_bitmap(filepath: &str) -> Vec2D<f32> {
    println!("parsing greyscale texture at {}", filepath);
    let path = Path::new(filepath);
    let img = image::open(path).unwrap();
    let greyscale = img.into_luma8();
    let (width, height) = greyscale.dimensions();
    let mut new_film = Vec2D::new(width as usize, height as usize, 0.0);
    for (x, y, pixel) in greyscale.enumerate_pixels() {
        let grey: [u8; 1] = pixel.0.into();
        new_film.write_at(x as usize, y as usize, grey[0] as f32 / 256.0);
    }
    new_film
}

pub fn select_channel(film: &Vec2D<f32x4>, channel: usize) -> Vec2D<f32> {
    assert!(channel < 4);

    Vec2D {
        buffer: film.buffer.iter().map(|v| v[channel]).collect(),
        width: film.width,
        height: film.height,
    }
}

fn convert_to_array(vec: Vec<CurveWithCDF>) -> [CurveWithCDF; 4] {
    let mut arr: [CurveWithCDF; 4] = [
        CurveWithCDF::default(),
        CurveWithCDF::default(),
        CurveWithCDF::default(),
        CurveWithCDF::default(),
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

pub fn attempt_write(film: &mut Vec2D<u32>, px: usize, py: usize, c: u32) {
    if py * film.width + px >= film.buffer.len() {
        return;
    }
    film.buffer[py * film.width + px] = c;
}

pub fn blit_circle(film: &mut Vec2D<u32>, radius: f32, x: usize, y: usize, c: u32) {
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
pub struct Texture4 {
    pub curves: [CurveWithCDF; 4],
    pub texture: Vec2D<f32x4>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture4 {
    // evaluate the 4 CurveWithCDFs with the mixing ratios specified by the texture.
    // not clamped to 0 to 1, so that should be done by the callee
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factors = self.texture.at_uv(uv);
        let eval = f32x4::from_array([
            self.curves[0].evaluate_power(lambda),
            self.curves[1].evaluate_power(lambda),
            self.curves[2].evaluate_power(lambda),
            self.curves[3].evaluate_power(lambda),
        ]);
        (factors * eval).reduce_sum()
    }
}
#[derive(Clone)]
pub struct Texture1 {
    pub curve: CurveWithCDF,
    pub texture: Vec2D<f32>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture1 {
    // evaluate the 4 CurveWithCDFs with the mixing ratios specified by the texture.
    // not clamped to 0 to 1, so that should be done by the callee
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factor = self.texture.at_uv(uv);
        let eval = self.curve.evaluate_power(lambda);
        factor * eval
    }
}
#[derive(Clone)]
pub enum Texture {
    Texture1(Texture1),
    Texture4(Texture4),
}

impl Texture {
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        match self {
            Texture::Texture1(tex) => tex.eval_at(lambda, uv),
            Texture::Texture4(tex) => tex.eval_at(lambda, uv),
        }
    }
}

#[derive(Clone)]
pub struct TexStack {
    pub textures: Vec<Texture>,
}
impl TexStack {
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

pub fn parse_texture(texture: TextureData, bounds: Bounds1D) -> Texture {
    match texture {
        TextureData::Texture1 { curve, filename } => {
            let cdf: CurveWithCDF = parse_curve(curve).to_cdf(bounds, 50);
            Texture::Texture1(Texture1 {
                curve: cdf,
                texture: parse_bitmap(&filename),
                interpolation_mode: InterpolationMode::Nearest,
            })
        }
        TextureData::Texture4 { curves, filename } => {
            let cdfs: [CurveWithCDF; 4] = convert_to_array(
                curves
                    .iter()
                    .map(|curve| parse_curve(curve.clone()).to_cdf(bounds, 50))
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

pub fn parse_texture_stack(tex_stack: TextureStackData, bounds: Bounds1D) -> TexStack {
    let mut textures = Vec::new();
    for v in tex_stack.texture_stack.iter() {
        textures.push(parse_texture(v.clone(), bounds));
    }
    TexStack { textures }
}
