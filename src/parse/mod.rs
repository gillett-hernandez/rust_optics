pub mod curves;
pub mod texture;

pub use texture::*;

use serde::{Deserialize, Serialize};
use toml;

use std::fs::File;
use std::io::Read;

pub type Vec3Data = [f32; 3];
pub type Point3Data = [f32; 3];

#[derive(Serialize, Deserialize, Clone)]
pub struct Scene {
    pub textures: Vec<TextureStackData>,
}

pub fn get_scene(filepath: &str) -> Result<Scene, toml::de::Error> {
    // will return None in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    File::open(filepath)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();
    // uncomment the following line to print out the raw contents
    // println!("{:?}", input);
    let scene: Scene = toml::from_str(&input)?;
    // for render_settings in scene.render_settings.iter_mut() {
    //     render_settings.threads = match render_settings.threads {
    //         Some(expr) => Some(expr),
    //         None => Some(num_cpus as u16),
    //     };
    // }
    return Ok(scene);
}
