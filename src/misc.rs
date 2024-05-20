use crate::vec2d::Vec2D;
pub use crate::math::*;
#[cfg(feature = "dev")]
use line_drawing;

pub trait Cycle {
    fn cycle(self) -> Self;
}

#[derive(Copy, Clone, Debug)]
pub enum SceneMode {
    // diffuse emitter texture
    TexturedWall { distance: f32, texture_scale: f32 },
    // small diffuse lights
    PinLight,

    // spot light shining with a specific angle
    SpotLight { pos: Vec3, size: f32, span: f32 },
}

impl Cycle for SceneMode {
    fn cycle(self) -> Self {
        match self {
            SceneMode::TexturedWall { .. } => SceneMode::PinLight,
            SceneMode::PinLight => SceneMode::SpotLight {
                pos: Vec3::ZERO + 100.0 * Vec3::Z,
                size: 0.1,
                span: 0.99,
            },
            // defaults to 5000mm == 5meters away
            SceneMode::SpotLight { .. } => SceneMode::TexturedWall {
                distance: 5000.0,
                texture_scale: 1.0,
            },
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ViewMode {
    Film,
    SpotOnFilm(f32, f32),
    XRay { bounds: Bounds2D },
}
impl Cycle for ViewMode {
    fn cycle(self) -> Self {
        match self {
            ViewMode::Film => ViewMode::XRay {
                bounds: Bounds2D::new((-1.0, 1.0).into(), (-1.0, 1.0).into()),
            },
            ViewMode::SpotOnFilm(_, _) => ViewMode::Film,
            ViewMode::XRay { .. } => ViewMode::SpotOnFilm(0.0, 0.0),
        }
    }
}

pub fn project<F>(point: Point3, plane_normal: Vec3, swizzle: F) -> Point3
where
    F: Fn(f32x4) -> f32x4,
{
    let as_vec = point - Point3::ORIGIN;
    let normal_component = plane_normal * (as_vec * plane_normal);
    let projected = as_vec - normal_component;

    Point3::ORIGIN + Vec3(swizzle(projected.0))
}

#[derive(Copy, Clone)]
pub enum DrawMode {
    Midpoint,
    XiaolinWu,
    Bresenham,
}

#[cfg(feature = "dev")]
pub fn draw_line(
    film: &mut Vec2D<XYZColor>,
    clip_window: Bounds2D,
    pt0: Point3,
    pt1: Point3,
    lambda: f32,
    tau: f32,
    draw_mode: DrawMode,
) {
    let we = SingleWavelength::new(lambda, tau.into());
    let (film_width, film_height) = (film.width, film.height);

    let r = Ray::new(pt0, pt1 - pt0);
    let (mut min_t, mut max_t) = (f32::INFINITY, 0.0f32);
    match r.direction.x() {
        dx if dx > 0.0 => {
            max_t = max_t.max((clip_window.x.upper - r.origin.x()) / dx);
            min_t = min_t.min((clip_window.x.upper - r.origin.x()) / dx);
        }
        dx if dx < 0.0 => {
            max_t = max_t.max((clip_window.x.lower - r.origin.x()) / dx);
            min_t = min_t.min((clip_window.x.lower - r.origin.x()) / dx);
        }
        _ => {
            // no left or right movement, up or down will be computed later
        }
    }
    match r.direction.y() {
        dy if dy > 0.0 => {
            max_t = max_t.max((clip_window.y.upper - r.origin.y()) / dy);
            min_t = min_t.min((clip_window.y.upper - r.origin.y()) / dy);
        }
        dy if dy < 0.0 => {
            max_t = max_t.max((clip_window.y.lower - r.origin.y()) / dy);
            min_t = min_t.min((clip_window.y.lower - r.origin.y()) / dy);
        }
        _ => {
            // no up or down movement.

            // left or right clip bounds should have been computed in other match statement.
            // unless dx and dy are both 0, in which case
            return;
        }
    }
    let clipped0 = if clip_window.x.contains(&pt0.x()) && clip_window.y.contains(&pt0.y()) {
        pt0
    } else {
        r.point_at_parameter(max_t)
    };
    let clipped1 = if clip_window.x.contains(&pt1.x()) && clip_window.y.contains(&pt1.y()) {
        pt1
    } else {
        r.point_at_parameter(min_t)
    };

    let (px0, py0) = (
        (film_width as f32 * (clipped0.x() - clip_window.x.lower) / clip_window.x.span()) as usize,
        (film_height as f32 * (1.0 - (clipped0.y() - clip_window.y.lower) / clip_window.y.span()))
            as usize,
    );
    let (px1, py1) = (
        (film_width as f32 * (clipped1.x() - clip_window.x.lower) / clip_window.x.span()) as usize,
        (film_height as f32 * (1.0 - (clipped1.y() - clip_window.y.lower) / clip_window.y.span()))
            as usize,
    );

    let (dx, dy) = (px1 as isize - px0 as isize, py1 as isize - py0 as isize);
    if dx == 0 && dy == 0 {
        if px0 as usize >= film_width || py0 as usize >= film_height {
            return;
        }
        film.buffer[py0 as usize * film_width + px0 as usize] += XYZColor::from(we);
        return;
    }
    let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
    match draw_mode {
        DrawMode::Midpoint => {
            for (x, y) in line_drawing::Midpoint::<f32, isize>::new(
                (px0 as f32, py0 as f32),
                (px1 as f32, py1 as f32),
            ) {
                if x as usize >= film_width || y as usize >= film_height || x < 0 || y < 0 {
                    continue;
                }
                assert!(!b.is_nan(), "{} {}", dx, dy);
                film.buffer[y as usize * film_width + x as usize] +=
                    XYZColor::from(we.replace_energy((we.energy * b).into()));
            }
        }
        DrawMode::XiaolinWu => {
            // let b = 1.0f32;
            for ((x, y), a) in line_drawing::XiaolinWu::<f32, isize>::new(
                (px0 as f32, py0 as f32),
                (px1 as f32, py1 as f32),
            ) {
                if x as usize >= film_width || y as usize >= film_height || x < 0 || y < 0 {
                    continue;
                }
                assert!(!b.is_nan(), "{} {}", dx, dy);
                film.buffer[y as usize * film_width + x as usize] +=
                    XYZColor::from(we.replace_energy((we.energy * a * b).into()));
            }
        }
        DrawMode::Bresenham => {
            for (x, y) in line_drawing::Bresenham::new(
                (px0 as isize, py0 as isize),
                (px1 as isize, py1 as isize),
            ) {
                if x as usize >= film_width || y as usize >= film_height || x < 0 || y < 0 {
                    continue;
                }
                assert!(!b.is_nan(), "{} {}", dx, dy);
                film.buffer[y as usize * film_width + x as usize] +=
                    XYZColor::from(we.replace_energy((we.energy * b).into()));
            }
        }
    }
}
