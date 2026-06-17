pub use crate::math::*;
#[cfg(feature = "dev")]
use crate::vec2d::Vec2D;
#[cfg(feature = "dev")]
use line_drawing;

pub trait Cycle {
    fn cycle(self) -> Self;
}

#[derive(Copy, Clone, Debug)]
pub enum SceneMode<S: SimdBackend> {
    // diffuse emitter texture
    TexturedWall { distance: f32, texture_scale: f32 },
    // small diffuse lights
    PinLight,

    // spot light shining with a specific angle
    SpotLight { pos: Vec3<S>, size: f32, span: f32 },
}

impl<S: SimdBackend> Cycle for SceneMode<S> {
    fn cycle(self) -> Self {
        match self {
            SceneMode::TexturedWall { .. } => SceneMode::PinLight,
            SceneMode::PinLight => SceneMode::SpotLight {
                pos: Vec3::ZERO + 100.0 * Vec3::z_axis(),
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

#[inline(always)]
pub fn project<S: SimdBackend, F>(point: Point3<S>, plane_normal: Vec3<S>, swizzle: F) -> Point3<S>
where
    F: Fn(F32x4<S>) -> F32x4<S>,
{
    let as_vec = point - Point3::origin();
    let normal_component = plane_normal * (as_vec * plane_normal);
    let projected = as_vec - normal_component;

    Point3::origin() + Vec3(swizzle(projected.0))
}

#[derive(Copy, Clone)]
pub enum DrawMode {
    Midpoint,
    XiaolinWu,
    Bresenham,
}

#[cfg(feature = "dev")]
pub fn draw_line<S: SimdBackend>(
    film: &mut Vec2D<XYZColor<S>>,
    clip_window: Bounds2D,
    pt0: Point3<S>,
    pt1: Point3<S>,
    lambda: f32,
    tau: f32,
    draw_mode: DrawMode,
) {
    let we = SingleWavelength::new(lambda, tau.into());
    let (film_width, film_height) = (film.width, film.height);

    // Liang–Barsky: clip the segment pt0 -> pt1 to `clip_window` parametrically.
    // We accumulate the entering/leaving parameter interval [t0, t1] within [0, 1]
    // over the four window edges. If the interval collapses (t0 > t1) the segment
    // lies entirely outside the window and we draw nothing. Clamping the endpoints
    // to the window border (rather than leaving them off-screen) is what keeps the
    // later `as usize` pixel casts from saturating negatives to 0 and snapping
    // lines to the top-left / bottom-left corner.
    let delta = pt1 - pt0;
    let (dx, dy) = (delta.x(), delta.y());
    // For each edge: p is the (signed) rate toward the boundary, q the slack at pt0.
    // p < 0 => entering edge (raises t0); p > 0 => leaving edge (lowers t1).
    let p = [-dx, dx, -dy, dy];
    let q = [
        pt0.x() - clip_window.x.lower,
        clip_window.x.upper - pt0.x(),
        pt0.y() - clip_window.y.lower,
        clip_window.y.upper - pt0.y(),
    ];
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for i in 0..4 {
        if p[i] == 0.0 {
            // parallel to this edge; if it starts outside the slab, reject entirely
            // (this also handles the degenerate dx == dy == 0 point case).
            if q[i] < 0.0 {
                return;
            }
        } else {
            let t = q[i] / p[i];
            if p[i] < 0.0 {
                if t > t1 {
                    return;
                }
                if t > t0 {
                    t0 = t;
                }
            } else {
                if t < t0 {
                    return;
                }
                if t < t1 {
                    t1 = t;
                }
            }
        }
    }
    if t0 > t1 {
        return;
    }
    let clipped0 = pt0 + delta * t0;
    let clipped1 = pt0 + delta * t1;

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
