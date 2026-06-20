//! Polynomial optics (Hullin, Hanika, Heidrich — EGSR 2012).
//!
//! Replaces ray *tracing* through a [`LensAssembly`](crate::lens::LensAssembly)
//! with a single truncated multivariate **polynomial** that maps a reduced input
//! ray `r = [px, py, dx, dy]` to a reduced output ray. Each optical element
//! (spherical refraction, free-space propagation, reflection) is approximated by a
//! Taylor series in the four ray parameters; elements are composed by polynomial
//! substitution with truncation back to a fixed total degree `n`.
//!
//! Coefficients are produced by *truncated multivariate power-series automatic
//! differentiation*: [`TruncPoly`] supports `+ - * / sqrt`, the four reduced-ray
//! inputs are seeded as identity polynomials centered on the optical axis, and a
//! branch-free analytic surface map (see [`surfaces`]) is run over them — the
//! output polynomials are read straight off. No symbolic algebra package needed.
//!
//! This module is backend-agnostic; only the world↔reduced-ray endpoint
//! conversions and validation are generic over `S: SimdBackend`.

pub mod assembly;
pub mod flare;
pub mod surfaces;
pub mod system;
pub mod trunc_poly;

pub use assembly::{build_forward, build_reverse, PolyAssembly, PolyLens};
pub use flare::{build_ghost, enumerate_ghosts, render_ghost};
pub use surfaces::{propagation, reflection_spherical, refraction_spherical};
pub use system::PolySystem;
pub use trunc_poly::{Basis, TruncPoly, NUM_VARS};
