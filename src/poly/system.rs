//! A `PolySystem` is a map `V → V` (`R⁴ → R⁴`) given by four [`TruncPoly`]s
//! sharing one [`Basis`] — the reduced-ray transfer of one element or a whole
//! assembly. Systems compose by polynomial substitution with truncation (paper
//! Eq. 9–10) and expose the Jacobian / degree-1 "system matrix".

use std::sync::Arc;

use nalgebra::{SMatrix, SVector};

use super::trunc_poly::{Basis, TruncPoly, NUM_VARS};

/// Four polynomials mapping `[px, py, dx, dy] → [px', py', dx', dy']`.
#[derive(Clone, Debug)]
pub struct PolySystem {
    pub polys: [TruncPoly; NUM_VARS],
}

impl PolySystem {
    pub fn new(polys: [TruncPoly; NUM_VARS]) -> Self {
        PolySystem { polys }
    }

    pub fn basis(&self) -> &Arc<Basis> {
        &self.polys[0].basis
    }

    /// The identity map (each output equals its input).
    pub fn identity(basis: Arc<Basis>) -> Self {
        PolySystem {
            polys: std::array::from_fn(|v| TruncPoly::var(basis.clone(), v)),
        }
    }

    /// Evaluate the system at a concrete reduced ray.
    pub fn eval(&self, r: SVector<f32, NUM_VARS>) -> SVector<f32, NUM_VARS> {
        let point: [f32; NUM_VARS] = [r[0], r[1], r[2], r[3]];
        SVector::from_fn(|i, _| self.polys[i].eval(&point))
    }

    /// `self ∘ inner`: substitute `inner`'s four polynomials into `self`'s
    /// variables, truncating to the shared degree. `inner` is applied first, so
    /// for optical elements built rear→front this is `outer.compose(inner)` where
    /// `inner` is nearer the input ray.
    pub fn compose(&self, inner: &PolySystem) -> PolySystem {
        debug_assert!(Arc::ptr_eq(self.basis(), inner.basis()));
        let polys = std::array::from_fn(|i| substitute(&self.polys[i], &inner.polys));
        PolySystem { polys }
    }

    /// Jacobian `∂output_i/∂input_j` at a concrete reduced ray.
    pub fn jacobian(&self, r: SVector<f32, NUM_VARS>) -> SMatrix<f32, NUM_VARS, NUM_VARS> {
        let point: [f32; NUM_VARS] = [r[0], r[1], r[2], r[3]];
        SMatrix::from_fn(|i, j| self.polys[i].partial(j).eval(&point))
    }

    /// The degree-1 coefficient matrix — the ABCD/system matrix of matrix optics,
    /// recovered as the linear part of the polynomial model.
    pub fn linear_part(&self) -> SMatrix<f32, NUM_VARS, NUM_VARS> {
        SMatrix::from_fn(|i, j| {
            let mut exp = [0u8; NUM_VARS];
            exp[j] = 1;
            self.polys[i].coeff_of(exp)
        })
    }

    /// The constant (degree-0) offset of each output component — zero for a
    /// system centered on the optical axis, but nonzero after recentering.
    pub fn constant_part(&self) -> SVector<f32, NUM_VARS> {
        SVector::from_fn(|i, _| self.polys[i].constant_term())
    }
}

/// Substitute polynomials `inputs` into the variables of `poly` (compose),
/// truncating to the shared basis degree.
fn substitute(poly: &TruncPoly, inputs: &[TruncPoly; NUM_VARS]) -> TruncPoly {
    let basis = poly.basis.clone();
    let n = basis.degree;

    // Precompute powers inputs[v]^e for e in 0..=n.
    let pows: [Vec<TruncPoly>; NUM_VARS] = std::array::from_fn(|v| {
        let mut acc = Vec::with_capacity(n + 1);
        acc.push(TruncPoly::constant(basis.clone(), 1.0));
        for e in 1..=n {
            let prev = &acc[e - 1];
            acc.push(prev.mul(&inputs[v]));
        }
        acc
    });

    let mut result = TruncPoly::zero(basis.clone());
    for (i, &c) in poly.coeffs.iter().enumerate() {
        if c == 0.0 {
            continue;
        }
        let e = basis.exps()[i];
        let mut term = TruncPoly::constant(basis.clone(), c);
        for v in 0..NUM_VARS {
            let p = e[v] as usize;
            if p > 0 {
                term = term.mul(&pows[v][p]);
            }
        }
        result = result.add(&term);
    }
    result
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compose_with_identity_is_noop() {
        let basis = Basis::cached(3);
        // A nontrivial system: px' = px + dx, dx' = 2*dx, etc.
        let p = |v: usize| TruncPoly::var(basis.clone(), v);
        let sys = PolySystem::new([
            p(0).add(&p(2)),
            p(1).add(&p(3)),
            p(2).scale(2.0),
            p(3).scale(2.0),
        ]);
        let id = PolySystem::identity(basis.clone());
        let left = sys.compose(&id);
        let right = id.compose(&sys);
        let pt = SVector::from([0.1, -0.2, 0.05, 0.03]);
        let a = sys.eval(pt);
        assert!((left.eval(pt) - a).norm() < 1e-6);
        assert!((right.eval(pt) - a).norm() < 1e-6);
    }

    #[test]
    fn linear_part_matches_jacobian_at_origin() {
        let basis = Basis::cached(3);
        let p = |v: usize| TruncPoly::var(basis.clone(), v);
        // include a quadratic term to ensure it does NOT leak into the linear part.
        let sys = PolySystem::new([
            p(0).add(&p(2)).add(&p(0).mul(&p(0))),
            p(1),
            p(2),
            p(3),
        ]);
        let lin = sys.linear_part();
        let jac = sys.jacobian(SVector::zeros());
        assert!((lin - jac).norm() < 1e-6);
    }
}
