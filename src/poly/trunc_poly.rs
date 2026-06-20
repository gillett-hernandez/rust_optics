//! Truncated multivariate polynomial over [`NUM_VARS`] variables, total degree
//! `≤ n`, with the arithmetic and elementary functions needed to run a lens
//! surface map symbolically (power-series automatic differentiation).
//!
//! The variables are the four reduced-ray components `[px, py, dx, dy]`. A
//! [`Basis`] fixes the monomial ordering and a precomputed product table for a
//! given degree; all polynomials in one computation share a `Basis` via `Arc`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Number of variables: the reduced ray `[px, py, dx, dy]`.
pub const NUM_VARS: usize = 4;

/// Per-degree monomial basis: ordering, exponent table, and a product table for
/// truncated multiplication. Build once and share via [`Basis::cached`].
#[derive(Debug)]
pub struct Basis {
    pub degree: usize,
    /// `exps[i]` = exponent multi-index of monomial `i`. Index 0 is the constant
    /// `(0,0,0,0)`; monomials are graded (ascending total degree).
    exps: Vec<[u8; NUM_VARS]>,
    /// Index of the degree-1 monomial for each variable (the identity seeds).
    var_index: [usize; NUM_VARS],
    /// Flat `M*M` product table: `prod[i*M + j]` is the index of monomial `i*j`,
    /// or `-1` if its total degree exceeds `degree` (truncated away).
    prod: Vec<i32>,
    m: usize,
}

impl Basis {
    /// Number of monomials (terms), `C(degree + NUM_VARS, NUM_VARS)`.
    pub fn num_terms(&self) -> usize {
        self.m
    }

    pub fn exps(&self) -> &[[u8; NUM_VARS]] {
        &self.exps
    }

    fn build(degree: usize) -> Self {
        // Enumerate all multi-indices with total degree <= degree, graded so that
        // index 0 is the constant term (required by the series ops below).
        let mut exps: Vec<[u8; NUM_VARS]> = Vec::new();
        for d in 0..=degree {
            for a in 0..=d {
                for b in 0..=(d - a) {
                    for c in 0..=(d - a - b) {
                        let e = d - a - b - c;
                        exps.push([a as u8, b as u8, c as u8, e as u8]);
                    }
                }
            }
        }
        let m = exps.len();

        let mut index: HashMap<[u8; NUM_VARS], usize> = HashMap::with_capacity(m);
        for (i, e) in exps.iter().enumerate() {
            index.insert(*e, i);
        }

        let mut var_index = [0usize; NUM_VARS];
        for (v, slot) in var_index.iter_mut().enumerate() {
            let mut e = [0u8; NUM_VARS];
            e[v] = 1;
            *slot = index[&e];
        }

        let mut prod = vec![-1i32; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut e = exps[i];
                let mut ok = true;
                for v in 0..NUM_VARS {
                    let s = e[v] as usize + exps[j][v] as usize;
                    if s > u8::MAX as usize {
                        ok = false;
                        break;
                    }
                    e[v] = s as u8;
                }
                if ok {
                    let td: usize = e.iter().map(|&x| x as usize).sum();
                    if td <= degree {
                        prod[i * m + j] = index[&e] as i32;
                    }
                }
            }
        }

        Basis {
            degree,
            exps,
            var_index,
            prod,
            m,
        }
    }

    /// Returns a process-wide cached `Basis` for the given degree, building it on
    /// first use. Sharing avoids rebuilding the product table per polynomial.
    pub fn cached(degree: usize) -> Arc<Basis> {
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Basis>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().unwrap();
        guard
            .entry(degree)
            .or_insert_with(|| Arc::new(Basis::build(degree)))
            .clone()
    }
}

/// A truncated multivariate polynomial. Coefficients are dense over its `Basis`.
#[derive(Clone, Debug)]
pub struct TruncPoly {
    pub basis: Arc<Basis>,
    /// One coefficient per monomial, indexed as in `basis.exps`.
    pub coeffs: Vec<f32>,
}

impl TruncPoly {
    /// The zero polynomial.
    pub fn zero(basis: Arc<Basis>) -> Self {
        let m = basis.num_terms();
        TruncPoly {
            basis,
            coeffs: vec![0.0; m],
        }
    }

    /// A constant.
    pub fn constant(basis: Arc<Basis>, c: f32) -> Self {
        let mut p = Self::zero(basis);
        p.coeffs[0] = c;
        p
    }

    /// The identity seed for variable `v` (`0=px, 1=py, 2=dx, 3=dy`): the
    /// polynomial whose value is exactly that input component.
    pub fn var(basis: Arc<Basis>, v: usize) -> Self {
        let idx = basis.var_index[v];
        let mut p = Self::zero(basis);
        p.coeffs[idx] = 1.0;
        p
    }

    pub fn degree(&self) -> usize {
        self.basis.degree
    }

    /// The constant term (value at the expansion center `r = 0`).
    pub fn constant_term(&self) -> f32 {
        self.coeffs[0]
    }

    fn same_basis(&self, other: &TruncPoly) -> bool {
        Arc::ptr_eq(&self.basis, &other.basis)
    }

    pub fn add(&self, other: &TruncPoly) -> TruncPoly {
        debug_assert!(self.same_basis(other));
        let coeffs = self
            .coeffs
            .iter()
            .zip(&other.coeffs)
            .map(|(a, b)| a + b)
            .collect();
        TruncPoly {
            basis: self.basis.clone(),
            coeffs,
        }
    }

    pub fn sub(&self, other: &TruncPoly) -> TruncPoly {
        debug_assert!(self.same_basis(other));
        let coeffs = self
            .coeffs
            .iter()
            .zip(&other.coeffs)
            .map(|(a, b)| a - b)
            .collect();
        TruncPoly {
            basis: self.basis.clone(),
            coeffs,
        }
    }

    pub fn scale(&self, s: f32) -> TruncPoly {
        TruncPoly {
            basis: self.basis.clone(),
            coeffs: self.coeffs.iter().map(|c| c * s).collect(),
        }
    }

    /// Add a constant in place-free fashion.
    pub fn add_constant(&self, c: f32) -> TruncPoly {
        let mut out = self.clone();
        out.coeffs[0] += c;
        out
    }

    pub fn mul(&self, other: &TruncPoly) -> TruncPoly {
        debug_assert!(self.same_basis(other));
        let m = self.basis.num_terms();
        let prod = &self.basis.prod;
        let mut coeffs = vec![0.0f32; m];
        for i in 0..m {
            let a = self.coeffs[i];
            if a == 0.0 {
                continue;
            }
            let row = i * m;
            for j in 0..m {
                let b = other.coeffs[j];
                if b == 0.0 {
                    continue;
                }
                let k = prod[row + j];
                if k >= 0 {
                    coeffs[k as usize] += a * b;
                }
            }
        }
        TruncPoly {
            basis: self.basis.clone(),
            coeffs,
        }
    }

    /// Compose this polynomial `p` with a scalar function `f` whose Taylor
    /// coefficients about `p`'s constant term `c0` are `series[k] = f^{(k)}(c0)/k!`.
    /// Computes `Σ_k series[k] · h^k` where `h = p - c0`. Because `h` has no
    /// constant term, `h^k` has minimum total degree `k`, so the sum terminates at
    /// `k = degree` with no truncation loss beyond the model's own truncation.
    fn compose_scalar_series(&self, series: &[f32]) -> TruncPoly {
        let n = self.basis.degree;
        debug_assert!(series.len() >= n + 1);
        // h = p with the constant term removed.
        let mut h = self.clone();
        h.coeffs[0] = 0.0;

        let mut result = TruncPoly::constant(self.basis.clone(), series[0]);
        // hpow starts at h^1.
        let mut hpow = h.clone();
        for &s in series.iter().take(n + 1).skip(1) {
            if s != 0.0 {
                result = result.add(&hpow.scale(s));
            }
            hpow = hpow.mul(&h);
        }
        result
    }

    /// Reciprocal `1/p` as a truncated series. Requires `p`'s constant term ≠ 0
    /// (true at the optical axis, where every reduced-ray denominator is nonzero).
    pub fn recip(&self) -> TruncPoly {
        let c0 = self.constant_term();
        debug_assert!(c0 != 0.0, "recip of polynomial with zero constant term");
        let n = self.basis.degree;
        // f(x)=1/x: f^{(k)}(c0)/k! = (-1)^k / c0^{k+1}.
        let mut series = vec![0.0f32; n + 1];
        let mut inv = 1.0 / c0;
        for (k, s) in series.iter_mut().enumerate() {
            *s = if k % 2 == 0 { inv } else { -inv };
            inv /= c0;
        }
        self.compose_scalar_series(&series)
    }

    /// Square root `sqrt(p)` as a truncated series. Requires `p`'s constant term
    /// > 0 (the axial value of every radicand in the surface maps).
    pub fn sqrt(&self) -> TruncPoly {
        let c0 = self.constant_term();
        debug_assert!(c0 > 0.0, "sqrt of polynomial with non-positive constant term");
        let n = self.basis.degree;
        // f(x)=x^{1/2}: f^{(k)}(c0)/k! = binom(1/2, k) * c0^{1/2 - k}.
        let mut series = vec![0.0f32; n + 1];
        let sqrt_c0 = c0.sqrt();
        for (k, s) in series.iter_mut().enumerate() {
            let binom = generalized_binomial(0.5, k);
            // c0^{0.5 - k} = sqrt(c0) / c0^k.
            *s = binom * sqrt_c0 / c0.powi(k as i32);
        }
        self.compose_scalar_series(&series)
    }

    pub fn div(&self, other: &TruncPoly) -> TruncPoly {
        self.mul(&other.recip())
    }

    /// Evaluate the polynomial at a concrete reduced ray.
    pub fn eval(&self, point: &[f32; NUM_VARS]) -> f32 {
        let mut acc = 0.0f32;
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let e = &self.basis.exps[i];
            let mut term = c;
            for v in 0..NUM_VARS {
                let p = e[v];
                for _ in 0..p {
                    term *= point[v];
                }
            }
            acc += term;
        }
        acc
    }

    /// Partial derivative with respect to variable `v`, as a polynomial in the
    /// same basis (degree drops by one, so it always fits).
    pub fn partial(&self, v: usize) -> TruncPoly {
        let mut out = TruncPoly::zero(self.basis.clone());
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let e = self.basis.exps[i];
            if e[v] == 0 {
                continue;
            }
            let mut de = e;
            de[v] -= 1;
            let coeff = c * e[v] as f32;
            // Find the index of the lowered exponent.
            let idx = self
                .basis
                .exps
                .iter()
                .position(|x| *x == de)
                .expect("lowered monomial must exist in basis");
            out.coeffs[idx] += coeff;
        }
        out
    }

    /// Coefficient of the monomial with the given exponent multi-index, or 0.
    pub fn coeff_of(&self, exp: [u8; NUM_VARS]) -> f32 {
        match self.basis.exps.iter().position(|x| *x == exp) {
            Some(i) => self.coeffs[i],
            None => 0.0,
        }
    }
}

/// Generalized binomial coefficient `C(alpha, k) = Π_{i=0}^{k-1}(alpha-i) / k!`.
fn generalized_binomial(alpha: f32, k: usize) -> f32 {
    let mut num = 1.0f32;
    let mut den = 1.0f32;
    for i in 0..k {
        num *= alpha - i as f32;
        den *= (i + 1) as f32;
    }
    num / den
}

#[cfg(test)]
mod test {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn product_drops_high_degree() {
        // (1 + x)(1 - x) = 1 - x^2, at degree 2 this is exact.
        let basis = Basis::cached(2);
        let x = TruncPoly::var(basis.clone(), 0);
        let one = TruncPoly::constant(basis.clone(), 1.0);
        let lhs = one.add(&x);
        let rhs = one.sub(&x);
        let p = lhs.mul(&rhs);
        assert!(approx(p.coeff_of([0, 0, 0, 0]), 1.0, 1e-6));
        assert!(approx(p.coeff_of([1, 0, 0, 0]), 0.0, 1e-6));
        assert!(approx(p.coeff_of([2, 0, 0, 0]), -1.0, 1e-6));
    }

    #[test]
    fn product_truncates() {
        // At degree 1, (1 + x)(1 + x) loses the x^2 term.
        let basis = Basis::cached(1);
        let x = TruncPoly::var(basis.clone(), 0);
        let one = TruncPoly::constant(basis.clone(), 1.0);
        let p = one.add(&x).mul(&one.add(&x));
        assert!(approx(p.coeff_of([1, 0, 0, 0]), 2.0, 1e-6));
        assert!(approx(p.coeff_of([2, 0, 0, 0]), 0.0, 1e-6)); // dropped
    }

    #[test]
    fn recip_times_self_is_one() {
        let basis = Basis::cached(4);
        // p = 2 + x + 0.5*y
        let x = TruncPoly::var(basis.clone(), 0);
        let y = TruncPoly::var(basis.clone(), 1);
        let p = TruncPoly::constant(basis.clone(), 2.0)
            .add(&x)
            .add(&y.scale(0.5));
        let prod = p.mul(&p.recip());
        // Should be 1 to truncation order at a few sample points.
        for pt in [[0.0, 0.0, 0.0, 0.0], [0.05, -0.03, 0.0, 0.0]] {
            assert!(approx(prod.eval(&pt), 1.0, 1e-4), "{:?}", prod.eval(&pt));
        }
    }

    #[test]
    fn sqrt_squared_is_self() {
        let basis = Basis::cached(4);
        let x = TruncPoly::var(basis.clone(), 2); // dx
        let y = TruncPoly::var(basis.clone(), 3); // dy
        // radicand 1 - dx^2 - dy^2, constant term 1 (axial dz^2).
        let rad = TruncPoly::constant(basis.clone(), 1.0)
            .sub(&x.mul(&x))
            .sub(&y.mul(&y));
        let s = rad.sqrt();
        let sq = s.mul(&s);
        for pt in [[0.0, 0.0, 0.02, -0.01], [0.0, 0.0, 0.05, 0.03]] {
            assert!(
                approx(sq.eval(&pt), rad.eval(&pt), 1e-4),
                "{} vs {}",
                sq.eval(&pt),
                rad.eval(&pt)
            );
        }
    }

    #[test]
    fn var_evaluates_to_input() {
        let basis = Basis::cached(3);
        let dx = TruncPoly::var(basis.clone(), 2);
        let pt = [0.1, 0.2, 0.3, 0.4];
        assert!(approx(dx.eval(&pt), 0.3, 1e-6));
    }

    #[test]
    fn partial_of_square() {
        let basis = Basis::cached(3);
        let x = TruncPoly::var(basis.clone(), 0);
        let x2 = x.mul(&x);
        let d = x2.partial(0); // d/dx x^2 = 2x
        assert!(approx(d.coeff_of([1, 0, 0, 0]), 2.0, 1e-6));
    }

    #[test]
    fn num_terms_matches_formula() {
        // C(n+4, 4)
        assert_eq!(Basis::cached(1).num_terms(), 5);
        assert_eq!(Basis::cached(3).num_terms(), 35);
        assert_eq!(Basis::cached(5).num_terms(), 126);
    }
}
