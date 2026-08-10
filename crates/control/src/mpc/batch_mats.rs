//! Condensed/batch MPC prediction matrices — direct port of
//! `BuildBatchMPCMats.m`. `x_{k+1} = Ad*x_k + Bd*u_k + Gd*ref`,
//! `x = [r,p,rdot,pdot,ydot, e_r,e_p, d1,d2,d3,d4]`,
//! `u` = absolute commanded positions `[d1*,d2*,d3*,d4*]`.

use nalgebra::{DMatrix, DVector};

pub struct BatchMats {
    pub n: usize,
    pub m: usize,
    pub horizon: usize,
    /// Free (autonomous) response: stacked `Ad^i`.
    pub omega: DMatrix<f64>,
    /// Forced response to `U = [u_0;...;u_{N-1}]`: block-Toeplitz `Ad^(i-j)*Bd`.
    pub gamma: DMatrix<f64>,
    /// Reference response: block-Toeplitz `Ad^(i-j)*Gd`.
    pub phi: DMatrix<f64>,
    /// Stacked reference: `kron(ones(N,1), ref)`.
    pub r_stack: DVector<f64>,
    /// First-difference operator on `U` (block-bidiagonal I/-I), for
    /// penalizing/bounding `Delta-u = u_k - u_{k-1}`.
    pub d: DMatrix<f64>,
    /// `[u_prev; 0; ...; 0]` — subtract from `D*U` to get the actual
    /// increment sequence starting from `u_prev`.
    pub d0: DVector<f64>,
    /// `Omega*x0 + Phi*r_stack` — the horizon's autonomous+reference response.
    pub x_free: DVector<f64>,
}

pub fn build(
    ad: &DMatrix<f64>,
    bd: &DMatrix<f64>,
    gd: &DMatrix<f64>,
    horizon: usize,
    x0: &DVector<f64>,
    ref_: &DVector<f64>,
    u_prev: &DVector<f64>,
) -> BatchMats {
    let n = ad.nrows();
    let m = bd.ncols();
    let p = gd.ncols();
    let big_n = horizon;

    let mut omega = DMatrix::<f64>::zeros(n * big_n, n);
    let mut gamma = DMatrix::<f64>::zeros(n * big_n, m * big_n);
    let mut phi = DMatrix::<f64>::zeros(n * big_n, p * big_n);

    // Powers of Ad, computed incrementally (Ad^0..Ad^N) rather than
    // recomputing Ad^(i-j) from scratch each time.
    let mut powers = Vec::with_capacity(big_n + 1);
    powers.push(DMatrix::<f64>::identity(n, n));
    for i in 1..=big_n {
        powers.push(&powers[i - 1] * ad);
    }

    for i in 1..=big_n {
        let rows = (i - 1) * n..i * n;
        omega.view_mut((rows.start, 0), (n, n)).copy_from(&powers[i]);

        for j in 1..=i {
            let cols_u = (j - 1) * m..j * m;
            let cols_r = (j - 1) * p..j * p;
            let a_pow = &powers[i - j];
            gamma
                .view_mut((rows.start, cols_u.start), (n, m))
                .copy_from(&(a_pow * bd));
            phi.view_mut((rows.start, cols_r.start), (n, p))
                .copy_from(&(a_pow * gd));
        }
    }

    let mut r_stack = DVector::<f64>::zeros(p * big_n);
    for k in 0..big_n {
        r_stack.rows_mut(k * p, p).copy_from(ref_);
    }

    let mut d = DMatrix::<f64>::zeros(m * big_n, m * big_n);
    for k in 1..=big_n {
        let rows = (k - 1) * m..k * m;
        d.view_mut((rows.start, rows.start), (m, m))
            .copy_from(&DMatrix::identity(m, m));
        if k >= 2 {
            let prev = (k - 2) * m..(k - 1) * m;
            d.view_mut((rows.start, prev.start), (m, m))
                .copy_from(&(-DMatrix::<f64>::identity(m, m)));
        }
    }

    let mut d0 = DVector::<f64>::zeros(m * big_n);
    d0.rows_mut(0, m).copy_from(u_prev);

    let x_free = &omega * x0 + &phi * &r_stack;

    BatchMats {
        n,
        m,
        horizon: big_n,
        omega,
        gamma,
        phi,
        r_stack,
        d,
        d0,
        x_free,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_1_reduces_to_single_step() {
        let ad = DMatrix::<f64>::identity(2, 2);
        let bd = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let gd = DMatrix::<f64>::zeros(2, 1);
        let x0 = DVector::from_vec(vec![1.0, 2.0]);
        let ref_ = DVector::from_vec(vec![0.0]);
        let u_prev = DVector::from_vec(vec![0.5]);

        let mats = build(&ad, &bd, &gd, 1, &x0, &ref_, &u_prev);
        assert_eq!(mats.gamma.shape(), (2, 1));
        assert_eq!(mats.gamma.column(0), bd.column(0));
        assert_eq!(mats.x_free, x0);
        assert_eq!(mats.d[(0, 0)], 1.0);
        assert_eq!(mats.d0[0], 0.5);
    }

    #[test]
    fn difference_operator_is_bidiagonal() {
        let ad = DMatrix::<f64>::identity(1, 1);
        let bd = DMatrix::from_row_slice(1, 1, &[1.0]);
        let gd = DMatrix::<f64>::zeros(1, 1);
        let x0 = DVector::from_vec(vec![0.0]);
        let ref_ = DVector::from_vec(vec![0.0]);
        let u_prev = DVector::from_vec(vec![0.0]);

        let mats = build(&ad, &bd, &gd, 3, &x0, &ref_, &u_prev);
        // D = [[1,0,0],[-1,1,0],[0,-1,1]]
        assert_eq!(mats.d.shape(), (3, 3));
        assert_eq!(mats.d[(1, 0)], -1.0);
        assert_eq!(mats.d[(1, 1)], 1.0);
        assert_eq!(mats.d[(2, 1)], -1.0);
        assert_eq!(mats.d[(2, 2)], 1.0);
    }
}
