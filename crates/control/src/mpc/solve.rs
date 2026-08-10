//! Constrained batch MPC QP solve — direct port of
//! `ConstrainedBatchMPC11AbsLag_dU.m`, the solver actually called by the live
//! `MainConstantTs11.m` loop. `R` penalizes `Delta-u` (rate of change of the
//! *absolute* position command), not `u` itself.

use super::batch_mats::{self, BatchMats};
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus, SupportedConeT,
    ZeroConeT,
};
use nalgebra::{DMatrix, DVector};

/// Extra decision-variable-space constraints supplied by the caller
/// (`MainConstantTs11.m` builds these itself: `Aineq`/`bineq` for the
/// `Delta-u`-vs-`du_max` rate limit, `Aeq`/`beq` for the `keepZ` equality).
/// `Aineq*U <= bineq`, `Aeq*U = beq`, where `U = [u0;u1;...;u_{N-1}]`.
#[derive(Default)]
pub struct ExtraConstraints {
    pub a_ineq: Option<DMatrix<f64>>,
    pub b_ineq: Option<DVector<f64>>,
    pub a_eq: Option<DMatrix<f64>>,
    pub b_eq: Option<DVector<f64>>,
}

pub struct SolveResult {
    pub u0: DVector<f64>,
    pub u_opt: DVector<f64>,
    pub feasible: bool,
    pub mats: BatchMats,
}

#[allow(clippy::too_many_arguments)]
pub fn solve(
    ad: &DMatrix<f64>,
    bd: &DMatrix<f64>,
    gd: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    pf: &DMatrix<f64>,
    horizon: usize,
    x0: &DVector<f64>,
    ref_: &DVector<f64>,
    u_prev: &DVector<f64>,
    dmin: &DVector<f64>,
    dmax: &DVector<f64>,
    extra: &ExtraConstraints,
) -> SolveResult {
    let mats = batch_mats::build(ad, bd, gd, horizon, x0, ref_, u_prev);
    let n = mats.n;
    let m = mats.m;
    let big_n = mats.horizon;

    // Qbar = kron(eye(N),Q) with the last block replaced by Pf.
    let mut qbar = DMatrix::<f64>::zeros(n * big_n, n * big_n);
    for k in 0..big_n {
        let block = if k == big_n - 1 { pf } else { q };
        qbar.view_mut((k * n, k * n), (n, n)).copy_from(block);
    }
    let rdu_bar = kron_eye_block(big_n, r);

    let mut h =
        mats.gamma.transpose() * &qbar * &mats.gamma + mats.d.transpose() * &rdu_bar * &mats.d;
    let f =
        mats.gamma.transpose() * &qbar * &mats.x_free - mats.d.transpose() * &rdu_bar * &mats.d0;

    h = (&h + h.transpose()) * 0.5;
    for i in 0..h.nrows() {
        h[(i, i)] += 1e-9;
    }

    let lb = kron_ones(big_n, dmin);
    let ub = kron_ones(big_n, dmax);

    let nu = m * big_n;

    // --- assemble clarabel constraint rows: equalities first (ZeroConeT),
    // then all inequalities (NonnegativeConeT): extra Aineq, then box ub,
    // then box lb, then extra Aeq.
    let mut rows_eq: Vec<Vec<f64>> = Vec::new();
    let mut b_eq: Vec<f64> = Vec::new();
    if let (Some(a), Some(b)) = (&extra.a_eq, &extra.b_eq) {
        for i in 0..a.nrows() {
            rows_eq.push(a.row(i).iter().copied().collect());
            b_eq.push(b[i]);
        }
    }

    let mut rows_ineq: Vec<Vec<f64>> = Vec::new();
    let mut b_ineq: Vec<f64> = Vec::new();
    if let (Some(a), Some(b)) = (&extra.a_ineq, &extra.b_ineq) {
        for i in 0..a.nrows() {
            rows_ineq.push(a.row(i).iter().copied().collect());
            b_ineq.push(b[i]);
        }
    }
    // U <= ub
    for i in 0..nu {
        let mut row = vec![0.0; nu];
        row[i] = 1.0;
        rows_ineq.push(row);
        b_ineq.push(ub[i]);
    }
    // -U <= -lb
    for i in 0..nu {
        let mut row = vec![0.0; nu];
        row[i] = -1.0;
        rows_ineq.push(row);
        b_ineq.push(-lb[i]);
    }

    let mut a_rows = rows_eq;
    let mut b_vals = b_eq;
    let n_eq = a_rows.len();
    a_rows.extend(rows_ineq);
    b_vals.extend(b_ineq);
    let n_ineq = a_rows.len() - n_eq;

    let p_csc = dense_to_csc(&h);
    let q_vec: Vec<f64> = f.iter().copied().collect();
    let a_csc = CscMatrix::from(&a_rows);
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    if n_eq > 0 {
        cones.push(ZeroConeT(n_eq));
    }
    if n_ineq > 0 {
        cones.push(NonnegativeConeT(n_ineq));
    }

    let settings = DefaultSettings {
        verbose: false,
        max_iter: 500,
        ..Default::default()
    };

    let (u_opt, feasible) = if cones.is_empty() {
        // No constraints at all beyond the box (shouldn't happen in practice
        // since box bounds always produce at least one Nonneg row) — fall
        // back to previous command, matching the MATLAB infeasible-fallback.
        (kron_ones(big_n, u_prev), false)
    } else {
        let mut solver = match DefaultSolver::new(&p_csc, &q_vec, &a_csc, &b_vals, &cones, settings)
        {
            Ok(s) => s,
            Err(_) => {
                return SolveResult {
                    u0: u_prev.clone(),
                    u_opt: kron_ones(big_n, u_prev),
                    feasible: false,
                    mats,
                };
            }
        };
        solver.solve();
        let ok = matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) && solver.solution.x.iter().all(|v| v.is_finite());
        if ok {
            (DVector::from_vec(solver.solution.x.clone()), true)
        } else {
            (kron_ones(big_n, u_prev), false)
        }
    };

    let u0 = if feasible {
        u_opt.rows(0, m).into_owned()
    } else {
        u_prev.clone()
    };

    SolveResult {
        u0,
        u_opt,
        feasible,
        mats,
    }
}

fn kron_eye_block(n: usize, block: &DMatrix<f64>) -> DMatrix<f64> {
    let d = block.nrows();
    let mut out = DMatrix::<f64>::zeros(d * n, d * n);
    for k in 0..n {
        out.view_mut((k * d, k * d), (d, d)).copy_from(block);
    }
    out
}

fn kron_ones(n: usize, v: &DVector<f64>) -> DVector<f64> {
    let d = v.len();
    let mut out = DVector::<f64>::zeros(d * n);
    for k in 0..n {
        out.rows_mut(k * d, d).copy_from(v);
    }
    out
}

fn dense_to_csc(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let rows: Vec<Vec<f64>> = (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect();
    CscMatrix::from(&rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconstrained_tracking_drives_state_to_reference() {
        // Trivial 1-state, 1-input plant: x_{k+1} = x_k + u_k. Track ref=0,
        // penalize (x-ref)^2 heavily and u^2*Delta lightly -> u0 should move
        // toward -x0.
        let ad = DMatrix::from_row_slice(1, 1, &[1.0]);
        let bd = DMatrix::from_row_slice(1, 1, &[1.0]);
        let gd = DMatrix::<f64>::zeros(1, 0);
        let q = DMatrix::from_row_slice(1, 1, &[100.0]);
        let r = DMatrix::from_row_slice(1, 1, &[0.01]);
        let pf = q.clone();
        let x0 = DVector::from_vec(vec![1.0]);
        let ref_ = DVector::from_vec(vec![]);
        let u_prev = DVector::from_vec(vec![0.0]);
        let dmin = DVector::from_vec(vec![-10.0]);
        let dmax = DVector::from_vec(vec![10.0]);

        let result = solve(
            &ad,
            &bd,
            &gd,
            &q,
            &r,
            &pf,
            3,
            &x0,
            &ref_,
            &u_prev,
            &dmin,
            &dmax,
            &ExtraConstraints::default(),
        );
        assert!(result.feasible);
        assert!(result.u0[0] < -0.5, "u0={}", result.u0[0]);
    }

    /// Locks in that `dense_to_csc` passing a full dense symmetric `H` (not
    /// just its upper triangle) is safe: Clarabel reads only the upper
    /// triangle, so extra lower-triangle nonzeros are silently ignored, not
    /// double-counted. Verified empirically (see the module's dev history)
    /// against `min 0.5x'[[2,1],[1,2]]x - x1 - x2` on a loose box, whose
    /// analytic unconstrained optimum is `x=[1/3,1/3]` — a diagonal-only
    /// misread would give `[0.5,0.5]`, a doubled-offdiagonal misread would
    /// give neither.
    #[test]
    fn hessian_off_diagonal_is_read_once_not_doubled() {
        let p = dense_to_csc(&DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]));
        let q = vec![-1.0, -1.0];
        let a = dense_to_csc(&DMatrix::from_row_slice(
            4,
            2,
            &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        ));
        let b = vec![10.0, 10.0, 10.0, 10.0];
        let cones: Vec<SupportedConeT<f64>> = vec![NonnegativeConeT(4)];
        let settings = DefaultSettings {
            verbose: false,
            ..Default::default()
        };
        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
        solver.solve();
        assert!((solver.solution.x[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((solver.solution.x[1] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn box_bounds_are_respected() {
        let ad = DMatrix::from_row_slice(1, 1, &[1.0]);
        let bd = DMatrix::from_row_slice(1, 1, &[1.0]);
        let gd = DMatrix::<f64>::zeros(1, 0);
        let q = DMatrix::from_row_slice(1, 1, &[100.0]);
        let r = DMatrix::from_row_slice(1, 1, &[0.001]);
        let pf = q.clone();
        let x0 = DVector::from_vec(vec![10.0]);
        let ref_ = DVector::from_vec(vec![]);
        let u_prev = DVector::from_vec(vec![0.0]);
        let dmin = DVector::from_vec(vec![-1.0]);
        let dmax = DVector::from_vec(vec![1.0]);

        let result = solve(
            &ad,
            &bd,
            &gd,
            &q,
            &r,
            &pf,
            2,
            &x0,
            &ref_,
            &u_prev,
            &dmin,
            &dmax,
            &ExtraConstraints::default(),
        );
        assert!(result.feasible);
        assert!(result.u0[0] >= -1.0 - 1e-6 && result.u0[0] <= 1.0 + 1e-6);
    }
}
