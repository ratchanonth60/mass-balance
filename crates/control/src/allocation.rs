//! Torque -> mass-position allocation for the LQI path (`MainConstantTs`'s
//! in-loop QP/pinv block, `app.MassSolver == "Pinv" | "QP"`). Given a desired
//! body-frame torque `tgt` and gravity vector `gb`, solves for a 4-vector of
//! mass-position deltas `d` minimizing `||A*d - tgt||^2` where `A = [cross(c_i,
//! gb)]` for the 4 rail direction vectors.
//!
//! Geometry here is intentionally [`crate::mpc_model::geometry`] (same
//! `sind(60)`, same ordering as `ModelInit.m`) — **not** `InitLQR.m`'s own
//! `cosd(60)` geometry, even though the `Kx`/`Ki` gains being allocated here
//! were synthesized against the `InitLQR.m` model. This mismatch is real in
//! the live app (`MainConstantTs` rebuilds geometry inline rather than
//! reusing whatever `InitLQR.m` used) — replicated, not "fixed".
//!
//! Rate-limiting (`du_max`), the `d_track_tol` gate, and the `+d0` nominal
//! center offset are session/loop state, not pure math — those live in the
//! `io` crate's LQI loop (M3), not here.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus, SupportedConeT, ZeroConeT};
use nalgebra::{Matrix3x4, Matrix4, SVector, Vector3, Vector4};

/// `A = [cross(c1,gb), cross(c2,gb), cross(c3,gb), cross(c4,gb)]`.
pub fn build_a(gb: Vector3<f64>) -> Matrix3x4<f64> {
    let c = super::mpc_model::geometry::c_vectors();
    let cols: [Vector3<f64>; 4] = c.map(|ci| Vector3::from(ci).cross(&gb));
    Matrix3x4::from_columns(&cols)
}

/// Minimum-norm solve via Moore-Penrose pseudo-inverse (`pinv(A)*tgt`).
pub fn allocate_pinv(a: &Matrix3x4<f64>, tgt: Vector3<f64>) -> Vector4<f64> {
    let svd = (*a).svd(true, true);
    let pinv = svd.pseudo_inverse(1e-12).expect("SVD pseudo-inverse failed");
    pinv * tgt
}

/// Constrained QP solve: `min ||A*d - tgt||^2` s.t. `sum(d) = 0`,
/// `lb <= d <= ub`. Matches the live `MassSolver == "QP"` branch:
/// `H = 2*A'A`, `f = -2*A'*tgt`.
pub fn allocate_qp(
    a: &Matrix3x4<f64>,
    tgt: Vector3<f64>,
    lb: Vector4<f64>,
    ub: Vector4<f64>,
) -> Option<Vector4<f64>> {
    let h: Matrix4<f64> = a.transpose() * a * 2.0;
    let f: SVector<f64, 4> = a.transpose() * tgt * -2.0;

    let mut rows: Vec<Vec<f64>> = vec![vec![1.0; 4]]; // sum(d) = 0 (equality)
    let mut b: Vec<f64> = vec![0.0];
    let n_eq = 1;

    for i in 0..4 {
        let mut row = vec![0.0; 4];
        row[i] = 1.0;
        rows.push(row);
        b.push(ub[i]);
    }
    for i in 0..4 {
        let mut row = vec![0.0; 4];
        row[i] = -1.0;
        rows.push(row);
        b.push(-lb[i]);
    }

    let p_rows: Vec<Vec<f64>> = (0..4).map(|i| h.row(i).iter().copied().collect()).collect();
    let p_csc = CscMatrix::from(&p_rows);
    let q_vec: Vec<f64> = f.iter().copied().collect();
    let a_csc = CscMatrix::from(&rows);
    let cones: Vec<SupportedConeT<f64>> = vec![ZeroConeT(n_eq), NonnegativeConeT(8)];

    let settings = DefaultSettings {
        verbose: false,
        max_iter: 200,
        ..Default::default()
    };

    let mut solver = DefaultSolver::new(&p_csc, &q_vec, &a_csc, &b, &cones, settings).ok()?;
    solver.solve();
    if !matches!(solver.solution.status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        return None;
    }
    if !solver.solution.x.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(Vector4::from_column_slice(&solver.solution.x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinv_and_qp_agree_when_unconstrained_ish() {
        let gb = Vector3::new(0.0, 0.0, -9.81);
        let a = build_a(gb);
        let tgt = Vector3::new(0.1, -0.05, 0.0);

        let d_pinv = allocate_pinv(&a, tgt);
        let lb = Vector4::from_element(-1.0);
        let ub = Vector4::from_element(1.0);
        let d_qp = allocate_qp(&a, tgt, lb, ub).unwrap();

        // Both minimize ||A*d-tgt||^2 but pinv gives the minimum-norm solution
        // (which happens to already satisfy sum(d)=0 for this geometry since
        // the columns of A sum to zero along that null direction); QP adds
        // sum(d)=0 explicitly. Just check both reduce the residual similarly.
        let res_pinv = (a * d_pinv - tgt).norm();
        let res_qp = (a * d_qp - tgt).norm();
        assert!((res_pinv - res_qp).abs() < 1e-3, "{res_pinv} vs {res_qp}");
    }
}
