//! Discrete-time infinite-horizon LQR gain via backward Riccati recursion
//! iterated to convergence — no `dlqr` equivalent exists as a mainstream Rust
//! crate, so this replicates what `dlqr` computes numerically (the same
//! recursion as `my_DP.m`, run until `P` stops changing instead of for a
//! fixed horizon). Must run live (Q/R come from the GUI at runtime), not be
//! precomputed.

use nalgebra::DMatrix;

pub struct DareResult {
    pub p: DMatrix<f64>,
    pub k: DMatrix<f64>,
    pub iterations: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DareError;

/// Solves the discrete algebraic Riccati equation for `(A, B, Q, R)` via
/// backward-recursion-to-convergence:
/// `P_{k+1} = Q + A'P_kA - A'P_kB(R+B'P_kB)^-1 B'P_kA`, `K = (R+B'PB)^-1 B'PA`.
pub fn solve(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<DareResult, DareError> {
    let mut p = q.clone();
    let at = a.transpose();
    let bt = b.transpose();

    for it in 0..max_iter {
        let pb = &p * b;
        let s = r + &bt * &pb; // R + B'PB
        let s_inv = s.try_inverse().ok_or(DareError)?;
        let k = &s_inv * &bt * &p * a; // (R+B'PB)^-1 B'PA
        let p_next = q + &at * &p * a - &at * &pb * &k;

        let diff = (&p_next - &p).norm();
        p = p_next;
        if diff < tol {
            return Ok(DareResult { p, k, iterations: it + 1 });
        }
    }

    // Return best-effort result even without full convergence; caller can
    // inspect `iterations == max_iter` to detect this.
    let pb = &p * b;
    let s = r + &bt * &pb;
    let s_inv = s.try_inverse().ok_or(DareError)?;
    let k = &s_inv * &bt * &p * a;
    Ok(DareResult { p, k, iterations: max_iter })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lqi_model;
    use nalgebra::DMatrix;
    use std::fs;

    #[test]
    fn matches_init_lqr_fixture_gains() {
        let fixture: serde_json::Value =
            serde_json::from_str(&fs::read_to_string("../../fixtures/ctrl_lqr.json").unwrap())
                .unwrap();

        let to_mat = |v: &serde_json::Value| -> DMatrix<f64> {
            let rows = v.as_array().unwrap();
            let nrows = rows.len();
            let ncols = rows[0].as_array().unwrap().len();
            let data: Vec<f64> = rows
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
                .collect();
            DMatrix::from_row_slice(nrows, ncols, &data)
        };

        let qx = to_mat(&fixture["Qx"]);
        let qi = to_mat(&fixture["Qi"]);
        let r = to_mat(&fixture["R"]);
        let kx_expected = to_mat(&fixture["Kx"]);
        let ki_expected = to_mat(&fixture["Ki"]);

        let (ad, bd) = lqi_model::build_augmented(1.0);

        let mut q_aug = DMatrix::<f64>::zeros(7, 7);
        q_aug.view_mut((0, 0), (5, 5)).copy_from(&qx);
        q_aug.view_mut((5, 5), (2, 2)).copy_from(&qi);

        let result = solve(&ad, &bd, &q_aug, &r, 1e-12, 5000).unwrap();

        let kx = result.k.columns(0, 5);
        let ki = result.k.columns(5, 2);

        for i in 0..3 {
            for j in 0..5 {
                assert!(
                    (kx[(i, j)] - kx_expected[(i, j)]).abs() < 1e-6,
                    "Kx[{i},{j}]: got {}, expected {}",
                    kx[(i, j)],
                    kx_expected[(i, j)]
                );
            }
            for j in 0..2 {
                assert!(
                    (ki[(i, j)] - ki_expected[(i, j)]).abs() < 1e-6,
                    "Ki[{i},{j}]: got {}, expected {}",
                    ki[(i, j)],
                    ki_expected[(i, j)]
                );
            }
        }
    }
}
