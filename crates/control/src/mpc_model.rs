//! Closed-form linearization of the 5-state attitude plant used by the live
//! MPC pipeline (`ModelInit.m` / `ModelInit_PostBalance.m` — both share this
//! exact physics, differing only in tuning-preset constants, see
//! [`Preset`]). Ported from a from-scratch sympy re-derivation of
//! `ModelInit.m`'s symbolic linearization (not hand-transcribed — see
//! `derive_mpc_model.py` in this crate's `tools/` dir for the script), because
//! `d_op` genuinely varies per run (unlike the LQI model, see
//! [`crate::lqi_model`]) so this cannot be a frozen matrix.
//!
//! `A_op(d_op, R0z)` is affine in `[d1,d2,d3,d4,R0z]` (`Rtot` is affine in `d`
//! and `R0`, and only `Rtot` carries the `d`/`R0` dependence into the
//! Jacobian), verified numerically against the raw sympy Jacobian to ~1e-22
//! residual before these coefficients were transcribed. `E_op` (Jacobian wrt
//! `d`) works out to a true constant, independent of `d_op`/`R0z` — `Rtot`'s
//! *own* Jacobian wrt `d` doesn't depend on `d`.

use nalgebra::{Matrix5, Matrix5x4};

/// Geometry/mass constants shared by [`linearize`] and the `allocation`
/// module's inline torque-allocation matrix (`MainConstantTs`'s in-loop
/// `cross(c_i, gb)` build uses this *exact* same `sind(60)`/ordering — not
/// `InitLQR.m`'s `cosd(60)` geometry, even though `InitLQR.m` is what
/// produces the LQI gains applied through that allocation). See
/// `lqi_model` module docs for why these are intentionally not unified.
pub mod geometry {
    /// Lead-screw rail direction unit vectors (`ModelInit.m`: `cz = sind(60)`).
    pub fn c_vectors() -> [[f64; 3]; 4] {
        let cxy = (60f64).to_radians().cos() * (45f64).to_radians().cos();
        let cz = (60f64).to_radians().sin();
        [
            [-cxy, -cxy, cz],
            [-cxy, cxy, cz],
            [cxy, cxy, cz],
            [cxy, -cxy, cz],
        ]
    }

    pub const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];
}

/// Tuning/config preset selected by `XYPreCheckbox` in the live app.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    /// Platform CoM z-offset (`R0` in `ModelInit.m`, only the z component is
    /// ever nonzero in either preset).
    pub r0z: f64,
    /// Actuator first-order lag pole blend factor.
    pub alpha: f64,
    /// MPC horizon.
    pub n_mpc: usize,
}

/// `ModelInit.m` (XYPreCheckbox off): `R0=[0,0,-0.000]`, `alpha=0.2`.
pub const PRESET_MODEL_INIT: Preset = Preset {
    r0z: -0.000,
    alpha: 0.2,
    n_mpc: 5,
};

/// `ModelInit_PostBalance.m` (XYPreCheckbox on): `R0=[0,0,-0.007]`, `alpha=0.1`.
pub const PRESET_POST_BALANCE: Preset = Preset {
    r0z: -0.007,
    alpha: 0.1,
    n_mpc: 5,
};

/// GUI-tunable weights (`app.q1EditField.Value` etc.) — only consulted by
/// the `ModelInit.m` preset. `ModelInit_PostBalance.m` hardcodes its own
/// weights and ignores the GUI entirely (see [`build_ctrl`]).
#[derive(Debug, Clone, Copy)]
pub struct TuningWeights {
    pub q: [f64; 5],
    pub qi: [f64; 2],
    pub qd: f64,
    pub r: f64,
    pub du_max: f64,
    pub d_track_tol: f64,
}

/// Fully assembled 11-state model + MPC tuning, ready for
/// `control::mpc::solve`. `Ts` is always `1.0` in both presets.
pub struct CtrlMats {
    pub ad: nalgebra::DMatrix<f64>,
    pub bd: nalgebra::DMatrix<f64>,
    pub gd: nalgebra::DMatrix<f64>,
    pub q11: nalgebra::DMatrix<f64>,
    pub r: nalgebra::DMatrix<f64>,
    pub pf11: nalgebra::DMatrix<f64>,
    pub n_mpc: usize,
    pub du_max: f64,
    pub d_track_tol: f64,
}

fn diag(vals: &[f64]) -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(vals))
}

/// Replicates `ModelInit(app, d0)` (weights required, GUI-tunable) or
/// `ModelInit_PostBalance(d0)` (weights ignored, hardcoded tuning) — which
/// branch runs is selected by the live app's `XYPreCheckbox`.
pub fn build_ctrl(preset: Preset, d_op: [f64; 4], weights: Option<&TuningWeights>) -> CtrlMats {
    let (a, e) = linearize(d_op, preset.r0z);
    let (ad, bd, gd) = build_11state(a, e, 1.0, preset.alpha);

    let is_post_balance = (preset.r0z - PRESET_POST_BALANCE.r0z).abs() < 1e-12;

    let (q11, r, pf11, du_max, d_track_tol) = if is_post_balance {
        // ModelInit_PostBalance.m: hardcoded, GUI weights not consulted.
        let qa = diag(&[100.0, 100.0, 50.0, 50.0, 25.0]);
        let qi = diag(&[0.3, 0.3]);
        let qd = diag(&[15.0, 15.0, 15.0, 15.0]);
        let mut q11 = nalgebra::DMatrix::<f64>::zeros(11, 11);
        q11.view_mut((0, 0), (5, 5)).copy_from(&qa);
        q11.view_mut((5, 5), (2, 2)).copy_from(&qi);
        q11.view_mut((7, 7), (4, 4)).copy_from(&qd);
        let r = diag(&[150.0; 4]);
        let pf11 = &q11 * 5.0;
        (q11, r, pf11, 0.010, 0.010)
    } else {
        // ModelInit.m: q1..q5/qi1/qi2/qd/R/du_max/d_track_tol from the GUI.
        let w = weights.expect("ModelInit preset requires GUI tuning weights");
        let qa = diag(&w.q);
        let qi = diag(&w.qi);
        let qd = diag(&[w.qd; 4]);
        let mut q11 = nalgebra::DMatrix::<f64>::zeros(11, 11);
        q11.view_mut((0, 0), (5, 5)).copy_from(&qa);
        q11.view_mut((5, 5), (2, 2)).copy_from(&qi);
        q11.view_mut((7, 7), (4, 4)).copy_from(&qd);
        let r = diag(&[w.r; 4]);
        let pf11 = &q11 * 0.3;
        (q11, r, pf11, w.du_max, w.d_track_tol)
    };

    CtrlMats {
        ad,
        bd,
        gd,
        q11,
        r,
        pf11,
        n_mpc: preset.n_mpc,
        du_max,
        d_track_tol,
    }
}

// Affine coefficient matrices for A_op(d,R0z) = A_BASE + sum(d_i*A_Di) + R0z*A_R0Z,
// generated by tools/derive_mpc_model.py from a from-scratch sympy re-derivation
// of ModelInit.m, verified affine to ~1e-22 residual before transcription.
#[rustfmt::skip]
const A_BASE: [[f64; 5]; 5] = [
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [9.193_809_597_677_091e-1, 0.0, 0.0, 0.0, 0.0],
    [0.0, 9.193_809_597_677_091e-1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
];
#[rustfmt::skip]
const A_D1: [[f64; 5]; 5] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.149_226_199_709_636_2, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.149_226_199_709_636_2, 0.0, 0.0, 0.0],
    [-2.894_926_435_933_835e-1, -2.894_926_435_933_835e-1, 0.0, 0.0, 0.0],
];
#[rustfmt::skip]
const A_D2: [[f64; 5]; 5] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.149_226_199_709_636_2, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.149_226_199_709_636_2, 0.0, 0.0, 0.0],
    [-2.894_926_435_933_835e-1, 2.894_926_435_933_835e-1, 0.0, 0.0, 0.0],
];
#[rustfmt::skip]
const A_D3: [[f64; 5]; 5] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.149_226_199_709_636_2, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.149_226_199_709_636_2, 0.0, 0.0, 0.0],
    [2.894_926_435_933_835e-1, 2.894_926_435_933_835e-1, 0.0, 0.0, 0.0],
];
#[rustfmt::skip]
const A_D4: [[f64; 5]; 5] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.149_226_199_709_636_2, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.149_226_199_709_636_2, 0.0, 0.0, 0.0],
    [2.894_926_435_933_835e-1, -2.894_926_435_933_835e-1, 0.0, 0.0, 0.0],
];
#[rustfmt::skip]
const A_R0Z: [[f64; 5]; 5] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [-9.448_326_234_052_719e1, 0.0, 0.0, 0.0, 0.0],
    [0.0, -9.448_326_234_052_719e1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
];
#[rustfmt::skip]
const E_CONST: [[f64; 4]; 5] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [4.691_696_313_877_410_4e-1, -4.691_696_313_877_410_4e-1, -4.691_696_313_877_410_4e-1, 4.691_696_313_877_410_4e-1],
    [-4.691_696_313_877_410_4e-1, -4.691_696_313_877_410_4e-1, 4.691_696_313_877_410_4e-1, 4.691_696_313_877_410_4e-1],
    [0.0, 0.0, 0.0, 0.0],
];

fn from_rows(rows: [[f64; 5]; 5]) -> Matrix5<f64> {
    Matrix5::from_row_iterator(rows.into_iter().flatten())
}

/// Linearizes the 5-state attitude plant at `x_att=0`, `d=d_op`, returning
/// `(A_op, E_op)` where `x_{k+1} = ... A_op*x_att + E_op*d + ...` (continuous-time
/// Jacobians, as in `ModelInit.m` — discretization/augmentation into the
/// 11-state model happens in [`build_11state`]).
pub fn linearize(d_op: [f64; 4], r0z: f64) -> (Matrix5<f64>, Matrix5x4<f64>) {
    let mut a = from_rows(A_BASE);
    for (di, coeffs) in d_op.iter().zip([A_D1, A_D2, A_D3, A_D4]) {
        a += from_rows(coeffs) * *di;
    }
    a += from_rows(A_R0Z) * r0z;

    let e = Matrix5x4::from_row_iterator(E_CONST.into_iter().flatten());
    (a, e)
}

/// Builds the 11-state discrete augmented model
/// `[attitude(5); integral-error(2); actuator-lag(4)]` from a linearization,
/// matching `ModelInit.m`'s hand-assembled `A11`/`B11`/`G11` (not a proper
/// ZOH discretization — `Ts*Aa` isn't even added to the identity on the
/// attitude block; this is exactly what the live model does).
pub fn build_11state(
    a_op: Matrix5<f64>,
    e_op: Matrix5x4<f64>,
    ts: f64,
    alpha: f64,
) -> (
    nalgebra::DMatrix<f64>,
    nalgebra::DMatrix<f64>,
    nalgebra::DMatrix<f64>,
) {
    use nalgebra::DMatrix;

    let mut ad = DMatrix::<f64>::zeros(11, 11);
    ad.view_mut((0, 0), (5, 5)).copy_from(&a_op);
    ad.view_mut((0, 7), (5, 4)).copy_from(&e_op);
    // Ts*Cz: rows 5,6 pick roll,pitch (cols 0,1) scaled by Ts.
    ad[(5, 0)] = ts;
    ad[(6, 1)] = ts;
    ad[(5, 5)] = 1.0;
    ad[(6, 6)] = 1.0;
    for i in 7..11 {
        ad[(i, i)] = alpha;
    }

    let mut bd = DMatrix::<f64>::zeros(11, 4);
    for i in 0..4 {
        bd[(7 + i, i)] = 1.0 - alpha;
    }

    let mut gd = DMatrix::<f64>::zeros(11, 2);
    gd[(5, 0)] = -ts;
    gd[(6, 1)] = -ts;

    (ad, bd, gd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e_op_matches_default_linearization() {
        let (_, e) = linearize([0.2, 0.2, 0.2, 0.2], 0.0);
        assert!((e[(2, 0)] - 0.4691696314).abs() < 1e-6);
    }

    #[test]
    fn a_op_matches_default_linearization() {
        let (a, _) = linearize([0.2, 0.2, 0.2, 0.2], 0.0);
        // From the sympy numeric check at d_op=[0.2]*4, R0z=0.
        assert!((a[(0, 2)] - 1.0).abs() < 1e-9);
        assert!((a[(1, 3)] - 1.0).abs() < 1e-9);
        assert!(a[(2, 0)].abs() < 1e-9); // A_base + 4*0.2*A_d ~ 0 at this d_op
    }

    /// External validation against a real `ModelInit_PostBalance.m` run's
    /// saved `Ctrl.mat` (`fixtures/ctrl_mpc.json`) — not just internal
    /// self-consistency. `alpha=0.1` and `Pf11=5*Q11` in that fixture both
    /// identify it as the `PostBalance` preset (`r0z=-0.007`). The fixture's
    /// `d_op` isn't recorded, but `A_op`'s `d`-dependence lives in only a
    /// 3-dimensional subspace of the 4-dim `d` space (every affine
    /// coefficient matrix's nonzero rows are multiples of just two "shapes"),
    /// so the 3 recoverable combinations (`sum(d)`, and two antisymmetric
    /// pairings) are solved from the fixture's own entries and fed back
    /// through `linearize` to check every affected entry reproduces exactly
    /// — not just the ones used to solve for `d`.
    #[test]
    fn linearize_matches_real_model_init_post_balance_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("../../fixtures/ctrl_mpc.json").unwrap(),
        )
        .unwrap();
        let ad_op = fixture["Ad_op"].as_array().unwrap();
        let row = |i: usize| -> Vec<f64> {
            ad_op[i].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
        };
        let alpha = fixture["alpha"].as_f64().unwrap();
        assert!((alpha - PRESET_POST_BALANCE.alpha).abs() < 1e-12);
        let r0z = PRESET_POST_BALANCE.r0z;

        // Solve the 3 recoverable d-combinations from the fixture itself.
        let e20 = row(2)[0];
        let e40 = row(4)[0];
        let e41 = row(4)[1];
        let a_base_20 = A_BASE[2][0];
        let a_d_20 = A_D1[2][0]; // same coefficient for all 4 masses at [2][0]
        let a_r0z_20 = A_R0Z[2][0];
        let sum_d = (e20 - a_base_20 - r0z * a_r0z_20) / a_d_20;
        let c = A_D1[4][0].abs(); // 0.2895...
        let diff_34_12 = e40 / c; // d3+d4-d1-d2
        let diff_23_14 = e41 / c; // d2+d3-d1-d4

        let d34 = (sum_d + diff_34_12) / 2.0;
        let d12 = (sum_d - diff_34_12) / 2.0;
        // A 4th combination (d1-d2-d3+d4) doesn't appear in A_op at all, so
        // it's unrecoverable from the fixture and irrelevant to the check
        // below — fix it arbitrarily (s=0) and solve the other free
        // parameter (t) from the remaining constraint.
        let t = diff_23_14 / 2.0;
        let d_op = [d12 / 2.0 - t, d12 / 2.0 + t, d34 / 2.0, d34 / 2.0];

        let (a, e) = linearize(d_op, r0z);

        // Every A_op entry the fixture actually has nonzero data for.
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (a[(i, j)] - row(i)[j]).abs() < 1e-8,
                    "A_op[{i}][{j}]: got {}, fixture {}",
                    a[(i, j)],
                    row(i)[j]
                );
            }
        }
        // E_op occupies fixture columns 7..11 of the same rows.
        for i in 0..5 {
            for j in 0..4 {
                assert!(
                    (e[(i, j)] - row(i)[7 + j]).abs() < 1e-8,
                    "E_op[{i}][{j}]: got {}, fixture {}",
                    e[(i, j)],
                    row(i)[7 + j]
                );
            }
        }
    }

    #[test]
    fn build_11state_matches_model_init_shape() {
        let (a, e) = linearize([0.2; 4], PRESET_MODEL_INIT.r0z);
        let (ad, bd, gd) = build_11state(a, e, 1.0, PRESET_MODEL_INIT.alpha);
        assert_eq!(ad.shape(), (11, 11));
        assert_eq!(bd.shape(), (11, 4));
        assert_eq!(gd.shape(), (11, 2));
        // actuator-lag block
        assert!((ad[(7, 7)] - 0.2).abs() < 1e-12);
        assert!((bd[(7, 0)] - 0.8).abs() < 1e-12);
    }
}
