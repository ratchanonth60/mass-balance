//! LQI plant model — `InitLQR.m`'s linearization, discretization, and
//! augmentation. Unlike [`crate::mpc_model`], this operating point is
//! genuinely fixed (`x_op=0`, `rm_op=[0,0,0,0]`, `R0=[0,0,0]` always), so
//! `Ad_op`/`Bd_op` are true constants — derived once by
//! `tools/derive_lqi_model.py` (sympy linearization + `scipy.linalg.expm`
//! ZOH discretization, `Ts=1`) and verified bit-for-bit against a real
//! `InitLQR.m` run's saved `Ctrl_LQR.mat` (`fixtures/ctrl_lqr.json`) before
//! being transcribed here.
//!
//! **Not the same geometry as [`crate::mpc_model`].** `InitLQR.m` uses
//! `M=5.18` (vs `ModelInit.m`'s `17.8`), `cz=cosd(60)` (vs `sind(60)`), and a
//! different direction-vector ordering (`dir2`/`dir3` swapped relative to
//! `ModelInit.m`'s `c2`/`c3`) with no `b_i` base offsets. The live app is
//! itself inconsistent here: `MainConstantTs`'s in-loop torque-allocation
//! matrix (see `crate::allocation`) is built from `ModelInit.m`-style
//! geometry, not `InitLQR.m`'s — even though the gains it's allocating
//! (`Kx`/`Ki`) were synthesized against the `InitLQR.m` model. Replicated
//! as-is; do not "fix" by unifying the two geometries.

use nalgebra::{DMatrix, Matrix5, Matrix5x3};

#[rustfmt::skip]
const AD_OP: [[f64; 5]; 5] = [
    [1.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
];
#[rustfmt::skip]
const BD_OP: [[f64; 3]; 5] = [
    [2.705_427_342_557_101e-1, 0.0, 0.0],
    [0.0, 2.705_427_342_557_101e-1, 0.0],
    [5.410_854_685_114_203e-1, 0.0, 0.0],
    [0.0, 5.410_854_685_114_203e-1, 0.0],
    [0.0, 0.0, 3.338_670_114_389_427_6e-1],
];

pub fn ad_op() -> Matrix5<f64> {
    Matrix5::from_row_iterator(AD_OP.into_iter().flatten())
}

pub fn bd_op() -> Matrix5x3<f64> {
    Matrix5x3::from_row_iterator(BD_OP.into_iter().flatten())
}

/// Tracking output matrix (`Cz` in `InitLQR.m`): picks roll, pitch.
pub fn cz() -> nalgebra::Matrix2x5<f64> {
    #[rustfmt::skip]
    let rows = [
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
    ];
    nalgebra::Matrix2x5::from_row_slice(&rows)
}

/// Augments `(Ad_op, Bd_op)` with roll/pitch integral-tracking states:
/// `Ad = [[Ad_op, 0], [Cz*Ts, I]]`, `Bd = [Bd_op; 0]` — matches `InitLQR.m`.
pub fn build_augmented(ts: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let ad_op = ad_op();
    let bd_op = bd_op();
    let cz = cz();

    let mut ad = DMatrix::<f64>::zeros(7, 7);
    ad.view_mut((0, 0), (5, 5)).copy_from(&ad_op);
    ad.view_mut((5, 0), (2, 5)).copy_from(&(cz * ts));
    ad[(5, 5)] = 1.0;
    ad[(6, 6)] = 1.0;

    let mut bd = DMatrix::<f64>::zeros(7, 3);
    bd.view_mut((0, 0), (5, 3)).copy_from(&bd_op);

    (ad, bd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn augmented_matches_ctrl_lqr_fixture_shape() {
        let (ad, bd) = build_augmented(1.0);
        assert_eq!(ad.shape(), (7, 7));
        assert_eq!(bd.shape(), (7, 3));
        // Cz*Ts row for roll -> ad[(5,0)] == 1.0
        assert!((ad[(5, 0)] - 1.0).abs() < 1e-12);
        assert!((ad[(6, 1)] - 1.0).abs() < 1e-12);
    }
}
