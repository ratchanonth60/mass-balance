//! Linear Kalman filter — direct port of `app.KFpredict`/`app.KFupdate`
//! (identical in both `AutoMass_MPC.mlapp` and `AutoMass.mlapp`).

use nalgebra::{DMatrix, DVector};

pub fn predict(
    x: &DVector<f64>,
    p: &DMatrix<f64>,
    a: &DMatrix<f64>,
    q: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let x_pred = a * x;
    let p_pred = a * p * a.transpose() + q;
    (x_pred, p_pred)
}

pub fn update(
    x: &DVector<f64>,
    p: &DMatrix<f64>,
    y: &DVector<f64>,
    c: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let s = c * p * c.transpose() + r;
    let s_inv = s
        .clone()
        .try_inverse()
        .expect("KF innovation covariance singular");
    let k = p * c.transpose() * s_inv;
    let v = y - c * x;
    let x_new = x + &k * v;
    let p_new = p - &k * &s * k.transpose();
    (x_new, p_new)
}

/// Divergence guard matching `MainConstantTs11.m`'s `badKF` checks: NaN/Inf
/// in the updated state, `|roll|`/`|pitch|` > 50deg, or a >20deg jump in
/// roll/pitch from the previous estimate (only checked from the 2nd update
/// onward, `j > 2` in the MATLAB indexing).
pub fn is_diverged(x_kf: &DVector<f64>, x_prev: Option<&DVector<f64>>) -> bool {
    if x_kf.iter().any(|v| !v.is_finite()) {
        return true;
    }
    let deg = |rad: f64| rad.to_degrees();
    if deg(x_kf[0]).abs() > 50.0 || deg(x_kf[1]).abs() > 50.0 {
        return true;
    }
    if let Some(prev) = x_prev {
        let d_roll = deg(x_kf[0] - prev[0]).abs();
        let d_pitch = deg(x_kf[1] - prev[1]).abs();
        if d_roll > 20.0 || d_pitch > 20.0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn predict_update_identity_model() {
        let x = DVector::from_vec(vec![0.0; 5]);
        let p = DMatrix::identity(5, 5) * 1e4;
        let a = DMatrix::identity(5, 5);
        let q = DMatrix::identity(5, 5);
        let (xp, pp) = predict(&x, &p, &a, &q);
        assert_eq!(xp, x);
        assert!(pp[(0, 0)] > p[(0, 0)]); // covariance grows on predict

        let y = DVector::from_vec(vec![0.1, 0.0, 0.0, 0.0, 0.0]);
        let c = DMatrix::identity(5, 5);
        let r = DMatrix::identity(5, 5);
        let (xu, _) = update(&xp, &pp, &y, &c, &r);
        assert!(xu[0] > 0.0 && xu[0] < 0.1); // pulled toward measurement
    }

    #[test]
    fn diverged_on_nan_and_large_angle_and_jump() {
        let bad_nan = DVector::from_vec(vec![f64::NAN, 0.0, 0.0, 0.0, 0.0]);
        assert!(is_diverged(&bad_nan, None));

        let bad_angle = DVector::from_vec(vec![60f64.to_radians(), 0.0, 0.0, 0.0, 0.0]);
        assert!(is_diverged(&bad_angle, None));

        let prev = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let jumped = DVector::from_vec(vec![25f64.to_radians(), 0.0, 0.0, 0.0, 0.0]);
        assert!(is_diverged(&jumped, Some(&prev)));

        let fine = DVector::from_vec(vec![1f64.to_radians(), 0.0, 0.0, 0.0, 0.0]);
        assert!(!is_diverged(&fine, Some(&prev)));
    }
}
