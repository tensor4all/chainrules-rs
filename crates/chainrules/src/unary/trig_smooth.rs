use crate::unary::one;
use crate::ScalarAd;

/// Primal `tan`.
///
/// # Examples
///
/// ```rust
/// use chainrules::tan;
///
/// assert!((tan(0.5_f64) - 0.5_f64.tan()).abs() < 1e-12);
/// ```
pub fn tan<S: ScalarAd>(x: S) -> S {
    x.tan()
}

/// Forward rule for `tan`.
///
/// # Examples
///
/// ```rust
/// use chainrules::tan_frule;
///
/// let (y, dy) = tan_frule(0.25_f64, 1.0);
/// assert!((dy - (1.0 + 0.25_f64.tan().powi(2))).abs() < 1e-12);
/// ```
pub fn tan_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.tan();
    (y, dx * (one::<S>() + y * y).conj())
}

/// Reverse rule for `tan`.
///
/// # Examples
///
/// ```rust
/// use chainrules::tan_rrule;
///
/// let dy = tan_rrule(0.25_f64.tan(), 1.0);
/// assert!((dy - (1.0 + 0.25_f64.tan().powi(2))).abs() < 1e-12);
/// ```
pub fn tan_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (one::<S>() + result * result).conj()
}

/// Primal `sincos`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sincos;
///
/// let (s, c) = sincos(0.5_f64);
/// assert!((s - 0.5_f64.sin()).abs() < 1e-12);
/// assert!((c - 0.5_f64.cos()).abs() < 1e-12);
/// ```
pub fn sincos<S: ScalarAd>(x: S) -> (S, S) {
    (x.sin(), x.cos())
}

/// Forward rule for `sincos`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sincos_frule;
///
/// let ((s, c), (ds, dc)) = sincos_frule(0.25_f64, 1.0);
/// assert!((ds - 0.25_f64.cos()).abs() < 1e-12);
/// assert!((dc + 0.25_f64.sin()).abs() < 1e-12);
/// ```
pub fn sincos_frule<S: ScalarAd>(x: S, dx: S) -> ((S, S), (S, S)) {
    let sin_x = x.sin();
    let cos_x = x.cos();
    ((sin_x, cos_x), (dx * cos_x.conj(), dx * (-sin_x).conj()))
}

/// Reverse rule for `sincos`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sincos_rrule;
///
/// let dx = sincos_rrule(0.25_f64, 1.0, 1.0);
/// assert!((dx - (0.25_f64.cos() - 0.25_f64.sin())).abs() < 1e-12);
/// ```
pub fn sincos_rrule<S: ScalarAd>(x: S, cotangent_sin: S, cotangent_cos: S) -> S {
    let sin_x = x.sin();
    let cos_x = x.cos();
    cotangent_sin * cos_x.conj() + cotangent_cos * (-sin_x).conj()
}
