use crate::unary::{
    cosh, cosh_frule, cosh_rrule, inv, inv_frule, inv_rrule, sinh, sinh_frule, sinh_rrule, tanh,
    tanh_frule, tanh_rrule,
};
use crate::ScalarAd;

/// Primal `sech`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sech;
///
/// assert!((sech(0.5_f64) - 1.0 / 0.5_f64.cosh()).abs() < 1e-12);
/// ```
pub fn sech<S: ScalarAd>(x: S) -> S {
    inv(cosh(x))
}

/// Forward rule for `sech`.
pub fn sech_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let (y, dy) = cosh_frule(x, dx);
    inv_frule(y, dy)
}

/// Reverse rule for `sech`.
pub fn sech_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let y = sech(x);
    let d_y = inv_rrule(y, cotangent);
    cosh_rrule(x, d_y)
}

/// Primal `csch`.
pub fn csch<S: ScalarAd>(x: S) -> S {
    inv(sinh(x))
}

/// Forward rule for `csch`.
pub fn csch_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let (y, dy) = sinh_frule(x, dx);
    inv_frule(y, dy)
}

/// Reverse rule for `csch`.
pub fn csch_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let y = csch(x);
    let d_y = inv_rrule(y, cotangent);
    sinh_rrule(x, d_y)
}

/// Primal `coth`.
pub fn coth<S: ScalarAd>(x: S) -> S {
    inv(tanh(x))
}

/// Forward rule for `coth`.
pub fn coth_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let (y, dy) = tanh_frule(x, dx);
    inv_frule(y, dy)
}

/// Reverse rule for `coth`.
pub fn coth_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let y = coth(x);
    let d_y = inv_rrule(y, cotangent);
    tanh_rrule(tanh(x), d_y)
}
