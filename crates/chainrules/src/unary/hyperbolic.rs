use crate::unary::one;
use crate::ScalarAd;

/// Primal `tanh`.
pub fn tanh<S: ScalarAd>(x: S) -> S {
    x.tanh()
}

/// Forward rule for `tanh`.
pub fn tanh_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.tanh();
    let scale = one::<S>() - y * y;
    (y, dx * scale)
}

/// Reverse rule for `tanh`.
///
/// Takes the forward **result** `tanh(x)`, not the input `x`.
pub fn tanh_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (one::<S>() - result * result).conj()
}

/// Primal `sinh`.
pub fn sinh<S: ScalarAd>(x: S) -> S {
    x.sinh()
}

/// Forward rule for `sinh`.
pub fn sinh_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.sinh();
    (y, dx * x.cosh())
}

/// Reverse rule for `sinh`.
///
/// Takes the original **input** `x`, not the result.
pub fn sinh_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * x.cosh().conj()
}

/// Primal `cosh`.
pub fn cosh<S: ScalarAd>(x: S) -> S {
    x.cosh()
}

/// Forward rule for `cosh`.
pub fn cosh_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.cosh();
    (y, dx * x.sinh())
}

/// Reverse rule for `cosh`.
///
/// Takes the original **input** `x`, not the result.
pub fn cosh_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * x.sinh().conj()
}

fn inverse_sqrt_one_plus_square<S: ScalarAd>(x: S) -> S {
    one::<S>() / (one::<S>() + x * x).sqrt()
}

/// Primal `asinh`.
pub fn asinh<S: ScalarAd>(x: S) -> S {
    x.asinh()
}

/// Forward rule for `asinh`.
pub fn asinh_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.asinh();
    let scale = inverse_sqrt_one_plus_square(x);
    (y, dx * scale)
}

/// Reverse rule for `asinh`.
pub fn asinh_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * inverse_sqrt_one_plus_square(x).conj()
}

fn inverse_acosh_scale<S: ScalarAd>(x: S) -> S {
    one::<S>() / ((x - one::<S>()).sqrt() * (x + one::<S>()).sqrt())
}

/// Primal `acosh`.
pub fn acosh<S: ScalarAd>(x: S) -> S {
    x.acosh()
}

/// Forward rule for `acosh`.
pub fn acosh_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.acosh();
    let scale = inverse_acosh_scale(x);
    (y, dx * scale)
}

/// Reverse rule for `acosh`.
pub fn acosh_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * inverse_acosh_scale(x).conj()
}

/// Primal `atanh`.
pub fn atanh<S: ScalarAd>(x: S) -> S {
    x.atanh()
}

/// Forward rule for `atanh`.
pub fn atanh_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.atanh();
    let scale = one::<S>() / (one::<S>() - x * x);
    (y, dx * scale)
}

/// Reverse rule for `atanh`.
pub fn atanh_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (one::<S>() - x * x)).conj()
}
