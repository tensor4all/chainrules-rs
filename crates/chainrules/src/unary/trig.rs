use crate::unary::{neg_one, one};
use crate::ScalarAd;

/// Primal `sin`.
pub fn sin<S: ScalarAd>(x: S) -> S {
    x.sin()
}

/// Forward rule for `sin`.
pub fn sin_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.sin();
    (y, dx * x.cos().conj())
}

/// Reverse rule for `sin`.
pub fn sin_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * x.cos().conj()
}

/// Primal `cos`.
pub fn cos<S: ScalarAd>(x: S) -> S {
    x.cos()
}

/// Forward rule for `cos`.
pub fn cos_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.cos();
    (y, dx * (neg_one::<S>() * x.sin()).conj())
}

/// Reverse rule for `cos`.
pub fn cos_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (neg_one::<S>() * x.sin()).conj()
}

fn inverse_sqrt_one_minus_square<S: ScalarAd>(x: S) -> S {
    one::<S>() / (one::<S>() - x * x).sqrt()
}

/// Primal `asin`.
pub fn asin<S: ScalarAd>(x: S) -> S {
    x.asin()
}

/// Forward rule for `asin`.
pub fn asin_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.asin();
    let scale = inverse_sqrt_one_minus_square(x);
    (y, dx * scale.conj())
}

/// Reverse rule for `asin`.
pub fn asin_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * inverse_sqrt_one_minus_square(x).conj()
}

/// Primal `acos`.
pub fn acos<S: ScalarAd>(x: S) -> S {
    x.acos()
}

/// Forward rule for `acos`.
pub fn acos_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.acos();
    let scale = neg_one::<S>() * inverse_sqrt_one_minus_square(x);
    (y, dx * scale.conj())
}

/// Reverse rule for `acos`.
pub fn acos_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (neg_one::<S>() * inverse_sqrt_one_minus_square(x)).conj()
}

/// Primal `atan`.
pub fn atan<S: ScalarAd>(x: S) -> S {
    x.atan()
}

/// Forward rule for `atan`.
pub fn atan_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.atan();
    let scale = one::<S>() / (one::<S>() + x * x);
    (y, dx * scale.conj())
}

/// Reverse rule for `atan`.
pub fn atan_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (one::<S>() + x * x)).conj()
}
