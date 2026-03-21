#[path = "exp_log_smooth.rs"]
mod smooth_rules;

use crate::unary::one;
use crate::ScalarAd;

pub use smooth_rules::{
    exp10, exp10_frule, exp10_rrule, exp2, exp2_frule, exp2_rrule, log10, log10_frule, log10_rrule,
    log2, log2_frule, log2_rrule,
};

/// Primal `exp`.
pub fn exp<S: ScalarAd>(x: S) -> S {
    x.exp()
}

/// Forward rule for `exp`.
pub fn exp_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp();
    (y, dx * y.conj())
}

/// Reverse rule for `exp`.
pub fn exp_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * result.conj()
}

/// Primal `exp(x) - 1`.
pub fn expm1<S: ScalarAd>(x: S) -> S {
    x.expm1()
}

/// Forward rule for `exp(x) - 1`.
pub fn expm1_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.expm1();
    let scale = y + one::<S>();
    (y, dx * scale.conj())
}

/// Reverse rule for `exp(x) - 1`.
pub fn expm1_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (result + one::<S>()).conj()
}

/// Primal `log`.
pub fn log<S: ScalarAd>(x: S) -> S {
    x.ln()
}

/// Forward rule for `log`.
pub fn log_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.ln();
    let dy = dx * (one::<S>() / x).conj();
    (y, dy)
}

/// Reverse rule for `log`.
pub fn log_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / x).conj()
}

/// Primal `log(1 + x)`.
pub fn log1p<S: ScalarAd>(x: S) -> S {
    x.log1p()
}

/// Forward rule for `log(1 + x)`.
pub fn log1p_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.log1p();
    let dy = dx * (one::<S>() / (one::<S>() + x)).conj();
    (y, dy)
}

/// Reverse rule for `log(1 + x)`.
pub fn log1p_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (one::<S>() + x)).conj()
}
