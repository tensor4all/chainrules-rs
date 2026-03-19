mod basic;
mod exp_log;
mod hyperbolic;
mod trig;

use crate::ScalarAd;

fn one<S: ScalarAd>() -> S {
    S::from_i32(1)
}

fn neg_one<S: ScalarAd>() -> S {
    S::from_i32(-1)
}

pub use basic::{conj, conj_frule, conj_rrule, sqrt, sqrt_frule, sqrt_rrule};
pub use exp_log::{
    exp, exp_frule, exp_rrule, expm1, expm1_frule, expm1_rrule, log, log1p, log1p_frule,
    log1p_rrule, log_frule, log_rrule,
};
pub use hyperbolic::{
    acosh, acosh_frule, acosh_rrule, asinh, asinh_frule, asinh_rrule, atanh, atanh_frule,
    atanh_rrule, cosh, cosh_frule, cosh_rrule, sinh, sinh_frule, sinh_rrule, tanh, tanh_frule,
    tanh_rrule,
};
pub use trig::{
    acos, acos_frule, acos_rrule, asin, asin_frule, asin_rrule, atan, atan_frule, atan_rrule, cos,
    cos_frule, cos_rrule, sin, sin_frule, sin_rrule,
};
