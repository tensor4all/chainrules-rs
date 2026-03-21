mod basic;
mod complex_parts;
mod exp_log;
mod hyperbolic;
mod hyperbolic_extra;
mod roots;
mod smooth;
mod trig;
mod trig_extra;

use crate::ScalarAd;

fn one<S: ScalarAd>() -> S {
    S::from_i32(1)
}

pub use basic::{conj, conj_frule, conj_rrule, sqrt, sqrt_frule, sqrt_rrule};
pub use complex_parts::{
    abs, abs2, abs2_frule, abs2_rrule, angle, angle_rrule, complex, imag, imag_rrule, real,
    real_rrule,
};
pub use exp_log::{
    exp, exp10, exp10_frule, exp10_rrule, exp2, exp2_frule, exp2_rrule, exp_frule, exp_rrule,
    expm1, expm1_frule, expm1_rrule, log, log10, log10_frule, log10_rrule, log1p, log1p_frule,
    log1p_rrule, log2, log2_frule, log2_rrule, log_frule, log_rrule,
};
pub use hyperbolic::{
    acosh, acosh_frule, acosh_rrule, asinh, asinh_frule, asinh_rrule, atanh, atanh_frule,
    atanh_rrule, cosh, cosh_frule, cosh_rrule, sinh, sinh_frule, sinh_rrule, tanh, tanh_frule,
    tanh_rrule,
};
pub use hyperbolic_extra::{
    coth, coth_frule, coth_rrule, csch, csch_frule, csch_rrule, sech, sech_frule, sech_rrule,
};
pub use roots::{cbrt, cbrt_frule, cbrt_rrule, inv, inv_frule, inv_rrule};
pub use smooth::{
    hypot, hypot_frule, hypot_rrule, pow, pow_frule, pow_rrule, sincos, sincos_frule, sincos_rrule,
    tan, tan_frule, tan_rrule,
};
pub use trig::{
    acos, acos_frule, acos_rrule, asin, asin_frule, asin_rrule, atan, atan_frule, atan_rrule, cos,
    cos_frule, cos_rrule, sin, sin_frule, sin_rrule,
};
pub use trig_extra::{
    cosd, cosd_frule, cosd_rrule, cospi, cospi_frule, cospi_rrule, cot, cot_frule, cot_rrule, csc,
    csc_frule, csc_rrule, sec, sec_frule, sec_rrule, sincospi, sincospi_frule, sincospi_rrule,
    sind, sind_frule, sind_rrule, sinpi, sinpi_frule, sinpi_rrule, tand, tand_frule, tand_rrule,
};
