#![doc = include_str!("../README.md")]

pub use chainrules_core::{
    AdResult, AutodiffError, Differentiable, ForwardRule, NodeId, PullbackEntry,
    PullbackWithTangentsEntry, ReverseRule, SavePolicy,
};

mod binary;
mod binary_special;
mod power;
mod real_ops;
mod scalar_ad;
mod unary;

#[doc(inline)]
pub use binary::{
    add, add_frule, add_rrule, div, div_frule, div_rrule, mul, mul_frule, mul_rrule, sub,
    sub_frule, sub_rrule,
};
#[doc(inline)]
pub use binary_special::{max, max_frule, max_rrule, min, min_frule, min_rrule};
#[doc(inline)]
pub use power::{powf, powf_frule, powf_rrule, powi, powi_frule, powi_rrule};
#[doc(inline)]
pub use real_ops::{atan2, atan2_frule, atan2_rrule};
#[doc(inline)]
pub use scalar_ad::{handle_r_to_c_f32, handle_r_to_c_f64, ScalarAd};
#[doc(inline)]
pub use unary::{
    abs, abs2, abs2_frule, abs2_rrule, acos, acos_frule, acos_rrule, acosh, acosh_frule,
    acosh_rrule, angle, angle_rrule, asin, asin_frule, asin_rrule, asinh, asinh_frule, asinh_rrule,
    atan, atan_frule, atan_rrule, atanh, atanh_frule, atanh_rrule, cbrt, cbrt_frule, cbrt_rrule,
    ceil, ceil_frule, ceil_rrule, complex, conj, conj_frule, conj_rrule, cos, cos_frule, cos_rrule,
    cosd, cosd_frule, cosd_rrule, cosh, cosh_frule, cosh_rrule, cospi, cospi_frule, cospi_rrule,
    cot, cot_frule, cot_rrule, coth, coth_frule, coth_rrule, csc, csc_frule, csc_rrule, csch,
    csch_frule, csch_rrule, exp, exp10, exp10_frule, exp10_rrule, exp2, exp2_frule, exp2_rrule,
    exp_frule, exp_rrule, expm1, expm1_frule, expm1_rrule, floor, floor_frule, floor_rrule, hypot,
    hypot_frule, hypot_rrule, imag, imag_rrule, inv, inv_frule, inv_rrule, log, log10, log10_frule,
    log10_rrule, log1p, log1p_frule, log1p_rrule, log2, log2_frule, log2_rrule, log_frule,
    log_rrule, pow, pow_frule, pow_rrule, real, real_rrule, round, round_frule, round_rrule, sec,
    sec_frule, sec_rrule, sech, sech_frule, sech_rrule, sign, sign_frule, sign_rrule, sin,
    sin_frule, sin_rrule, sincos, sincos_frule, sincos_rrule, sincospi, sincospi_frule,
    sincospi_rrule, sind, sind_frule, sind_rrule, sinh, sinh_frule, sinh_rrule, sinpi, sinpi_frule,
    sinpi_rrule, sqrt, sqrt_frule, sqrt_rrule, tan, tan_frule, tan_rrule, tand, tand_frule,
    tand_rrule, tanh, tanh_frule, tanh_rrule,
};

#[cfg(test)]
mod tests;
