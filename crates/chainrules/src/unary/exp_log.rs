use crate::unary::one;
use crate::ScalarAd;
use num_traits::FloatConst;
fn ln_2<S: ScalarAd>() -> S {
    S::from_real(S::Real::LN_2())
}
fn ln_10<S: ScalarAd>() -> S {
    S::from_real(S::Real::LN_10())
}
/// Primal `exp`.
pub fn exp<S: ScalarAd>(x: S) -> S {
    x.exp()
}
/// Forward rule for `exp`.
pub fn exp_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp();
    (y, dx * y)
}
/// Reverse rule for `exp`.
///
/// Takes the forward **result** `exp(x)`, not the input `x`.
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
    (y, dx * scale)
}
/// Reverse rule for `exp(x) - 1`.
///
/// Takes the forward **result** `expm1(x)`, not the input `x`.
pub fn expm1_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (result + one::<S>()).conj()
}
#[doc = "Primal `2^x`.\n\n# Examples\n```rust\nuse chainrules::exp2;\n\nassert!((exp2(3.0_f64) - 8.0).abs() < 1e-12);\n```"]
pub fn exp2<S: ScalarAd>(x: S) -> S {
    x.exp2()
}
#[doc = "Forward rule for `2^x`.\n\n# Examples\n```rust\nuse chainrules::exp2_frule;\n\nlet (y, dy) = exp2_frule(3.0_f64, 1.0);\nassert!((y - 8.0).abs() < 1e-12);\nassert!((dy - 8.0_f64 * std::f64::consts::LN_2).abs() < 1e-12);\n```"]
pub fn exp2_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp2();
    (y, dx * (y * ln_2::<S>()))
}
#[doc = "Reverse rule for `2^x`.\n\n# Examples\n```rust\nuse chainrules::exp2_rrule;\n\nlet dy = exp2_rrule(8.0_f64, 1.0);\nassert!((dy - 8.0_f64 * std::f64::consts::LN_2).abs() < 1e-12);\n```"]
pub fn exp2_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (result * ln_2::<S>()).conj()
}
#[doc = "Primal `10^x`.\n\n# Examples\n```rust\nuse chainrules::exp10;\n\nassert!((exp10(2.0_f64) - 100.0).abs() < 1e-12);\n```"]
pub fn exp10<S: ScalarAd>(x: S) -> S {
    x.exp10()
}
#[doc = "Forward rule for `10^x`.\n\n# Examples\n```rust\nuse chainrules::exp10_frule;\n\nlet (y, dy) = exp10_frule(2.0_f64, 0.5);\nassert!((y - 100.0).abs() < 1e-12);\nassert!((dy - 0.5_f64 * 100.0_f64 * std::f64::consts::LN_10).abs() < 1e-12);\n```"]
pub fn exp10_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp10();
    (y, dx * (y * ln_10::<S>()))
}
#[doc = "Reverse rule for `10^x`.\n\n# Examples\n```rust\nuse chainrules::exp10_rrule;\n\nlet dy = exp10_rrule(100.0_f64, 0.5);\nassert!((dy - 0.5_f64 * 100.0_f64 * std::f64::consts::LN_10).abs() < 1e-12);\n```"]
pub fn exp10_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (result * ln_10::<S>()).conj()
}
/// Primal `log`.
pub fn log<S: ScalarAd>(x: S) -> S {
    x.ln()
}
/// Forward rule for `log`.
pub fn log_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.ln();
    let dy = dx * (one::<S>() / x);
    (y, dy)
}
/// Reverse rule for `log`.
///
/// Takes the original **input** `x`, not the result.
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
    let dy = dx * (one::<S>() / (one::<S>() + x));
    (y, dy)
}
/// Reverse rule for `log(1 + x)`.
///
/// Takes the original **input** `x`, not the result.
pub fn log1p_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (one::<S>() + x)).conj()
}
#[doc = "Primal `log2`.\n\n# Examples\n```rust\nuse chainrules::log2;\n\nassert_eq!(log2(8.0_f64), 3.0);\n```"]
pub fn log2<S: ScalarAd>(x: S) -> S {
    x.log2()
}
#[doc = "Forward rule for `log2`.\n\n# Examples\n```rust\nuse chainrules::log2_frule;\n\nlet (y, dy) = log2_frule(8.0_f64, 2.0);\nassert!((y - 3.0).abs() < 1e-12);\nassert!((dy - (2.0_f64 / (8.0_f64 * std::f64::consts::LN_2))).abs() < 1e-12);\n```"]
pub fn log2_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.log2();
    let scale = one::<S>() / (x * ln_2::<S>());
    (y, dx * scale)
}
#[doc = "Reverse rule for `log2`.\n\n# Examples\n```rust\nuse chainrules::log2_rrule;\n\nlet dy = log2_rrule(8.0_f64, 2.0);\nassert!((dy - (2.0_f64 / (8.0_f64 * std::f64::consts::LN_2))).abs() < 1e-12);\n```"]
pub fn log2_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (x * ln_2::<S>())).conj()
}
#[doc = "Primal `log10`.\n\n# Examples\n```rust\nuse chainrules::log10;\n\nassert_eq!(log10(100.0_f64), 2.0);\n```"]
pub fn log10<S: ScalarAd>(x: S) -> S {
    x.log10()
}
#[doc = "Forward rule for `log10`.\n\n# Examples\n```rust\nuse chainrules::log10_frule;\n\nlet (y, dy) = log10_frule(100.0_f64, 2.0);\nassert!((y - 2.0).abs() < 1e-12);\nassert!((dy - (2.0_f64 / (100.0_f64 * std::f64::consts::LN_10))).abs() < 1e-12);\n```"]
pub fn log10_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.log10();
    let scale = one::<S>() / (x * ln_10::<S>());
    (y, dx * scale)
}
#[doc = "Reverse rule for `log10`.\n\n# Examples\n```rust\nuse chainrules::log10_rrule;\n\nlet dy = log10_rrule(100.0_f64, 2.0);\nassert!((dy - (2.0_f64 / (100.0_f64 * std::f64::consts::LN_10))).abs() < 1e-12);\n```"]
pub fn log10_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (x * ln_10::<S>())).conj()
}
