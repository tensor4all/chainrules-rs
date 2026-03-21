use crate::unary::one;
use crate::ScalarAd;

/// Primal `sin`.
pub fn sin<S: ScalarAd>(x: S) -> S {
    x.sin()
}

/// Forward rule for `sin`.
pub fn sin_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.sin();
    (y, dx * x.cos())
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
    (y, dx * -x.sin())
}

/// Reverse rule for `cos`.
pub fn cos_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (-x.sin()).conj()
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
    (y, dx * scale)
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
    let scale = -inverse_sqrt_one_minus_square(x);
    (y, dx * scale)
}

/// Reverse rule for `acos`.
pub fn acos_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (-inverse_sqrt_one_minus_square(x)).conj()
}

/// Primal `atan`.
pub fn atan<S: ScalarAd>(x: S) -> S {
    x.atan()
}

/// Forward rule for `atan`.
pub fn atan_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.atan();
    let scale = one::<S>() / (one::<S>() + x * x);
    (y, dx * scale)
}

/// Reverse rule for `atan`.
pub fn atan_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (one::<S>() + x * x)).conj()
}

#[doc = "Primal `tan`.\n\n# Examples\n```rust\nuse chainrules::tan;\n\nassert!((tan(0.5_f64) - 0.5_f64.tan()).abs() < 1e-12);\n```"]
pub fn tan<S: ScalarAd>(x: S) -> S {
    x.tan()
}

#[doc = "Forward rule for `tan`.\n\n# Examples\n```rust\nuse chainrules::tan_frule;\n\nlet (y, dy) = tan_frule(0.25_f64, 1.0);\nassert!((dy - (1.0 + 0.25_f64.tan().powi(2))).abs() < 1e-12);\n```"]
pub fn tan_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.tan();
    (y, dx * (one::<S>() + y * y))
}

#[doc = "Reverse rule for `tan`.\n\n# Examples\n```rust\nuse chainrules::tan_rrule;\n\nlet dy = tan_rrule(0.25_f64.tan(), 1.0);\nassert!((dy - (1.0 + 0.25_f64.tan().powi(2))).abs() < 1e-12);\n```"]
pub fn tan_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (one::<S>() + result * result).conj()
}

#[doc = "Primal `sincos`.\n\n# Examples\n```rust\nuse chainrules::sincos;\n\nlet (s, c) = sincos(0.5_f64);\nassert!((s - 0.5_f64.sin()).abs() < 1e-12);\nassert!((c - 0.5_f64.cos()).abs() < 1e-12);\n```"]
pub fn sincos<S: ScalarAd>(x: S) -> (S, S) {
    (x.sin(), x.cos())
}

#[doc = "Forward rule for `sincos`.\n\n# Examples\n```rust\nuse chainrules::sincos_frule;\n\nlet ((s, c), (ds, dc)) = sincos_frule(0.25_f64, 1.0);\nassert!((ds - 0.25_f64.cos()).abs() < 1e-12);\nassert!((dc + 0.25_f64.sin()).abs() < 1e-12);\n```"]
pub fn sincos_frule<S: ScalarAd>(x: S, dx: S) -> ((S, S), (S, S)) {
    let sin_x = x.sin();
    let cos_x = x.cos();
    ((sin_x, cos_x), (dx * cos_x, dx * -sin_x))
}

#[doc = "Reverse rule for `sincos`.\n\n# Examples\n```rust\nuse chainrules::sincos_rrule;\n\nlet dx = sincos_rrule(0.25_f64, (1.0, 1.0));\nassert!((dx - (0.25_f64.cos() - 0.25_f64.sin())).abs() < 1e-12);\n```"]
pub fn sincos_rrule<S: ScalarAd>(x: S, cotangents: (S, S)) -> S {
    let (cotangent_sin, cotangent_cos) = cotangents;
    let sin_x = x.sin();
    let cos_x = x.cos();
    cotangent_sin * cos_x.conj() + cotangent_cos * (-sin_x).conj()
}
