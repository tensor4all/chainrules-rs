use crate::binary::{mul_frule, mul_rrule};
use crate::unary::{
    cos, cos_frule, cos_rrule, inv, inv_frule, inv_rrule, sin, sin_frule, sin_rrule, sincos,
    sincos_frule, sincos_rrule, tan, tan_frule, tan_rrule,
};
use crate::ScalarAd;
use num_traits::FloatConst;

fn pi<S: ScalarAd>() -> S {
    S::from_real(S::Real::PI())
}

fn deg2rad<S: ScalarAd>() -> S {
    pi::<S>() / S::from_i32(180)
}

/// Primal `sec`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sec;
///
/// assert!((sec(0.5_f64) - 1.0 / 0.5_f64.cos()).abs() < 1e-12);
/// ```
pub fn sec<S: ScalarAd>(x: S) -> S {
    inv(cos(x))
}

/// Forward rule for `sec`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sec_frule;
///
/// let (y, dy) = sec_frule(0.5_f64, 1.0);
/// assert!((y - 1.0 / 0.5_f64.cos()).abs() < 1e-12);
/// assert!((dy - (0.5_f64.sin() / 0.5_f64.cos().powi(2))).abs() < 1e-12);
/// ```
pub fn sec_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let (y, dy) = cos_frule(x, dx);
    inv_frule(y, dy)
}

/// Reverse rule for `sec`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sec_rrule;
///
/// let dy = sec_rrule(0.5_f64, 1.0);
/// assert!((dy - (0.5_f64.sin() / 0.5_f64.cos().powi(2))).abs() < 1e-12);
/// ```
pub fn sec_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let y = sec(x);
    let d_y = inv_rrule(y, cotangent);
    cos_rrule(x, d_y)
}

/// Primal `csc`.
pub fn csc<S: ScalarAd>(x: S) -> S {
    inv(sin(x))
}

/// Forward rule for `csc`.
pub fn csc_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let (y, dy) = sin_frule(x, dx);
    inv_frule(y, dy)
}

/// Reverse rule for `csc`.
pub fn csc_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let y = csc(x);
    let d_y = inv_rrule(y, cotangent);
    sin_rrule(x, d_y)
}

/// Primal `cot`.
pub fn cot<S: ScalarAd>(x: S) -> S {
    inv(tan(x))
}

/// Forward rule for `cot`.
pub fn cot_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let (y, dy) = tan_frule(x, dx);
    inv_frule(y, dy)
}

/// Reverse rule for `cot`.
pub fn cot_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let y = cot(x);
    let d_y = inv_rrule(y, cotangent);
    tan_rrule(tan(x), d_y)
}

/// Primal `sinpi`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sinpi;
///
/// assert!((sinpi(0.25_f64) - 0.25_f64.mul_add(std::f64::consts::PI, 0.0).sin()).abs() < 1e-12);
/// ```
pub fn sinpi<S: ScalarAd>(x: S) -> S {
    sincospi(x).0
}

/// Forward rule for `sinpi`.
pub fn sinpi_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let ((y, _), (dy, _)) = sincospi_frule(x, dx);
    (y, dy)
}

/// Reverse rule for `sinpi`.
pub fn sinpi_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    sincospi_rrule(x, (cotangent, S::from_i32(0)))
}

/// Primal `cospi`.
pub fn cospi<S: ScalarAd>(x: S) -> S {
    sincospi(x).1
}

/// Forward rule for `cospi`.
pub fn cospi_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let ((_, y), (_, dy)) = sincospi_frule(x, dx);
    (y, dy)
}

/// Reverse rule for `cospi`.
pub fn cospi_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    sincospi_rrule(x, (S::from_i32(0), cotangent))
}

/// Primal `sincospi`.
///
/// # Examples
///
/// ```rust
/// use chainrules::sincospi;
///
/// let (s, c) = sincospi(0.25_f64);
/// assert!((s - (std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-12);
/// assert!((c - (std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-12);
/// ```
pub fn sincospi<S: ScalarAd>(x: S) -> (S, S) {
    sincos(pi::<S>() * x)
}

/// Forward rule for `sincospi`.
pub fn sincospi_frule<S: ScalarAd>(x: S, dx: S) -> ((S, S), (S, S)) {
    let scale = pi::<S>();
    let (scaled_x, dscaled_x) = mul_frule(scale, x, S::from_i32(0), dx);
    sincos_frule(scaled_x, dscaled_x)
}

/// Reverse rule for `sincospi`.
pub fn sincospi_rrule<S: ScalarAd>(x: S, cotangents: (S, S)) -> S {
    let scale = pi::<S>();
    let scaled_x = scale * x;
    let dscaled_x = sincos_rrule(scaled_x, cotangents);
    let (_, dx) = mul_rrule(scale, x, dscaled_x);
    dx
}

/// Primal `sind`.
pub fn sind<S: ScalarAd>(x: S) -> S {
    sin(deg2rad::<S>() * x)
}

/// Forward rule for `sind`.
pub fn sind_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let scale = deg2rad::<S>();
    let (scaled_x, dscaled_x) = mul_frule(scale, x, S::from_i32(0), dx);
    sin_frule(scaled_x, dscaled_x)
}

/// Reverse rule for `sind`.
pub fn sind_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let scale = deg2rad::<S>();
    let scaled_x = scale * x;
    let dscaled_x = sin_rrule(scaled_x, cotangent);
    let (_, dx) = mul_rrule(scale, x, dscaled_x);
    dx
}

/// Primal `cosd`.
pub fn cosd<S: ScalarAd>(x: S) -> S {
    cos(deg2rad::<S>() * x)
}

/// Forward rule for `cosd`.
pub fn cosd_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let scale = deg2rad::<S>();
    let (scaled_x, dscaled_x) = mul_frule(scale, x, S::from_i32(0), dx);
    cos_frule(scaled_x, dscaled_x)
}

/// Reverse rule for `cosd`.
pub fn cosd_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let scale = deg2rad::<S>();
    let scaled_x = scale * x;
    let dscaled_x = cos_rrule(scaled_x, cotangent);
    let (_, dx) = mul_rrule(scale, x, dscaled_x);
    dx
}

/// Primal `tand`.
pub fn tand<S: ScalarAd>(x: S) -> S {
    tan(deg2rad::<S>() * x)
}

/// Forward rule for `tand`.
pub fn tand_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let scale = deg2rad::<S>();
    let (scaled_x, dscaled_x) = mul_frule(scale, x, S::from_i32(0), dx);
    tan_frule(scaled_x, dscaled_x)
}

/// Reverse rule for `tand`.
pub fn tand_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let scale = deg2rad::<S>();
    let scaled_x = scale * x;
    let y = tan(scaled_x);
    let dscaled_x = tan_rrule(y, cotangent);
    let (_, dx) = mul_rrule(scale, x, dscaled_x);
    dx
}
