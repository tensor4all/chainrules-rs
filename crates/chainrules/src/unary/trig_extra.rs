use crate::binary::{mul_frule, mul_rrule};
use crate::unary::{
    cos, cos_frule, cos_rrule, inv, inv_frule, inv_rrule, sin, sin_frule, sin_rrule, sincos, tan,
    tan_frule, tan_rrule,
};
use crate::ScalarAd;
use num_traits::{Float, FloatConst, Zero};

fn pi<S: ScalarAd>() -> S {
    S::from_real(S::Real::PI())
}

fn real<R: Float>(value: f64) -> R {
    match R::from(value) {
        Some(value) => value,
        None => unreachable!("float constant conversion should succeed"),
    }
}

fn deg2rad<S: ScalarAd>() -> S {
    pi::<S>() / S::from_real(real::<S::Real>(180.0))
}

fn real_input<S: ScalarAd>(x: S) -> Option<S::Real> {
    if x.imag().is_zero() {
        Some(x.real())
    } else {
        None
    }
}

fn sinpi_real<R: Float + FloatConst>(x: R) -> R {
    let two = real::<R>(2.0);
    let reduced = x - (x / two).floor() * two;
    let zero = real::<R>(0.0);
    let one = real::<R>(1.0);
    let half = real::<R>(0.5);
    let three_half = real::<R>(1.5);
    if reduced == zero || reduced == one {
        zero
    } else if reduced == half {
        one
    } else if reduced == three_half {
        -one
    } else {
        (R::PI() * reduced).sin()
    }
}

fn cospi_real<R: Float + FloatConst>(x: R) -> R {
    let two = real::<R>(2.0);
    let reduced = x - (x / two).floor() * two;
    let zero = real::<R>(0.0);
    let one = real::<R>(1.0);
    let half = real::<R>(0.5);
    let three_half = real::<R>(1.5);
    if reduced == zero {
        one
    } else if reduced == one {
        -one
    } else if reduced == half || reduced == three_half {
        zero
    } else {
        (R::PI() * reduced).cos()
    }
}

fn tand_real<R: Float + FloatConst>(x: R) -> R {
    let one_eighty = real::<R>(180.0);
    let reduced = x - (x / one_eighty).floor() * one_eighty;
    let zero = real::<R>(0.0);
    let forty_five = real::<R>(45.0);
    let ninety = real::<R>(90.0);
    let one_thirty_five = real::<R>(135.0);
    if reduced == zero {
        zero
    } else if reduced == forty_five {
        real::<R>(1.0)
    } else if reduced == ninety {
        R::infinity()
    } else if reduced == one_thirty_five {
        real::<R>(-1.0)
    } else {
        (R::PI() * reduced / one_eighty).tan()
    }
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
    if let Some(x_real) = real_input(x) {
        return S::from_real(sinpi_real(x_real));
    }
    sincos(pi::<S>() * x).0
}

/// Forward rule for `sinpi`.
pub fn sinpi_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = sinpi(x);
    let scale = pi::<S>() * cospi(x);
    (y, dx * scale.conj())
}

/// Reverse rule for `sinpi`.
pub fn sinpi_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (pi::<S>() * cospi(x)).conj()
}

/// Primal `cospi`.
pub fn cospi<S: ScalarAd>(x: S) -> S {
    if let Some(x_real) = real_input(x) {
        return S::from_real(cospi_real(x_real));
    }
    sincos(pi::<S>() * x).1
}

/// Forward rule for `cospi`.
pub fn cospi_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = cospi(x);
    let scale = -(pi::<S>() * sinpi(x));
    (y, dx * scale.conj())
}

/// Reverse rule for `cospi`.
pub fn cospi_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (-(pi::<S>() * sinpi(x))).conj()
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
    (sinpi(x), cospi(x))
}

/// Forward rule for `sincospi`.
pub fn sincospi_frule<S: ScalarAd>(x: S, dx: S) -> ((S, S), (S, S)) {
    let sin_x = sinpi(x);
    let cos_x = cospi(x);
    (
        (sin_x, cos_x),
        (
            dx * (pi::<S>() * cos_x).conj(),
            dx * (-(pi::<S>() * sin_x)).conj(),
        ),
    )
}

/// Reverse rule for `sincospi`.
pub fn sincospi_rrule<S: ScalarAd>(x: S, cotangents: (S, S)) -> S {
    let (cotangent_sin, cotangent_cos) = cotangents;
    sinpi_rrule(x, cotangent_sin) + cospi_rrule(x, cotangent_cos)
}

/// Primal `sind`.
pub fn sind<S: ScalarAd>(x: S) -> S {
    sinpi(x / S::from_real(real::<S::Real>(180.0)))
}

/// Forward rule for `sind`.
pub fn sind_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let scale = S::from_real(real::<S::Real>(1.0 / 180.0));
    let (scaled_x, dscaled_x) = mul_frule(scale, x, S::from_i32(0), dx);
    sinpi_frule(scaled_x, dscaled_x)
}

/// Reverse rule for `sind`.
pub fn sind_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let scale = S::from_real(real::<S::Real>(1.0 / 180.0));
    let scaled_x = scale * x;
    let dscaled_x = sinpi_rrule(scaled_x, cotangent);
    let (_, dx) = mul_rrule(scale, x, dscaled_x);
    dx
}

/// Primal `cosd`.
pub fn cosd<S: ScalarAd>(x: S) -> S {
    cospi(x / S::from_real(real::<S::Real>(180.0)))
}

/// Forward rule for `cosd`.
pub fn cosd_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let scale = S::from_real(real::<S::Real>(1.0 / 180.0));
    let (scaled_x, dscaled_x) = mul_frule(scale, x, S::from_i32(0), dx);
    cospi_frule(scaled_x, dscaled_x)
}

/// Reverse rule for `cosd`.
pub fn cosd_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let scale = S::from_real(real::<S::Real>(1.0 / 180.0));
    let scaled_x = scale * x;
    let dscaled_x = cospi_rrule(scaled_x, cotangent);
    let (_, dx) = mul_rrule(scale, x, dscaled_x);
    dx
}

/// Primal `tand`.
pub fn tand<S: ScalarAd>(x: S) -> S {
    if let Some(x_real) = real_input(x) {
        return S::from_real(tand_real(x_real));
    }
    tan(deg2rad::<S>() * x)
}

/// Forward rule for `tand`.
pub fn tand_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = tand(x);
    let scale = deg2rad::<S>() * (S::from_i32(1) + y * y);
    (y, dx * scale.conj())
}

/// Reverse rule for `tand`.
pub fn tand_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    let y = tand(x);
    let scale = deg2rad::<S>() * (S::from_i32(1) + y * y);
    cotangent * scale.conj()
}
