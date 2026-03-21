use num_traits::{One, Zero};

use crate::ScalarAd;

/// Primal `powf`.
///
/// # Examples
///
/// ```rust
/// use chainrules::powf;
///
/// assert_eq!(powf(2.0_f64, 3.0), 8.0);
/// ```
pub fn powf<S: ScalarAd>(x: S, exponent: S::Real) -> S {
    x.powf(exponent)
}

/// Forward rule for `powf` with fixed exponent.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules::powf_frule;
///
/// let (y, dy) = powf_frule(2.0_f64, 3.0, 1.0);
/// assert_eq!(y, 8.0);
/// assert_eq!(dy, 12.0);
/// ```
pub fn powf_frule<S: ScalarAd>(x: S, exponent: S::Real, dx: S) -> (S, S) {
    let y = x.powf(exponent);
    let dy = if exponent == S::Real::zero() {
        S::from_real(S::Real::zero())
    } else {
        dx * (S::from_real(exponent) * x.powf(exponent - S::Real::one())).conj()
    };
    (y, dy)
}

/// Reverse rule for `powf` with fixed exponent.
///
/// # Examples
///
/// ```rust
/// use chainrules::powf_rrule;
///
/// let dx = powf_rrule(2.0_f64, 3.0, 1.0);
/// assert_eq!(dx, 12.0);
/// ```
pub fn powf_rrule<S: ScalarAd>(x: S, exponent: S::Real, cotangent: S) -> S {
    if exponent == S::Real::zero() {
        return S::from_real(S::Real::zero());
    }
    cotangent * (S::from_real(exponent) * x.powf(exponent - S::Real::one())).conj()
}

/// Primal `powi`.
///
/// # Examples
///
/// ```rust
/// use chainrules::powi;
///
/// assert_eq!(powi(2.0_f64, 4), 16.0);
/// ```
pub fn powi<S: ScalarAd>(x: S, exponent: i32) -> S {
    x.powi(exponent)
}

/// Forward rule for `powi` with fixed integer exponent.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules::powi_frule;
///
/// let (y, dy) = powi_frule(2.0_f64, 4, 1.0);
/// assert_eq!(y, 16.0);
/// assert_eq!(dy, 32.0);
/// ```
pub fn powi_frule<S: ScalarAd>(x: S, exponent: i32, dx: S) -> (S, S) {
    let y = x.powi(exponent);
    let dy = if exponent == 0 {
        S::from_i32(0)
    } else {
        dx * (S::from_i32(exponent) * x.powi(exponent - 1)).conj()
    };
    (y, dy)
}

/// Reverse rule for `powi` with fixed integer exponent.
///
/// # Examples
///
/// ```rust
/// use chainrules::powi_rrule;
///
/// let dx = powi_rrule(2.0_f64, 4, 1.0);
/// assert_eq!(dx, 32.0);
/// ```
pub fn powi_rrule<S: ScalarAd>(x: S, exponent: i32, cotangent: S) -> S {
    if exponent == 0 {
        return S::from_i32(0);
    }
    cotangent * (S::from_i32(exponent) * x.powi(exponent - 1)).conj()
}

#[doc = "Primal `pow(x, exponent)`.\n\n# Examples\n```rust\nuse chainrules::pow;\n\nassert_eq!(pow(2.0_f64, 3.0_f64), 8.0);\n```"]
pub fn pow<S: ScalarAd>(x: S, exponent: S) -> S {
    x.pow(exponent)
}
fn zero<S: ScalarAd>() -> S {
    S::from_i32(0)
}
fn pow_x_scale<S: ScalarAd>(x: S, exponent: S) -> S {
    if exponent == zero::<S>() {
        zero::<S>()
    } else {
        (exponent * x.pow(exponent - S::from_i32(1))).conj()
    }
}
fn pow_exp_scale<S: ScalarAd>(x: S, exponent: S) -> S {
    if x.real() == S::Real::zero()
        && exponent.imag() == S::Real::zero()
        && exponent.real() >= S::Real::zero()
    {
        zero::<S>()
    } else {
        (x.pow(exponent) * x.ln()).conj()
    }
}
#[doc = "Forward rule for `pow(x, exponent)`.\n\n# Examples\n```rust\nuse chainrules::pow_frule;\n\nlet (y, dy) = pow_frule(2.0_f64, 3.0_f64, 1.0, 0.0);\nassert_eq!(y, 8.0);\nassert!((dy - 12.0).abs() < 1e-12);\n```"]
pub fn pow_frule<S: ScalarAd>(x: S, exponent: S, dx: S, dexponent: S) -> (S, S) {
    let y = x.pow(exponent);
    let dfdx = if dx == zero::<S>() {
        zero::<S>()
    } else {
        dx * pow_x_scale(x, exponent)
    };
    let dfde = if dexponent == zero::<S>() {
        zero::<S>()
    } else {
        dexponent * pow_exp_scale(x, exponent)
    };
    (y, dfdx + dfde)
}
#[doc = "Reverse rule for `pow(x, exponent)`.\n\n# Examples\n```rust\nuse chainrules::pow_rrule;\n\nlet (dx, dexp) = pow_rrule(2.0_f64, 3.0_f64, 1.0);\nassert_eq!(dx, 12.0);\nassert!((dexp - 8.0_f64 * std::f64::consts::LN_2).abs() < 1e-12);\n```"]
pub fn pow_rrule<S: ScalarAd>(x: S, exponent: S, cotangent: S) -> (S, S) {
    let dfdx = if cotangent == zero::<S>() {
        zero::<S>()
    } else {
        cotangent * pow_x_scale(x, exponent)
    };
    let dfde = if cotangent == zero::<S>() {
        zero::<S>()
    } else {
        cotangent * pow_exp_scale(x, exponent)
    };
    (dfdx, dfde)
}
