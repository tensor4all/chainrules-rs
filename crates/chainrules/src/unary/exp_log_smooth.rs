use crate::unary::one;
use crate::ScalarAd;
use num_traits::FloatConst;

fn ln_2<S: ScalarAd>() -> S {
    S::from_real(S::Real::LN_2())
}

fn ln_10<S: ScalarAd>() -> S {
    S::from_real(S::Real::LN_10())
}

/// Primal `2^x`.
///
/// # Examples
///
/// ```rust
/// use chainrules::exp2;
///
/// assert!((exp2(3.0_f64) - 8.0).abs() < 1e-12);
/// ```
pub fn exp2<S: ScalarAd>(x: S) -> S {
    x.exp2()
}

/// Forward rule for `2^x`.
///
/// # Examples
///
/// ```rust
/// use chainrules::exp2_frule;
///
/// let (y, dy) = exp2_frule(3.0_f64, 1.0);
/// assert_eq!(y, 8.0);
/// assert!((dy - 8.0_f64 * std::f64::consts::LN_2).abs() < 1e-12);
/// ```
pub fn exp2_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp2();
    (y, dx * (y * ln_2::<S>()).conj())
}

/// Reverse rule for `2^x`.
///
/// # Examples
///
/// ```rust
/// use chainrules::exp2_rrule;
///
/// let dy = exp2_rrule(8.0_f64, 1.0);
/// assert!((dy - 8.0_f64 * std::f64::consts::LN_2).abs() < 1e-12);
/// ```
pub fn exp2_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (result * ln_2::<S>()).conj()
}

/// Primal `10^x`.
///
/// # Examples
///
/// ```rust
/// use chainrules::exp10;
///
/// assert!((exp10(2.0_f64) - 100.0).abs() < 1e-12);
/// ```
pub fn exp10<S: ScalarAd>(x: S) -> S {
    x.exp10()
}

/// Forward rule for `10^x`.
///
/// # Examples
///
/// ```rust
/// use chainrules::exp10_frule;
///
/// let (y, dy) = exp10_frule(2.0_f64, 0.5);
/// assert!((y - 100.0).abs() < 1e-12);
/// assert!((dy - 0.5_f64 * 100.0_f64 * std::f64::consts::LN_10).abs() < 1e-12);
/// ```
pub fn exp10_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp10();
    (y, dx * (y * ln_10::<S>()).conj())
}

/// Reverse rule for `10^x`.
///
/// # Examples
///
/// ```rust
/// use chainrules::exp10_rrule;
///
/// let dy = exp10_rrule(100.0_f64, 0.5);
/// assert!((dy - 0.5_f64 * 100.0_f64 * std::f64::consts::LN_10).abs() < 1e-12);
/// ```
pub fn exp10_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (result * ln_10::<S>()).conj()
}

/// Primal `log2`.
///
/// # Examples
///
/// ```rust
/// use chainrules::log2;
///
/// assert_eq!(log2(8.0_f64), 3.0);
/// ```
pub fn log2<S: ScalarAd>(x: S) -> S {
    x.log2()
}

/// Forward rule for `log2`.
///
/// # Examples
///
/// ```rust
/// use chainrules::log2_frule;
///
/// let (y, dy) = log2_frule(8.0_f64, 2.0);
/// assert!((y - 3.0).abs() < 1e-12);
/// assert!((dy - (2.0_f64 / (8.0_f64 * std::f64::consts::LN_2))).abs() < 1e-12);
/// ```
pub fn log2_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.log2();
    let scale = one::<S>() / (x * ln_2::<S>());
    (y, dx * scale.conj())
}

/// Reverse rule for `log2`.
///
/// # Examples
///
/// ```rust
/// use chainrules::log2_rrule;
///
/// let dy = log2_rrule(8.0_f64, 2.0);
/// assert!((dy - (2.0_f64 / (8.0_f64 * std::f64::consts::LN_2))).abs() < 1e-12);
/// ```
pub fn log2_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (x * ln_2::<S>())).conj()
}

/// Primal `log10`.
///
/// # Examples
///
/// ```rust
/// use chainrules::log10;
///
/// assert_eq!(log10(100.0_f64), 2.0);
/// ```
pub fn log10<S: ScalarAd>(x: S) -> S {
    x.log10()
}

/// Forward rule for `log10`.
///
/// # Examples
///
/// ```rust
/// use chainrules::log10_frule;
///
/// let (y, dy) = log10_frule(100.0_f64, 2.0);
/// assert!((y - 2.0).abs() < 1e-12);
/// assert!((dy - (2.0_f64 / (100.0_f64 * std::f64::consts::LN_10))).abs() < 1e-12);
/// ```
pub fn log10_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.log10();
    let scale = one::<S>() / (x * ln_10::<S>());
    (y, dx * scale.conj())
}

/// Reverse rule for `log10`.
///
/// # Examples
///
/// ```rust
/// use chainrules::log10_rrule;
///
/// let dy = log10_rrule(100.0_f64, 2.0);
/// assert!((dy - (2.0_f64 / (100.0_f64 * std::f64::consts::LN_10))).abs() < 1e-12);
/// ```
pub fn log10_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (one::<S>() / (x * ln_10::<S>())).conj()
}
