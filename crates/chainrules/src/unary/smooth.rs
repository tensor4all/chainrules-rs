use crate::ScalarAd;
use num_traits::Float;

/// Primal `cbrt`.
///
/// # Examples
///
/// ```rust
/// use chainrules::cbrt;
///
/// assert_eq!(cbrt(8.0_f64), 2.0);
/// ```
pub fn cbrt<S: ScalarAd>(x: S) -> S {
    x.cbrt()
}

/// Primal `exp2`.
///
/// # Examples
///
/// ```rust
/// use chainrules::exp2;
///
/// assert_eq!(exp2(3.0_f64), 8.0);
/// ```
pub fn exp2<S: ScalarAd>(x: S) -> S {
    x.exp2()
}

/// Primal `hypot`.
///
/// # Examples
///
/// ```rust
/// use chainrules::hypot;
///
/// assert_eq!(hypot(3.0_f64, 4.0_f64), 5.0);
/// ```
pub fn hypot<R: Float>(x: R, y: R) -> R {
    x.hypot(y)
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

/// Primal `pow`.
///
/// # Examples
///
/// ```rust
/// use chainrules::pow;
///
/// assert_eq!(pow(2.0_f64, 3.0_f64), 8.0);
/// ```
pub fn pow<S: ScalarAd>(x: S, exponent: S) -> S {
    x.pow(exponent)
}

/// Primal `tan`.
///
/// # Examples
///
/// ```rust
/// use chainrules::tan;
///
/// assert!((tan(0.5_f64) - 0.5_f64.tan()).abs() < 1e-12);
/// ```
pub fn tan<S: ScalarAd>(x: S) -> S {
    x.tan()
}
