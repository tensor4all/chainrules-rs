use crate::ScalarAd;

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

/// Forward rule for `cbrt`.
///
/// # Examples
///
/// ```rust
/// use chainrules::cbrt_frule;
///
/// let (y, dy) = cbrt_frule(8.0_f64, 1.0);
/// assert_eq!(y, 2.0);
/// assert!((dy - (1.0 / 12.0)).abs() < 1e-12);
/// ```
pub fn cbrt_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.cbrt();
    let scale = S::from_i32(1) / (S::from_i32(3) * y * y);
    (y, dx * scale.conj())
}

/// Reverse rule for `cbrt`.
///
/// # Examples
///
/// ```rust
/// use chainrules::cbrt_rrule;
///
/// let dx = cbrt_rrule(2.0_f64, 1.0);
/// assert!((dx - (1.0 / 12.0)).abs() < 1e-12);
/// ```
pub fn cbrt_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (S::from_i32(1) / (S::from_i32(3) * result * result)).conj()
}

/// Primal `inv`.
///
/// # Examples
///
/// ```rust
/// use chainrules::inv;
///
/// assert_eq!(inv(4.0_f64), 0.25);
/// ```
pub fn inv<S: ScalarAd>(x: S) -> S {
    x.recip()
}

/// Forward rule for `inv`.
///
/// # Examples
///
/// ```rust
/// use chainrules::inv_frule;
///
/// let (y, dy) = inv_frule(4.0_f64, 2.0);
/// assert_eq!(y, 0.25);
/// assert!((dy + 0.125).abs() < 1e-12);
/// ```
pub fn inv_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.recip();
    (y, dx * (-(y * y)).conj())
}

/// Reverse rule for `inv`.
///
/// # Examples
///
/// ```rust
/// use chainrules::inv_rrule;
///
/// let dx = inv_rrule(0.25_f64, 2.0);
/// assert!((dx + 0.125).abs() < 1e-12);
/// ```
pub fn inv_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * (-(result * result)).conj()
}
