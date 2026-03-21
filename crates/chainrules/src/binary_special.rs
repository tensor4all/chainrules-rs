use num_traits::Float;

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

/// Forward rule for `hypot`.
///
/// # Examples
///
/// ```rust
/// use chainrules::hypot_frule;
///
/// let (r, dr) = hypot_frule(3.0_f64, 4.0_f64, 0.5_f64, 0.25_f64);
/// assert_eq!(r, 5.0);
/// assert!((dr - 0.5).abs() < 1e-12);
/// ```
pub fn hypot_frule<R: Float>(x: R, y: R, dx: R, dy: R) -> (R, R) {
    let r = x.hypot(y);
    let inv_r = R::one() / r;
    (r, dx * (x * inv_r) + dy * (y * inv_r))
}

/// Reverse rule for `hypot`.
///
/// # Examples
///
/// ```rust
/// use chainrules::hypot_rrule;
///
/// let (dx, dy) = hypot_rrule(3.0_f64, 4.0_f64, 1.0_f64);
/// assert!((dx - 0.6).abs() < 1e-12);
/// assert!((dy - 0.8).abs() < 1e-12);
/// ```
pub fn hypot_rrule<R: Float>(x: R, y: R, cotangent: R) -> (R, R) {
    let r = x.hypot(y);
    let inv_r = R::one() / r;
    (cotangent * (x * inv_r), cotangent * (y * inv_r))
}
