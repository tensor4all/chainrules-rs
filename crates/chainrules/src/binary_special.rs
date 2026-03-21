use num_traits::Float;

fn select_first_for_min<R: Float>(x: R, y: R) -> bool {
    !x.is_nan() && (y.is_nan() || x < y)
}

fn select_first_for_max<R: Float>(x: R, y: R) -> bool {
    !x.is_nan() && (y.is_nan() || x > y)
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

/// Primal `min`.
///
/// The primal follows `Float::min`; tie behavior routes the derivative to the
/// second argument.
///
/// # Examples
///
/// ```rust
/// use chainrules::min;
///
/// assert_eq!(min(1.5_f64, 2.5_f64), 1.5);
/// assert_eq!(min(2.0_f64, 2.0_f64), 2.0);
/// ```
pub fn min<R: Float>(x: R, y: R) -> R {
    x.min(y)
}

/// Forward rule for `min`.
///
/// When `x == y`, the tangent comes from `y`.
///
/// # Examples
///
/// ```rust
/// use chainrules::min_frule;
///
/// let (z, dz) = min_frule(1.0_f64, 2.0_f64, 0.25, 0.5);
/// assert_eq!(z, 1.0);
/// assert_eq!(dz, 0.25);
/// ```
pub fn min_frule<R: Float>(x: R, y: R, dx: R, dy: R) -> (R, R) {
    let z = x.min(y);
    if select_first_for_min(x, y) {
        (z, dx)
    } else {
        (z, dy)
    }
}

/// Reverse rule for `min`.
///
/// When `x == y`, the cotangent goes to `y`.
///
/// # Examples
///
/// ```rust
/// use chainrules::min_rrule;
///
/// let (dx, dy) = min_rrule(1.0_f64, 2.0_f64, 0.5);
/// assert_eq!(dx, 0.5);
/// assert_eq!(dy, 0.0);
/// ```
pub fn min_rrule<R: Float>(x: R, y: R, cotangent: R) -> (R, R) {
    if select_first_for_min(x, y) {
        (cotangent, R::zero())
    } else {
        (R::zero(), cotangent)
    }
}

/// Primal `max`.
///
/// The primal follows `Float::max`; tie behavior routes the derivative to the
/// second argument.
///
/// # Examples
///
/// ```rust
/// use chainrules::max;
///
/// assert_eq!(max(1.5_f64, 2.5_f64), 2.5);
/// assert_eq!(max(2.0_f64, 2.0_f64), 2.0);
/// ```
pub fn max<R: Float>(x: R, y: R) -> R {
    x.max(y)
}

/// Forward rule for `max`.
///
/// When `x == y`, the tangent comes from `y`.
///
/// # Examples
///
/// ```rust
/// use chainrules::max_frule;
///
/// let (z, dz) = max_frule(1.0_f64, 2.0_f64, 0.25, 0.5);
/// assert_eq!(z, 2.0);
/// assert_eq!(dz, 0.5);
/// ```
pub fn max_frule<R: Float>(x: R, y: R, dx: R, dy: R) -> (R, R) {
    let z = x.max(y);
    if select_first_for_max(x, y) {
        (z, dx)
    } else {
        (z, dy)
    }
}

/// Reverse rule for `max`.
///
/// When `x == y`, the cotangent goes to `y`.
///
/// # Examples
///
/// ```rust
/// use chainrules::max_rrule;
///
/// let (dx, dy) = max_rrule(1.0_f64, 2.0_f64, 0.5);
/// assert_eq!(dx, 0.0);
/// assert_eq!(dy, 0.5);
/// ```
pub fn max_rrule<R: Float>(x: R, y: R, cotangent: R) -> (R, R) {
    if select_first_for_max(x, y) {
        (cotangent, R::zero())
    } else {
        (R::zero(), cotangent)
    }
}
