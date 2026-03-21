#![allow(dead_code)]

use num_traits::Float;

fn select_first_for_min<R: Float>(x: R, y: R) -> bool {
    !x.is_nan() && (y.is_nan() || x < y)
}

fn select_first_for_max<R: Float>(x: R, y: R) -> bool {
    !x.is_nan() && (y.is_nan() || x > y)
}

/// Primal `round`.
///
/// The corresponding forward and reverse rules use a zero-gradient policy at
/// every point, including integer inputs.
///
/// # Examples
///
/// ```rust
/// use chainrules::round;
///
/// assert_eq!(round(1.4_f64), 1.0);
/// assert_eq!(round(1.5_f64), 2.0);
/// ```
pub fn round<R: Float>(x: R) -> R {
    x.round()
}

/// Forward rule for `round`.
///
/// The tangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::round_frule;
///
/// let (y, dy) = round_frule(1.6_f64, 0.75);
/// assert_eq!(y, 2.0);
/// assert_eq!(dy, 0.0);
/// ```
pub fn round_frule<R: Float>(x: R, _dx: R) -> (R, R) {
    (x.round(), R::zero())
}

/// Reverse rule for `round`.
///
/// The cotangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::round_rrule;
///
/// assert_eq!(round_rrule(1.0_f64), 0.0);
/// ```
pub fn round_rrule<R: Float>(_cotangent: R) -> R {
    R::zero()
}

/// Primal `floor`.
///
/// The corresponding forward and reverse rules use a zero-gradient policy at
/// every point.
///
/// # Examples
///
/// ```rust
/// use chainrules::floor;
///
/// assert_eq!(floor(1.9_f64), 1.0);
/// ```
pub fn floor<R: Float>(x: R) -> R {
    x.floor()
}

/// Forward rule for `floor`.
///
/// The tangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::floor_frule;
///
/// let (y, dy) = floor_frule(1.6_f64, 0.75);
/// assert_eq!(y, 1.0);
/// assert_eq!(dy, 0.0);
/// ```
pub fn floor_frule<R: Float>(x: R, _dx: R) -> (R, R) {
    (x.floor(), R::zero())
}

/// Reverse rule for `floor`.
///
/// The cotangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::floor_rrule;
///
/// assert_eq!(floor_rrule(1.0_f64), 0.0);
/// ```
pub fn floor_rrule<R: Float>(_cotangent: R) -> R {
    R::zero()
}

/// Primal `ceil`.
///
/// The corresponding forward and reverse rules use a zero-gradient policy at
/// every point.
///
/// # Examples
///
/// ```rust
/// use chainrules::ceil;
///
/// assert_eq!(ceil(1.1_f64), 2.0);
/// ```
pub fn ceil<R: Float>(x: R) -> R {
    x.ceil()
}

/// Forward rule for `ceil`.
///
/// The tangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::ceil_frule;
///
/// let (y, dy) = ceil_frule(1.1_f64, 0.75);
/// assert_eq!(y, 2.0);
/// assert_eq!(dy, 0.0);
/// ```
pub fn ceil_frule<R: Float>(x: R, _dx: R) -> (R, R) {
    (x.ceil(), R::zero())
}

/// Reverse rule for `ceil`.
///
/// The cotangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::ceil_rrule;
///
/// assert_eq!(ceil_rrule(1.0_f64), 0.0);
/// ```
pub fn ceil_rrule<R: Float>(_cotangent: R) -> R {
    R::zero()
}

/// Primal `sign`.
///
/// The primal follows `Float::signum`, and the corresponding forward and
/// reverse rules use a zero-gradient policy at every point.
///
/// # Examples
///
/// ```rust
/// use chainrules::sign;
///
/// assert_eq!(sign(-3.0_f64), -1.0);
/// assert_eq!(sign(0.0_f64), 1.0);
/// ```
pub fn sign<R: Float>(x: R) -> R {
    x.signum()
}

/// Forward rule for `sign`.
///
/// The tangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::sign_frule;
///
/// let (y, dy) = sign_frule(-2.0_f64, 0.75);
/// assert_eq!(y, -1.0);
/// assert_eq!(dy, 0.0);
/// ```
pub fn sign_frule<R: Float>(x: R, _dx: R) -> (R, R) {
    (x.signum(), R::zero())
}

/// Reverse rule for `sign`.
///
/// The cotangent is always zero.
///
/// # Examples
///
/// ```rust
/// use chainrules::sign_rrule;
///
/// assert_eq!(sign_rrule(1.0_f64), 0.0);
/// ```
pub fn sign_rrule<R: Float>(_cotangent: R) -> R {
    R::zero()
}

/// Primal `min`.
///
/// Tie behavior routes the derivative to the second argument.
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
/// Tie behavior routes the derivative to the second argument.
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
