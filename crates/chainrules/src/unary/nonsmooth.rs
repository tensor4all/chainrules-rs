use num_traits::Float;

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
/// assert_eq!(round_rrule(1.0_f64, 0.5), 0.0);
/// ```
pub fn round_rrule<R: Float>(_x: R, _cotangent: R) -> R {
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
/// assert_eq!(floor_rrule(1.0_f64, 0.5), 0.0);
/// ```
pub fn floor_rrule<R: Float>(_x: R, _cotangent: R) -> R {
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
/// assert_eq!(ceil_rrule(1.0_f64, 0.5), 0.0);
/// ```
pub fn ceil_rrule<R: Float>(_x: R, _cotangent: R) -> R {
    R::zero()
}

/// Primal `sign`.
///
/// The primal follows Julia-style `sign`: it returns signed zero for zero
/// inputs and otherwise `x / abs(x)`.
///
/// The corresponding forward and reverse rules use a zero-gradient policy at
/// every point.
///
/// # Examples
///
/// ```rust
/// use chainrules::sign;
///
/// assert_eq!(sign(-3.0_f64), -1.0);
/// assert_eq!(sign(0.0_f64), 0.0);
/// assert_eq!(sign(-0.0_f64).is_sign_negative(), true);
/// ```
pub fn sign<R: Float>(x: R) -> R {
    if x == R::zero() {
        x
    } else {
        x / x.abs()
    }
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
    (sign(x), R::zero())
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
/// assert_eq!(sign_rrule(1.0_f64, 0.5), 0.0);
/// ```
pub fn sign_rrule<R: Float>(_x: R, _cotangent: R) -> R {
    R::zero()
}
