use core::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::{Float, FloatConst};

/// Scalar trait used by elementary AD rule helpers.
///
/// # Examples
///
/// ```rust
/// use chainrules::ScalarAd;
///
/// fn takes_scalar<S: ScalarAd>(_x: S) {}
///
/// takes_scalar(1.0_f32);
/// takes_scalar(1.0_f64);
/// ```
pub trait ScalarAd:
    Copy
    + PartialEq
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// Real exponent type for `powf`.
    type Real: Copy + Float + FloatConst;

    /// Complex conjugate (identity for real scalars).
    fn conj(self) -> Self;

    /// Reciprocal.
    fn recip(self) -> Self;

    /// Cubic root.
    fn cbrt(self) -> Self;

    /// Square root.
    fn sqrt(self) -> Self;

    /// Exponential.
    fn exp(self) -> Self;

    /// `2^self`.
    fn exp2(self) -> Self;

    /// `10^self`.
    fn exp10(self) -> Self;

    /// `exp(self) - 1`.
    fn expm1(self) -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// `ln(1 + self)`.
    fn log1p(self) -> Self;

    /// Base-2 logarithm.
    fn log2(self) -> Self;

    /// Base-10 logarithm.
    fn log10(self) -> Self;

    /// Sine.
    fn sin(self) -> Self;

    /// Cosine.
    fn cos(self) -> Self;

    /// Tangent.
    fn tan(self) -> Self;

    /// Hyperbolic tangent.
    fn tanh(self) -> Self;

    /// Arc sine.
    fn asin(self) -> Self;

    /// Arc cosine.
    fn acos(self) -> Self;

    /// Arc tangent.
    fn atan(self) -> Self;

    /// Hyperbolic sine.
    fn sinh(self) -> Self;

    /// Hyperbolic cosine.
    fn cosh(self) -> Self;

    /// Area hyperbolic sine.
    fn asinh(self) -> Self;

    /// Area hyperbolic cosine.
    fn acosh(self) -> Self;

    /// Area hyperbolic tangent.
    fn atanh(self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self::Real;

    /// Absolute value squared.
    fn abs2(self) -> Self::Real;

    /// Real part.
    fn real(self) -> Self::Real;

    /// Imaginary part.
    fn imag(self) -> Self::Real;

    /// Polar angle.
    fn angle(self) -> Self::Real;

    /// Power by real exponent.
    fn powf(self, exponent: Self::Real) -> Self;

    /// Power by integer exponent.
    fn powi(self, exponent: i32) -> Self;

    /// Power by same-scalar exponent.
    fn pow(self, exponent: Self) -> Self;

    /// Convert real scalar to this scalar type.
    fn from_real(value: Self::Real) -> Self;

    /// Convert signed integer to this scalar type.
    fn from_i32(value: i32) -> Self;
}

mod complex;
mod real;
