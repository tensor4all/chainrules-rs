use core::ops::{Add, Div, Mul, Neg, Sub};

use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{Float, FloatConst, Zero};

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

macro_rules! impl_scalar_ad_real {
    ($ty:ty) => {
        impl ScalarAd for $ty {
            type Real = $ty;

            fn conj(self) -> Self {
                self
            }

            fn recip(self) -> Self {
                <$ty as Float>::recip(self)
            }

            fn cbrt(self) -> Self {
                <$ty as Float>::cbrt(self)
            }

            fn sqrt(self) -> Self {
                <$ty as Float>::sqrt(self)
            }

            fn exp(self) -> Self {
                <$ty as Float>::exp(self)
            }

            fn exp2(self) -> Self {
                <$ty as Float>::exp2(self)
            }

            fn exp10(self) -> Self {
                (<$ty as Float>::exp(self * <$ty as FloatConst>::LN_10()))
            }

            fn expm1(self) -> Self {
                <$ty as Float>::exp_m1(self)
            }

            fn ln(self) -> Self {
                <$ty as Float>::ln(self)
            }

            fn log1p(self) -> Self {
                <$ty as Float>::ln_1p(self)
            }

            fn log2(self) -> Self {
                <$ty as Float>::log2(self)
            }

            fn log10(self) -> Self {
                <$ty as Float>::log10(self)
            }

            fn sin(self) -> Self {
                <$ty as Float>::sin(self)
            }

            fn cos(self) -> Self {
                <$ty as Float>::cos(self)
            }

            fn tan(self) -> Self {
                <$ty as Float>::tan(self)
            }

            fn tanh(self) -> Self {
                <$ty as Float>::tanh(self)
            }

            fn asin(self) -> Self {
                <$ty as Float>::asin(self)
            }

            fn acos(self) -> Self {
                <$ty as Float>::acos(self)
            }

            fn atan(self) -> Self {
                <$ty as Float>::atan(self)
            }

            fn sinh(self) -> Self {
                <$ty as Float>::sinh(self)
            }

            fn cosh(self) -> Self {
                <$ty as Float>::cosh(self)
            }

            fn asinh(self) -> Self {
                <$ty as Float>::asinh(self)
            }

            fn acosh(self) -> Self {
                <$ty as Float>::acosh(self)
            }

            fn atanh(self) -> Self {
                <$ty as Float>::atanh(self)
            }

            fn abs(self) -> Self::Real {
                <$ty as Float>::abs(self)
            }

            fn abs2(self) -> Self::Real {
                self * self
            }

            fn real(self) -> Self::Real {
                self
            }

            fn imag(self) -> Self::Real {
                <$ty as Zero>::zero()
            }

            fn angle(self) -> Self::Real {
                <$ty as Float>::atan2(<$ty as Zero>::zero(), self)
            }

            fn powf(self, exponent: Self::Real) -> Self {
                <$ty as Float>::powf(self, exponent)
            }

            fn powi(self, exponent: i32) -> Self {
                <$ty as Float>::powi(self, exponent)
            }

            fn pow(self, exponent: Self) -> Self {
                <$ty as Float>::powf(self, exponent)
            }

            fn from_real(value: Self::Real) -> Self {
                value
            }

            fn from_i32(value: i32) -> Self {
                value as $ty
            }
        }
    };
}

macro_rules! impl_scalar_ad_complex {
    ($complex_ty:ty, $real_ty:ty, $one:expr) => {
        impl ScalarAd for $complex_ty {
            type Real = $real_ty;

            fn conj(self) -> Self {
                <$complex_ty>::conj(&self)
            }

            fn recip(self) -> Self {
                <$complex_ty as ComplexFloat>::recip(self)
            }

            fn cbrt(self) -> Self {
                <$complex_ty as ComplexFloat>::cbrt(self)
            }

            fn sqrt(self) -> Self {
                <$complex_ty as ComplexFloat>::sqrt(self)
            }

            fn exp(self) -> Self {
                <$complex_ty as ComplexFloat>::exp(self)
            }

            fn exp2(self) -> Self {
                <$complex_ty as ComplexFloat>::exp2(self)
            }

            fn exp10(self) -> Self {
                (<$complex_ty as ComplexFloat>::exp(
                    self * <$complex_ty>::new(
                        <$real_ty as FloatConst>::LN_10(),
                        <$real_ty as Zero>::zero(),
                    ),
                ))
            }

            fn expm1(self) -> Self {
                <$complex_ty as ComplexFloat>::exp(self) - $one
            }

            fn ln(self) -> Self {
                <$complex_ty as ComplexFloat>::ln(self)
            }

            fn log1p(self) -> Self {
                <$complex_ty as ComplexFloat>::ln(self + $one)
            }

            fn log2(self) -> Self {
                <$complex_ty as ComplexFloat>::log2(self)
            }

            fn log10(self) -> Self {
                <$complex_ty as ComplexFloat>::log10(self)
            }

            fn sin(self) -> Self {
                <$complex_ty as ComplexFloat>::sin(self)
            }

            fn cos(self) -> Self {
                <$complex_ty as ComplexFloat>::cos(self)
            }

            fn tan(self) -> Self {
                <$complex_ty as ComplexFloat>::tan(self)
            }

            fn tanh(self) -> Self {
                <$complex_ty as ComplexFloat>::tanh(self)
            }

            fn asin(self) -> Self {
                <$complex_ty as ComplexFloat>::asin(self)
            }

            fn acos(self) -> Self {
                <$complex_ty as ComplexFloat>::acos(self)
            }

            fn atan(self) -> Self {
                <$complex_ty as ComplexFloat>::atan(self)
            }

            fn sinh(self) -> Self {
                <$complex_ty as ComplexFloat>::sinh(self)
            }

            fn cosh(self) -> Self {
                <$complex_ty as ComplexFloat>::cosh(self)
            }

            fn asinh(self) -> Self {
                <$complex_ty as ComplexFloat>::asinh(self)
            }

            fn acosh(self) -> Self {
                <$complex_ty as ComplexFloat>::acosh(self)
            }

            fn atanh(self) -> Self {
                <$complex_ty as ComplexFloat>::atanh(self)
            }

            fn abs(self) -> Self::Real {
                <$complex_ty as ComplexFloat>::abs(self)
            }

            fn abs2(self) -> Self::Real {
                <$complex_ty>::norm_sqr(&self)
            }

            fn real(self) -> Self::Real {
                self.re
            }

            fn imag(self) -> Self::Real {
                self.im
            }

            fn angle(self) -> Self::Real {
                <$complex_ty as ComplexFloat>::arg(self)
            }

            fn powf(self, exponent: Self::Real) -> Self {
                <$complex_ty as ComplexFloat>::powf(self, exponent)
            }

            fn powi(self, exponent: i32) -> Self {
                <$complex_ty as ComplexFloat>::powi(self, exponent)
            }

            fn pow(self, exponent: Self) -> Self {
                <$complex_ty as ComplexFloat>::powc(self, exponent)
            }

            fn from_real(value: Self::Real) -> Self {
                <$complex_ty>::new(value, 0.0)
            }

            fn from_i32(value: i32) -> Self {
                <$complex_ty>::new(value as $real_ty, 0.0)
            }
        }
    };
}

impl_scalar_ad_real!(f32);
impl_scalar_ad_real!(f64);
impl_scalar_ad_complex!(Complex32, f32, Complex32::new(1.0, 0.0));
impl_scalar_ad_complex!(Complex64, f64, Complex64::new(1.0, 0.0));

/// PyTorch-style real-input / complex-gradient projection helper (`handle_r_to_c`).
///
/// This is equivalent to taking the real part when a gradient for real input
/// becomes complex during intermediate algebra.
///
/// # Examples
///
/// ```rust
/// use chainrules::handle_r_to_c_f64;
/// use num_complex::Complex64;
///
/// let g = Complex64::new(1.25, -3.0);
/// assert_eq!(handle_r_to_c_f64(g), 1.25);
/// ```
pub fn handle_r_to_c_f64(gradient: Complex64) -> f64 {
    gradient.re
}

/// `f32` variant of [`handle_r_to_c_f64`].
///
/// # Examples
///
/// ```rust
/// use chainrules::handle_r_to_c_f32;
/// use num_complex::Complex32;
///
/// let g = Complex32::new(2.0, 4.0);
/// assert_eq!(handle_r_to_c_f32(g), 2.0);
/// ```
pub fn handle_r_to_c_f32(gradient: Complex32) -> f32 {
    gradient.re
}
