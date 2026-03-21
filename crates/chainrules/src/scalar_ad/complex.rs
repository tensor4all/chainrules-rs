use super::ScalarAd;
use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{FloatConst, Zero};

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
                <$complex_ty as ComplexFloat>::exp(
                    self * <$complex_ty>::new(
                        <$real_ty as FloatConst>::LN_10(),
                        <$real_ty as Zero>::zero(),
                    ),
                )
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

impl_scalar_ad_complex!(Complex32, f32, Complex32::new(1.0, 0.0));
impl_scalar_ad_complex!(Complex64, f64, Complex64::new(1.0, 0.0));
