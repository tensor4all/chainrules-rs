use super::ScalarAd;
use num_traits::{Float, FloatConst, Zero};

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
                <$ty as Float>::exp(self * <$ty as FloatConst>::LN_10())
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

impl_scalar_ad_real!(f32);
impl_scalar_ad_real!(f64);
