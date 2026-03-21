use crate::ScalarAd;
use num_complex::Complex;
use num_traits::{Float, One, Zero};

trait ComplexProjectionScalar: ScalarAd {
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;
}

impl ComplexProjectionScalar for num_complex::Complex32 {
    fn from_parts(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }
}

impl ComplexProjectionScalar for num_complex::Complex64 {
    fn from_parts(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }
}

/// Primal `abs`.
///
/// # Examples
///
/// ```rust
/// use chainrules::abs;
/// use num_complex::Complex64;
///
/// assert_eq!(abs(Complex64::new(3.0, 4.0)), 5.0);
/// ```
#[inline]
pub fn abs<S: ScalarAd>(x: S) -> S::Real {
    x.abs()
}

/// Primal `abs2`.
///
/// # Examples
///
/// ```rust
/// use chainrules::abs2;
/// use num_complex::Complex64;
///
/// assert_eq!(abs2(Complex64::new(3.0, 4.0)), 25.0);
/// ```
#[inline]
pub fn abs2<S: ScalarAd>(x: S) -> S::Real {
    x.abs2()
}

/// Primal `real`.
///
/// # Examples
///
/// ```rust
/// use chainrules::real;
/// use num_complex::Complex64;
///
/// assert_eq!(real(Complex64::new(3.0, 4.0)), 3.0);
/// ```
#[inline]
pub fn real<S: ScalarAd>(x: S) -> S::Real {
    x.real()
}

/// Primal `imag`.
///
/// # Examples
///
/// ```rust
/// use chainrules::imag;
/// use num_complex::Complex64;
///
/// assert_eq!(imag(Complex64::new(3.0, 4.0)), 4.0);
/// ```
#[inline]
pub fn imag<S: ScalarAd>(x: S) -> S::Real {
    x.imag()
}

/// Primal `angle`.
///
/// # Examples
///
/// ```rust
/// use chainrules::angle;
/// use num_complex::Complex64;
///
/// assert!((angle(Complex64::new(3.0, 4.0)) - 0.9272952180016122).abs() < 1e-12);
/// ```
#[inline]
pub fn angle<S: ScalarAd>(x: S) -> S::Real {
    x.angle()
}

/// Construct a complex number from real and imaginary parts.
///
/// # Examples
///
/// ```rust
/// use chainrules::complex;
/// use num_complex::Complex64;
///
/// assert_eq!(complex(3.0_f64, 4.0_f64), Complex64::new(3.0, 4.0));
/// ```
#[inline]
pub fn complex<R: Float>(re: R, im: R) -> Complex<R> {
    Complex::new(re, im)
}

/// Forward rule for `abs2`.
///
/// # Examples
///
/// ```rust
/// use chainrules::abs2_frule;
/// use num_complex::Complex64;
///
/// let z = Complex64::new(3.0, 4.0);
/// let dz = Complex64::new(1.0, -2.0);
/// let (y, dy) = abs2_frule(z, dz);
/// assert_eq!(y, 25.0);
/// assert_eq!(dy, -10.0);
/// ```
#[inline]
pub fn abs2_frule<S: ScalarAd>(x: S, dx: S) -> (S::Real, S::Real) {
    let y = x.abs2();
    let two = S::Real::one() + S::Real::one();
    let dy = two * (x.real() * dx.real() + x.imag() * dx.imag());
    (y, dy)
}

/// Reverse rule for `abs2`.
///
/// # Examples
///
/// ```rust
/// use chainrules::abs2_rrule;
/// use num_complex::Complex64;
///
/// let z = Complex64::new(3.0, 4.0);
/// assert_eq!(abs2_rrule(z, 1.25), Complex64::new(7.5, 10.0));
/// ```
#[inline]
pub fn abs2_rrule<R: Float>(x: Complex<R>, cotangent: R) -> Complex<R> {
    let two = R::one() + R::one();
    Complex::new(two * cotangent * x.re, two * cotangent * x.im)
}

/// Reverse rule for `real`.
///
/// # Examples
///
/// ```rust
/// use chainrules::real_rrule;
/// use num_complex::Complex64;
///
/// let grad: Complex64 = real_rrule(2.0);
/// assert_eq!(grad, Complex64::new(2.0, 0.0));
/// ```
#[inline]
#[allow(private_bounds)]
pub fn real_rrule<S: ComplexProjectionScalar>(cotangent: S::Real) -> S {
    S::from_parts(cotangent, S::Real::zero())
}

/// Reverse rule for `imag`.
///
/// # Examples
///
/// ```rust
/// use chainrules::imag_rrule;
/// use num_complex::Complex64;
///
/// let grad: Complex64 = imag_rrule(2.0);
/// assert_eq!(grad, Complex64::new(0.0, 2.0));
/// ```
#[inline]
#[allow(private_bounds)]
pub fn imag_rrule<S: ComplexProjectionScalar>(cotangent: S::Real) -> S {
    S::from_parts(S::Real::zero(), cotangent)
}

/// Reverse rule for `angle`.
///
/// # Examples
///
/// ```rust
/// use chainrules::angle_rrule;
/// use num_complex::Complex64;
///
/// assert_eq!(angle_rrule(Complex64::new(3.0, 4.0), 1.0), Complex64::new(-0.16, 0.12));
/// ```
#[inline]
pub fn angle_rrule<R: Float>(x: Complex<R>, cotangent: R) -> Complex<R> {
    let denom = x.re * x.re + x.im * x.im;
    Complex::new(-x.im * cotangent / denom, x.re * cotangent / denom)
}
