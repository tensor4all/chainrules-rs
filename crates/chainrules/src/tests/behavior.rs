use num_complex::{Complex32, Complex64};

use crate::{
    acos, acos_frule, acos_rrule, acosh, acosh_frule, acosh_rrule, asin, asin_frule, asin_rrule,
    asinh, asinh_frule, asinh_rrule, atan, atan2, atan2_frule, atan_frule, atan_rrule, atanh,
    atanh_frule, atanh_rrule, conj, conj_frule, conj_rrule, cos, cos_frule, cos_rrule, cosh,
    cosh_frule, cosh_rrule, exp, exp_frule, exp_rrule, expm1_frule, expm1_rrule, handle_r_to_c_f32,
    handle_r_to_c_f64, log, log1p_frule, log1p_rrule, log_frule, log_rrule, sin, sin_frule,
    sin_rrule, sinh, sinh_frule, sinh_rrule, sqrt, sqrt_frule, sqrt_rrule, tanh, tanh_frule,
    tanh_rrule, ScalarAd,
};

fn assert_close_f32(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1.0e-5,
        "actual={actual}, expected={expected}",
    );
}

fn assert_close_f64(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual}, expected={expected}",
    );
}

fn assert_close_c32(actual: Complex32, expected: Complex32) {
    assert_close_f32(actual.re, expected.re);
    assert_close_f32(actual.im, expected.im);
}

fn assert_close_c64(actual: Complex64, expected: Complex64) {
    assert_close_f64(actual.re, expected.re);
    assert_close_f64(actual.im, expected.im);
}

#[test]
fn scalar_ad_real_impls_match_std_real_ops() {
    let x32 = 0.25_f32;
    assert_close_f32(<f32 as ScalarAd>::expm1(x32), x32.exp_m1());
    assert_close_f32(<f32 as ScalarAd>::log1p(x32), x32.ln_1p());
    assert_close_f32(<f32 as ScalarAd>::sin(x32), x32.sin());
    assert_close_f32(<f32 as ScalarAd>::cos(x32), x32.cos());
    assert_close_f32(<f32 as ScalarAd>::tanh(x32), x32.tanh());
    assert_close_f32(<f32 as ScalarAd>::asin(x32), x32.asin());
    assert_close_f32(<f32 as ScalarAd>::acos(x32), x32.acos());
    assert_close_f32(<f32 as ScalarAd>::atan(x32), x32.atan());
    assert_close_f32(<f32 as ScalarAd>::sinh(x32), x32.sinh());
    assert_close_f32(<f32 as ScalarAd>::cosh(x32), x32.cosh());
    assert_close_f32(<f32 as ScalarAd>::asinh(x32), x32.asinh());
    assert_close_f32(<f32 as ScalarAd>::acosh(1.25_f32), 1.25_f32.acosh());
    assert_close_f32(<f32 as ScalarAd>::atanh(x32), x32.atanh());
    assert_close_f32(<f32 as ScalarAd>::powf(x32, 2.5), x32.powf(2.5));
    assert_eq!(<f32 as ScalarAd>::powi(x32, 3), x32.powi(3));
    assert_eq!(<f32 as ScalarAd>::from_real(1.5), 1.5);
    assert_eq!(<f32 as ScalarAd>::from_i32(-2), -2.0);

    let x64 = 0.5_f64;
    assert_close_f64(<f64 as ScalarAd>::expm1(x64), x64.exp_m1());
    assert_close_f64(<f64 as ScalarAd>::log1p(x64), x64.ln_1p());
    assert_close_f64(<f64 as ScalarAd>::sin(x64), x64.sin());
    assert_close_f64(<f64 as ScalarAd>::cos(x64), x64.cos());
    assert_close_f64(<f64 as ScalarAd>::tanh(x64), x64.tanh());
    assert_close_f64(<f64 as ScalarAd>::asin(x64), x64.asin());
    assert_close_f64(<f64 as ScalarAd>::acos(x64), x64.acos());
    assert_close_f64(<f64 as ScalarAd>::atan(x64), x64.atan());
    assert_close_f64(<f64 as ScalarAd>::sinh(x64), x64.sinh());
    assert_close_f64(<f64 as ScalarAd>::cosh(x64), x64.cosh());
    assert_close_f64(<f64 as ScalarAd>::asinh(x64), x64.asinh());
    assert_close_f64(<f64 as ScalarAd>::acosh(1.5_f64), 1.5_f64.acosh());
    assert_close_f64(<f64 as ScalarAd>::atanh(x64), x64.atanh());
    assert_close_f64(<f64 as ScalarAd>::powf(x64, 1.5), x64.powf(1.5));
    assert_eq!(<f64 as ScalarAd>::powi(x64, 4), x64.powi(4));
    assert_eq!(<f64 as ScalarAd>::from_real(2.5), 2.5);
    assert_eq!(<f64 as ScalarAd>::from_i32(3), 3.0);
}

#[test]
fn scalar_ad_complex_impls_match_std_complex_ops() {
    let x32 = Complex32::new(0.25, -0.5);
    assert_close_c32(<Complex32 as ScalarAd>::conj(x32), x32.conj());
    assert_close_c32(<Complex32 as ScalarAd>::sqrt(x32), x32.sqrt());
    assert_close_c32(<Complex32 as ScalarAd>::exp(x32), x32.exp());
    assert_close_c32(
        <Complex32 as ScalarAd>::expm1(x32),
        x32.exp() - Complex32::new(1.0, 0.0),
    );
    assert_close_c32(<Complex32 as ScalarAd>::ln(x32), x32.ln());
    assert_close_c32(
        <Complex32 as ScalarAd>::log1p(x32),
        (x32 + Complex32::new(1.0, 0.0)).ln(),
    );
    assert_close_c32(<Complex32 as ScalarAd>::sin(x32), x32.sin());
    assert_close_c32(<Complex32 as ScalarAd>::cos(x32), x32.cos());
    assert_close_c32(<Complex32 as ScalarAd>::tanh(x32), x32.tanh());
    assert_close_c32(<Complex32 as ScalarAd>::asin(x32), x32.asin());
    assert_close_c32(<Complex32 as ScalarAd>::acos(x32), x32.acos());
    assert_close_c32(<Complex32 as ScalarAd>::atan(x32), x32.atan());
    assert_close_c32(<Complex32 as ScalarAd>::sinh(x32), x32.sinh());
    assert_close_c32(<Complex32 as ScalarAd>::cosh(x32), x32.cosh());
    assert_close_c32(<Complex32 as ScalarAd>::asinh(x32), x32.asinh());
    assert_close_c32(<Complex32 as ScalarAd>::acosh(x32), x32.acosh());
    assert_close_c32(<Complex32 as ScalarAd>::atanh(x32), x32.atanh());
    assert_close_c32(<Complex32 as ScalarAd>::powf(x32, 2.0), x32.powf(2.0));
    assert_close_c32(<Complex32 as ScalarAd>::powi(x32, 3), x32.powi(3));
    assert_eq!(
        <Complex32 as ScalarAd>::from_real(1.5),
        Complex32::new(1.5, 0.0)
    );
    assert_eq!(
        <Complex32 as ScalarAd>::from_i32(-2),
        Complex32::new(-2.0, 0.0)
    );

    let x64 = Complex64::new(0.5, 0.75);
    assert_close_c64(<Complex64 as ScalarAd>::conj(x64), x64.conj());
    assert_close_c64(<Complex64 as ScalarAd>::sqrt(x64), x64.sqrt());
    assert_close_c64(<Complex64 as ScalarAd>::exp(x64), x64.exp());
    assert_close_c64(
        <Complex64 as ScalarAd>::expm1(x64),
        x64.exp() - Complex64::new(1.0, 0.0),
    );
    assert_close_c64(<Complex64 as ScalarAd>::ln(x64), x64.ln());
    assert_close_c64(
        <Complex64 as ScalarAd>::log1p(x64),
        (x64 + Complex64::new(1.0, 0.0)).ln(),
    );
    assert_close_c64(<Complex64 as ScalarAd>::sin(x64), x64.sin());
    assert_close_c64(<Complex64 as ScalarAd>::cos(x64), x64.cos());
    assert_close_c64(<Complex64 as ScalarAd>::tanh(x64), x64.tanh());
    assert_close_c64(<Complex64 as ScalarAd>::asin(x64), x64.asin());
    assert_close_c64(<Complex64 as ScalarAd>::acos(x64), x64.acos());
    assert_close_c64(<Complex64 as ScalarAd>::atan(x64), x64.atan());
    assert_close_c64(<Complex64 as ScalarAd>::sinh(x64), x64.sinh());
    assert_close_c64(<Complex64 as ScalarAd>::cosh(x64), x64.cosh());
    assert_close_c64(<Complex64 as ScalarAd>::asinh(x64), x64.asinh());
    assert_close_c64(<Complex64 as ScalarAd>::acosh(x64), x64.acosh());
    assert_close_c64(<Complex64 as ScalarAd>::atanh(x64), x64.atanh());
    assert_close_c64(<Complex64 as ScalarAd>::powf(x64, 1.5), x64.powf(1.5));
    assert_close_c64(<Complex64 as ScalarAd>::powi(x64, 2), x64.powi(2));
    assert_eq!(
        <Complex64 as ScalarAd>::from_real(2.5),
        Complex64::new(2.5, 0.0)
    );
    assert_eq!(
        <Complex64 as ScalarAd>::from_i32(4),
        Complex64::new(4.0, 0.0)
    );
}

#[test]
fn direct_entrypoints_match_real_projection_and_atan2_formulas() {
    assert_eq!(handle_r_to_c_f32(Complex32::new(2.0, -5.0)), 2.0);
    assert_eq!(handle_r_to_c_f64(Complex64::new(-3.0, 1.5)), -3.0);

    let primal = atan2(3.0_f64, 4.0_f64);
    assert_close_f64(primal, 3.0_f64.atan2(4.0));

    let (atan2_y, atan2_dy) = atan2_frule(3.0_f64, 4.0_f64, 0.5_f64, 0.25_f64);
    assert_close_f64(atan2_y, primal);
    assert_close_f64(atan2_dy, 0.05);
}

#[test]
fn unary_entrypoints_match_forward_and_reverse_formulas() {
    let complex = Complex32::new(1.0, -2.0);
    assert_eq!(conj(complex), complex.conj());
    let (_y, dy) = conj_frule(complex, Complex32::new(3.0, 4.0));
    assert_eq!(dy, Complex32::new(3.0, -4.0));
    assert_eq!(conj_rrule(complex), complex.conj());

    assert_eq!(sqrt(9.0_f32), 3.0);
    let (sqrt_y, sqrt_dy) = sqrt_frule(9.0_f32, 2.0_f32);
    assert_eq!(sqrt_y, 3.0);
    assert_close_f32(sqrt_dy, 1.0 / 3.0);
    assert_close_f32(sqrt_rrule(3.0_f32, 2.0_f32), 1.0 / 3.0);

    let exp_y = exp(1.0_f32);
    assert_close_f32(exp_y, std::f32::consts::E);
    let (exp_primal, exp_tangent) = exp_frule(1.0_f32, 0.25_f32);
    assert_close_f32(exp_primal, std::f32::consts::E);
    assert_close_f32(exp_tangent, 0.25 * std::f32::consts::E);
    assert_close_f32(exp_rrule(exp_primal, 0.5_f32), 0.5 * std::f32::consts::E);

    let log_y = log(std::f32::consts::E);
    assert_close_f32(log_y, 1.0);
    let (log_primal, log_tangent) = log_frule(2.0_f32, 3.0_f32);
    assert_close_f32(log_primal, 2.0_f32.ln());
    assert_close_f32(log_tangent, 1.5);
    assert_close_f32(log_rrule(2.0_f32, 3.0_f32), 1.5);
}

#[test]
fn extended_real_unary_rules_match_expected_formulas() {
    let x = 0.25_f64;
    let dx = -0.5_f64;
    let cotangent = 0.75_f64;

    let (expm1_y, expm1_dy) = expm1_frule(x, dx);
    assert_close_f64(expm1_y, x.exp_m1());
    assert_close_f64(expm1_dy, dx * x.exp());
    assert_close_f64(expm1_rrule(expm1_y, cotangent), cotangent * x.exp());

    let (log1p_y, log1p_dy) = log1p_frule(x, dx);
    assert_close_f64(log1p_y, x.ln_1p());
    assert_close_f64(log1p_dy, dx / (1.0 + x));
    assert_close_f64(log1p_rrule(x, cotangent), cotangent / (1.0 + x));

    let (sin_y, sin_dy) = sin_frule(x, dx);
    assert_close_f64(sin_y, x.sin());
    assert_close_f64(sin_dy, dx * x.cos());
    assert_close_f64(sin_rrule(x, cotangent), cotangent * x.cos());

    let (cos_y, cos_dy) = cos_frule(x, dx);
    assert_close_f64(cos_y, x.cos());
    assert_close_f64(cos_dy, dx * -x.sin());
    assert_close_f64(cos_rrule(x, cotangent), cotangent * -x.sin());

    let (tanh_y, tanh_dy) = tanh_frule(x, dx);
    assert_close_f64(tanh_y, x.tanh());
    assert_close_f64(tanh_dy, dx * (1.0 - x.tanh().powi(2)));
    assert_close_f64(
        tanh_rrule(tanh_y, cotangent),
        cotangent * (1.0 - x.tanh().powi(2)),
    );

    let trig_x = 0.4_f64;
    let trig_dx = 1.25_f64;
    let (asin_y, asin_dy) = asin_frule(trig_x, trig_dx);
    assert_close_f64(asin_y, trig_x.asin());
    assert_close_f64(asin_dy, trig_dx / (1.0 - trig_x * trig_x).sqrt());
    assert_close_f64(
        asin_rrule(trig_x, cotangent),
        cotangent / (1.0 - trig_x * trig_x).sqrt(),
    );

    let (acos_y, acos_dy) = acos_frule(trig_x, trig_dx);
    assert_close_f64(acos_y, trig_x.acos());
    assert_close_f64(acos_dy, -trig_dx / (1.0 - trig_x * trig_x).sqrt());
    assert_close_f64(
        acos_rrule(trig_x, cotangent),
        -cotangent / (1.0 - trig_x * trig_x).sqrt(),
    );

    let (atan_y, atan_dy) = atan_frule(trig_x, trig_dx);
    assert_close_f64(atan_y, trig_x.atan());
    assert_close_f64(atan_dy, trig_dx / (1.0 + trig_x * trig_x));
    assert_close_f64(
        atan_rrule(trig_x, cotangent),
        cotangent / (1.0 + trig_x * trig_x),
    );

    let (sinh_y, sinh_dy) = sinh_frule(x, dx);
    assert_close_f64(sinh_y, x.sinh());
    assert_close_f64(sinh_dy, dx * x.cosh());
    assert_close_f64(sinh_rrule(x, cotangent), cotangent * x.cosh());

    let (cosh_y, cosh_dy) = cosh_frule(x, dx);
    assert_close_f64(cosh_y, x.cosh());
    assert_close_f64(cosh_dy, dx * x.sinh());
    assert_close_f64(cosh_rrule(x, cotangent), cotangent * x.sinh());

    let (asinh_y, asinh_dy) = asinh_frule(x, dx);
    assert_close_f64(asinh_y, x.asinh());
    assert_close_f64(asinh_dy, dx / (1.0 + x * x).sqrt());
    assert_close_f64(asinh_rrule(x, cotangent), cotangent / (1.0 + x * x).sqrt());

    let acosh_x = 1.75_f64;
    let (acosh_y, acosh_dy) = acosh_frule(acosh_x, dx);
    assert_close_f64(acosh_y, acosh_x.acosh());
    assert_close_f64(
        acosh_dy,
        dx / ((acosh_x - 1.0).sqrt() * (acosh_x + 1.0).sqrt()),
    );
    assert_close_f64(
        acosh_rrule(acosh_x, cotangent),
        cotangent / ((acosh_x - 1.0).sqrt() * (acosh_x + 1.0).sqrt()),
    );

    let atanh_x = 0.2_f64;
    let (atanh_y, atanh_dy) = atanh_frule(atanh_x, dx);
    assert_close_f64(atanh_y, atanh_x.atanh());
    assert_close_f64(atanh_dy, dx / (1.0 - atanh_x * atanh_x));
    assert_close_f64(
        atanh_rrule(atanh_x, cotangent),
        cotangent / (1.0 - atanh_x * atanh_x),
    );
}

#[test]
fn trig_and_hyperbolic_primal_entrypoints_match_std_ops() {
    let real = 0.25_f64;
    assert_close_f64(sin(real), real.sin());
    assert_close_f64(cos(real), real.cos());
    assert_close_f64(tanh(real), real.tanh());
    assert_close_f64(asin(real), real.asin());
    assert_close_f64(acos(real), real.acos());
    assert_close_f64(atan(real), real.atan());
    assert_close_f64(sinh(real), real.sinh());
    assert_close_f64(cosh(real), real.cosh());
    assert_close_f64(asinh(real), real.asinh());
    assert_close_f64(atanh(real), real.atanh());

    let acosh_real = 1.75_f64;
    assert_close_f64(acosh(acosh_real), acosh_real.acosh());

    let complex = Complex64::new(0.25, -0.5);
    assert_close_c64(sin(complex), complex.sin());
    assert_close_c64(cos(complex), complex.cos());
    assert_close_c64(tanh(complex), complex.tanh());
    assert_close_c64(asin(complex), complex.asin());
    assert_close_c64(acos(complex), complex.acos());
    assert_close_c64(atan(complex), complex.atan());
    assert_close_c64(sinh(complex), complex.sinh());
    assert_close_c64(cosh(complex), complex.cosh());
    assert_close_c64(asinh(complex), complex.asinh());
    assert_close_c64(acosh(complex), complex.acosh());
    assert_close_c64(atanh(complex), complex.atanh());
}

#[test]
fn extended_complex_unary_rules_conjugate_their_jacobians() {
    let x = Complex64::new(0.25, -0.5);
    let dx = Complex64::new(-0.75, 0.5);
    let cotangent = Complex64::new(0.5, -1.25);

    let (_sin_y, sin_dy) = sin_frule(x, dx);
    assert_close_c64(sin_dy, dx * x.cos().conj());
    assert_close_c64(sin_rrule(x, cotangent), cotangent * x.cos().conj());

    let (_cos_y, cos_dy) = cos_frule(x, dx);
    assert_close_c64(cos_dy, dx * (-x.sin()).conj());
    assert_close_c64(cos_rrule(x, cotangent), cotangent * (-x.sin()).conj());

    let tanh_y = x.tanh();
    let (_tanh_primal, tanh_dy) = tanh_frule(x, dx);
    assert_close_c64(
        tanh_dy,
        dx * (Complex64::new(1.0, 0.0) - tanh_y * tanh_y).conj(),
    );
    assert_close_c64(
        tanh_rrule(tanh_y, cotangent),
        cotangent * (Complex64::new(1.0, 0.0) - tanh_y * tanh_y).conj(),
    );

    let (_asinh_y, asinh_dy) = asinh_frule(x, dx);
    let asinh_scale = (Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + x * x).sqrt()).conj();
    assert_close_c64(asinh_dy, dx * asinh_scale);
    assert_close_c64(asinh_rrule(x, cotangent), cotangent * asinh_scale);
}
