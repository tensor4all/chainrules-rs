use chainrules::{
    cbrt, cbrt_frule, cbrt_rrule, exp10, exp10_frule, exp10_rrule, exp2, exp2_frule, exp2_rrule,
    hypot, hypot_frule, hypot_rrule, inv, inv_frule, inv_rrule, log10, log10_frule, log10_rrule,
    log2, log2_frule, log2_rrule, pow, pow_frule, pow_rrule, sincos, sincos_frule, sincos_rrule,
    tan, tan_frule, tan_rrule,
};
use num_complex::{Complex64, ComplexFloat};

#[test]
fn smooth_basis_helpers_are_reexported_from_chainrules() {
    assert!((cbrt(8.0_f64) - 2.0).abs() < 1.0e-12);
    assert!((inv(4.0_f64) - 0.25).abs() < 1.0e-12);
    assert!((exp2(3.0_f64) - 8.0).abs() < 1.0e-12);
    assert!((exp10(2.0_f64) - 100.0).abs() < 1.0e-12);
    assert!((hypot(3.0_f64, 4.0_f64) - 5.0).abs() < 1.0e-12);
    assert!((log2(8.0_f64) - 3.0).abs() < 1.0e-12);
    assert!((log10(100.0_f64) - 2.0).abs() < 1.0e-12);
    assert!((pow(2.0_f64, 3.0_f64) - 8.0).abs() < 1.0e-12);
    assert!((tan(0.5_f64) - 0.5_f64.tan()).abs() < 1.0e-12);
    let (sin_x, cos_x) = sincos(0.5_f64);
    assert!((sin_x - 0.5_f64.sin()).abs() < 1.0e-12);
    assert!((cos_x - 0.5_f64.cos()).abs() < 1.0e-12);

    let z = Complex64::new(1.0, 2.0);
    let _ = pow(z, Complex64::new(2.0, 0.0));
}

#[test]
fn smooth_basis_frules_and_rrules_match_expected_derivatives() {
    let (tan_y, tan_dy) = tan_frule(0.25_f64, 1.0_f64);
    assert!((tan_y - 0.25_f64.tan()).abs() < 1.0e-12);
    assert!((tan_dy - (1.0_f64 + 0.25_f64.tan().powi(2))).abs() < 1.0e-12);
    assert!(
        (tan_rrule(0.25_f64.tan(), 1.0_f64) - (1.0_f64 + 0.25_f64.tan().powi(2))).abs() < 1.0e-12
    );

    let (exp2_y, exp2_dy) = exp2_frule(3.0_f64, 1.0_f64);
    assert!((exp2_y - 8.0).abs() < 1.0e-12);
    assert!((exp2_dy - 8.0_f64 * std::f64::consts::LN_2).abs() < 1.0e-12);
    assert!((exp2_rrule(8.0_f64, 1.0_f64) - 8.0_f64 * std::f64::consts::LN_2).abs() < 1.0e-12);

    let (hypot_y, hypot_dy) = hypot_frule(3.0_f64, 4.0_f64, 0.5_f64, 0.25_f64);
    assert!((hypot_y - 5.0).abs() < 1.0e-12);
    assert!((hypot_dy - 0.5).abs() < 1.0e-12);
    assert!((hypot_rrule(3.0_f64, 4.0_f64, 1.0_f64).0 - 0.6_f64).abs() < 1.0e-12);
    assert!((hypot_rrule(3.0_f64, 4.0_f64, 1.0_f64).1 - 0.8_f64).abs() < 1.0e-12);

    let (pow_y, pow_dy) = pow_frule(2.0_f64, 3.0_f64, 1.0_f64, 0.0_f64);
    assert!((pow_y - 8.0).abs() < 1.0e-12);
    assert!((pow_dy - 12.0).abs() < 1.0e-12);
    let (pow_dx, pow_dexp) = pow_rrule(2.0_f64, 3.0_f64, 1.0_f64);
    assert!((pow_dx - 12.0).abs() < 1.0e-12);
    assert!((pow_dexp - (8.0_f64 * std::f64::consts::LN_2)).abs() < 1.0e-12);

    let (sincos_y, sincos_dy) = sincos_frule(0.25_f64, 1.0_f64);
    assert!((sincos_y.0 - 0.25_f64.sin()).abs() < 1.0e-12);
    assert!((sincos_y.1 - 0.25_f64.cos()).abs() < 1.0e-12);
    assert!((sincos_dy.0 - 0.25_f64.cos()).abs() < 1.0e-12);
    assert!((sincos_dy.1 + 0.25_f64.sin()).abs() < 1.0e-12);
    assert!(
        (sincos_rrule(0.25_f64, (1.0_f64, 1.0_f64)) - (0.25_f64.cos() - 0.25_f64.sin())).abs()
            < 1.0e-12
    );

    let (cbrt_y, cbrt_dy) = cbrt_frule(8.0_f64, 1.0_f64);
    assert!((cbrt_y - 2.0).abs() < 1.0e-12);
    assert!((cbrt_dy - (1.0_f64 / (3.0_f64 * 4.0_f64))).abs() < 1.0e-12);
    assert!((cbrt_rrule(2.0_f64, 1.0_f64) - (1.0_f64 / (3.0_f64 * 4.0_f64))).abs() < 1.0e-12);

    let (inv_y, inv_dy) = inv_frule(4.0_f64, 2.0_f64);
    assert!((inv_y - 0.25).abs() < 1.0e-12);
    assert!((inv_dy + 0.125).abs() < 1.0e-12);
    assert!((inv_rrule(0.25_f64, 2.0_f64) + 0.125).abs() < 1.0e-12);

    let (log2_y, log2_dy) = log2_frule(8.0_f64, 2.0_f64);
    assert!((log2_y - 3.0).abs() < 1.0e-12);
    let expected_log2 = 2.0_f64 / (8.0_f64 * std::f64::consts::LN_2);
    assert!((log2_dy - expected_log2).abs() < 1.0e-12);
    assert!((log2_rrule(8.0_f64, 2.0_f64) - expected_log2).abs() < 1.0e-12);

    let (log10_y, log10_dy) = log10_frule(100.0_f64, 2.0_f64);
    assert!((log10_y - 2.0).abs() < 1.0e-12);
    assert!((log10_dy - (2.0_f64 / (100.0_f64 * std::f64::consts::LN_10))).abs() < 1.0e-12);
    assert!(
        (log10_rrule(100.0_f64, 2.0_f64) - (2.0_f64 / (100.0_f64 * std::f64::consts::LN_10))).abs()
            < 1.0e-12
    );

    let (exp10_y, exp10_dy) = exp10_frule(2.0_f64, 0.5_f64);
    assert!((exp10_y - 100.0).abs() < 1.0e-12);
    assert!((exp10_dy - (100.0_f64 * std::f64::consts::LN_10 * 0.5_f64)).abs() < 1.0e-12);
    assert!(
        (exp10_rrule(100.0_f64, 0.5_f64) - (100.0_f64 * std::f64::consts::LN_10 * 0.5_f64)).abs()
            < 1.0e-12
    );
}

#[test]
fn pow_rules_handle_zero_and_negative_real_paths() {
    let (neg_y, neg_dy) = pow_frule(-2.0_f64, 3.0_f64, 1.0_f64, 0.0_f64);
    assert!((neg_y + 8.0).abs() < 1.0e-12);
    assert!((neg_dy - 12.0).abs() < 1.0e-12);

    let (zero_y, zero_dy) = pow_frule(0.0_f64, 2.0_f64, 1.0_f64, 0.0_f64);
    assert!((zero_y - 0.0).abs() < 1.0e-12);
    assert!((zero_dy - 0.0).abs() < 1.0e-12);

    let (dx, dexp) = pow_rrule(0.0_f64, 2.0_f64, 1.0_f64);
    assert!((dx - 0.0).abs() < 1.0e-12);
    assert!((dexp - 0.0).abs() < 1.0e-12);
}

#[test]
fn pow_rules_cover_complex_frule_and_rrule_paths() {
    let x = Complex64::new(1.0, 1.0);
    let exponent = Complex64::new(2.0, 0.0);
    let dx = Complex64::new(0.5, -0.25);
    let dexp = Complex64::new(0.0, 0.0);

    let (y, dy) = pow_frule(x, exponent, dx, dexp);
    let expected_y = x.powc(exponent);
    let expected_dy = dx * (exponent * x.powc(exponent - Complex64::new(1.0, 0.0))).conj();
    assert!((y - expected_y).norm() < 1.0e-12);
    assert!((dy - expected_dy).norm() < 1.0e-12);

    let cotangent = Complex64::new(0.5, -0.25);
    let (dx_rr, dexp_rr) = pow_rrule(x, exponent, cotangent);
    let expected_dx_rr =
        cotangent * (exponent * x.powc(exponent - Complex64::new(1.0, 0.0))).conj();
    let expected_dexp_rr = cotangent * (expected_y * x.ln()).conj();
    assert!((dx_rr - expected_dx_rr).norm() < 1.0e-12);
    assert!((dexp_rr - expected_dexp_rr).norm() < 1.0e-12);

    let imag_x = Complex64::new(0.0, 1.0);
    let (_, imag_dexp_rr) = pow_rrule(imag_x, Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0));
    let expected_imag_dexp_rr = (imag_x.powc(Complex64::new(2.0, 0.0)) * imag_x.ln()).conj();
    assert!((imag_dexp_rr - expected_imag_dexp_rr).norm() < 1.0e-12);
}
