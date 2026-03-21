mod common;

use chainrules::{
    cbrt, cbrt_frule, cbrt_rrule, exp10, exp10_frule, exp10_rrule, exp2, exp2_frule, exp2_rrule,
    hypot, hypot_frule, hypot_rrule, inv, inv_frule, inv_rrule, log10, log10_frule, log10_rrule,
    log2, log2_frule, log2_rrule, pow, pow_frule, pow_rrule, sincos, sincos_frule, sincos_rrule,
    tan, tan_frule, tan_rrule,
};
use num_complex::{Complex64, ComplexFloat};

use common::{assert_close_complex64, assert_close_f64};

#[test]
fn smooth_basis_helpers_are_reexported_from_chainrules() {
    assert_close_f64(cbrt(8.0_f64), 2.0, 1.0e-12, 0.0, "cbrt");
    assert_close_f64(inv(4.0_f64), 0.25, 1.0e-12, 0.0, "inv");
    assert_close_f64(exp2(3.0_f64), 8.0, 1.0e-12, 0.0, "exp2");
    assert_close_f64(exp10(2.0_f64), 100.0, 1.0e-12, 0.0, "exp10");
    assert_close_f64(hypot(3.0_f64, 4.0_f64), 5.0, 1.0e-12, 0.0, "hypot");
    assert_close_f64(log2(8.0_f64), 3.0, 1.0e-12, 0.0, "log2");
    assert_close_f64(log10(100.0_f64), 2.0, 1.0e-12, 0.0, "log10");
    assert_close_f64(pow(2.0_f64, 3.0_f64), 8.0, 1.0e-12, 0.0, "pow");
    assert_close_f64(tan(0.5_f64), 0.5_f64.tan(), 1.0e-12, 0.0, "tan");
    let (sin_x, cos_x) = sincos(0.5_f64);
    assert_close_f64(sin_x, 0.5_f64.sin(), 1.0e-12, 0.0, "sincos.sin");
    assert_close_f64(cos_x, 0.5_f64.cos(), 1.0e-12, 0.0, "sincos.cos");

    let z = Complex64::new(1.0, 2.0);
    let _ = pow(z, Complex64::new(2.0, 0.0));
}

#[test]
fn smooth_basis_frules_and_rrules_match_expected_derivatives() {
    let (tan_y, tan_dy) = tan_frule(0.25_f64, 1.0_f64);
    assert_close_f64(tan_y, 0.25_f64.tan(), 1.0e-12, 0.0, "tan.y");
    assert_close_f64(
        tan_dy,
        1.0_f64 + 0.25_f64.tan().powi(2),
        1.0e-12,
        0.0,
        "tan.dy",
    );
    assert_close_f64(
        tan_rrule(0.25_f64.tan(), 1.0_f64),
        1.0_f64 + 0.25_f64.tan().powi(2),
        1.0e-12,
        0.0,
        "tan.rrule",
    );

    let (exp2_y, exp2_dy) = exp2_frule(3.0_f64, 1.0_f64);
    assert_close_f64(exp2_y, 8.0, 1.0e-12, 0.0, "exp2.y");
    assert_close_f64(
        exp2_dy,
        8.0_f64 * std::f64::consts::LN_2,
        1.0e-12,
        0.0,
        "exp2.dy",
    );
    assert_close_f64(
        exp2_rrule(8.0_f64, 1.0_f64),
        8.0_f64 * std::f64::consts::LN_2,
        1.0e-12,
        0.0,
        "exp2.rrule",
    );

    let (hypot_y, hypot_dy) = hypot_frule(3.0_f64, 4.0_f64, 0.5_f64, 0.25_f64);
    assert_close_f64(hypot_y, 5.0, 1.0e-12, 0.0, "hypot.y");
    assert_close_f64(hypot_dy, 0.5, 1.0e-12, 0.0, "hypot.dy");
    let (hypot_dx, hypot_dy) = hypot_rrule(3.0_f64, 4.0_f64, 1.0_f64);
    assert_close_f64(hypot_dx, 0.6_f64, 1.0e-12, 0.0, "hypot.rrule.dx");
    assert_close_f64(hypot_dy, 0.8_f64, 1.0e-12, 0.0, "hypot.rrule.dy");

    let (pow_y, pow_dy) = pow_frule(2.0_f64, 3.0_f64, 1.0_f64, 0.0_f64);
    assert_close_f64(pow_y, 8.0, 1.0e-12, 0.0, "pow.y");
    assert_close_f64(pow_dy, 12.0, 1.0e-12, 0.0, "pow.dy");
    let (pow_dx, pow_dexp) = pow_rrule(2.0_f64, 3.0_f64, 1.0_f64);
    assert_close_f64(pow_dx, 12.0, 1.0e-12, 0.0, "pow.rrule.dx");
    assert_close_f64(
        pow_dexp,
        8.0_f64 * std::f64::consts::LN_2,
        1.0e-12,
        0.0,
        "pow.rrule.dexp",
    );

    let (pow_y, pow_dy) = pow_frule(2.0_f64, 3.0_f64, 1.0_f64, 0.5_f64);
    assert_close_f64(pow_y, 8.0, 1.0e-12, 0.0, "pow.y.dexp");
    assert_close_f64(
        pow_dy,
        12.0 + 0.5_f64 * 8.0_f64 * std::f64::consts::LN_2,
        1.0e-12,
        0.0,
        "pow.dy.dexp",
    );

    let (sincos_y, sincos_dy) = sincos_frule(0.25_f64, 1.0_f64);
    assert_close_f64(sincos_y.0, 0.25_f64.sin(), 1.0e-12, 0.0, "sincos.y.sin");
    assert_close_f64(sincos_y.1, 0.25_f64.cos(), 1.0e-12, 0.0, "sincos.y.cos");
    assert_close_f64(sincos_dy.0, 0.25_f64.cos(), 1.0e-12, 0.0, "sincos.dy.sin");
    assert_close_f64(sincos_dy.1, -0.25_f64.sin(), 1.0e-12, 0.0, "sincos.dy.cos");
    assert_close_f64(
        sincos_rrule(0.25_f64, (1.0_f64, 1.0_f64)),
        0.25_f64.cos() - 0.25_f64.sin(),
        1.0e-12,
        0.0,
        "sincos.rrule",
    );

    let (cbrt_y, cbrt_dy) = cbrt_frule(8.0_f64, 1.0_f64);
    assert_close_f64(cbrt_y, 2.0, 1.0e-12, 0.0, "cbrt.y");
    assert_close_f64(
        cbrt_dy,
        1.0_f64 / (3.0_f64 * 4.0_f64),
        1.0e-12,
        0.0,
        "cbrt.dy",
    );
    assert_close_f64(
        cbrt_rrule(2.0_f64, 1.0_f64),
        1.0_f64 / (3.0_f64 * 4.0_f64),
        1.0e-12,
        0.0,
        "cbrt.rrule",
    );

    let (inv_y, inv_dy) = inv_frule(4.0_f64, 2.0_f64);
    assert_close_f64(inv_y, 0.25, 1.0e-12, 0.0, "inv.y");
    assert_close_f64(inv_dy, -0.125, 1.0e-12, 0.0, "inv.dy");
    assert_close_f64(
        inv_rrule(0.25_f64, 2.0_f64),
        -0.125,
        1.0e-12,
        0.0,
        "inv.rrule",
    );

    let (log2_y, log2_dy) = log2_frule(8.0_f64, 2.0_f64);
    assert_close_f64(log2_y, 3.0, 1.0e-12, 0.0, "log2.y");
    let expected_log2 = 2.0_f64 / (8.0_f64 * std::f64::consts::LN_2);
    assert_close_f64(log2_dy, expected_log2, 1.0e-12, 0.0, "log2.dy");
    assert_close_f64(
        log2_rrule(8.0_f64, 2.0_f64),
        expected_log2,
        1.0e-12,
        0.0,
        "log2.rrule",
    );

    let (log10_y, log10_dy) = log10_frule(100.0_f64, 2.0_f64);
    assert_close_f64(log10_y, 2.0, 1.0e-12, 0.0, "log10.y");
    let expected_log10 = 2.0_f64 / (100.0_f64 * std::f64::consts::LN_10);
    assert_close_f64(log10_dy, expected_log10, 1.0e-12, 0.0, "log10.dy");
    assert_close_f64(
        log10_rrule(100.0_f64, 2.0_f64),
        expected_log10,
        1.0e-12,
        0.0,
        "log10.rrule",
    );

    let (exp10_y, exp10_dy) = exp10_frule(2.0_f64, 0.5_f64);
    assert_close_f64(exp10_y, 100.0, 1.0e-12, 0.0, "exp10.y");
    let expected_exp10 = 100.0_f64 * std::f64::consts::LN_10 * 0.5_f64;
    assert_close_f64(exp10_dy, expected_exp10, 1.0e-12, 0.0, "exp10.dy");
    assert_close_f64(
        exp10_rrule(100.0_f64, 0.5_f64),
        expected_exp10,
        1.0e-12,
        0.0,
        "exp10.rrule",
    );
}

#[test]
fn smooth_basis_complex_frules_match_expected_derivatives() {
    let z = Complex64::new(0.25, -0.5);
    let dz = Complex64::new(0.5, -0.25);

    let (tan_y, tan_dy) = tan_frule(z, dz);
    let tan_scale = (Complex64::new(1.0, 0.0) + tan_y * tan_y).conj();
    assert_close_complex64(tan_y, z.tan(), 1.0e-12, 0.0, "tan.z");
    assert_close_complex64(tan_dy, dz * tan_scale, 1.0e-12, 0.0, "tan.dz");

    let (exp2_y, exp2_dy) = exp2_frule(z, dz);
    let exp2_scale = (exp2_y * Complex64::new(std::f64::consts::LN_2, 0.0)).conj();
    assert_close_complex64(exp2_y, z.exp2(), 1.0e-12, 0.0, "exp2.z");
    assert_close_complex64(exp2_dy, dz * exp2_scale, 1.0e-12, 0.0, "exp2.dz");

    let (log2_y, log2_dy) = log2_frule(z, dz);
    let log2_scale =
        (Complex64::new(1.0, 0.0) / (z * Complex64::new(std::f64::consts::LN_2, 0.0))).conj();
    assert_close_complex64(log2_y, z.log2(), 1.0e-12, 0.0, "log2.z");
    assert_close_complex64(log2_dy, dz * log2_scale, 1.0e-12, 0.0, "log2.dz");
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
fn pow_rules_mark_zero_base_exponent_singularities_for_real_inputs() {
    let (_, zero_zero_dy) = pow_frule(0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64);
    assert!(zero_zero_dy.is_nan());

    let (_, zero_neg_dy) = pow_frule(0.0_f64, -1.0_f64, 0.0_f64, 1.0_f64);
    assert!(zero_neg_dy.is_nan());

    let (_, zero_zero_dexp) = pow_rrule(0.0_f64, 0.0_f64, 1.0_f64);
    assert!(zero_zero_dexp.is_nan());

    let (_, zero_neg_dexp) = pow_rrule(0.0_f64, -1.0_f64, 1.0_f64);
    assert!(zero_neg_dexp.is_nan());

    let (_, zero_zero_dy32) = pow_frule(0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32);
    assert!(zero_zero_dy32.is_nan());

    let (_, zero_neg_dexp32) = pow_rrule(0.0_f32, -1.0_f32, 1.0_f32);
    assert!(zero_neg_dexp32.is_nan());
}

#[test]
fn pow_rules_cover_complex_frule_and_rrule_paths() {
    let x = Complex64::new(1.0, 1.0);
    let exponent = Complex64::new(2.0, 0.5);
    let dx = Complex64::new(0.5, -0.25);
    let dexp = Complex64::new(0.1, -0.2);

    let (y, dy) = pow_frule(x, exponent, dx, dexp);
    let expected_y = x.powc(exponent);
    let expected_dy = dx * (exponent * x.powc(exponent - Complex64::new(1.0, 0.0))).conj()
        + dexp * (expected_y * x.ln()).conj();
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
