mod common;

use chainrules::{
    cosd, cosd_frule, cosd_rrule, cospi, cospi_frule, cospi_rrule, cot, cot_frule, cot_rrule, coth,
    coth_frule, coth_rrule, csc, csc_frule, csc_rrule, csch, csch_frule, csch_rrule, sec,
    sec_frule, sec_rrule, sech, sech_frule, sech_rrule, sincospi, sincospi_frule, sincospi_rrule,
    sind, sind_frule, sind_rrule, sinpi, sinpi_frule, sinpi_rrule, tand, tand_frule, tand_rrule,
};
use common::{assert_close_complex64, assert_close_f64};
use num_complex::Complex64;

#[test]
fn julia_compat_landmark_real_inputs_match_julia_style_values() {
    assert_eq!(sinpi(1.0_f64), 0.0_f64);
    assert_eq!(sinpi(0.5_f64), 1.0_f64);
    assert_eq!(cospi(0.5_f64), 0.0_f64);
    let (s, c) = sincospi(0.5_f64);
    assert_eq!(s, 1.0_f64);
    assert_eq!(c, 0.0_f64);
    assert_eq!(sind(180.0_f64), 0.0_f64);
    assert_eq!(cosd(90.0_f64), 0.0_f64);
    assert_eq!(tand(45.0_f64), 1.0_f64);
    assert_eq!(tand(90.0_f64), f64::INFINITY);
    assert_eq!(tand(-90.0_f64), f64::NEG_INFINITY);
    assert_eq!(tand(270.0_f64), f64::NEG_INFINITY);
}

#[test]
fn julia_compat_primal_helpers_match_expected_values() {
    let x = 0.25_f64;
    assert_close_f64(sec(x), 1.0 / x.cos(), 1e-12, 0.0, "sec");
    assert_close_f64(csc(x), 1.0 / x.sin(), 1e-12, 0.0, "csc");
    assert_close_f64(cot(x), 1.0 / x.tan(), 1e-12, 0.0, "cot");
    assert_close_f64(
        sinpi(x),
        (std::f64::consts::PI * x).sin(),
        1e-12,
        0.0,
        "sinpi",
    );
    assert_close_f64(
        cospi(x),
        (std::f64::consts::PI * x).cos(),
        1e-12,
        0.0,
        "cospi",
    );
    let (s, c) = sincospi(x);
    assert_close_f64(
        s,
        (std::f64::consts::PI * x).sin(),
        1e-12,
        0.0,
        "sincospi.sin",
    );
    assert_close_f64(
        c,
        (std::f64::consts::PI * x).cos(),
        1e-12,
        0.0,
        "sincospi.cos",
    );
    assert_close_f64(sind(30.0_f64), 0.5_f64, 1e-12, 0.0, "sind");
    assert_close_f64(cosd(60.0_f64), 0.5_f64, 1e-12, 0.0, "cosd");
    assert_close_f64(tand(45.0_f64), 1.0_f64, 1e-12, 0.0, "tand");
    assert_close_f64(sech(x), 1.0 / x.cosh(), 1e-12, 0.0, "sech");
    assert_close_f64(csch(x), 1.0 / x.sinh(), 1e-12, 0.0, "csch");
    assert_close_f64(coth(x), 1.0 / x.tanh(), 1e-12, 0.0, "coth");
}

#[test]
fn julia_compat_derivative_helpers_match_expected_values() {
    let x = 0.25_f64;
    let g = 1.0_f64;

    let (_, dsec) = sec_frule(x, g);
    assert_close_f64(dsec, x.sin() / x.cos().powi(2), 1e-12, 0.0, "sec_frule");
    assert_close_f64(
        sec_rrule(x, g),
        x.sin() / x.cos().powi(2),
        1e-12,
        0.0,
        "sec_rrule",
    );

    let (_, dsinpi_landmark) = sinpi_frule(1.0_f64, g);
    assert_close_f64(
        dsinpi_landmark,
        -std::f64::consts::PI,
        1e-12,
        0.0,
        "sinpi_frule landmark",
    );
    assert_close_f64(
        sinpi_rrule(1.0_f64, g),
        -std::f64::consts::PI,
        1e-12,
        0.0,
        "sinpi_rrule landmark",
    );

    let (_, dcospi_landmark) = cospi_frule(0.5_f64, g);
    assert_close_f64(
        dcospi_landmark,
        -std::f64::consts::PI,
        1e-12,
        0.0,
        "cospi_frule landmark",
    );
    assert_close_f64(
        cospi_rrule(0.5_f64, g),
        -std::f64::consts::PI,
        1e-12,
        0.0,
        "cospi_rrule landmark",
    );

    let (_, dsinpi) = sinpi_frule(x, g);
    assert_close_f64(
        dsinpi,
        std::f64::consts::PI * (std::f64::consts::PI * x).cos(),
        1e-12,
        0.0,
        "sinpi_frule",
    );
    assert_close_f64(
        sinpi_rrule(x, g),
        std::f64::consts::PI * (std::f64::consts::PI * x).cos(),
        1e-12,
        0.0,
        "sinpi_rrule",
    );

    let (_, dtand) = tand_frule(45.0_f64, g);
    assert_close_f64(
        dtand,
        std::f64::consts::PI / 180.0 * 2.0,
        1e-12,
        0.0,
        "tand_frule",
    );
    assert_close_f64(
        tand_rrule(45.0_f64, g),
        std::f64::consts::PI / 180.0 * 2.0,
        1e-12,
        0.0,
        "tand_rrule",
    );

    let (_, dsech) = sech_frule(x, g);
    let sech_x: f64 = 1.0 / x.cosh();
    assert_close_f64(dsech, -sech_x * x.tanh(), 1e-12, 0.0, "sech_frule");
    assert_close_f64(
        sech_rrule(x, g),
        -sech_x * x.tanh(),
        1e-12,
        0.0,
        "sech_rrule",
    );

    let (_, dcsc) = csc_frule(x, g);
    assert_close_f64(dcsc, -(x.cos() / x.sin().powi(2)), 1e-12, 0.0, "csc_frule");
    assert_close_f64(
        csc_rrule(x, g),
        -(x.cos() / x.sin().powi(2)),
        1e-12,
        0.0,
        "csc_rrule",
    );

    let (_, dcot) = cot_frule(x, g);
    assert_close_f64(dcot, -(1.0 / x.sin().powi(2)), 1e-12, 0.0, "cot_frule");
    assert_close_f64(
        cot_rrule(x, g),
        -(1.0 / x.sin().powi(2)),
        1e-12,
        0.0,
        "cot_rrule",
    );

    let (_, dcsch) = csch_frule(x, g);
    let csch_x: f64 = 1.0 / x.sinh();
    assert_close_f64(
        dcsch,
        -csch_x * x.cosh() / x.sinh(),
        1e-12,
        0.0,
        "csch_frule",
    );
    assert_close_f64(
        csch_rrule(x, g),
        -csch_x * x.cosh() / x.sinh(),
        1e-12,
        0.0,
        "csch_rrule",
    );

    let (_, dcoth) = coth_frule(x, g);
    assert_close_f64(dcoth, -(1.0 / x.sinh().powi(2)), 1e-12, 0.0, "coth_frule");
    assert_close_f64(
        coth_rrule(x, g),
        -(1.0 / x.sinh().powi(2)),
        1e-12,
        0.0,
        "coth_rrule",
    );

    let (_, dsind) = sind_frule(30.0_f64, g);
    assert_close_f64(
        dsind,
        std::f64::consts::PI / 180.0 * (30.0_f64.to_radians()).cos(),
        1e-12,
        0.0,
        "sind_frule",
    );
    assert_close_f64(
        sind_rrule(30.0_f64, g),
        std::f64::consts::PI / 180.0 * (30.0_f64.to_radians()).cos(),
        1e-12,
        0.0,
        "sind_rrule",
    );

    let (_, dcospi) = cospi_frule(x, g);
    assert_close_f64(
        dcospi,
        -std::f64::consts::PI * (std::f64::consts::PI * x).sin(),
        1e-12,
        0.0,
        "cospi_frule",
    );
    assert_close_f64(
        cospi_rrule(x, g),
        -std::f64::consts::PI * (std::f64::consts::PI * x).sin(),
        1e-12,
        0.0,
        "cospi_rrule",
    );

    let (_, dcosd) = cosd_frule(60.0_f64, g);
    assert_close_f64(
        dcosd,
        -std::f64::consts::PI / 180.0 * (60.0_f64.to_radians()).sin(),
        1e-12,
        0.0,
        "cosd_frule",
    );
    assert_close_f64(
        cosd_rrule(60.0_f64, g),
        -std::f64::consts::PI / 180.0 * (60.0_f64.to_radians()).sin(),
        1e-12,
        0.0,
        "cosd_rrule",
    );

    let (_, dsincospi) = sincospi_frule(x, g);
    assert_close_f64(
        dsincospi.0,
        std::f64::consts::PI * (std::f64::consts::PI * x).cos(),
        1e-12,
        0.0,
        "sincospi_frule.sin",
    );
    assert_close_f64(
        dsincospi.1,
        -std::f64::consts::PI * (std::f64::consts::PI * x).sin(),
        1e-12,
        0.0,
        "sincospi_frule.cos",
    );
    assert_close_f64(
        sincospi_rrule(x, (g, g)),
        std::f64::consts::PI * (std::f64::consts::PI * x).cos()
            - std::f64::consts::PI * (std::f64::consts::PI * x).sin(),
        1e-12,
        0.0,
        "sincospi_rrule",
    );
}

#[test]
fn julia_compat_helpers_cover_complex_primal_and_cotangent_paths() {
    let z = Complex64::new(0.25, -0.1);
    let dz = Complex64::new(1.0, -0.25);
    let cotangent = Complex64::new(0.75, -0.5);
    let pi_z = Complex64::new(std::f64::consts::PI, 0.0) * z;

    assert_close_complex64(sinpi(z), pi_z.sin(), 1e-12, 0.0, "sinpi(z)");
    assert_close_complex64(cospi(z), pi_z.cos(), 1e-12, 0.0, "cospi(z)");

    let (_, dsinpi) = sinpi_frule(z, dz);
    assert_close_complex64(
        dsinpi,
        dz * (Complex64::new(std::f64::consts::PI, 0.0) * pi_z.cos()).conj(),
        1e-12,
        0.0,
        "sinpi_frule(z)",
    );

    assert_close_complex64(
        sec_rrule(z, cotangent),
        cotangent * (z.sin() / z.cos().powi(2)).conj(),
        1e-12,
        0.0,
        "sec_rrule(z)",
    );
    assert_close_complex64(
        sinpi_rrule(z, cotangent),
        cotangent * (Complex64::new(std::f64::consts::PI, 0.0) * pi_z.cos()).conj(),
        1e-12,
        0.0,
        "sinpi_rrule(z)",
    );
}

#[test]
fn julia_compat_complex_inputs_cover_generic_surface() {
    let z = Complex64::new(0.25, -0.5);

    let pi_z = Complex64::new(std::f64::consts::PI, 0.0) * z;
    assert_close_complex64(sinpi(z), pi_z.sin(), 1e-12, 0.0, "sinpi(z)");
    assert_close_complex64(cospi(z), pi_z.cos(), 1e-12, 0.0, "cospi(z)");
    let (s, c) = sincospi(z);
    assert_close_complex64(s, pi_z.sin(), 1e-12, 0.0, "sincospi.sin(z)");
    assert_close_complex64(c, pi_z.cos(), 1e-12, 0.0, "sincospi.cos(z)");

    let deg_z = Complex64::new(std::f64::consts::PI / 180.0, 0.0) * z;
    assert_close_complex64(sind(z), deg_z.sin(), 1e-12, 0.0, "sind(z)");
    assert_close_complex64(cosd(z), deg_z.cos(), 1e-12, 0.0, "cosd(z)");
    assert_close_complex64(tand(z), deg_z.tan(), 1e-12, 0.0, "tand(z)");
    assert_close_complex64(
        coth(z),
        Complex64::new(1.0, 0.0) / z.tanh(),
        1e-12,
        0.0,
        "coth(z)",
    );
}
