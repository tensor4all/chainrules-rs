use chainrules::{
    cosd, cosd_frule, cosd_rrule, cospi, cospi_frule, cospi_rrule, cot, cot_frule, cot_rrule, coth,
    coth_frule, coth_rrule, csc, csc_frule, csc_rrule, csch, csch_frule, csch_rrule, sec,
    sec_frule, sec_rrule, sech, sech_frule, sech_rrule, sincospi, sincospi_frule, sincospi_rrule,
    sind, sind_frule, sind_rrule, sinpi, sinpi_frule, sinpi_rrule, tand, tand_frule, tand_rrule,
};
use num_complex::Complex64;

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!((actual - expected).norm() < 1e-12);
}

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
}

#[test]
fn julia_compat_primal_helpers_match_expected_values() {
    let x = 0.25_f64;
    assert!((sec(x) - (1.0 / x.cos())).abs() < 1e-12);
    assert!((csc(x) - (1.0 / x.sin())).abs() < 1e-12);
    assert!((cot(x) - (1.0 / x.tan())).abs() < 1e-12);
    assert!((sinpi(x) - (std::f64::consts::PI * x).sin()).abs() < 1e-12);
    assert!((cospi(x) - (std::f64::consts::PI * x).cos()).abs() < 1e-12);
    let (s, c) = sincospi(x);
    assert!((s - (std::f64::consts::PI * x).sin()).abs() < 1e-12);
    assert!((c - (std::f64::consts::PI * x).cos()).abs() < 1e-12);
    assert!((sind(30.0_f64) - 0.5_f64).abs() < 1e-12);
    assert!((cosd(60.0_f64) - 0.5_f64).abs() < 1e-12);
    assert!((tand(45.0_f64) - 1.0_f64).abs() < 1e-12);
    assert!((sech(x) - (1.0 / x.cosh())).abs() < 1e-12);
    assert!((csch(x) - (1.0 / x.sinh())).abs() < 1e-12);
    assert!((coth(x) - (1.0 / x.tanh())).abs() < 1e-12);
}

#[test]
fn julia_compat_derivative_helpers_match_expected_values() {
    let x = 0.25_f64;
    let g = 1.0_f64;

    let (_, dsec) = sec_frule(x, g);
    assert!((dsec - (x.sin() / x.cos().powi(2))).abs() < 1e-12);
    assert!((sec_rrule(x, g) - (x.sin() / x.cos().powi(2))).abs() < 1e-12);

    let (_, dsinpi_landmark) = sinpi_frule(1.0_f64, g);
    assert!((dsinpi_landmark + std::f64::consts::PI).abs() < 1e-12);
    assert!((sinpi_rrule(1.0_f64, g) + std::f64::consts::PI).abs() < 1e-12);

    let (_, dcospi_landmark) = cospi_frule(0.5_f64, g);
    assert!((dcospi_landmark + std::f64::consts::PI).abs() < 1e-12);
    assert!((cospi_rrule(0.5_f64, g) + std::f64::consts::PI).abs() < 1e-12);

    let (_, dsinpi) = sinpi_frule(x, g);
    assert!((dsinpi - std::f64::consts::PI * (std::f64::consts::PI * x).cos()).abs() < 1e-12);
    assert!(
        (sinpi_rrule(x, g) - std::f64::consts::PI * (std::f64::consts::PI * x).cos()).abs() < 1e-12
    );

    let (_, dtand) = tand_frule(45.0_f64, g);
    assert!((dtand - std::f64::consts::PI / 180.0 * 2.0).abs() < 1e-12);
    assert!((tand_rrule(45.0_f64, g) - std::f64::consts::PI / 180.0 * 2.0).abs() < 1e-12);

    let (_, dsech) = sech_frule(x, g);
    let sech_x: f64 = 1.0 / x.cosh();
    assert!((dsech - (-sech_x * x.tanh())).abs() < 1e-12);
    assert!((sech_rrule(x, g) - (-sech_x * x.tanh())).abs() < 1e-12);

    let (_, dcsc) = csc_frule(x, g);
    assert!((dcsc - (-(x.cos() / x.sin().powi(2)))).abs() < 1e-12);
    assert!((csc_rrule(x, g) - (-(x.cos() / x.sin().powi(2)))).abs() < 1e-12);

    let (_, dcot) = cot_frule(x, g);
    assert!((dcot - (-(1.0 / x.sin().powi(2)))).abs() < 1e-12);
    assert!((cot_rrule(x, g) - (-(1.0 / x.sin().powi(2)))).abs() < 1e-12);

    let (_, dcsch) = csch_frule(x, g);
    let csch_x: f64 = 1.0 / x.sinh();
    assert!((dcsch - (-csch_x * x.cosh() / x.sinh())).abs() < 1e-12);
    assert!((csch_rrule(x, g) - (-csch_x * x.cosh() / x.sinh())).abs() < 1e-12);

    let (_, dcoth) = coth_frule(x, g);
    assert!((dcoth - (-(1.0 / x.sinh().powi(2)))).abs() < 1e-12);
    assert!((coth_rrule(x, g) - (-(1.0 / x.sinh().powi(2)))).abs() < 1e-12);

    let (_, dsind) = sind_frule(30.0_f64, g);
    assert!((dsind - (std::f64::consts::PI / 180.0 * (30.0_f64.to_radians()).cos())).abs() < 1e-12);
    assert!(
        (sind_rrule(30.0_f64, g) - (std::f64::consts::PI / 180.0 * (30.0_f64.to_radians()).cos()))
            .abs()
            < 1e-12
    );

    let (_, dcospi) = cospi_frule(x, g);
    assert!((dcospi + std::f64::consts::PI * (std::f64::consts::PI * x).sin()).abs() < 1e-12);
    assert!(
        (cospi_rrule(x, g) + std::f64::consts::PI * (std::f64::consts::PI * x).sin()).abs() < 1e-12
    );

    let (_, dcosd) = cosd_frule(60.0_f64, g);
    assert!((dcosd + std::f64::consts::PI / 180.0 * (60.0_f64.to_radians()).sin()).abs() < 1e-12);
    assert!(
        (cosd_rrule(60.0_f64, g) + std::f64::consts::PI / 180.0 * (60.0_f64.to_radians()).sin())
            .abs()
            < 1e-12
    );

    let (_, dsincospi) = sincospi_frule(x, g);
    assert!((dsincospi.0 - std::f64::consts::PI * (std::f64::consts::PI * x).cos()).abs() < 1e-12);
    assert!((dsincospi.1 + std::f64::consts::PI * (std::f64::consts::PI * x).sin()).abs() < 1e-12);
    assert!(
        (sincospi_rrule(x, (g, g))
            - (std::f64::consts::PI * (std::f64::consts::PI * x).cos()
                - std::f64::consts::PI * (std::f64::consts::PI * x).sin()))
        .abs()
            < 1e-12
    );
}

#[test]
fn julia_compat_landmark_inputs_match_expected_values() {
    assert_eq!(sinpi(1.0_f64), 0.0);
    assert_eq!(cospi(0.5_f64), 0.0);

    let (s, c) = sincospi(0.5_f64);
    assert_eq!(s, 1.0);
    assert_eq!(c, 0.0);

    assert_eq!(sind(180.0_f64), 0.0);
    assert_eq!(cosd(90.0_f64), 0.0);
    assert_eq!(tand(45.0_f64), 1.0);
    assert!(tand(90.0_f64).is_infinite());
}

#[test]
fn julia_compat_helpers_cover_complex_primal_and_cotangent_paths() {
    let z = Complex64::new(0.25, -0.1);
    let dz = Complex64::new(1.0, -0.25);
    let cotangent = Complex64::new(0.75, -0.5);
    let pi_z = Complex64::new(std::f64::consts::PI, 0.0) * z;

    assert_complex_close(sinpi(z), pi_z.sin());
    assert_complex_close(cospi(z), pi_z.cos());

    let (_, dsinpi) = sinpi_frule(z, dz);
    assert_complex_close(
        dsinpi,
        dz * (Complex64::new(std::f64::consts::PI, 0.0) * pi_z.cos()).conj(),
    );

    assert_complex_close(
        sec_rrule(z, cotangent),
        cotangent * (z.sin() / z.cos().powi(2)).conj(),
    );
    assert_complex_close(
        sinpi_rrule(z, cotangent),
        cotangent * (Complex64::new(std::f64::consts::PI, 0.0) * pi_z.cos()).conj(),
    );
}

#[test]
fn julia_compat_complex_inputs_cover_generic_surface() {
    let z = Complex64::new(0.25, -0.5);

    let pi_z = Complex64::new(std::f64::consts::PI, 0.0) * z;
    assert!((sinpi(z) - pi_z.sin()).norm() < 1e-12);
    assert!((cospi(z) - pi_z.cos()).norm() < 1e-12);
    let (s, c) = sincospi(z);
    assert!((s - pi_z.sin()).norm() < 1e-12);
    assert!((c - pi_z.cos()).norm() < 1e-12);

    let deg_z = Complex64::new(std::f64::consts::PI / 180.0, 0.0) * z;
    assert!((sind(z) - deg_z.sin()).norm() < 1e-12);
    assert!((cosd(z) - deg_z.cos()).norm() < 1e-12);
    assert!((tand(z) - deg_z.tan()).norm() < 1e-12);
    assert!((coth(z) - Complex64::new(1.0, 0.0) / z.tanh()).norm() < 1e-12);
}
