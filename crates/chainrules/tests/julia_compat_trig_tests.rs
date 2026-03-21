use chainrules::{
    cosd, cosd_frule, cosd_rrule, cospi, cospi_frule, cospi_rrule, cot, cot_frule, cot_rrule, csc,
    csc_frule, csc_rrule, csch, csch_frule, csch_rrule, sec, sec_frule, sec_rrule, sech,
    sech_frule, sech_rrule, sincospi, sincospi_frule, sincospi_rrule, sind, sind_frule, sind_rrule,
    sinpi, sinpi_frule, sinpi_rrule, tand, tand_frule, tand_rrule,
};

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
}

#[test]
fn julia_compat_derivative_helpers_match_expected_values() {
    let x = 0.25_f64;
    let g = 1.0_f64;

    let (_, dsec) = sec_frule(x, g);
    assert!((dsec - (x.sin() / x.cos().powi(2))).abs() < 1e-12);
    assert!((sec_rrule(x, g) - (x.sin() / x.cos().powi(2))).abs() < 1e-12);

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
