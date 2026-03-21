mod common;

use chainrules::{
    abs, abs2, abs2_frule, abs2_rrule, angle, angle_rrule, complex, imag, imag_rrule, real,
    real_rrule,
};
use num_complex::Complex64;

use common::{assert_close_complex64, assert_close_f64};

#[test]
fn complex_helpers_match_expected_formulas() {
    let x = 3.0_f64;
    let z = Complex64::new(3.0, 4.0);
    let dz = Complex64::new(1.0, -2.0);

    let constructed: Complex64 = complex(3.0, 4.0);
    assert_eq!(constructed, z);
    assert_close_f64(abs(x), 3.0, 1.0e-12, 0.0, "abs(x)");
    assert_close_f64(abs2(x), 9.0, 1.0e-12, 0.0, "abs2(x)");
    assert_close_f64(real(x), 3.0, 1.0e-12, 0.0, "real(x)");
    assert_close_f64(imag(x), 0.0, 1.0e-12, 0.0, "imag(x)");
    assert_close_f64(angle(x), 0.0_f64.atan2(x), 1.0e-12, 0.0, "angle(x)");
    assert_close_f64(abs(z), 5.0, 1.0e-12, 0.0, "abs(z)");
    assert_close_f64(abs2(z), 25.0, 1.0e-12, 0.0, "abs2(z)");
    assert_close_f64(real(z), 3.0, 1.0e-12, 0.0, "real(z)");
    assert_close_f64(imag(z), 4.0, 1.0e-12, 0.0, "imag(z)");
    assert_close_f64(angle(z), z.arg(), 1.0e-12, 0.0, "angle(z)");

    let (abs2_y, abs2_dy) = abs2_frule(z, dz);
    assert_close_f64(abs2_y, 25.0, 1.0e-12, 0.0, "abs2.y");
    assert_close_f64(
        abs2_dy,
        2.0 * (z.re * dz.re + z.im * dz.im),
        1.0e-12,
        0.0,
        "abs2.dy",
    );

    assert_close_complex64(
        abs2_rrule(z, 1.25),
        Complex64::new(7.5, 10.0),
        1.0e-12,
        0.0,
        "abs2.rrule",
    );
    let real_grad: Complex64 = real_rrule(2.0);
    assert_close_complex64(
        real_grad,
        Complex64::new(2.0, 0.0),
        1.0e-12,
        0.0,
        "real.rrule",
    );
    let imag_grad: Complex64 = imag_rrule(2.0);
    assert_close_complex64(
        imag_grad,
        Complex64::new(0.0, 2.0),
        1.0e-12,
        0.0,
        "imag.rrule",
    );
    assert_close_complex64(
        angle_rrule(z, 1.0),
        Complex64::new(-0.16, 0.12),
        1.0e-12,
        0.0,
        "angle.rrule",
    );
}
