use chainrules::{
    abs, abs2, abs2_frule, abs2_rrule, angle, angle_rrule, complex, imag, imag_rrule, real,
    real_rrule,
};
use num_complex::Complex64;

#[test]
fn complex_helpers_match_expected_formulas() {
    let z = Complex64::new(3.0, 4.0);
    let dz = Complex64::new(1.0, -2.0);

    let constructed: Complex64 = complex(3.0, 4.0);
    assert_eq!(constructed, z);
    assert_eq!(abs(z), 5.0);
    assert_eq!(abs2(z), 25.0);
    assert_eq!(real(z), 3.0);
    assert_eq!(imag(z), 4.0);
    assert_eq!(angle(z), z.arg());

    let (abs2_y, abs2_dy) = abs2_frule(z, dz);
    assert_eq!(abs2_y, 25.0);
    assert_eq!(abs2_dy, 2.0 * (z.re * dz.re + z.im * dz.im));

    assert_eq!(abs2_rrule(z, 1.25), Complex64::new(7.5, 10.0));
    assert_eq!(real_rrule::<Complex64>(2.0), Complex64::new(2.0, 0.0));
    assert_eq!(imag_rrule::<Complex64>(2.0), Complex64::new(0.0, 2.0));
    assert_eq!(angle_rrule(z, 1.0), Complex64::new(-0.16, 0.12));
}
