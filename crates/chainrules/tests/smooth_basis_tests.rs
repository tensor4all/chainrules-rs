use chainrules::{cbrt, exp2, hypot, log2, pow, tan};
use num_complex::Complex64;

#[test]
fn smooth_basis_helpers_are_reexported_from_chainrules() {
    assert!((cbrt(8.0_f64) - 2.0).abs() < 1.0e-12);
    assert!((exp2(3.0_f64) - 8.0).abs() < 1.0e-12);
    assert!((hypot(3.0_f64, 4.0_f64) - 5.0).abs() < 1.0e-12);
    assert!((log2(8.0_f64) - 3.0).abs() < 1.0e-12);
    assert!((pow(2.0_f64, 3.0_f64) - 8.0).abs() < 1.0e-12);
    assert!((tan(0.5_f64) - 0.5_f64.tan()).abs() < 1.0e-12);

    let z = Complex64::new(1.0, 2.0);
    let _ = pow(z, Complex64::new(2.0, 0.0));
}
