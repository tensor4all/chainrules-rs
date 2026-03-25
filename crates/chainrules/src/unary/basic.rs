use crate::ScalarAd;

/// Primal `conj`.
pub fn conj<S: ScalarAd>(x: S) -> S {
    x.conj()
}

/// Forward rule for `conj`.
pub fn conj_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    (x.conj(), dx.conj())
}

/// Reverse rule for `conj`.
pub fn conj_rrule<S: ScalarAd>(cotangent: S) -> S {
    cotangent.conj()
}

/// Primal `sqrt`.
pub fn sqrt<S: ScalarAd>(x: S) -> S {
    x.sqrt()
}

/// Forward rule for `sqrt`.
pub fn sqrt_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.sqrt();
    let dy = dx / (S::from_i32(2) * y);
    (y, dy)
}

/// Reverse rule for `sqrt`.
///
/// Takes the forward **result** `sqrt(x)`, not the input `x`.
pub fn sqrt_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent / (S::from_i32(2) * result.conj())
}
