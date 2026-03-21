use chainrules::{
    ceil, ceil_frule, ceil_rrule, floor, floor_frule, floor_rrule, max, max_frule, max_rrule, min,
    min_frule, min_rrule, round, round_frule, round_rrule, sign, sign_frule, sign_rrule,
};
use num_traits::Float;

fn assert_close<T>(actual: T, expected: T)
where
    T: core::fmt::Debug + PartialEq,
{
    assert_eq!(actual, expected);
}

fn assert_zero<T>(actual: T)
where
    T: core::fmt::Debug + PartialEq + Float,
{
    assert_eq!(actual, T::zero());
}

fn cast<T>(value: f32) -> T
where
    T: Float,
{
    T::from(value).expect("cast to float")
}

fn check_nonsmooth_scalar_rules<T>()
where
    T: Copy + core::fmt::Debug + PartialEq + Float,
{
    let x = cast::<T>(1.6_f32);
    let y = cast::<T>(-2.4_f32);

    assert_close(round(x), cast::<T>(2.0_f32));
    assert_close(floor(x), cast::<T>(1.0_f32));
    assert_close(ceil(y), cast::<T>(-2.0_f32));
    assert_close(sign(y), cast::<T>(-1.0_f32));

    let (round_y, round_dy) = round_frule(x, cast::<T>(7.0_f32));
    assert_close(round_y, cast::<T>(2.0_f32));
    assert_zero(round_dy);
    assert_zero(round_rrule(cast::<T>(7.0_f32)));

    let (floor_y, floor_dy) = floor_frule(x, cast::<T>(5.0_f32));
    assert_close(floor_y, cast::<T>(1.0_f32));
    assert_zero(floor_dy);
    assert_zero(floor_rrule(cast::<T>(5.0_f32)));

    let (ceil_y, ceil_dy) = ceil_frule(y, cast::<T>(11.0_f32));
    assert_close(ceil_y, cast::<T>(-2.0_f32));
    assert_zero(ceil_dy);
    assert_zero(ceil_rrule(cast::<T>(11.0_f32)));

    let (sign_y, sign_dy) = sign_frule(y, cast::<T>(3.0_f32));
    assert_close(sign_y, cast::<T>(-1.0_f32));
    assert_zero(sign_dy);
    assert_zero(sign_rrule(cast::<T>(3.0_f32)));

    let (min_y, min_dy) = min_frule(
        cast::<T>(1.0_f32),
        cast::<T>(2.0_f32),
        cast::<T>(4.0_f32),
        cast::<T>(8.0_f32),
    );
    assert_close(
        min(cast::<T>(1.0_f32), cast::<T>(2.0_f32)),
        cast::<T>(1.0_f32),
    );
    assert_close(min_y, cast::<T>(1.0_f32));
    assert_close(min_dy, cast::<T>(4.0_f32));
    let (min_tie_y, min_tie_dy) = min_frule(
        cast::<T>(3.0_f32),
        cast::<T>(3.0_f32),
        cast::<T>(4.0_f32),
        cast::<T>(8.0_f32),
    );
    assert_close(min_tie_y, cast::<T>(3.0_f32));
    assert_close(min_tie_dy, cast::<T>(8.0_f32));
    let (min_dx, min_dy) = min_rrule(cast::<T>(3.0_f32), cast::<T>(3.0_f32), cast::<T>(6.0_f32));
    assert_zero(min_dx);
    assert_close(min_dy, cast::<T>(6.0_f32));

    let (max_y, max_dy) = max_frule(
        cast::<T>(1.0_f32),
        cast::<T>(2.0_f32),
        cast::<T>(4.0_f32),
        cast::<T>(8.0_f32),
    );
    assert_close(
        max(cast::<T>(1.0_f32), cast::<T>(2.0_f32)),
        cast::<T>(2.0_f32),
    );
    assert_close(max_y, cast::<T>(2.0_f32));
    assert_close(max_dy, cast::<T>(8.0_f32));
    let (max_tie_y, max_tie_dy) = max_frule(
        cast::<T>(3.0_f32),
        cast::<T>(3.0_f32),
        cast::<T>(4.0_f32),
        cast::<T>(8.0_f32),
    );
    assert_close(max_tie_y, cast::<T>(3.0_f32));
    assert_close(max_tie_dy, cast::<T>(8.0_f32));
    let (max_dx, max_dy) = max_rrule(cast::<T>(3.0_f32), cast::<T>(3.0_f32), cast::<T>(6.0_f32));
    assert_zero(max_dx);
    assert_close(max_dy, cast::<T>(6.0_f32));
}

#[test]
fn nonsmooth_scalar_rules_match_expected_policy_for_f64() {
    check_nonsmooth_scalar_rules::<f64>();
}

#[test]
fn nonsmooth_scalar_rules_match_expected_policy_for_f32() {
    check_nonsmooth_scalar_rules::<f32>();
}
