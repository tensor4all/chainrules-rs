# chainrules

`chainrules` provides a shared scalar rule basis for Rust automatic
differentiation crates.

It is designed for reusable scalar calculus, not for tapes, traced values, or
tensor-specific execution engines. The crate focuses on stateless helpers that
can be called from downstream AD runtimes and tensor libraries.

## What It Provides

- stateless scalar primal helpers
- stateless scalar `foo_frule` helpers
- stateless scalar `foo_rrule` helpers
- real/complex projection helpers for common scalar formulas

Supported scalar domains:

- `f32`
- `f64`
- `Complex32`
- `Complex64`

## Supported Functions

Current shipped scalar families:

- arithmetic: `add`, `sub`, `mul`, `div`
- powers and roots: `powf`, `powi`, `sqrt`
- exponentials and logs: `exp`, `expm1`, `log`, `log1p`
- trigonometric: `sin`, `cos`, `asin`, `acos`, `atan`
- hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Julia-compatible trigonometric helpers: `sec`, `csc`, `cot`, `sinpi`, `cospi`, `sincospi`, `sind`, `cosd`, `tand`
- Julia-compatible hyperbolic helpers: `sech`, `csch`, `coth`
- non-smooth real helpers: `round`, `floor`, `ceil`, `sign`, `min`, `max`
- smooth helpers: `cbrt`, `inv`, `exp2`, `exp10`, `log2`, `log10`, `hypot`, `pow`, `sincos`, `tan`
- complex and projection helpers: `conj`, `abs`, `abs2`, `angle`, `real`, `imag`, `complex`
- real-valued binary helpers: `atan2`

This crate is a landing zone for scalar rules ported or adapted from Julia's
`ChainRules.jl` where they fit this crate boundary, but it is not a full port
of `ChainRules.jl`.

## Validation

Rules in this crate are not accepted on provenance alone. They are checked with
repository-local tests:

- `tests/scalarops_tests.rs` covers direct formulas, edge cases, and smooth
  real/complex behavior
- `tests/julia_compat_trig_tests.rs` covers Julia migration helpers, including
  landmark real inputs and representative Complex64 behavior
- `tests/nonsmooth_scalar_tests.rs` covers the documented zero-gradient and
  tie-routing policies for non-smooth helpers
- `tests/complex_helper_tests.rs` covers the projection helpers and complex
  constructor surface
- `tests/oracle_scalar_rules.rs` replays vendored published oracle cases from
  `../../third_party/tensor-ad-oracles`, with direct float64 replay and
  selected direct Complex64 replay for `tan`, `exp2`, and `log2`
- complex forward-mode checks use the standard JVP convention on `C ~= R^2`
  in repository-local formula tests such as `tests/smooth_basis_tests.rs`
- complex reverse-mode checks remain conjugate-Wirtinger for real-valued losses

## rrule first-argument convention

Some rrule helpers accept the **forward result** as their first argument
(when the derivative can be expressed in terms of the output), while others
accept the **input `x`** (when the derivative depends on the original
input). The parameter name in each function signature tells you which:

| First parameter | Functions |
|-----------------|-----------|
| `result` | `exp_rrule`, `expm1_rrule`, `exp2_rrule`, `exp10_rrule`, `sqrt_rrule`, `cbrt_rrule`, `inv_rrule`, `tanh_rrule`, `tan_rrule` |
| `x` | `log_rrule`, `log1p_rrule`, `log2_rrule`, `log10_rrule`, `sin_rrule`, `cos_rrule`, `sinh_rrule`, `cosh_rrule`, all inverse-trig/hyperbolic rrules, `powf_rrule`, `powi_rrule`, `pow_rrule`, Julia-compat trig/hyperbolic rrules |
| both inputs | `mul_rrule(x, y, …)`, `div_rrule(x, y, …)`, `atan2_rrule(y, x, …)`, `hypot_rrule(x, y, …)`, `min_rrule(x, y, …)`, `max_rrule(x, y, …)` |
| cotangent only | `add_rrule`, `sub_rrule`, `conj_rrule`, `real_rrule`, `imag_rrule` |

## Complex scalar convention

For complex scalars (`Complex64`, `Complex32`):

- **Forward-mode** (frule): uses the standard JVP convention on **C ≅ R²**
- **Reverse-mode** (rrule): uses the **conjugate-Wirtinger** convention for
  real-valued losses — gradients include `conj(df/dz)`

For real scalars `conj` is the identity, so the convention is invisible.

See the validation section below for how each convention is tested.

## Examples

```rust
use chainrules::{powf, powf_frule, powf_rrule};

let y = powf(2.0_f64, 3.0_f64);
assert_eq!(y, 8.0_f64);

let (y, dy) = powf_frule(2.0_f64, 3.0_f64, 1.0_f64);
assert_eq!(y, 8.0_f64);
assert_eq!(dy, 12.0_f64);

let dx = powf_rrule(2.0_f64, 3.0_f64, 1.0_f64);
assert_eq!(dx, 12.0_f64);
```

## Notes

This crate is the landing zone for shared scalar rule logic, including
Julia-style convenience functions when they help migration.

It does not define tensor, array, broadcast, reduction, or engine-specific
rules.
