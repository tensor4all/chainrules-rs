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
- complex and projection helpers: `conj`, `handle_r_to_c_f32`, `handle_r_to_c_f64`
- real-valued binary helpers: `atan2`

This crate is intended as a landing zone for scalar rules ported or adapted
from Julia's `ChainRules.jl` where they fit this crate boundary, but it is not
a full port of `ChainRules.jl`.

## Validation

Rules in this crate are not accepted on provenance alone. They are checked with
repository-local tests:

- `tests/scalarops_tests.rs` covers direct formulas, edge cases, and
  real/complex behavior
- `tests/oracle_scalar_rules.rs` replays vendored published oracle cases from
  `../../third_party/tensor-ad-oracles`

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
