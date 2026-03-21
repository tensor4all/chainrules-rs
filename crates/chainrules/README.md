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
