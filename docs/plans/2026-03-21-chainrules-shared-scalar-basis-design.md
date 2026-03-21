# ChainRules Shared Scalar Basis Design

## Goal

Expand `chainrules-rs` into a broader shared scalar automatic-differentiation
basis for Rust crates while preserving the existing crate boundary:

- `chainrules-core` stays protocol-only
- `chainrules` grows into the reusable scalar rule library
- runtime execution remains in engine crates such as `tidu-rs`
- tensor, array, and operation-specific rules stay in downstream crates

The target scalar domains are:

- `f32`
- `f64`
- `Complex32`
- `Complex64`

## Boundary

`chainrules-core` remains responsible for:

- `Differentiable`
- `ReverseRule`
- `ForwardRule`
- `AutodiffError`
- shared AD result and node types

`chainrules` remains responsible for:

- stateless scalar primal helpers
- stateless scalar `*_frule` helpers
- stateless scalar `*_rrule` helpers
- PyTorch-style real-input/complex-gradient projection helpers

Explicitly out of scope for this repository:

- tape or traced-value runtimes
- `RuleConfig`, `rrule_via_ad`, `frule_via_ad`
- generic `ProjectTo`, `Thunk`, or `InplaceableThunk` machinery
- tensor, array, broadcast, reduction, or factorization rules
- operation-specific AD rules such as einsum or SVD

## Documentation Layout

Documentation is split by responsibility.

- `README.md`
  Repository boundary, crate roles, and representative function families.
- `crates/chainrules/README.md`
  Canonical list of supported scalar functions and examples.
- `crates/chainrules-core/README.md`
  Protocol-only documentation. It explicitly states that this crate does not
  ship function rules.

This keeps the supported function inventory attached to the crate that
actually provides it, while still making the repository-level boundary obvious.

## API Shape

The public API keeps the existing flat naming style:

- `foo`
- `foo_frule`
- `foo_rrule`

The implementation stays internally modular and family-oriented, with
small focused source files and flat re-exports from `chainrules::lib`.

`rrule` helpers take the minimum saved values needed for reverse-mode formulas.
Examples:

- `exp_rrule(result, cotangent)`
- `log_rrule(x, cotangent)`
- `pow_rrule(x, p, cotangent) -> (dx, dp)`
- `sincos_rrule(x, (dsin, dcos)) -> dx`

## Function Families

The end state is a broad scalar rule basis, not a narrow Rust-idiomatic subset.
This keeps `chainrules` useful as a landing zone for Julia-to-Rust rule ports.

### Core Numeric Basis

- arithmetic: `add`, `sub`, `mul`, `div`
- powers and roots: `pow`, `powf`, `powi`, `sqrt`, `cbrt`, `inv`
- exponentials and logs:
  `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`, `log1p`
- trigonometric:
  `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sincos`
- hyperbolic:
  `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- combined real/complex helpers:
  `hypot`

### Complex And Projection Helpers

- `conj`
- `real`
- `imag`
- `abs`
- `abs2`
- `angle`
- `complex`
- `handle_r_to_c`

### Julia Migration Convenience Surface

- reciprocal trig: `sec`, `csc`, `cot`
- reciprocal hyperbolic: `sech`, `csch`, `coth`
- pi-based trig: `sinpi`, `cospi`, `sincospi`
- degree-based trig: `sind`, `cosd`, `tand`
- unit conversion: `deg2rad`, `rad2deg`

### Non-Smooth Utilities

- `sign`
- `copysign`
- `min`
- `max`
- `round`
- `floor`
- `ceil`
- `rem`
- `mod`
- `fma`
- `muladd`

## Compatibility Policy

The crate prefers semantic compatibility with the Julia scalar surface where
that improves migration and shared rule reuse, but it does not attempt to
become a full port of `ChainRules.jl`.

Compatibility rules:

- include Julia-style convenience scalar functions when they are useful as
  migration landing zones
- keep tensor and runtime-specific abstractions out of scope
- document branch cuts, singularities, and tie behavior explicitly
- preserve the existing complex-gradient convention used by the crate

## Implementation Shape

The public API remains flat, but source files should stay small. The expected
end-state module layout in `crates/chainrules/src` is:

- `binary.rs` for basic binary arithmetic
- `binary_special.rs` for `atan2`, `hypot`, `min/max`, `copysign`,
  `rem/mod`, `fma`, `muladd`
- `power.rs` for `pow`, `powf`, `powi`
- `scalar_ad.rs` for the scalar trait surface and shared scalar helpers
- `unary/basic.rs`
- `unary/roots.rs`
- `unary/exp_log.rs`
- `unary/trig.rs`
- `unary/trig_extra.rs`
- `unary/hyperbolic.rs`
- `unary/hyperbolic_extra.rs`
- `unary/complex_parts.rs`
- `unary/nonsmooth.rs`

The exact split may adjust slightly while keeping file sizes under control.

## Testing Strategy

Testing extends the current two-layer approach.

### Formula Tests

Add family-specific integration tests for:

- smooth numeric basis
- complex and projection helpers
- Julia compatibility helpers
- non-smooth utilities

Use generic helpers to avoid duplicating logic across scalar types. Favor
`f64` and `Complex64` as the primary correctness targets, with focused
sanity checks for `f32` and `Complex32`.

### Oracle Tests

Continue using the vendored `tensor-ad-oracles` data for published references.
Expand coverage for functions that already exist in the oracle set, such as:

- `tan`
- `exp2`
- `exp10`
- `log2`
- `log10`
- `abs`
- `angle`
- `hypot`
- `deg2rad`
- `rad2deg`
- `sign`

### Behavioral Tests

Keep hand-written tests for:

- singularities such as `sqrt(0)`
- branch-cut-sensitive complex functions
- non-smooth behavior for `sign`, `round`, `floor`, `ceil`
- tie behavior for `min` and `max`

## Rollout

Implement in four ordered phases.

1. README reorganization and crate-boundary documentation.
2. Smooth core scalar basis and complex/projection helpers.
3. Julia convenience scalar surface.
4. Non-smooth and tie-sensitive utility rules.

This sequencing makes the public boundary clear first, lands the highest-value
smooth basis next, and defers specification-heavy non-smooth behavior until the
rest of the surface is stable.
