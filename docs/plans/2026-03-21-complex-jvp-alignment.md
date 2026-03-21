# Complex JVP Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align complex `frule` semantics with PyTorch and standard JVP over `C ~= R^2`, keep complex `rrule` on the existing conjugate-Wirtinger convention for real-valued losses, remove the unused `handle_r_to_c_*` helpers, and validate complex forward rules directly against vendored oracle JVPs.

**Architecture:** Treat this as a family-wide semantic bug, not a one-off formula mismatch. Any complex `frule` that currently applies `conj(local_derivative)` must switch to the plain pushforward `J * dx`, while `rrule` keeps the existing `J^H * cotangent` behavior. Audit the shared scalar basis by module family, update local formula tests first, then re-enable direct complex oracle replay so the repository validates both float64 and Complex64 entrypoints against the same published PyTorch JVP/VJP data.

**Tech Stack:** Rust 2021, `chainrules`, `chainrules-core`, `num-complex`, vendored `tensor-ad-oracles`, `cargo fmt`, `cargo nextest`, `cargo test --doc`, `cargo clippy`, `cargo llvm-cov`.

---

### Task 1: Re-enable Direct Complex Oracle Replay

**Files:**
- Modify: `crates/chainrules/tests/common.rs`
- Modify: `crates/chainrules/tests/oracle_scalar_rules.rs`
- Modify: `crates/chainrules/src/unary/trig.rs`
- Modify: `crates/chainrules/src/unary/exp_log.rs`

**Step 1: Write the failing test**

Replace the reverse-only Complex64 oracle entrypoint with a direct `UnaryOracleCase<Complex64>` replay for the currently covered ops:

```rust
#[test]
fn published_complex128_oracles_match_unary_rule_entrypoints() {
    let cases: [UnaryOracleCase<Complex64>; 3] = [
        UnaryOracleCase {
            op: "tan",
            frule: tan_frule,
            rrule: |_x: Complex64, result, cotangent| tan_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "exp2",
            frule: exp2_frule,
            rrule: |_x: Complex64, result, cotangent| exp2_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "log2",
            frule: log2_frule,
            rrule: |x: Complex64, _result, cotangent| log2_rrule(x, cotangent),
        },
    ];

    run_unary_oracle_cases(&cases);
}
```

Delete the reverse-only helper path and its comment in `tests/common.rs`.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo nextest run --release --test oracle_scalar_rules
```

Expected: FAIL with Complex64 JVP mismatches for `tan`, `exp2`, or `log2`.

**Step 3: Write minimal implementation**

Make the smallest source change that matches the oracle:

```rust
pub fn tan_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.tan();
    (y, dx * (one::<S>() + y * y))
}

pub fn exp2_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp2();
    (y, dx * (y * ln_2::<S>()))
}

pub fn log2_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.log2();
    let scale = one::<S>() / (x * ln_2::<S>());
    (y, dx * scale)
}
```

Keep the corresponding `rrule` implementations unchanged.

**Step 4: Run test to verify it passes**

Run:

```bash
cargo fmt --all
cargo nextest run --release --test oracle_scalar_rules
```

Expected: PASS for the Complex64 oracle replay test.

**Step 5: Commit**

```bash
git add crates/chainrules/tests/common.rs crates/chainrules/tests/oracle_scalar_rules.rs crates/chainrules/src/unary/trig.rs crates/chainrules/src/unary/exp_log.rs
git commit -m "refactor: replay complex unary oracles with direct jvp"
```

### Task 2: Align Smooth Unary Complex `frule` Families

**Files:**
- Modify: `crates/chainrules/src/unary/basic.rs`
- Modify: `crates/chainrules/src/unary/exp_log.rs`
- Modify: `crates/chainrules/src/unary/trig.rs`
- Modify: `crates/chainrules/src/unary/hyperbolic.rs`
- Modify: `crates/chainrules/src/unary/roots.rs`
- Modify: `crates/chainrules/src/tests/behavior.rs`
- Modify: `crates/chainrules/tests/smooth_basis_tests.rs`

**Step 1: Write the failing test**

Rename the behavior test so it states the new rule, then switch the expectations to plain JVP:

```rust
#[test]
fn extended_complex_unary_frules_match_standard_jvp_while_rrules_stay_conjugate() {
    let x = Complex64::new(0.25, -0.5);
    let dx = Complex64::new(-0.75, 0.5);
    let cotangent = Complex64::new(0.5, -1.25);

    let (_sin_y, sin_dy) = sin_frule(x, dx);
    assert_close_c64(sin_dy, dx * ComplexFloat::cos(x));
    assert_close_c64(
        sin_rrule(x, cotangent),
        cotangent * ComplexFloat::conj(ComplexFloat::cos(x)),
    );
}
```

Update `smooth_basis_complex_frules_match_expected_derivatives` the same way for `tan_frule`, `exp2_frule`, and `log2_frule`: remove `.conj()` from the expected forward scales but keep the reverse expectations unchanged.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo nextest run --release --test smooth_basis_tests --test chainrules \
  smooth_basis_complex_frules_match_expected_derivatives \
  extended_complex_unary_frules_match_standard_jvp_while_rrules_stay_conjugate
```

Expected: FAIL because the remaining unary `frule` implementations still conjugate their local derivatives.

**Step 3: Write minimal implementation**

Remove forward-mode conjugation from every smooth unary helper in these files:

```rust
pub fn sqrt_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.sqrt();
    let dy = dx / (S::from_i32(2) * y);
    (y, dy)
}

pub fn exp_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp();
    (y, dx * y)
}

pub fn asinh_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.asinh();
    let scale = inverse_sqrt_one_plus_square(x);
    (y, dx * scale)
}
```

Audit every `*_frule` in the listed modules and change only the forward formulas. Keep `rrule` code as-is.

**Step 4: Run test to verify it passes**

Run:

```bash
cargo fmt --all
cargo nextest run --release --test smooth_basis_tests
cargo nextest run --release --test chainrules extended_complex_unary_frules_match_standard_jvp_while_rrules_stay_conjugate
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/chainrules/src/unary/basic.rs crates/chainrules/src/unary/exp_log.rs crates/chainrules/src/unary/trig.rs crates/chainrules/src/unary/hyperbolic.rs crates/chainrules/src/unary/roots.rs crates/chainrules/src/tests/behavior.rs crates/chainrules/tests/smooth_basis_tests.rs
git commit -m "refactor: align smooth unary complex frules with standard jvp"
```

### Task 3: Align Binary And Power Complex `frule` Families

**Files:**
- Modify: `crates/chainrules/src/binary.rs`
- Modify: `crates/chainrules/src/power.rs`
- Modify: `crates/chainrules/tests/scalarops_tests.rs`
- Modify: `crates/chainrules/tests/smooth_basis_tests.rs`

**Step 1: Write the failing test**

Change the complex forward expectations so they use the plain Jacobian instead of the conjugated one:

```rust
#[test]
fn mul_div_rules_match_formula_complex64() {
    let x = Complex64::new(1.5, -0.5);
    let y = Complex64::new(-0.25, 2.0);
    let dx = Complex64::new(0.3, -0.2);
    let dy = Complex64::new(-0.1, 0.4);

    let (mul_y, mul_dy) = mul_frule(x, y, dx, dy);
    assert!((mul_y - (x * y)).norm() < 1e-12);
    assert!((mul_dy - (dx * y + dy * x)).norm() < 1e-12);
}
```

Update the power tests likewise:

```rust
let expected_dy = dx * (exponent * x.powc(exponent - Complex64::new(1.0, 0.0)))
    + dexp * (expected_y * x.ln());
assert!((dy - expected_dy).norm() < 1e-12);
```

Keep all `rrule` expectations on the conjugated derivative.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo nextest run --release --test scalarops_tests --test smooth_basis_tests
```

Expected: FAIL in the Complex64 forward checks for `mul`, `div`, `powf`, `powi`, or `pow`.

**Step 3: Write minimal implementation**

Update only the forward formulas:

```rust
pub fn mul_frule<S: ScalarAd>(x: S, y: S, dx: S, dy: S) -> (S, S) {
    let primal = x * y;
    let tangent = dx * y + dy * x;
    (primal, tangent)
}

pub fn div_frule<S: ScalarAd>(x: S, y: S, dx: S, dy: S) -> (S, S) {
    let primal = x / y;
    let inv_y = one::<S>() / y;
    let dfdx = inv_y;
    let dfdy = -(x * inv_y * inv_y);
    (primal, dx * dfdx + dy * dfdy)
}
```

Mirror the same change in `powf_frule`, `powi_frule`, and `pow_frule`. Keep the singularity behavior at `pow(0, 0)` and zero-base negative exponents unchanged.

**Step 4: Run test to verify it passes**

Run:

```bash
cargo fmt --all
cargo nextest run --release --test scalarops_tests
cargo nextest run --release --test smooth_basis_tests
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/chainrules/src/binary.rs crates/chainrules/src/power.rs crates/chainrules/tests/scalarops_tests.rs crates/chainrules/tests/smooth_basis_tests.rs
git commit -m "refactor: align binary and power complex frules with standard jvp"
```

### Task 4: Align Julia Compatibility Helpers With Standard JVP

**Files:**
- Modify: `crates/chainrules/src/unary/trig_extra.rs`
- Modify: `crates/chainrules/src/unary/hyperbolic_extra.rs`
- Modify: `crates/chainrules/tests/julia_compat_trig_tests.rs`

**Step 1: Write the failing test**

Change the Complex64 forward expectations in the Julia-compat tests and add one extra representative check so the family is covered:

```rust
let (_, dsinpi) = sinpi_frule(z, dz);
assert_close_complex64(
    dsinpi,
    dz * (Complex64::new(std::f64::consts::PI, 0.0) * pi_z.cos()),
    1e-12,
    0.0,
    "sinpi_frule(z)",
);

let ((_sin_y, _cos_y), (dsin, dcos)) = sincospi_frule(z, dz);
assert_close_complex64(dsin, dz * (Complex64::new(std::f64::consts::PI, 0.0) * pi_z.cos()), 1e-12, 0.0, "sincospi_frule.sin");
assert_close_complex64(dcos, dz * (-(Complex64::new(std::f64::consts::PI, 0.0) * pi_z.sin())), 1e-12, 0.0, "sincospi_frule.cos");
```

Keep the `rrule` expectations on the conjugated derivative.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo nextest run --release --test julia_compat_trig_tests
```

Expected: FAIL for Complex64 forward checks in `sinpi_frule`, `sincospi_frule`, or related helpers.

**Step 3: Write minimal implementation**

Audit the Julia migration helpers and remove forward-mode conjugation everywhere it still appears:

```rust
pub fn sinpi_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = sinpi(x);
    let scale = pi::<S>() * cospi(x);
    (y, dx * scale)
}
```

Repeat the same cleanup for `cospi_frule`, `sincospi_frule`, `tand_frule`, `sec_frule`, `csc_frule`, `cot_frule`, `sech_frule`, `csch_frule`, and `coth_frule`. If a degree-based helper delegates entirely to corrected primitives, prefer reuse over re-deriving a new formula.

**Step 4: Run test to verify it passes**

Run:

```bash
cargo fmt --all
cargo nextest run --release --test julia_compat_trig_tests
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/chainrules/src/unary/trig_extra.rs crates/chainrules/src/unary/hyperbolic_extra.rs crates/chainrules/tests/julia_compat_trig_tests.rs
git commit -m "refactor: align Julia compatibility frules with standard jvp"
```

### Task 5: Remove `handle_r_to_c_*` And Update Public Docs

**Files:**
- Modify: `crates/chainrules/src/scalar_ad/mod.rs`
- Modify: `crates/chainrules/src/lib.rs`
- Modify: `crates/chainrules/src/tests/behavior.rs`
- Modify: `crates/chainrules/tests/scalarops_tests.rs`
- Modify: `README.md`
- Modify: `crates/chainrules/README.md`

**Step 1: Write the failing cleanup**

Remove the public re-exports first:

```rust
#[doc(inline)]
pub use scalar_ad::ScalarAd;
```

Then update the test imports so any remaining helper references are explicit failures:

```rust
use chainrules::{
    add, add_frule, add_rrule, conj, conj_frule, conj_rrule, div, div_frule, div_rrule, mul,
    mul_frule, mul_rrule, powf, powf_frule, powf_rrule, powi, powi_frule, powi_rrule, sqrt,
    sqrt_frule, sqrt_rrule, sub, sub_frule, sub_rrule,
};
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo nextest run --release --test scalarops_tests --test chainrules
cargo test --doc --release -p chainrules
```

Expected: FAIL because `handle_r_to_c_*` doctests and dedicated tests still exist.

**Step 3: Write minimal implementation**

Delete the unused helpers and update the docs to state the new convention clearly:

- remove `handle_r_to_c_f32` and `handle_r_to_c_f64` from `scalar_ad/mod.rs`
- delete the dedicated helper checks from `src/tests/behavior.rs` and `tests/scalarops_tests.rs`
- update `README.md` so the testing section says Complex64 oracle replay is direct, not reverse-only
- update `crates/chainrules/README.md` so it:
  - removes `handle_r_to_c_*` from the supported surface
  - states that complex `frule` follows standard JVP on `C ~= R^2`
  - states that complex `rrule` remains conjugate-Wirtinger for real-valued losses
  - keeps the provenance note that this crate is a landing zone for scalar rules ported or adapted from `ChainRules.jl`, not a full port

**Step 4: Run test to verify it passes**

Run:

```bash
cargo fmt --all
cargo nextest run --release --test scalarops_tests --test chainrules
cargo test --doc --release -p chainrules
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/chainrules/src/scalar_ad/mod.rs crates/chainrules/src/lib.rs crates/chainrules/src/tests/behavior.rs crates/chainrules/tests/scalarops_tests.rs README.md crates/chainrules/README.md
git commit -m "cleanup: remove projection helpers and document complex jvp"
```

## Final Verification

Run the full repository verification after the last task:

```bash
cargo fmt --all --check
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo llvm-cov nextest --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: all commands PASS with no direct-complex-oracle failures, no remaining `handle_r_to_c_*` references, and no docs claiming that complex forward replay is local-only.
