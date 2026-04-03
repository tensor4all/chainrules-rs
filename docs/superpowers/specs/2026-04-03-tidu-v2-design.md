# tidu-rs v2 Implementation Spec

**Date:** 2026-04-03
**Status:** Approved
**Upstream design:** `tensor4all-meta/docs/design-v2/tidu-design.md`

---

## Goal

Rewrite tidu-rs on `feat/v2` branch as a thin AD transform crate providing
`differentiate` (JVP) and `transpose` (reverse linear flow), fully generic
over `Op: PrimitiveOp`. No graph infrastructure, no concrete primitives.

---

## Branch Strategy

Work on `feat/v2` branch. Delete all v1 code. Replace with a single
top-level crate.

---

## Module Layout

```
tidu-rs/  (feat/v2 branch)
├── Cargo.toml              # git deps on computegraph + chainrules
├── src/
│   ├── lib.rs              # re-exports
│   ├── linear_fragment.rs  # LinearFragment struct
│   ├── differentiate.rs    # differentiate()
│   └── transpose.rs        # transpose()
└── tests/
    └── scalar_ad_tests.rs  # end-to-end AD tests with ScalarOp
```

---

## 1. LinearFragment (`linear_fragment.rs`)

```rust
use computegraph::fragment::Fragment;
use computegraph::traits::GraphOp;
use computegraph::types::LocalValId;

/// A linear fragment produced by `differentiate`, consumable by `transpose`.
pub struct LinearFragment<Op: GraphOp> {
    /// The fragment containing linear ops.
    pub fragment: Fragment<Op>,
    /// (primal_input_key, tangent_local_val_id) pairs for each
    /// differentiated input.
    pub tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    /// Tangent outputs — `None` means the output has no tangent
    /// dependency on the requested inputs.
    pub tangent_outputs: Vec<Option<LocalValId>>,
}
```

---

## 2. differentiate (`differentiate.rs`)

```rust
use computegraph::resolve::ResolvedView;
use computegraph::types::GlobalValKey;
use chainrules::{DiffPassId, PrimitiveOp};
use crate::LinearFragment;

pub fn differentiate<Op: PrimitiveOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
) -> LinearFragment<Op>;
```

`pass` is supplied by the caller (no global counter). Each call creates
tangent input keys via `wrt_key.tangent_of(pass)`.

Algorithm:

1. Walk the reachable logical DAG from `outputs` in topological order
   (dependency-first).
2. For each `wrt` key, create a tangent input in the builder via
   `ADKey::tangent_of(pass)`. Record the `(primal_key, tangent_local_id)` pair.
3. For each reachable op, call `Op::linearize(builder, primal_in, primal_out,
   tangent_in)`. Primal values are referenced as `External(GlobalValKey)`.
4. Inactive tangents propagate as `None` — no op is emitted.
5. Collect tangent outputs for the requested `outputs`.
6. Build the fragment and return `LinearFragment`.

---

## 3. transpose (`transpose.rs`)

```rust
use chainrules::PrimitiveOp;
use crate::LinearFragment;

pub fn transpose<Op: PrimitiveOp>(
    linear: &LinearFragment<Op>,
) -> LinearFragment<Op>;
```

Algorithm:

1. Traverse the linear fragment's ops in **reverse** topological order.
2. For each op, call `Op::transpose_rule(builder, cotangent_out, inputs, mode)`.
3. Accumulate cotangent contributions to the same global key using
   `Operand::add` (via an `Add` op or direct accumulation in the builder).
4. The transposed fragment's inputs are cotangent seeds (one per original
   tangent output that was `Some`), and its outputs are cotangent values
   for each original tangent input.

---

## 4. Dependencies

```toml
[package]
name = "tidu"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
publish = false

[dependencies]
computegraph = { git = "https://github.com/tensor4all/computegraph-rs.git", rev = "76aeccc" }
chainrules = { git = "https://github.com/tensor4all/chainrules-rs.git", branch = "feat/v2" }
```

---

## 5. Test Strategy

Test file: `tests/scalar_ad_tests.rs`

Define a test `ScalarOp` enum (`Add`, `Mul`, `Exp`, `Neg`, `Dup`) that
implements both `GraphOp` and `PrimitiveOp`, plus a `ScalarKey` enum
implementing `ADKey`.

### Test cases

1. **JVP of `x + x`**: `differentiate` → `materialize_merge` → `compile` →
   `eval` with tangent seed `dx=1`. Expected: `dy = 2`.

2. **JVP of `x * y`**: tangent w.r.t. `x` with `dx=1, y=3`. Expected:
   `dy = y = 3`.

3. **JVP of `exp(a*x)`**: tangent w.r.t. `x` with `dx=1, a=2, x=1`.
   Expected: `dy = a * exp(a*x) = 2*exp(2)`.

4. **VJP of `exp(a*x)`**: `differentiate` → `transpose` → full pipeline.
   With `ct_y=1, a=2, x=1`. Expected: `ct_x = a * exp(a*x) = 2*exp(2)`.

5. **2nd derivative (FoF) of `x^2`**: `Mul(x,x)` differentiated twice.
   Expected: `d²y/dx² = 2`.

6. **FoR (HVP) of `exp(a*x)`**: `differentiate` → `transpose` → `resolve`
   → `differentiate` again → full pipeline. With unit seeds, `a=2, x=1`.
   Expected: `d²y/dx² = a² * exp(a*x) = 4*exp(2)`.

7. **Numerical gradient check**: compare VJP result against finite
   differences for `exp(a*x)`.

---

## 6. Not In Scope

- Concrete primitives beyond test helpers — tenferro-rs responsibility
- Graph infrastructure — computegraph-rs responsibility
- PrimitiveOp trait — chainrules-rs responsibility
- Compilation cache, optimization — later phases
- Cross-country mode, partial transpose — Phase 4
