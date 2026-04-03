# tidu-rs v2 Implementation Spec

**Date:** 2026-04-03
**Status:** Approved (v2 — Dup removed, internal accumulation)
**Upstream design:** `tensor4all-meta/docs/design-v2/tidu-design.md`

---

## Goal

Rewrite tidu-rs on `feat/v2` branch as a thin AD transform crate providing
`differentiate` (JVP) and `transpose` (reverse linear flow), fully generic
over `Op: PrimitiveOp`. No graph infrastructure, no concrete primitives.

Fan-out accumulation is handled internally by `transpose` using
`Op::add()` — no `Dup` primitive exists.

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
    ├── common/
    │   └── mod.rs           # ScalarOp, ScalarKey, f64 PrimitiveOp impl
    └── scalar_ad_tests.rs   # end-to-end AD tests
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
    /// (primal_input_key, tangent_local_val_id) pairs.
    pub tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    /// Tangent outputs — `None` means inactive.
    pub tangent_outputs: Vec<Option<LocalValId>>,
}
```

---

## 2. differentiate (`differentiate.rs`)

```rust
pub fn differentiate<Op: PrimitiveOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
) -> LinearFragment<Op>;
```

`pass` is supplied by the caller.

Algorithm:

1. Collect reachable keys from `outputs` via the resolver (topological order).
2. For each `wrt` key, create tangent input via `key.tangent_of(pass)`.
   Store `(wrt_key, tangent_local_id)`.
3. For each reachable op (in dependency order), resolve its input tangents:
   - Input key in `wrt` set → corresponding tangent local id
   - Derived key → tangent produced by a previous linearize call
   - Otherwise → `None` (inactive)
4. Call `Op::linearize(builder, primal_in_keys, primal_out_keys, tangent_in)`.
5. Primal values referenced as `External(GlobalValKey)`.
6. Collect tangent outputs for requested `outputs`.
7. Build and return `LinearFragment`.

---

## 3. transpose (`transpose.rs`)

```rust
pub fn transpose<Op: PrimitiveOp>(
    linear: &LinearFragment<Op>,
) -> LinearFragment<Op>;
```

Algorithm (from design doc):

1. **Seed cotangent inputs**: for each non-None tangent output, create a
   cotangent input in the builder. Store in `ct_env: HashMap<GlobalValKey, LocalValId>`.

2. **Reverse topological traversal**: iterate `linear.fragment.ops()` in
   reverse order. For each op:
   - Look up cotangent for each output from `ct_env`
   - Call `op.transpose_rule(builder, ct_outs, inputs, mode)`
   - For each returned cotangent input:
     - Resolve the corresponding input's `GlobalValKey`
     - If key already in `ct_env`: emit `Op::add()` node to accumulate
       (fan-out accumulation, `OpMode::Linear { active_mask: [true, true] }`)
     - If key not in `ct_env`: insert directly

3. **Collect cotangent outputs**: for each tangent input in the original
   `LinearFragment`, look up its key in `ct_env`.

4. Build and return the transposed `LinearFragment`.

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

## 5. Test Helpers (`tests/common/mod.rs`)

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarOp { Add, Mul, Exp, Neg }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarKey {
    User(String),
    Tangent { of: Box<ScalarKey>, pass: DiffPassId },
}
```

`ScalarOp` implements `GraphOp` + `PrimitiveOp`:
- `Add::linearize`: `d(x+y) = dx+dy` (emit Add in linear mode, or pass through)
- `Mul::linearize`: `d(x*y) = dx*y + x*dy` (emit Mul for each active, Add to combine)
- `Exp::linearize`: `d(exp(x)) = exp(x)*dx` (emit Mul with External primal output)
- `Neg::linearize`: `d(-x) = -dx` (emit Neg in linear mode)
- `Add::transpose`: both inputs get the cotangent
- `Mul::transpose`: per active input, emit Mul with the fixed input
- `Exp::transpose`: not directly called (Exp is primal only)
- `Neg::transpose`: emit Neg on cotangent
- `ScalarOp::add() -> ScalarOp::Add`

---

## 6. Test Cases

1. **JVP `x + x`**: dy = 2·dx. Tests implicit fan-out in forward mode.

2. **JVP `x * y`**: dy/dx = y. Two-input linearize.

3. **JVP `exp(a*x)`**: dy/dx = a·exp(a·x). Chain of linearize calls.

4. **VJP `exp(a*x)`**: ct_x = a·exp(a·x)·ct_y. Forward + transpose.

5. **VJP `(x+x)*x`**: ct_x = 4x·ct_y. Fan-out accumulation in transpose
   (the worked example from tidu-design.md).

6. **FoF `x²`**: d²y/dx² = 2. `Mul(x,x)` differentiated twice.

7. **FoR (HVP) `exp(a*x)`**: d²y/dx² = a²·exp(a·x). Forward-of-reverse.

8. **Numerical gradient check**: VJP vs finite differences for `exp(a*x)`.

---

## 7. Not In Scope

- Concrete primitives beyond test helpers
- Graph infrastructure (computegraph-rs)
- PrimitiveOp trait definition (chainrules-rs)
- `Dup` primitive (removed from design)
- Compilation cache, optimization
