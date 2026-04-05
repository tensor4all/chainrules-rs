# chainrules-rs v2 Implementation Spec

**Date:** 2026-04-03
**Status:** Approved
**Upstream design:** `tensor4all-meta/docs/design-v2/chainrules-design.md`

---

## Goal

Rewrite chainrules-rs on `feat/v2` branch as a thin trait-only crate that
defines `PrimitiveOp` (extends `GraphOp` with `linearize` + `transpose_rule`)
and `ADKey` (tangent input key generation). No concrete primitives, no graph
infrastructure.

---

## Branch Strategy

Work on `feat/v2` branch. Delete all v1 code (`crates/` directory and
workspace structure). Replace with a single top-level crate.

---

## Module Layout

```
chainrules-rs/  (feat/v2 branch)
├── Cargo.toml          # single crate, depends on computegraph via git
├── src/
│   ├── lib.rs          # pub re-exports
│   ├── primitive_op.rs # PrimitiveOp trait
│   └── ad_key.rs       # ADKey trait, DiffPassId
└── tests/
    └── trait_tests.rs   # compile tests + mock impl tests
```

---

## 1. ADKey Trait (`ad_key.rs`)

```rust
use std::hash::Hash;

/// Unique identifier for a `differentiate` call.
pub type DiffPassId = u64;

/// Constraint on `GraphOp::InputKey` for AD use.
///
/// `tidu-rs` uses this trait to generate tangent input keys
/// during `differentiate`.
pub trait ADKey: Clone + std::fmt::Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `differentiate` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

---

## 2. PrimitiveOp Trait (`primitive_op.rs`)

```rust
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::GraphOp;

use crate::ADKey;

/// Extends `GraphOp` with linearization and transpose rules for AD.
///
/// - `linearize` is called by `tidu::differentiate`
/// - `transpose_rule` is called by `tidu::transpose`
///
/// Both methods emit new ops into a `FragmentBuilder`. The downstream
/// implementor (e.g. tenferro-rs) is responsible for ensuring closure:
/// every op emitted must also implement `PrimitiveOp`.
pub trait PrimitiveOp: GraphOp
where
    Self::InputKey: ADKey,
{
    /// Emit the linear (JVP) rule for this primitive.
    ///
    /// Must be linear in tangent inputs. May reference primal inputs/outputs
    /// through `External(GlobalValKey)`. Must emit ops in `OpMode::Linear`.
    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;

    /// Emit the transpose rule for this linear primitive.
    ///
    /// Receives cotangent outputs and produces cotangent inputs.
    /// Must only emit ops that themselves implement `PrimitiveOp`.
    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;
}
```

---

## 3. lib.rs

```rust
pub mod ad_key;
pub mod primitive_op;

pub use ad_key::{ADKey, DiffPassId};
pub use primitive_op::PrimitiveOp;
```

---

## 4. Dependencies

```toml
[package]
name = "chainrules"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
publish = false

[dependencies]
computegraph = { git = "https://github.com/tensor4all/computegraph-rs.git" }
```

No other dependencies. v1 workspace dependencies (`num-complex`, `num-traits`,
`serde_json`, `thiserror`) are removed.

---

## 5. Test Strategy

Test file: `tests/trait_tests.rs`

Define a mock implementation inside tests:

- `MockOp` enum implementing `GraphOp` + `PrimitiveOp`
- `MockKey` enum implementing `ADKey`
- `f64` already implements `Operand` (from computegraph)

Test cases:

1. **ADKey**: `tangent_of` produces correct derived keys, different passes
   produce different keys
2. **PrimitiveOp compile check**: `MockOp` implements `PrimitiveOp` with
   trivial linearize/transpose_rule, verify it compiles and the methods
   are callable
3. **Linearize emits into builder**: call `linearize` on MockOp, verify
   the builder contains expected ops
4. **Transpose emits into builder**: call `transpose_rule` on MockOp,
   verify the builder contains expected ops

---

## 6. Not In Scope

- Concrete primitives (Add, Mul, Exp, etc.) — tenferro-rs responsibility
- AD transforms (differentiate, transpose) — tidu-rs responsibility
- Graph infrastructure — computegraph-rs responsibility
- v1 code preservation — v1 stays on `main`, `feat/v2` is a clean slate
