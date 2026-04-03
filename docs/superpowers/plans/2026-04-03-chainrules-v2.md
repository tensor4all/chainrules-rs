# chainrules-rs v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite chainrules-rs as a thin trait-only crate defining `PrimitiveOp` and `ADKey` for fragment-based AD.

**Architecture:** `feat/v2` branch with all v1 code deleted. Single crate depending on computegraph-rs via git. Two source files (`ad_key.rs`, `primitive_op.rs`) plus re-export `lib.rs`.

**Tech Stack:** Rust 2021, depends on `computegraph` (git)

**Spec:** `docs/superpowers/specs/2026-04-03-chainrules-v2-design.md`

---

## File Structure

```
chainrules-rs/  (feat/v2 branch)
├── Cargo.toml          # single crate, git dep on computegraph
├── src/
│   ├── lib.rs          # pub re-exports
│   ├── ad_key.rs       # ADKey trait, DiffPassId
│   └── primitive_op.rs # PrimitiveOp trait
└── tests/
    └── trait_tests.rs   # mock impl + integration tests
```

---

### Task 1: Branch Setup and v1 Cleanup

**Files:**
- Delete: `crates/` (entire directory)
- Delete: `Cargo.lock`
- Delete: `coverage-thresholds.json`, `coverage.json`
- Delete: `third_party/` (entire directory)
- Delete: `docs/plans/` (v1 plans)
- Delete: `scripts/` (v1 scripts)
- Delete: `.github/` (v1 CI)
- Modify: `Cargo.toml` — replace workspace with single crate
- Modify: `src/lib.rs` — replace with v2 placeholder

- [ ] **Step 1: Create feat/v2 branch**

```bash
cd /home/shinaoka/tensor4all/chainrules-rs
git checkout -b feat/v2
```

- [ ] **Step 2: Delete all v1 source code and config**

```bash
rm -rf crates/
rm -rf third_party/
rm -rf scripts/
rm -rf .github/
rm -rf docs/plans/
rm -f Cargo.lock coverage-thresholds.json coverage.json
rm -rf ai/
rm -rf .worktrees/
```

- [ ] **Step 3: Replace Cargo.toml with v2 single-crate config**

Replace the entire contents of `Cargo.toml` with:

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

- [ ] **Step 4: Create src/ directory and placeholder lib.rs**

```bash
mkdir -p src
```

Write `src/lib.rs`:

```rust
//! AD trait definitions for the tensor4all v2 stack.
//!
//! This crate defines `PrimitiveOp` (extends `GraphOp` with linearization
//! and transpose rules) and `ADKey` (tangent input key generation).
//! It contains no concrete primitives and no graph infrastructure.
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check`
Expected: success

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: clean v1 code and set up v2 single-crate structure"
```

---

### Task 2: ADKey Trait

**Files:**
- Create: `src/ad_key.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write tests for ADKey**

Create `tests/trait_tests.rs`:

```rust
use chainrules::{ADKey, DiffPassId};

/// Mock input key implementing ADKey for testing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MockKey {
    User(String),
    Tangent {
        of: Box<MockKey>,
        pass: DiffPassId,
    },
}

impl ADKey for MockKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        MockKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

#[test]
fn ad_key_tangent_of_produces_derived_key() {
    let key = MockKey::User("x".to_string());
    let tangent = key.tangent_of(1);
    assert_eq!(
        tangent,
        MockKey::Tangent {
            of: Box::new(MockKey::User("x".to_string())),
            pass: 1,
        }
    );
}

#[test]
fn ad_key_different_passes_produce_different_keys() {
    let key = MockKey::User("x".to_string());
    let t1 = key.tangent_of(1);
    let t2 = key.tangent_of(2);
    assert_ne!(t1, t2);
}

#[test]
fn ad_key_higher_order_tangent() {
    let key = MockKey::User("x".to_string());
    let t1 = key.tangent_of(1);
    let t2 = t1.tangent_of(2);
    // t2 = Tangent { of: Tangent { of: User("x"), pass: 1 }, pass: 2 }
    assert_eq!(
        t2,
        MockKey::Tangent {
            of: Box::new(MockKey::Tangent {
                of: Box::new(MockKey::User("x".to_string())),
                pass: 1,
            }),
            pass: 2,
        }
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release`
Expected: compilation error — `ADKey` and `DiffPassId` not found

- [ ] **Step 3: Implement ADKey**

Create `src/ad_key.rs`:

```rust
use std::hash::Hash;

/// Unique identifier for a `differentiate` call.
pub type DiffPassId = u64;

/// Constraint on `GraphOp::InputKey` for AD use.
///
/// `tidu-rs` uses this trait to generate tangent input keys
/// during `differentiate`.
///
/// # Examples
///
/// ```
/// use chainrules::{ADKey, DiffPassId};
///
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// enum MyKey {
///     User(String),
///     Tangent { of: Box<MyKey>, pass: DiffPassId },
/// }
///
/// impl ADKey for MyKey {
///     fn tangent_of(&self, pass: DiffPassId) -> Self {
///         MyKey::Tangent { of: Box::new(self.clone()), pass }
///     }
/// }
/// ```
pub trait ADKey: Clone + std::fmt::Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `differentiate` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

- [ ] **Step 4: Update lib.rs**

Replace `src/lib.rs` with:

```rust
//! AD trait definitions for the tensor4all v2 stack.
//!
//! This crate defines `PrimitiveOp` (extends `GraphOp` with linearization
//! and transpose rules) and `ADKey` (tangent input key generation).
//! It contains no concrete primitives and no graph infrastructure.

pub mod ad_key;

pub use ad_key::{ADKey, DiffPassId};
```

- [ ] **Step 5: Run tests**

Run: `cargo test --release`
Expected: 3 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/ad_key.rs src/lib.rs tests/trait_tests.rs
git commit -m "feat: add ADKey trait and DiffPassId type"
```

---

### Task 3: PrimitiveOp Trait

**Files:**
- Create: `src/primitive_op.rs`
- Modify: `src/lib.rs`
- Modify: `tests/trait_tests.rs`

- [ ] **Step 1: Write tests for PrimitiveOp**

Add the following to `tests/trait_tests.rs`:

```rust
use chainrules::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::{GraphOp, Operand};

/// Mock operation implementing both GraphOp and PrimitiveOp.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MockOp {
    Add,
    Scale, // multiplies by a fixed operand
}

impl GraphOp for MockOp {
    type Operand = f64;
    type Context = ();
    type InputKey = MockKey;

    fn n_inputs(&self) -> usize {
        match self {
            MockOp::Add => 2,
            MockOp::Scale => 2,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            MockOp::Add => vec![inputs[0] + inputs[1]],
            MockOp::Scale => vec![inputs[0] * inputs[1]],
        }
    }
}

impl PrimitiveOp for MockOp {
    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<computegraph::LocalValId>],
    ) -> Vec<Option<computegraph::LocalValId>> {
        match self {
            MockOp::Add => {
                // d(x + y) = dx + dy
                // If both tangents are present, emit Add in linear mode
                match (&tangent_in[0], &tangent_in[1]) {
                    (Some(dx), Some(dy)) => {
                        let out = builder.add_op(
                            MockOp::Add,
                            vec![ValRef::Local(*dx), ValRef::Local(*dy)],
                            OpMode::Linear {
                                active_mask: vec![true, true],
                            },
                        );
                        vec![Some(out[0])]
                    }
                    (Some(dx), None) => vec![Some(*dx)],
                    (None, Some(dy)) => vec![Some(*dy)],
                    (None, None) => vec![None],
                }
            }
            MockOp::Scale => {
                // d(a * x) where a is fixed = a * dx
                // tangent_in[0] is for fixed operand (None), tangent_in[1] is for active
                match &tangent_in[1] {
                    Some(dx) => {
                        let out = builder.add_op(
                            MockOp::Scale,
                            vec![
                                ValRef::External(_primal_in[0].clone()),
                                ValRef::Local(*dx),
                            ],
                            OpMode::Linear {
                                active_mask: vec![false, true],
                            },
                        );
                        vec![Some(out[0])]
                    }
                    None => vec![None],
                }
            }
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<computegraph::LocalValId>],
        inputs: &[ValRef<Self>],
        _mode: &OpMode,
    ) -> Vec<Option<computegraph::LocalValId>> {
        match self {
            MockOp::Add => {
                // transpose of Add: cotangent flows to both inputs
                match &cotangent_out[0] {
                    Some(ct) => vec![Some(*ct), Some(*ct)],
                    None => vec![None, None],
                }
            }
            MockOp::Scale => {
                // transpose of Scale(a, dx): ct_dx = a * ct_y
                match &cotangent_out[0] {
                    Some(ct) => {
                        let out = builder.add_op(
                            MockOp::Scale,
                            vec![inputs[0].clone(), ValRef::Local(*ct)],
                            OpMode::Linear {
                                active_mask: vec![false, true],
                            },
                        );
                        vec![None, Some(out[0])]
                    }
                    None => vec![None, None],
                }
            }
        }
    }
}

#[test]
fn primitive_op_linearize_add() {
    let mut builder = FragmentBuilder::<MockOp>::new();
    let dx = builder.add_input(MockKey::User("dx".to_string()));
    let dy = builder.add_input(MockKey::User("dy".to_string()));

    let primal_in = vec![
        GlobalValKey::Input(MockKey::User("x".to_string())),
        GlobalValKey::Input(MockKey::User("y".to_string())),
    ];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("sum".to_string()))];
    let tangent_in = vec![Some(dx), Some(dy)];

    let result = MockOp::Add.linearize(&mut builder, &primal_in, &primal_out, &tangent_in);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    // Builder should contain one Add op in Linear mode
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, MockOp::Add);
    assert_eq!(
        frag.ops()[0].mode,
        OpMode::Linear {
            active_mask: vec![true, true]
        }
    );
}

#[test]
fn primitive_op_linearize_skip_inactive() {
    let mut builder = FragmentBuilder::<MockOp>::new();
    let dx = builder.add_input(MockKey::User("dx".to_string()));

    let primal_in = vec![
        GlobalValKey::Input(MockKey::User("x".to_string())),
        GlobalValKey::Input(MockKey::User("y".to_string())),
    ];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("sum".to_string()))];
    // Only first tangent is active
    let tangent_in = vec![Some(dx), None];

    let result = MockOp::Add.linearize(&mut builder, &primal_in, &primal_out, &tangent_in);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    // No op emitted — tangent passes through
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 0);
}

#[test]
fn primitive_op_transpose_add() {
    let mut builder = FragmentBuilder::<MockOp>::new();
    let ct = builder.add_input(MockKey::User("ct".to_string()));

    let inputs = vec![
        ValRef::External(GlobalValKey::Input(MockKey::User("x".to_string()))),
        ValRef::External(GlobalValKey::Input(MockKey::User("y".to_string()))),
    ];
    let cotangent_out = vec![Some(ct)];

    let result =
        MockOp::Add.transpose_rule(&mut builder, &cotangent_out, &inputs, &OpMode::Primal);

    // Both inputs get the same cotangent (no op emitted, just pass-through)
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], Some(ct));
    assert_eq!(result[1], Some(ct));
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 0);
}

#[test]
fn primitive_op_transpose_scale() {
    let mut builder = FragmentBuilder::<MockOp>::new();
    let ct = builder.add_input(MockKey::User("ct".to_string()));

    let inputs = vec![
        ValRef::External(GlobalValKey::Input(MockKey::User("a".to_string()))),
        ValRef::External(GlobalValKey::Input(MockKey::User("x".to_string()))),
    ];
    let cotangent_out = vec![Some(ct)];
    let mode = OpMode::Linear {
        active_mask: vec![false, true],
    };

    let result = MockOp::Scale.transpose_rule(&mut builder, &cotangent_out, &inputs, &mode);

    // First input (fixed) gets None, second gets Scale(a, ct)
    assert_eq!(result.len(), 2);
    assert!(result[0].is_none());
    assert!(result[1].is_some());
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, MockOp::Scale);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release`
Expected: compilation error — `PrimitiveOp` not found

- [ ] **Step 3: Implement PrimitiveOp**

Create `src/primitive_op.rs`:

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

- [ ] **Step 4: Update lib.rs**

Replace `src/lib.rs` with:

```rust
//! AD trait definitions for the tensor4all v2 stack.
//!
//! This crate defines [`PrimitiveOp`] (extends [`computegraph::GraphOp`] with
//! linearization and transpose rules) and [`ADKey`] (tangent input key
//! generation). It contains no concrete primitives and no graph infrastructure.

pub mod ad_key;
pub mod primitive_op;

pub use ad_key::{ADKey, DiffPassId};
pub use primitive_op::PrimitiveOp;
```

- [ ] **Step 5: Run tests**

Run: `cargo test --release`
Expected: 7 tests pass (3 ADKey + 4 PrimitiveOp)

- [ ] **Step 6: Run fmt and clippy**

Run: `cargo fmt --all && cargo clippy --workspace`
Expected: no warnings or errors

- [ ] **Step 7: Commit**

```bash
git add src/primitive_op.rs src/lib.rs tests/trait_tests.rs
git commit -m "feat: add PrimitiveOp trait with linearize and transpose_rule"
```

---

## Summary

| Task | What it delivers | Tests |
|------|------------------|-------|
| 1 | Branch setup, v1 cleanup, single-crate scaffold | compilation only |
| 2 | `ADKey` trait, `DiffPassId` type | 3 tests |
| 3 | `PrimitiveOp` trait + mock impl tests | 4 tests |
| **Total** | | **7 tests** |
