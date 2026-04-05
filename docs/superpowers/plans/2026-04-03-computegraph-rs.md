# computegraph-rs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement computegraph-rs — an AD-agnostic tensor computation graph engine providing fragment-based graph construction, resolution, materialization, compilation, and evaluation.

**Architecture:** Fragment-based computation graph with global structural identity (`GlobalValKey`). Graphs are built as fragments, logically resolved, physically materialized, compiled to SSA, and evaluated. The crate is fully generic over `Op: GraphOp` and never references specific primitives.

**Tech Stack:** Rust 2021 edition, no external dependencies beyond `std`.

**Spec:** `docs/superpowers/specs/2026-04-03-computegraph-rs-design.md`

**Upstream design:** `tensor4all-meta/docs/design-v2/computegraph-design.md`

---

## File Structure

```
/home/shinaoka/tensor4all/computegraph-rs/
├── Cargo.toml
├── .gitignore
├── AGENTS.md
├── README.md
├── src/
│   ├── lib.rs              # pub re-exports of all modules
│   ├── traits.rs           # Operand, GraphOp traits
│   ├── types.rs            # LocalValId, LocalOpId, OpMode, ValRef, GlobalValKey, GlobalOpKey
│   ├── interner.rs         # KeyInterner, ValKeyId
│   ├── fragment.rs         # Fragment, ValNode, OpNode, FragmentBuilder
│   ├── resolve.rs          # Resolver trait, ResolvedView, resolve()
│   ├── materialize.rs      # MaterializedGraph, MaterializedVal, MaterializedOp, materialize_merge()
│   ├── compile.rs          # CompiledProgram, Instruction, compile()
│   └── eval.rs             # CompiledProgram::eval() impl
└── tests/
    ├── common/
    │   └── mod.rs           # ScalarOp enum, f64 Operand impl (shared test helpers)
    └── scalar_tests.rs      # End-to-end integration tests using ScalarOp
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `/home/shinaoka/tensor4all/computegraph-rs/Cargo.toml`
- Create: `/home/shinaoka/tensor4all/computegraph-rs/.gitignore`
- Create: `/home/shinaoka/tensor4all/computegraph-rs/README.md`
- Create: `/home/shinaoka/tensor4all/computegraph-rs/AGENTS.md`
- Create: `/home/shinaoka/tensor4all/computegraph-rs/src/lib.rs`

- [ ] **Step 1: Create project directory and initialize git**

```bash
mkdir -p /home/shinaoka/tensor4all/computegraph-rs/src
cd /home/shinaoka/tensor4all/computegraph-rs
git init
```

- [ ] **Step 2: Create Cargo.toml**

```toml
[package]
name = "computegraph"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
publish = false
```

- [ ] **Step 3: Create .gitignore**

```
/target
Cargo.lock
```

- [ ] **Step 4: Create src/lib.rs (empty placeholder)**

```rust
//! AD-agnostic tensor computation graph engine.
```

- [ ] **Step 5: Create README.md**

```markdown
# computegraph-rs

AD-agnostic tensor computation graph engine in Rust.

Provides fragment-based graph construction, logical resolution,
physical materialization, SSA compilation, and evaluation.

Fully generic over `Op: GraphOp` — never references specific primitives.

## Part of the tensor4all v2 stack

```text
computegraph-rs  ← this crate
chainrules-rs    ← AD trait definitions
tidu-rs          ← AD graph transforms
tenferro-rs      ← concrete tensor primitives
```
```

- [ ] **Step 6: Create AGENTS.md**

Copy `/home/shinaoka/tensor4all/chainrules-rs/AGENTS.md` and adapt:
- Remove coverage threshold references (none yet)
- Keep all Rust code style, testing, and git workflow conventions

- [ ] **Step 7: Verify the project compiles**

Run: `cargo check`
Expected: success (empty lib)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: initialize computegraph-rs project scaffold"
```

---

### Task 2: Core Traits and Types

**Files:**
- Create: `src/traits.rs`
- Create: `src/types.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create src/traits.rs**

```rust
use std::hash::Hash;

/// Runtime value type. Scalars are rank-0 tensors.
pub trait Operand: Clone + Send + Sync + 'static {
    /// Additive identity for the given shape.
    fn zero(shape: &[usize]) -> Self;

    /// Multiplicative identity for the given shape.
    fn one(shape: &[usize]) -> Self;

    fn reshape(&self, shape: &[usize]) -> Self;

    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self;

    fn add(&self, other: &Self) -> Self;

    fn multiply(&self, other: &Self) -> Self;

    fn reduce_sum(&self, axes: &[usize]) -> Self;

    fn dot_general(
        &self,
        other: &Self,
        lhs_contracting: &[usize],
        rhs_contracting: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self;

    fn conj(&self) -> Self;
}

/// Operation node trait. computegraph is fully generic over this.
pub trait GraphOp: Clone + std::fmt::Debug + Hash + Eq + Send + Sync + 'static {
    type Operand: Operand;
    type Context;
    type InputKey: Clone + std::fmt::Debug + Hash + Eq + Send + Sync + 'static;

    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;
    fn eval(&self, ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;
}
```

- [ ] **Step 2: Create src/types.rs**

```rust
use crate::traits::GraphOp;

/// Fragment-local value identifier.
pub type LocalValId = usize;

/// Fragment-local operation identifier.
pub type LocalOpId = usize;

/// Distinguishes primal nodes from linear (AD-generated) nodes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpMode {
    Primal,
    Linear { active_mask: Vec<bool> },
}

/// Reference to a value: either local to the current fragment or external.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValRef<Op: GraphOp> {
    Local(LocalValId),
    External(GlobalValKey<Op>),
}

/// Cross-fragment structural identity for a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GlobalValKey<Op: GraphOp> {
    Input(Op::InputKey),
    Derived {
        op: GlobalOpKey<Op>,
        output_slot: u8,
    },
}

/// Cross-fragment structural identity for an operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlobalOpKey<Op: GraphOp> {
    pub primitive: Op,
    pub inputs: Vec<GlobalValKey<Op>>,
    pub mode: OpMode,
}
```

- [ ] **Step 3: Update src/lib.rs with module declarations**

```rust
//! AD-agnostic tensor computation graph engine.

pub mod traits;
pub mod types;

pub use traits::{GraphOp, Operand};
pub use types::{GlobalOpKey, GlobalValKey, LocalOpId, LocalValId, OpMode, ValRef};
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check`
Expected: success

- [ ] **Step 5: Commit**

```bash
git add src/traits.rs src/types.rs src/lib.rs
git commit -m "feat: add core traits (Operand, GraphOp) and types"
```

---

### Task 3: Test Helpers (ScalarOp + f64 Operand)

**Files:**
- Create: `tests/common/mod.rs`

- [ ] **Step 1: Create tests/common/mod.rs**

```rust
use computegraph::{GraphOp, Operand};

/// Scalar operations for testing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarOp {
    Add,
    Mul,
    Exp,
    Neg,
    Dup,
}

impl GraphOp for ScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = String;

    fn n_inputs(&self) -> usize {
        match self {
            ScalarOp::Add | ScalarOp::Mul => 2,
            ScalarOp::Exp | ScalarOp::Neg | ScalarOp::Dup => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        match self {
            ScalarOp::Dup => 2,
            _ => 1,
        }
    }

    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            ScalarOp::Add => vec![inputs[0] + inputs[1]],
            ScalarOp::Mul => vec![inputs[0] * inputs[1]],
            ScalarOp::Exp => vec![inputs[0].exp()],
            ScalarOp::Neg => vec![-inputs[0]],
            ScalarOp::Dup => vec![*inputs[0], *inputs[0]],
        }
    }
}

impl Operand for f64 {
    fn zero(_shape: &[usize]) -> Self {
        0.0
    }

    fn one(_shape: &[usize]) -> Self {
        1.0
    }

    fn reshape(&self, _shape: &[usize]) -> Self {
        *self
    }

    fn broadcast_in_dim(&self, _shape: &[usize], _dims: &[usize]) -> Self {
        *self
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn multiply(&self, other: &Self) -> Self {
        self * other
    }

    fn reduce_sum(&self, _axes: &[usize]) -> Self {
        *self
    }

    fn dot_general(
        &self,
        other: &Self,
        _lhs_contracting: &[usize],
        _rhs_contracting: &[usize],
        _lhs_batch: &[usize],
        _rhs_batch: &[usize],
    ) -> Self {
        self * other
    }

    fn conj(&self) -> Self {
        *self
    }
}
```

- [ ] **Step 2: Create a smoke test to verify the helpers compile**

Create `tests/scalar_tests.rs`:

```rust
mod common;

use common::ScalarOp;
use computegraph::GraphOp;

#[test]
fn scalar_op_eval_add() {
    let op = ScalarOp::Add;
    assert_eq!(op.n_inputs(), 2);
    assert_eq!(op.n_outputs(), 1);
    let result = op.eval(&mut (), &[&3.0, &4.0]);
    assert_eq!(result, vec![7.0]);
}

#[test]
fn scalar_op_eval_exp() {
    let op = ScalarOp::Exp;
    let result = op.eval(&mut (), &[&0.0]);
    assert_eq!(result, vec![1.0]);
}

#[test]
fn scalar_op_eval_dup() {
    let op = ScalarOp::Dup;
    assert_eq!(op.n_outputs(), 2);
    let result = op.eval(&mut (), &[&5.0]);
    assert_eq!(result, vec![5.0, 5.0]);
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --release`
Expected: 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add ScalarOp and f64 Operand test helpers"
```

---

### Task 4: KeyInterner

**Files:**
- Create: `src/interner.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write tests for KeyInterner**

Add to `tests/scalar_tests.rs`:

```rust
use computegraph::{GlobalOpKey, GlobalValKey, OpMode};
use computegraph::interner::KeyInterner;

#[test]
fn interner_intern_and_resolve() {
    let mut interner = KeyInterner::<ScalarOp>::new();
    let key = GlobalValKey::Input("x".to_string());
    let id = interner.intern(key.clone());
    assert_eq!(interner.resolve(id), &key);
}

#[test]
fn interner_deduplicates() {
    let mut interner = KeyInterner::<ScalarOp>::new();
    let key = GlobalValKey::Input("x".to_string());
    let id1 = interner.intern(key.clone());
    let id2 = interner.intern(key);
    assert_eq!(id1, id2);
}

#[test]
fn interner_distinct_keys_get_distinct_ids() {
    let mut interner = KeyInterner::<ScalarOp>::new();
    let id_x = interner.intern(GlobalValKey::Input("x".to_string()));
    let id_y = interner.intern(GlobalValKey::Input("y".to_string()));
    assert_ne!(id_x, id_y);
}

#[test]
fn interner_get_returns_none_for_unknown() {
    let interner = KeyInterner::<ScalarOp>::new();
    let key = GlobalValKey::Input("x".to_string());
    assert_eq!(interner.get(&key), None);
}

#[test]
fn interner_derived_key() {
    let mut interner = KeyInterner::<ScalarOp>::new();
    let key = GlobalValKey::<ScalarOp>::Derived {
        op: GlobalOpKey {
            primitive: ScalarOp::Add,
            inputs: vec![
                GlobalValKey::Input("x".to_string()),
                GlobalValKey::Input("y".to_string()),
            ],
            mode: OpMode::Primal,
        },
        output_slot: 0,
    };
    let id = interner.intern(key.clone());
    assert_eq!(interner.resolve(id), &key);
    assert_eq!(interner.get(&key), Some(id));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release`
Expected: compilation error — `interner` module not found

- [ ] **Step 3: Implement KeyInterner**

Create `src/interner.rs`:

```rust
use std::collections::HashMap;

use crate::traits::GraphOp;
use crate::types::GlobalValKey;

/// Interned identity for O(1) equality comparison of `GlobalValKey`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ValKeyId(u32);

/// Maps `GlobalValKey` to `ValKeyId` for fast equality and deduplication.
pub struct KeyInterner<Op: GraphOp> {
    map: HashMap<GlobalValKey<Op>, ValKeyId>,
    keys: Vec<GlobalValKey<Op>>,
}

impl<Op: GraphOp> KeyInterner<Op> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            keys: Vec::new(),
        }
    }

    /// Intern a key, returning its unique id. Returns existing id if already interned.
    pub fn intern(&mut self, key: GlobalValKey<Op>) -> ValKeyId {
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        let id = ValKeyId(self.keys.len() as u32);
        self.keys.push(key.clone());
        self.map.insert(key, id);
        id
    }

    /// Look up the id for a key without interning. Returns `None` if not interned.
    pub fn get(&self, key: &GlobalValKey<Op>) -> Option<ValKeyId> {
        self.map.get(key).copied()
    }

    /// Retrieve the full key from an id.
    pub fn resolve(&self, id: ValKeyId) -> &GlobalValKey<Op> {
        &self.keys[id.0 as usize]
    }
}

impl<Op: GraphOp> Default for KeyInterner<Op> {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Add module to lib.rs**

Add to `src/lib.rs`:

```rust
pub mod interner;
```

- [ ] **Step 5: Run tests**

Run: `cargo test --release`
Expected: all tests pass (previous + 5 new interner tests)

- [ ] **Step 6: Commit**

```bash
git add src/interner.rs src/lib.rs tests/scalar_tests.rs
git commit -m "feat: add KeyInterner for O(1) GlobalValKey comparison"
```

---

### Task 5: Fragment and FragmentBuilder

**Files:**
- Create: `src/fragment.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write tests for FragmentBuilder**

Add to `tests/scalar_tests.rs`:

```rust
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::ValRef;

#[test]
fn fragment_builder_single_input() {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    assert_eq!(x, 0);
    builder.set_outputs(vec![x]);
    let frag = builder.build();
    assert_eq!(frag.inputs().len(), 1);
    assert_eq!(frag.outputs().len(), 1);
    assert_eq!(frag.vals()[x].key, GlobalValKey::Input("x".to_string()));
    assert!(frag.vals()[x].producer.is_none());
}

#[test]
fn fragment_builder_add_op() {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let y = builder.add_input("y".to_string());
    let outputs = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(y)],
        OpMode::Primal,
    );
    assert_eq!(outputs.len(), 1);
    let sum_id = outputs[0];
    builder.set_outputs(vec![sum_id]);
    let frag = builder.build();

    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, ScalarOp::Add);
    assert!(frag.vals()[sum_id].producer.is_some());

    // Verify GlobalValKey structure
    let expected_key = GlobalValKey::Derived {
        op: GlobalOpKey {
            primitive: ScalarOp::Add,
            inputs: vec![
                GlobalValKey::Input("x".to_string()),
                GlobalValKey::Input("y".to_string()),
            ],
            mode: OpMode::Primal,
        },
        output_slot: 0,
    };
    assert_eq!(frag.vals()[sum_id].key, expected_key);
}

#[test]
fn fragment_builder_chain() {
    // Build: Exp(Mul(x, a))
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let a = builder.add_input("a".to_string());
    let mul_out = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let exp_out = builder.add_op(
        ScalarOp::Exp,
        vec![ValRef::Local(mul_out[0])],
        OpMode::Primal,
    );
    builder.set_outputs(vec![exp_out[0]]);
    let frag = builder.build();

    assert_eq!(frag.ops().len(), 2);
    assert_eq!(frag.vals().len(), 4); // x, a, mul_out, exp_out
}

#[test]
fn fragment_builder_dup_two_outputs() {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let dup_outs = builder.add_op(
        ScalarOp::Dup,
        vec![ValRef::Local(x)],
        OpMode::Primal,
    );
    assert_eq!(dup_outs.len(), 2);
    builder.set_outputs(dup_outs.clone());
    let frag = builder.build();

    assert_eq!(frag.outputs().len(), 2);
    // Both outputs should be Derived with different output_slot
    let key0 = &frag.vals()[dup_outs[0]].key;
    let key1 = &frag.vals()[dup_outs[1]].key;
    match (key0, key1) {
        (
            GlobalValKey::Derived { output_slot: s0, .. },
            GlobalValKey::Derived { output_slot: s1, .. },
        ) => {
            assert_eq!(*s0, 0);
            assert_eq!(*s1, 1);
        }
        _ => panic!("expected Derived keys"),
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release`
Expected: compilation error — `fragment` module not found

- [ ] **Step 3: Implement Fragment and FragmentBuilder**

Create `src/fragment.rs`:

```rust
use std::sync::Arc;

use crate::traits::GraphOp;
use crate::types::{GlobalOpKey, GlobalValKey, LocalOpId, LocalValId, OpMode, ValRef};

/// A value node in a fragment.
pub struct ValNode<Op: GraphOp> {
    /// Cross-fragment structural identity.
    pub key: GlobalValKey<Op>,
    /// `None` for fragment inputs; `Some((op_id, output_slot))` for produced values.
    pub producer: Option<(LocalOpId, usize)>,
}

/// An operation node in a fragment.
pub struct OpNode<Op: GraphOp> {
    pub op: Op,
    pub inputs: Vec<ValRef<Op>>,
    pub outputs: Vec<LocalValId>,
    pub mode: OpMode,
}

/// The unit of graph construction. Owns local nodes, may reference
/// values in other fragments through external references.
pub struct Fragment<Op: GraphOp> {
    pub(crate) vals: Vec<ValNode<Op>>,
    pub(crate) ops: Vec<OpNode<Op>>,
    pub(crate) inputs: Vec<LocalValId>,
    pub(crate) outputs: Vec<LocalValId>,
    pub(crate) parents: Vec<Arc<Fragment<Op>>>,
}

impl<Op: GraphOp> Fragment<Op> {
    pub fn vals(&self) -> &[ValNode<Op>] {
        &self.vals
    }

    pub fn ops(&self) -> &[OpNode<Op>] {
        &self.ops
    }

    pub fn inputs(&self) -> &[LocalValId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[LocalValId] {
        &self.outputs
    }

    pub fn parents(&self) -> &[Arc<Fragment<Op>>] {
        &self.parents
    }
}

/// Builder for constructing fragments incrementally.
pub struct FragmentBuilder<Op: GraphOp> {
    vals: Vec<ValNode<Op>>,
    ops: Vec<OpNode<Op>>,
    inputs: Vec<LocalValId>,
    outputs: Vec<LocalValId>,
    parents: Vec<Arc<Fragment<Op>>>,
    /// Maps LocalValId -> GlobalValKey for resolving Local refs during add_op.
    local_keys: Vec<GlobalValKey<Op>>,
}

impl<Op: GraphOp> FragmentBuilder<Op> {
    pub fn new() -> Self {
        Self {
            vals: Vec::new(),
            ops: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            parents: Vec::new(),
            local_keys: Vec::new(),
        }
    }

    /// Add a fragment input with the given key. Returns its local id.
    pub fn add_input(&mut self, key: Op::InputKey) -> LocalValId {
        let val_id = self.vals.len();
        let global_key = GlobalValKey::Input(key);
        self.vals.push(ValNode {
            key: global_key.clone(),
            producer: None,
        });
        self.local_keys.push(global_key);
        self.inputs.push(val_id);
        val_id
    }

    /// Add an operation node. Returns local ids for each output.
    ///
    /// Computes `GlobalValKey` for each output by resolving input `ValRef`s
    /// to their global keys and constructing `GlobalOpKey`.
    pub fn add_op(
        &mut self,
        op: Op,
        inputs: Vec<ValRef<Op>>,
        mode: OpMode,
    ) -> Vec<LocalValId> {
        let op_id = self.ops.len();
        let n_outputs = op.n_outputs();

        // Resolve input ValRefs to GlobalValKeys
        let global_inputs: Vec<GlobalValKey<Op>> = inputs
            .iter()
            .map(|vr| match vr {
                ValRef::Local(id) => self.local_keys[*id].clone(),
                ValRef::External(key) => key.clone(),
            })
            .collect();

        let global_op_key = GlobalOpKey {
            primitive: op.clone(),
            inputs: global_inputs,
            mode: mode.clone(),
        };

        let mut output_ids = Vec::with_capacity(n_outputs);
        for slot in 0..n_outputs {
            let val_id = self.vals.len();
            let key = GlobalValKey::Derived {
                op: global_op_key.clone(),
                output_slot: slot as u8,
            };
            self.vals.push(ValNode {
                key: key.clone(),
                producer: Some((op_id, slot)),
            });
            self.local_keys.push(key);
            output_ids.push(val_id);
        }

        self.ops.push(OpNode {
            op,
            inputs,
            outputs: output_ids.clone(),
            mode,
        });

        output_ids
    }

    /// Declare which local values are the fragment's outputs.
    pub fn set_outputs(&mut self, outputs: Vec<LocalValId>) {
        self.outputs = outputs;
    }

    /// Register a parent fragment for external reference resolution.
    pub fn add_parent(&mut self, parent: Arc<Fragment<Op>>) {
        self.parents.push(parent);
    }

    /// Retrieve the `GlobalValKey` for a local value id.
    pub fn global_key(&self, local_id: LocalValId) -> &GlobalValKey<Op> {
        &self.local_keys[local_id]
    }

    /// Consume the builder and produce a `Fragment`.
    pub fn build(self) -> Fragment<Op> {
        Fragment {
            vals: self.vals,
            ops: self.ops,
            inputs: self.inputs,
            outputs: self.outputs,
            parents: self.parents,
        }
    }
}

impl<Op: GraphOp> Default for FragmentBuilder<Op> {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Add module to lib.rs**

Add to `src/lib.rs`:

```rust
pub mod fragment;
```

- [ ] **Step 5: Run tests**

Run: `cargo test --release`
Expected: all tests pass (previous + 4 new fragment tests)

- [ ] **Step 6: Commit**

```bash
git add src/fragment.rs src/lib.rs tests/scalar_tests.rs
git commit -m "feat: add Fragment and FragmentBuilder for graph construction"
```

---

### Task 6: Resolve

**Files:**
- Create: `src/resolve.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write tests for resolve**

Add to `tests/scalar_tests.rs`:

```rust
use std::sync::Arc;
use computegraph::resolve::{resolve, ValDef};

#[test]
fn resolve_single_fragment() {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let y = builder.add_input("y".to_string());
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(y)],
        OpMode::Primal,
    );
    builder.set_outputs(vec![sum[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag.clone()]);

    // Input keys should resolve
    let x_key = GlobalValKey::Input("x".to_string());
    match view.resolve_val(&x_key).unwrap() {
        ValDef::Input { key } => assert_eq!(key, "x"),
        _ => panic!("expected Input"),
    }

    // Derived key should resolve
    let sum_key = &frag.vals()[sum[0]].key;
    match view.resolve_val(sum_key).unwrap() {
        ValDef::Produced { op, input_keys, mode, output_slot } => {
            assert_eq!(op, ScalarOp::Add);
            assert_eq!(input_keys.len(), 2);
            assert_eq!(mode, OpMode::Primal);
            assert_eq!(output_slot, 0);
        }
        _ => panic!("expected Produced"),
    }
}

#[test]
fn resolve_external_ref_across_fragments() {
    // Fragment F0: x, a, mul = Mul(x, a)
    let mut b0 = FragmentBuilder::<ScalarOp>::new();
    let x = b0.add_input("x".to_string());
    let a = b0.add_input("a".to_string());
    let mul = b0.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let mul_key = b0.global_key(mul[0]).clone();
    b0.set_outputs(vec![mul[0]]);
    let f0 = Arc::new(b0.build());

    // Fragment F1: references F0's mul output via External, applies Exp
    let mut b1 = FragmentBuilder::<ScalarOp>::new();
    b1.add_parent(f0.clone());
    let exp = b1.add_op(
        ScalarOp::Exp,
        vec![ValRef::External(mul_key.clone())],
        OpMode::Primal,
    );
    b1.set_outputs(vec![exp[0]]);
    let f1 = Arc::new(b1.build());

    let view = resolve(vec![f0, f1.clone()]);

    // mul_key should be resolvable
    assert!(view.resolve_val(&mul_key).is_some());

    // exp output should be resolvable
    let exp_key = &f1.vals()[exp[0]].key;
    match view.resolve_val(exp_key).unwrap() {
        ValDef::Produced { op, input_keys, .. } => {
            assert_eq!(op, ScalarOp::Exp);
            assert_eq!(input_keys.len(), 1);
            assert_eq!(input_keys[0], mul_key);
        }
        _ => panic!("expected Produced"),
    }
}

#[test]
fn resolve_unknown_key_returns_none() {
    let builder = FragmentBuilder::<ScalarOp>::new();
    let frag = Arc::new(builder.build());
    let view = resolve(vec![frag]);
    let unknown = GlobalValKey::Input("unknown".to_string());
    assert!(view.resolve_val(&unknown).is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release`
Expected: compilation error — `resolve` module not found

- [ ] **Step 3: Implement resolve**

Create `src/resolve.rs`:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use crate::fragment::Fragment;
use crate::traits::GraphOp;
use crate::types::{GlobalValKey, OpMode, ValRef};

/// Definition of a value as seen through the resolver.
#[derive(Clone, Debug, PartialEq)]
pub enum ValDef<Op: GraphOp> {
    Input {
        key: Op::InputKey,
    },
    Produced {
        op: Op,
        /// Inputs resolved to global keys (no `ValRef::Local` remains).
        input_keys: Vec<GlobalValKey<Op>>,
        mode: OpMode,
        output_slot: usize,
    },
}

/// Trait for resolving `GlobalValKey` to its definition.
pub trait Resolver<Op: GraphOp> {
    fn resolve_val(&self, key: &GlobalValKey<Op>) -> Option<ValDef<Op>>;
}

/// Logical traversal view over one or more fragments.
pub struct ResolvedView<Op: GraphOp> {
    pub roots: Vec<Arc<Fragment<Op>>>,
    resolver: Box<dyn Resolver<Op>>,
}

impl<Op: GraphOp> ResolvedView<Op> {
    pub fn resolve_val(&self, key: &GlobalValKey<Op>) -> Option<ValDef<Op>> {
        self.resolver.resolve_val(key)
    }
}

/// HashMap-backed resolver.
struct HashMapResolver<Op: GraphOp> {
    map: HashMap<GlobalValKey<Op>, ValDef<Op>>,
}

impl<Op: GraphOp> Resolver<Op> for HashMapResolver<Op> {
    fn resolve_val(&self, key: &GlobalValKey<Op>) -> Option<ValDef<Op>> {
        self.map.get(key).cloned()
    }
}

/// Build a logical lookup view over fragments and their parent chains.
/// No node copying — just index construction.
pub fn resolve<Op: GraphOp>(roots: Vec<Arc<Fragment<Op>>>) -> ResolvedView<Op> {
    let mut map: HashMap<GlobalValKey<Op>, ValDef<Op>> = HashMap::new();

    fn walk_fragment<Op: GraphOp>(
        fragment: &Fragment<Op>,
        map: &mut HashMap<GlobalValKey<Op>, ValDef<Op>>,
    ) {
        // Walk parents first (earlier fragments define values that later ones reference)
        for parent in fragment.parents() {
            walk_fragment(parent, map);
        }

        // Register this fragment's values
        for val in fragment.vals() {
            if map.contains_key(&val.key) {
                continue; // already registered (e.g., from a parent)
            }
            match val.producer {
                None => {
                    // Input node
                    if let GlobalValKey::Input(ref input_key) = val.key {
                        map.insert(
                            val.key.clone(),
                            ValDef::Input {
                                key: input_key.clone(),
                            },
                        );
                    }
                }
                Some((op_id, slot)) => {
                    let op_node = &fragment.ops()[op_id];
                    // Resolve Local refs to GlobalValKeys
                    let input_keys: Vec<GlobalValKey<Op>> = op_node
                        .inputs
                        .iter()
                        .map(|vr| match vr {
                            ValRef::Local(id) => fragment.vals()[*id].key.clone(),
                            ValRef::External(key) => key.clone(),
                        })
                        .collect();
                    map.insert(
                        val.key.clone(),
                        ValDef::Produced {
                            op: op_node.op.clone(),
                            input_keys,
                            mode: op_node.mode.clone(),
                            output_slot: slot,
                        },
                    );
                }
            }
        }
    }

    for root in &roots {
        walk_fragment(root, &mut map);
    }

    ResolvedView {
        roots,
        resolver: Box::new(HashMapResolver { map }),
    }
}
```

- [ ] **Step 4: Add module to lib.rs**

Add to `src/lib.rs`:

```rust
pub mod resolve;
```

- [ ] **Step 5: Run tests**

Run: `cargo test --release`
Expected: all tests pass (previous + 3 new resolve tests)

- [ ] **Step 6: Commit**

```bash
git add src/resolve.rs src/lib.rs tests/scalar_tests.rs
git commit -m "feat: add resolve for logical fragment traversal"
```

---

### Task 7: materialize_merge

**Files:**
- Create: `src/materialize.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write tests for materialize_merge**

Add to `tests/scalar_tests.rs`:

```rust
use computegraph::materialize::materialize_merge;

#[test]
fn materialize_single_op() {
    // Add(x, y)
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let y = builder.add_input("y".to_string());
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(y)],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[sum_key]);

    assert_eq!(graph.ops.len(), 1);
    assert_eq!(graph.ops[0].op, ScalarOp::Add);
    // 3 vals: x, y, sum
    assert_eq!(graph.vals.len(), 3);
    assert_eq!(graph.inputs.len(), 2);
    assert_eq!(graph.outputs.len(), 1);
}

#[test]
fn materialize_chain() {
    // Exp(Mul(x, a))
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let a = builder.add_input("a".to_string());
    let mul = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let exp = builder.add_op(
        ScalarOp::Exp,
        vec![ValRef::Local(mul[0])],
        OpMode::Primal,
    );
    let exp_key = builder.global_key(exp[0]).clone();
    builder.set_outputs(vec![exp[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[exp_key]);

    assert_eq!(graph.ops.len(), 2); // Mul, Exp
    assert_eq!(graph.vals.len(), 4); // x, a, mul, exp
    // Ops should be in topological order: Mul before Exp
    assert_eq!(graph.ops[0].op, ScalarOp::Mul);
    assert_eq!(graph.ops[1].op, ScalarOp::Exp);
}

#[test]
fn materialize_cse_deduplicates() {
    // x + x: both inputs to Add reference the same GlobalValKey
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[sum_key]);

    // Only 1 input val (x is deduplicated), 1 op (Add), 1 output
    assert_eq!(graph.vals.len(), 2); // x, sum
    assert_eq!(graph.ops.len(), 1);
    // Both inputs to Add should reference the same val index
    assert_eq!(graph.ops[0].inputs[0], graph.ops[0].inputs[1]);
}

#[test]
fn materialize_across_fragments() {
    // F0: mul = Mul(x, a)
    let mut b0 = FragmentBuilder::<ScalarOp>::new();
    let x = b0.add_input("x".to_string());
    let a = b0.add_input("a".to_string());
    let mul = b0.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let mul_key = b0.global_key(mul[0]).clone();
    b0.set_outputs(vec![mul[0]]);
    let f0 = Arc::new(b0.build());

    // F1: exp = Exp(External(mul_key))
    let mut b1 = FragmentBuilder::<ScalarOp>::new();
    b1.add_parent(f0.clone());
    let exp = b1.add_op(
        ScalarOp::Exp,
        vec![ValRef::External(mul_key)],
        OpMode::Primal,
    );
    let exp_key = b1.global_key(exp[0]).clone();
    b1.set_outputs(vec![exp[0]]);
    let f1 = Arc::new(b1.build());

    let view = resolve(vec![f0, f1]);
    let graph = materialize_merge(&view, &[exp_key]);

    assert_eq!(graph.ops.len(), 2); // Mul, Exp
    assert_eq!(graph.vals.len(), 4); // x, a, mul, exp
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release`
Expected: compilation error — `materialize` module not found

- [ ] **Step 3: Implement materialize_merge**

Create `src/materialize.rs`:

```rust
use std::collections::HashMap;

use crate::resolve::{ResolvedView, ValDef};
use crate::traits::GraphOp;
use crate::types::{GlobalOpKey, GlobalValKey, OpMode};

/// A value in the materialized graph.
pub struct MaterializedVal<Op: GraphOp> {
    pub key: GlobalValKey<Op>,
    /// `None` for inputs; `Some((op_index, output_slot))` for produced values.
    pub producer: Option<(usize, usize)>,
}

/// An operation in the materialized graph.
pub struct MaterializedOp<Op: GraphOp> {
    pub op: Op,
    pub inputs: Vec<usize>,  // val indices
    pub outputs: Vec<usize>, // val indices
    pub mode: OpMode,
}

/// Fully flattened, deduplicated graph ready for compilation.
pub struct MaterializedGraph<Op: GraphOp> {
    pub vals: Vec<MaterializedVal<Op>>,
    pub ops: Vec<MaterializedOp<Op>>,
    pub inputs: Vec<GlobalValKey<Op>>,
    pub outputs: Vec<GlobalValKey<Op>>,
}

struct Materializer<'a, Op: GraphOp> {
    view: &'a ResolvedView<Op>,
    val_map: HashMap<GlobalValKey<Op>, usize>,
    op_map: HashMap<GlobalOpKey<Op>, usize>,
    vals: Vec<MaterializedVal<Op>>,
    ops: Vec<MaterializedOp<Op>>,
    input_keys: Vec<GlobalValKey<Op>>,
}

impl<'a, Op: GraphOp> Materializer<'a, Op> {
    fn new(view: &'a ResolvedView<Op>) -> Self {
        Self {
            view,
            val_map: HashMap::new(),
            op_map: HashMap::new(),
            vals: Vec::new(),
            ops: Vec::new(),
            input_keys: Vec::new(),
        }
    }

    fn visit(&mut self, key: &GlobalValKey<Op>) -> usize {
        if let Some(&idx) = self.val_map.get(key) {
            return idx;
        }

        let val_def = self
            .view
            .resolve_val(key)
            .unwrap_or_else(|| panic!("key not found in resolved view: {:?}", key));

        match val_def {
            ValDef::Input { .. } => {
                let idx = self.vals.len();
                self.vals.push(MaterializedVal {
                    key: key.clone(),
                    producer: None,
                });
                self.val_map.insert(key.clone(), idx);
                self.input_keys.push(key.clone());
                idx
            }
            ValDef::Produced {
                op,
                input_keys,
                mode,
                output_slot,
            } => {
                let op_key = GlobalOpKey {
                    primitive: op.clone(),
                    inputs: input_keys.clone(),
                    mode: mode.clone(),
                };

                let op_idx = if let Some(&idx) = self.op_map.get(&op_key) {
                    // Op already materialized — just add this output val
                    idx
                } else {
                    // First time seeing this op: visit all inputs first
                    let mat_input_indices: Vec<usize> = input_keys
                        .iter()
                        .map(|k| self.visit(k))
                        .collect();

                    let idx = self.ops.len();
                    self.ops.push(MaterializedOp {
                        op: op.clone(),
                        inputs: mat_input_indices,
                        outputs: vec![usize::MAX; op.n_outputs()],
                        mode,
                    });
                    self.op_map.insert(op_key, idx);
                    idx
                };

                // Add this output val
                let val_idx = self.vals.len();
                self.vals.push(MaterializedVal {
                    key: key.clone(),
                    producer: Some((op_idx, output_slot)),
                });
                self.val_map.insert(key.clone(), val_idx);
                self.ops[op_idx].outputs[output_slot] = val_idx;

                val_idx
            }
        }
    }
}

/// Flatten resolved fragments into a single materialized graph.
///
/// Walks from `outputs`, collects reachable definitions, deduplicates
/// by `GlobalValKey`, and produces topologically ordered vals and ops.
pub fn materialize_merge<Op: GraphOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
) -> MaterializedGraph<Op> {
    let mut mat = Materializer::new(view);

    for key in outputs {
        mat.visit(key);
    }

    MaterializedGraph {
        vals: mat.vals,
        ops: mat.ops,
        inputs: mat.input_keys,
        outputs: outputs.to_vec(),
    }
}
```

- [ ] **Step 4: Add module to lib.rs**

Add to `src/lib.rs`:

```rust
pub mod materialize;
```

- [ ] **Step 5: Run tests**

Run: `cargo test --release`
Expected: all tests pass (previous + 4 new materialize tests)

- [ ] **Step 6: Commit**

```bash
git add src/materialize.rs src/lib.rs tests/scalar_tests.rs
git commit -m "feat: add materialize_merge for graph flattening and CSE"
```

---

### Task 8: compile and eval

**Files:**
- Create: `src/compile.rs`
- Create: `src/eval.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write tests for compile + eval**

Add to `tests/scalar_tests.rs`:

```rust
use computegraph::compile::compile;

#[test]
fn compile_and_eval_add() {
    // Add(x, y) with x=3, y=4
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let y = builder.add_input("y".to_string());
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(y)],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[sum_key]);
    let prog = compile(&graph);

    assert_eq!(prog.input_slots.len(), 2);
    assert_eq!(prog.output_slots.len(), 1);
    assert_eq!(prog.instructions.len(), 1);

    let result = prog.eval(&mut (), &[&3.0, &4.0]);
    assert_eq!(result, vec![7.0]);
}

#[test]
fn compile_and_eval_chain() {
    // Exp(Mul(x, a)) with x=1, a=2 => exp(2)
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let a = builder.add_input("a".to_string());
    let mul = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let exp = builder.add_op(
        ScalarOp::Exp,
        vec![ValRef::Local(mul[0])],
        OpMode::Primal,
    );
    let exp_key = builder.global_key(exp[0]).clone();
    builder.set_outputs(vec![exp[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[exp_key]);
    let prog = compile(&graph);

    let result = prog.eval(&mut (), &[&1.0, &2.0]);
    assert!((result[0] - 2.0_f64.exp()).abs() < 1e-12);
}

#[test]
fn compile_and_eval_reuse() {
    // Compile once, eval multiple times with different inputs
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let y = builder.add_input("y".to_string());
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(y)],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[sum_key]);
    let prog = compile(&graph);

    assert_eq!(prog.eval(&mut (), &[&1.0, &2.0]), vec![3.0]);
    assert_eq!(prog.eval(&mut (), &[&10.0, &20.0]), vec![30.0]);
}

#[test]
fn compile_and_eval_multi_output() {
    // Request both mul and exp outputs
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let a = builder.add_input("a".to_string());
    let mul = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let exp = builder.add_op(
        ScalarOp::Exp,
        vec![ValRef::Local(mul[0])],
        OpMode::Primal,
    );
    let mul_key = builder.global_key(mul[0]).clone();
    let exp_key = builder.global_key(exp[0]).clone();
    builder.set_outputs(vec![mul[0], exp[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[mul_key, exp_key]);
    let prog = compile(&graph);

    let result = prog.eval(&mut (), &[&1.0, &2.0]);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 2.0).abs() < 1e-12);       // mul: 1*2
    assert!((result[1] - 2.0_f64.exp()).abs() < 1e-12); // exp(2)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release`
Expected: compilation error — `compile` module not found

- [ ] **Step 3: Implement compile**

Create `src/compile.rs`:

```rust
use std::collections::HashMap;

use crate::materialize::MaterializedGraph;
use crate::traits::GraphOp;
use crate::types::GlobalValKey;

/// A single instruction in the compiled program.
pub struct Instruction<Op: GraphOp> {
    pub op: Op,
    pub inputs: Vec<usize>,  // slot indices
    pub outputs: Vec<usize>, // slot indices
}

/// SSA-form compiled program. Each slot is written exactly once.
pub struct CompiledProgram<Op: GraphOp> {
    pub instructions: Vec<Instruction<Op>>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
}

/// Compile a materialized graph into an SSA instruction sequence.
///
/// Val indices map 1:1 to slot indices. Instructions are in topological order
/// (inherited from `materialize_merge`).
pub fn compile<Op: GraphOp>(graph: &MaterializedGraph<Op>) -> CompiledProgram<Op> {
    let n_slots = graph.vals.len();

    let instructions: Vec<Instruction<Op>> = graph
        .ops
        .iter()
        .map(|op_node| Instruction {
            op: op_node.op.clone(),
            inputs: op_node.inputs.clone(),
            outputs: op_node.outputs.clone(),
        })
        .collect();

    // Input slots: val indices with no producer
    let input_slots: Vec<usize> = graph
        .vals
        .iter()
        .enumerate()
        .filter(|(_, v)| v.producer.is_none())
        .map(|(i, _)| i)
        .collect();

    // Output slots: val indices for the requested outputs
    let key_to_idx: HashMap<&GlobalValKey<Op>, usize> = graph
        .vals
        .iter()
        .enumerate()
        .map(|(i, v)| (&v.key, i))
        .collect();

    let output_slots: Vec<usize> = graph
        .outputs
        .iter()
        .map(|key| key_to_idx[key])
        .collect();

    CompiledProgram {
        instructions,
        input_slots,
        output_slots,
        n_slots,
    }
}
```

- [ ] **Step 4: Implement eval**

Create `src/eval.rs`:

```rust
use crate::compile::CompiledProgram;
use crate::traits::GraphOp;

impl<Op: GraphOp> CompiledProgram<Op> {
    /// Execute the compiled program with the given inputs.
    ///
    /// Input order matches the order of input vals in the materialized graph
    /// (i.e., the order they were first encountered during `materialize_merge`).
    pub fn eval(&self, ctx: &mut Op::Context, inputs: &[&Op::Operand]) -> Vec<Op::Operand> {
        assert_eq!(
            inputs.len(),
            self.input_slots.len(),
            "expected {} inputs, got {}",
            self.input_slots.len(),
            inputs.len()
        );

        let mut slots: Vec<Option<Op::Operand>> = vec![None; self.n_slots];

        // Fill input slots
        for (i, &slot) in self.input_slots.iter().enumerate() {
            slots[slot] = Some(inputs[i].clone());
        }

        // Execute instructions in order
        for instr in &self.instructions {
            let input_vals: Vec<&Op::Operand> = instr
                .inputs
                .iter()
                .map(|&slot| {
                    slots[slot]
                        .as_ref()
                        .unwrap_or_else(|| panic!("slot {} not filled", slot))
                })
                .collect();

            let outputs = instr.op.eval(ctx, &input_vals);

            for (i, &slot) in instr.outputs.iter().enumerate() {
                slots[slot] = Some(outputs[i].clone());
            }
        }

        // Collect requested outputs
        self.output_slots
            .iter()
            .map(|&slot| {
                slots[slot]
                    .clone()
                    .unwrap_or_else(|| panic!("output slot {} not filled", slot))
            })
            .collect()
    }
}
```

- [ ] **Step 5: Add modules to lib.rs**

Add to `src/lib.rs`:

```rust
pub mod compile;
mod eval; // not `pub` — eval is an impl block on CompiledProgram, accessed via compile module
```

- [ ] **Step 6: Run tests**

Run: `cargo test --release`
Expected: all tests pass (previous + 4 new compile/eval tests)

- [ ] **Step 7: Commit**

```bash
git add src/compile.rs src/eval.rs src/lib.rs tests/scalar_tests.rs
git commit -m "feat: add compile and eval for SSA execution"
```

---

### Task 9: End-to-End Integration Tests

**Files:**
- Modify: `tests/scalar_tests.rs`

- [ ] **Step 1: Add end-to-end test for exp(a*x) across fragments**

Add to `tests/scalar_tests.rs`:

```rust
#[test]
fn e2e_exp_ax_single_fragment() {
    // f(x, a) = exp(a * x)
    // With x=2, a=3: exp(6)
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let a = builder.add_input("a".to_string());
    let mul = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let exp = builder.add_op(
        ScalarOp::Exp,
        vec![ValRef::Local(mul[0])],
        OpMode::Primal,
    );
    let exp_key = builder.global_key(exp[0]).clone();
    builder.set_outputs(vec![exp[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[exp_key]);
    let prog = compile(&graph);
    let result = prog.eval(&mut (), &[&2.0, &3.0]);

    assert!((result[0] - 6.0_f64.exp()).abs() < 1e-12);
}

#[test]
fn e2e_exp_ax_multi_fragment() {
    // F0: mul = Mul(x, a)
    let mut b0 = FragmentBuilder::<ScalarOp>::new();
    let x = b0.add_input("x".to_string());
    let a = b0.add_input("a".to_string());
    let mul = b0.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let mul_key = b0.global_key(mul[0]).clone();
    b0.set_outputs(vec![mul[0]]);
    let f0 = Arc::new(b0.build());

    // F1: exp = Exp(External(mul_key))
    let mut b1 = FragmentBuilder::<ScalarOp>::new();
    b1.add_parent(f0.clone());
    let exp = b1.add_op(
        ScalarOp::Exp,
        vec![ValRef::External(mul_key)],
        OpMode::Primal,
    );
    let exp_key = b1.global_key(exp[0]).clone();
    b1.set_outputs(vec![exp[0]]);
    let f1 = Arc::new(b1.build());

    // Full pipeline
    let view = resolve(vec![f0, f1]);
    let graph = materialize_merge(&view, &[exp_key]);
    let prog = compile(&graph);

    let result = prog.eval(&mut (), &[&2.0, &3.0]);
    assert!((result[0] - 6.0_f64.exp()).abs() < 1e-12);

    // Eval again with different values
    let result2 = prog.eval(&mut (), &[&1.0, &1.0]);
    assert!((result2[0] - 1.0_f64.exp()).abs() < 1e-12);
}

#[test]
fn e2e_x_plus_x() {
    // f(x) = x + x
    // Tests that CSE correctly handles the same input used twice
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[sum_key]);
    let prog = compile(&graph);

    let result = prog.eval(&mut (), &[&5.0]);
    assert_eq!(result, vec![10.0]);
}

#[test]
fn e2e_neg_exp() {
    // f(x) = -exp(x)
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let exp = builder.add_op(
        ScalarOp::Exp,
        vec![ValRef::Local(x)],
        OpMode::Primal,
    );
    let neg = builder.add_op(
        ScalarOp::Neg,
        vec![ValRef::Local(exp[0])],
        OpMode::Primal,
    );
    let neg_key = builder.global_key(neg[0]).clone();
    builder.set_outputs(vec![neg[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[neg_key]);
    let prog = compile(&graph);

    let result = prog.eval(&mut (), &[&0.0]);
    assert!((result[0] - (-1.0)).abs() < 1e-12);
}

#[test]
fn e2e_dup_and_add() {
    // f(x) = let (a, b) = dup(x); a + b  (== 2x)
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input("x".to_string());
    let dup = builder.add_op(
        ScalarOp::Dup,
        vec![ValRef::Local(x)],
        OpMode::Primal,
    );
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(dup[0]), ValRef::Local(dup[1])],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    let frag = Arc::new(builder.build());

    let view = resolve(vec![frag]);
    let graph = materialize_merge(&view, &[sum_key]);
    let prog = compile(&graph);

    let result = prog.eval(&mut (), &[&7.0]);
    assert_eq!(result, vec![14.0]);
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --release`
Expected: all tests pass (previous + 5 new e2e tests)

- [ ] **Step 3: Run clippy and fmt**

Run: `cargo fmt --all && cargo clippy --workspace`
Expected: no warnings or errors

- [ ] **Step 4: Commit**

```bash
git add tests/scalar_tests.rs
git commit -m "test: add end-to-end integration tests for full pipeline"
```

---

## Summary

| Task | What it delivers | Tests |
|------|------------------|-------|
| 1 | Project scaffold, compiles | - |
| 2 | `Operand`, `GraphOp` traits, all type definitions | compilation only |
| 3 | `ScalarOp` + `f64 Operand` test helpers | 3 tests |
| 4 | `KeyInterner` | 5 tests |
| 5 | `Fragment`, `FragmentBuilder` | 4 tests |
| 6 | `resolve`, `ResolvedView` | 3 tests |
| 7 | `materialize_merge`, `MaterializedGraph` | 4 tests |
| 8 | `compile`, `CompiledProgram::eval` | 4 tests |
| 9 | End-to-end integration | 5 tests |
| **Total** | | **28 tests** |
