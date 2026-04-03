# computegraph-rs Implementation Spec

**Date:** 2026-04-03
**Status:** Approved
**Upstream design:** `tensor4all-meta/docs/design-v2/computegraph-design.md`

---

## Goal

Implement `computegraph-rs` as a new Rust crate at
`/home/shinaoka/tensor4all/computegraph-rs/`. This is the AD-agnostic
tensor computation graph engine that chainrules-rs, tidu-rs, and tenferro-rs
depend on.

Scope: full Phase 1 (all functionality computegraph-rs owns).

---

## Module Layout

```
computegraph-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs              # pub re-exports
│   ├── traits.rs           # GraphOp, Operand
│   ├── types.rs            # LocalValId, LocalOpId, OpMode, ValRef, GlobalValKey, GlobalOpKey
│   ├── interner.rs         # KeyInterner, ValKeyId
│   ├── fragment.rs         # Fragment, ValNode, OpNode, FragmentBuilder
│   ├── resolve.rs          # Resolver, ResolvedView, resolve()
│   ├── materialize.rs      # MaterializedGraph, materialize_merge()
│   ├── compile.rs          # CompiledProgram, Instruction, compile()
│   └── eval.rs             # eval() on CompiledProgram
├── tests/
│   └── scalar_tests.rs     # ScalarOp + f64 Operand, end-to-end tests
├── AGENTS.md
└── README.md
```

---

## 1. Traits (`traits.rs`)

### Operand

Runtime value type. Scalars are rank-0 tensors.

```rust
pub trait Operand: Clone + Send + Sync + 'static {
    fn zero(shape: &[usize]) -> Self;
    fn one(shape: &[usize]) -> Self;
    fn reshape(&self, shape: &[usize]) -> Self;
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn reduce_sum(&self, axes: &[usize]) -> Self;
    fn dot_general(
        &self, other: &Self,
        lhs_contracting: &[usize], rhs_contracting: &[usize],
        lhs_batch: &[usize], rhs_batch: &[usize],
    ) -> Self;
    fn conj(&self) -> Self;
}
```

### GraphOp

Operation node trait. computegraph is fully generic over this.

```rust
pub trait GraphOp: Clone + Hash + Eq + Send + Sync + 'static {
    type Operand: Operand;
    type Context;
    type InputKey: Clone + Hash + Eq + Send + Sync + 'static;

    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;
    fn eval(&self, ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;
}
```

---

## 2. Types (`types.rs`)

```rust
pub type LocalValId = usize;
pub type LocalOpId = usize;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpMode {
    Primal,
    Linear { active_mask: Vec<bool> },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValRef<Op: GraphOp> {
    Local(LocalValId),
    External(GlobalValKey<Op>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GlobalValKey<Op: GraphOp> {
    Input(Op::InputKey),
    Derived {
        op: GlobalOpKey<Op>,
        output_slot: u8,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlobalOpKey<Op: GraphOp> {
    pub primitive: Op,
    pub inputs: Vec<GlobalValKey<Op>>,
    pub mode: OpMode,
}
```

---

## 3. Key Interner (`interner.rs`)

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ValKeyId(u32);

pub struct KeyInterner<Op: GraphOp> {
    map: HashMap<GlobalValKey<Op>, ValKeyId>,
    keys: Vec<GlobalValKey<Op>>,
}
```

Methods: `new`, `intern`, `get`, `resolve`.

Single-threaded for Phase 1. Passed as `&mut KeyInterner` to
`FragmentBuilder::new`.

---

## 4. Fragment (`fragment.rs`)

### Data structures

```rust
pub struct ValNode<Op: GraphOp> {
    pub key: GlobalValKey<Op>,
    pub producer: Option<(LocalOpId, usize)>,
}

pub struct OpNode<Op: GraphOp> {
    pub op: Op,
    pub inputs: Vec<ValRef<Op>>,
    pub outputs: Vec<LocalValId>,
    pub mode: OpMode,
}

pub struct Fragment<Op: GraphOp> {
    pub(crate) vals: Vec<ValNode<Op>>,
    pub(crate) ops: Vec<OpNode<Op>>,
    pub(crate) inputs: Vec<LocalValId>,
    pub(crate) outputs: Vec<LocalValId>,
    pub(crate) parents: Vec<Arc<Fragment<Op>>>,
}
```

### FragmentBuilder

```rust
pub struct FragmentBuilder<Op: GraphOp> { /* ... */ }

impl<Op: GraphOp> FragmentBuilder<Op> {
    pub fn new(interner: &mut KeyInterner<Op>) -> Self;
    pub fn add_input(&mut self, key: Op::InputKey) -> LocalValId;
    pub fn add_op(
        &mut self, op: Op, inputs: Vec<ValRef<Op>>, mode: OpMode,
    ) -> Vec<LocalValId>;
    pub fn set_outputs(&mut self, outputs: Vec<LocalValId>);
    pub fn add_parent(&mut self, parent: Arc<Fragment<Op>>);
    pub fn build(self) -> Fragment<Op>;
}
```

`add_op` computes `GlobalValKey` for each output by resolving input
`ValRef`s to their global keys and constructing `GlobalOpKey`.

---

## 5. Resolve (`resolve.rs`)

```rust
pub enum ValDef<Op: GraphOp> {
    Input { key: Op::InputKey },
    Produced {
        op: Op,
        inputs: Vec<ValRef<Op>>,
        mode: OpMode,
        output_slot: usize,
    },
}

pub trait Resolver<Op: GraphOp> {
    fn resolve_val(&self, key: &GlobalValKey<Op>) -> Option<ValDef<Op>>;
}

pub struct ResolvedView<Op: GraphOp> {
    pub roots: Vec<Arc<Fragment<Op>>>,
    resolver: Box<dyn Resolver<Op>>,
}

pub fn resolve<Op: GraphOp>(roots: Vec<Arc<Fragment<Op>>>) -> ResolvedView<Op>;
```

Implementation: build a `HashMap<GlobalValKey, ValDef>` by walking all
fragments and their parent chains. No node copying.

---

## 6. Materialize (`materialize.rs`)

```rust
pub struct MaterializedVal<Op: GraphOp> {
    pub key: GlobalValKey<Op>,
    pub producer: Option<(usize, usize)>,  // (op_index, output_slot)
}

pub struct MaterializedOp<Op: GraphOp> {
    pub op: Op,
    pub inputs: Vec<usize>,   // val indices
    pub outputs: Vec<usize>,  // val indices
    pub mode: OpMode,
}

pub struct MaterializedGraph<Op: GraphOp> {
    pub vals: Vec<MaterializedVal<Op>>,
    pub ops: Vec<MaterializedOp<Op>>,
    pub inputs: Vec<GlobalValKey<Op>>,
    pub outputs: Vec<GlobalValKey<Op>>,
}

pub fn materialize_merge<Op: GraphOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
) -> MaterializedGraph<Op>;
```

Walk from outputs, collect reachable definitions, deduplicate by
`GlobalValKey`, topological sort.

---

## 7. Compile + Eval (`compile.rs`, `eval.rs`)

```rust
pub struct Instruction<Op: GraphOp> {
    pub op: Op,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

pub struct CompiledProgram<Op: GraphOp> {
    pub instructions: Vec<Instruction<Op>>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
}

pub fn compile<Op: GraphOp>(graph: &MaterializedGraph<Op>) -> CompiledProgram<Op>;

impl<Op: GraphOp> CompiledProgram<Op> {
    pub fn eval(
        &self, ctx: &mut Op::Context, inputs: &[&Op::Operand],
    ) -> Vec<Op::Operand>;
}
```

SSA form. Each slot written once. `eval` allocates a `Vec<Option<Operand>>`
of size `n_slots`, fills input slots, executes instructions in order.

---

## 8. Test Strategy

Test-internal concrete types:

- `ScalarOp` enum: `Add`, `Mul`, `Exp`, `Neg`, `Dup`
- `f64` implements `Operand` (scalar = rank-0 tensor)
- `InputKey = String`, `Context = ()`

Test cases:

1. Single op: `Add(x, y)` build -> compile -> eval
2. Chain: `Exp(Mul(x, a))` build -> compile -> eval, verify `exp(a*x)`
3. External refs: child fragment references parent via `External`
4. CSE: `materialize_merge` deduplicates same `GlobalValKey`
5. Fragment parents: multi-fragment resolve correctness

---

## 9. Project Config

- **Dependencies:** `thiserror` only
- **Edition:** 2021
- **License:** MIT OR Apache-2.0
- **AGENTS.md:** copy from chainrules-rs (same conventions)

---

## 10. Not In Scope

- Thread-safe interner (Phase 1 is single-threaded)
- Compilation cache (defer until needed)
- Concrete tensor Operand implementation (tenferro-rs responsibility)
- AD traits and transforms (chainrules-rs, tidu-rs)
