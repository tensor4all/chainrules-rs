# ADContext for PrimitiveOp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `ADContext` associated type to `PrimitiveOp` so AD rules can receive runtime context (e.g., concrete tensor shapes and guard recording) during differentiation.

**Architecture:** `PrimitiveOp` gains `type ADContext: Default;`. Both `linearize` and `transpose_rule` gain a `ctx: &mut Self::ADContext` parameter. Existing implementors use `type ADContext = ();` for zero-cost backward compat. New tests verify: (1) backward compat, (2) custom context read/write, (3) context-dependent graph structure (simulating shape-branched linalg AD).

**Tech Stack:** Rust, `computegraph` crate (existing dependency)

**Downstream consumers that will need mechanical updates (NOT in this plan):**
- `tidu-rs`: 7 `impl PrimitiveOp` blocks in tests + `differentiate`/`transpose` functions
- `tenferro-rs/tenferro-ops`: `impl PrimitiveOp for StdTensorOp`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/primitive_op.rs` | Modify | Add `ADContext` type + `ctx` param |
| `src/lib.rs` | No change | Already re-exports `PrimitiveOp` |
| `src/ad_key.rs` | No change | Unrelated |
| `tests/trait_tests.rs` | Modify | Update `MockOp` + add new tests |

---

### Task 1: Add `ADContext` to `PrimitiveOp` trait and update `MockOp`

**Files:**
- Modify: `src/primitive_op.rs` (trait definition + doc example)
- Modify: `tests/trait_tests.rs` (MockOp impl + existing tests)

- [ ] **Step 1: Update the trait definition**

In `src/primitive_op.rs`, replace the trait body (lines 59–97) with:

```rust
pub trait PrimitiveOp: GraphOp
where
    Self::InputKey: ADKey,
{
    /// Context passed through `tidu::differentiate` and `tidu::transpose`
    /// into every `linearize` / `transpose_rule` call.
    ///
    /// Downstream implementors (e.g. tenferro-ops) use this to supply
    /// runtime information (concrete tensor shapes, guard recording)
    /// that is not available at graph-construction time.
    ///
    /// Implementors that do not need context should use `type ADContext = ();`.
    type ADContext: Default;

    /// Returns the addition operation used for cotangent accumulation
    /// in `tidu::transpose`. When multiple cotangents flow to the same
    /// `GlobalValKey`, transpose emits `Op::add()` nodes to sum them.
    fn add() -> Self
    where
        Self: Sized;

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
        ctx: &mut Self::ADContext,
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
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;
}
```

- [ ] **Step 2: Update the doc example in `primitive_op.rs`**

Replace the doc example (lines 18–57) with:

```rust
/// # Examples
///
/// ```
/// use chainrules::{ADKey, DiffPassId, PrimitiveOp};
/// use computegraph::fragment::FragmentBuilder;
/// use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
/// use computegraph::GraphOp;
///
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// enum Key { Base(String), Tan(Box<Key>, DiffPassId) }
///
/// impl ADKey for Key {
///     fn tangent_of(&self, p: DiffPassId) -> Self { Key::Tan(Box::new(self.clone()), p) }
/// }
///
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// struct AddOp;
///
/// impl GraphOp for AddOp {
///     type Operand = f64;
///     type Context = ();
///     type InputKey = Key;
///     fn n_inputs(&self) -> usize { 2 }
///     fn n_outputs(&self) -> usize { 1 }
/// }
///
/// impl PrimitiveOp for AddOp {
///     type ADContext = ();
///     fn add() -> Self { AddOp }
///     fn linearize(
///         &self, _b: &mut FragmentBuilder<Self>,
///         _pi: &[GlobalValKey<Self>], _po: &[GlobalValKey<Self>],
///         t: &[Option<LocalValId>], _ctx: &mut (),
///     ) -> Vec<Option<LocalValId>> {
///         vec![t[0].or(t[1])]
///     }
///     fn transpose_rule(
///         &self, _b: &mut FragmentBuilder<Self>,
///         ct: &[Option<LocalValId>], _i: &[ValRef<Self>], _m: &OpMode,
///         _ctx: &mut (),
///     ) -> Vec<Option<LocalValId>> {
///         vec![ct[0], ct[0]]
///     }
/// }
/// ```
```

- [ ] **Step 3: Update `MockOp` in `tests/trait_tests.rs`**

Add `type ADContext = ();` and `_ctx: &mut ()` to both methods. The full `impl PrimitiveOp for MockOp` block becomes:

```rust
impl PrimitiveOp for MockOp {
    type ADContext = ();

    fn add() -> Self {
        MockOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            MockOp::Add => match (&tangent_in[0], &tangent_in[1]) {
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
            },
            MockOp::Scale => match &tangent_in[1] {
                Some(dx) => {
                    let out = builder.add_op(
                        MockOp::Scale,
                        vec![ValRef::External(primal_in[0].clone()), ValRef::Local(*dx)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        _mode: &OpMode,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            MockOp::Add => match &cotangent_out[0] {
                Some(ct) => vec![Some(*ct), Some(*ct)],
                None => vec![None, None],
            },
            MockOp::Scale => match &cotangent_out[0] {
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
            },
        }
    }
}
```

- [ ] **Step 4: Update existing test call sites in `tests/trait_tests.rs`**

Every call to `.linearize(...)` and `.transpose_rule(...)` needs `&mut ()` appended. There are 4 tests to update:

`primitive_op_linearize_add` (line 171):
```rust
    let result = MockOp::Add.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ());
```

`primitive_op_linearize_skip_inactive` (line 198):
```rust
    let result = MockOp::Add.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ());
```

`primitive_op_transpose_add` (line 217):
```rust
    let result = MockOp::Add.transpose_rule(&mut builder, &cotangent_out, &inputs, &OpMode::Primal, &mut ());
```

`primitive_op_transpose_scale` (line 240):
```rust
    let result = MockOp::Scale.transpose_rule(&mut builder, &cotangent_out, &inputs, &mode, &mut ());
```

- [ ] **Step 5: Run all tests**

Run: `cd /home/shinaoka/tensor4all/chainrules-rs && cargo test --release`
Expected: All 7 existing tests pass + 2 doc-tests pass

- [ ] **Step 6: Run formatting and clippy**

Run: `cd /home/shinaoka/tensor4all/chainrules-rs && cargo fmt --all && cargo clippy --all-targets --release`
Expected: No warnings, no errors

---

### Task 2: Test custom ADContext read/write

**Files:**
- Modify: `tests/trait_tests.rs` (add new op + tests)

- [ ] **Step 1: Define `RecordingContext` and `RecordingOp`**

Append to `tests/trait_tests.rs`:

```rust
// === ADContext recording tests ===

/// A context that records which methods were called and how many times.
#[derive(Default)]
struct RecordingContext {
    linearize_calls: Vec<String>,
    transpose_calls: Vec<String>,
}

/// A trivial op that logs its name into the context.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RecordingOp {
    Add,
    Foo,
}

impl GraphOp for RecordingOp {
    type Operand = f64;
    type Context = ();
    type InputKey = MockKey;

    fn n_inputs(&self) -> usize {
        match self {
            RecordingOp::Add => 2,
            RecordingOp::Foo => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl PrimitiveOp for RecordingOp {
    type ADContext = RecordingContext;

    fn add() -> Self {
        RecordingOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut RecordingContext,
    ) -> Vec<Option<LocalValId>> {
        ctx.linearize_calls.push(format!("{:?}", self));
        match self {
            RecordingOp::Add => match (&tangent_in[0], &tangent_in[1]) {
                (Some(dx), Some(dy)) => {
                    let out = builder.add_op(
                        RecordingOp::Add,
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
            },
            RecordingOp::Foo => {
                // Identity JVP: tangent passes through
                vec![tangent_in[0]]
            }
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        ctx: &mut RecordingContext,
    ) -> Vec<Option<LocalValId>> {
        ctx.transpose_calls.push(format!("{:?}", self));
        match self {
            RecordingOp::Add => match &cotangent_out[0] {
                Some(ct) => vec![Some(*ct), Some(*ct)],
                None => vec![None, None],
            },
            RecordingOp::Foo => vec![cotangent_out[0]],
        }
    }
}
```

- [ ] **Step 2: Write test for linearize recording**

```rust
#[test]
fn adcontext_linearize_records_calls() {
    let mut builder = FragmentBuilder::<RecordingOp>::new();
    let dx = builder.add_input(MockKey::User("dx".to_string()));
    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];
    let tangent_in = vec![Some(dx)];

    let mut ctx = RecordingContext::default();
    let result = RecordingOp::Foo.linearize(
        &mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx,
    );

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    assert_eq!(ctx.linearize_calls, vec!["Foo"]);
    assert!(ctx.transpose_calls.is_empty());
}
```

- [ ] **Step 3: Write test for transpose recording**

```rust
#[test]
fn adcontext_transpose_records_calls() {
    let mut builder = FragmentBuilder::<RecordingOp>::new();
    let ct = builder.add_input(MockKey::User("ct".to_string()));
    let inputs = vec![ValRef::External(GlobalValKey::Input(MockKey::User("x".to_string())))];
    let cotangent_out = vec![Some(ct)];

    let mut ctx = RecordingContext::default();
    let result = RecordingOp::Foo.transpose_rule(
        &mut builder, &cotangent_out, &inputs, &OpMode::Primal, &mut ctx,
    );

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    assert!(ctx.linearize_calls.is_empty());
    assert_eq!(ctx.transpose_calls, vec!["Foo"]);
}
```

- [ ] **Step 4: Run tests**

Run: `cd /home/shinaoka/tensor4all/chainrules-rs && cargo test --release`
Expected: 9 tests pass (7 old + 2 new) + 2 doc-tests

- [ ] **Step 5: Commit**

```bash
cd /home/shinaoka/tensor4all/chainrules-rs
git add -A
git commit -m "feat: add ADContext associated type to PrimitiveOp

Add type ADContext: Default to PrimitiveOp trait. linearize and
transpose_rule now receive &mut Self::ADContext. Existing impls use
type ADContext = () for backward compat. New RecordingOp tests verify
context read/write.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Test context-dependent graph structure (shape branching simulation)

This is the most important test: it simulates the SVD `if m > n` pattern where the context controls which ops are emitted.

**Files:**
- Modify: `tests/trait_tests.rs` (add `BranchOp` + tests)

- [ ] **Step 1: Define `BranchContext` and `BranchOp`**

Append to `tests/trait_tests.rs`:

```rust
// === Context-dependent branching tests ===

/// Simulates ShapeGuardContext: holds a flag that controls which AD
/// graph is emitted, and records the guards that were checked.
#[derive(Default)]
struct BranchContext {
    /// If true, emit the "tall" correction path. If false, emit "wide".
    is_tall: bool,
    /// Records of branch decisions made during linearize/transpose.
    guards: Vec<bool>,
}

/// An op whose linearize emits different graph structure depending on
/// the ADContext, simulating SVD's `if m > n` branching.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum BranchOp {
    Add,
    /// A mock decomposition whose JVP depends on context.
    Decompose,
}

impl GraphOp for BranchOp {
    type Operand = f64;
    type Context = ();
    type InputKey = MockKey;

    fn n_inputs(&self) -> usize {
        match self {
            BranchOp::Add => 2,
            BranchOp::Decompose => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl PrimitiveOp for BranchOp {
    type ADContext = BranchContext;

    fn add() -> Self {
        BranchOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut BranchContext,
    ) -> Vec<Option<LocalValId>> {
        match self {
            BranchOp::Add => match (&tangent_in[0], &tangent_in[1]) {
                (Some(dx), Some(dy)) => {
                    let out = builder.add_op(
                        BranchOp::Add,
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
            },
            BranchOp::Decompose => {
                let Some(dx) = tangent_in[0] else {
                    return vec![None];
                };

                // Record the guard decision
                ctx.guards.push(ctx.is_tall);

                if ctx.is_tall {
                    // "Tall" path: emit Decompose (simulates tall correction)
                    let out = builder.add_op(
                        BranchOp::Decompose,
                        vec![ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(out[0])]
                } else {
                    // "Wide" path: emit Add with primal (simulates wide correction)
                    let out = builder.add_op(
                        BranchOp::Add,
                        vec![
                            ValRef::External(primal_in[0].clone()),
                            ValRef::Local(dx),
                        ],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    vec![Some(out[0])]
                }
            }
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        _mode: &OpMode,
        ctx: &mut BranchContext,
    ) -> Vec<Option<LocalValId>> {
        match self {
            BranchOp::Add => match &cotangent_out[0] {
                Some(ct) => vec![Some(*ct), Some(*ct)],
                None => vec![None, None],
            },
            BranchOp::Decompose => {
                ctx.guards.push(ctx.is_tall);

                match &cotangent_out[0] {
                    Some(ct) => {
                        if ctx.is_tall {
                            let out = builder.add_op(
                                BranchOp::Decompose,
                                vec![ValRef::Local(*ct)],
                                OpMode::Linear {
                                    active_mask: vec![true],
                                },
                            );
                            vec![Some(out[0])]
                        } else {
                            let out = builder.add_op(
                                BranchOp::Add,
                                vec![inputs[0].clone(), ValRef::Local(*ct)],
                                OpMode::Linear {
                                    active_mask: vec![false, true],
                                },
                            );
                            vec![Some(out[0])]
                        }
                    }
                    None => vec![None],
                }
            }
        }
    }
}
```

- [ ] **Step 2: Write test — tall path emits Decompose**

```rust
#[test]
fn adcontext_branch_tall_emits_decompose() {
    let mut builder = FragmentBuilder::<BranchOp>::new();
    let dx = builder.add_input(MockKey::User("dx".to_string()));
    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];
    let tangent_in = vec![Some(dx)];

    let mut ctx = BranchContext {
        is_tall: true,
        guards: vec![],
    };
    let result = BranchOp::Decompose.linearize(
        &mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx,
    );

    assert!(result[0].is_some());
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, BranchOp::Decompose);
    assert_eq!(ctx.guards, vec![true]);
}
```

- [ ] **Step 3: Write test — wide path emits Add**

```rust
#[test]
fn adcontext_branch_wide_emits_add() {
    let mut builder = FragmentBuilder::<BranchOp>::new();
    let dx = builder.add_input(MockKey::User("dx".to_string()));
    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];
    let tangent_in = vec![Some(dx)];

    let mut ctx = BranchContext {
        is_tall: false,
        guards: vec![],
    };
    let result = BranchOp::Decompose.linearize(
        &mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx,
    );

    assert!(result[0].is_some());
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, BranchOp::Add);
    assert_eq!(ctx.guards, vec![false]);
}
```

- [ ] **Step 4: Write test — same op, different context produces different graph**

```rust
#[test]
fn adcontext_same_op_different_context_different_graph() {
    let op = BranchOp::Decompose;
    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];

    // Tall path
    let mut builder_tall = FragmentBuilder::<BranchOp>::new();
    let dx_tall = builder_tall.add_input(MockKey::User("dx".to_string()));
    let mut ctx_tall = BranchContext { is_tall: true, guards: vec![] };
    op.linearize(&mut builder_tall, &primal_in, &primal_out, &[Some(dx_tall)], &mut ctx_tall);
    let frag_tall = builder_tall.build();

    // Wide path
    let mut builder_wide = FragmentBuilder::<BranchOp>::new();
    let dx_wide = builder_wide.add_input(MockKey::User("dx".to_string()));
    let mut ctx_wide = BranchContext { is_tall: false, guards: vec![] };
    op.linearize(&mut builder_wide, &primal_in, &primal_out, &[Some(dx_wide)], &mut ctx_wide);
    let frag_wide = builder_wide.build();

    // Different graph structure
    assert_eq!(frag_tall.ops()[0].op, BranchOp::Decompose);
    assert_eq!(frag_wide.ops()[0].op, BranchOp::Add);

    // Both recorded guards
    assert_eq!(ctx_tall.guards, vec![true]);
    assert_eq!(ctx_wide.guards, vec![false]);
}
```

- [ ] **Step 5: Write test — transpose also branches on context**

```rust
#[test]
fn adcontext_transpose_branches_on_context() {
    let op = BranchOp::Decompose;
    let inputs = vec![ValRef::External(GlobalValKey::Input(MockKey::User("x".to_string())))];

    // Tall
    let mut builder_tall = FragmentBuilder::<BranchOp>::new();
    let ct_tall = builder_tall.add_input(MockKey::User("ct".to_string()));
    let mut ctx_tall = BranchContext { is_tall: true, guards: vec![] };
    op.transpose_rule(&mut builder_tall, &[Some(ct_tall)], &inputs, &OpMode::Primal, &mut ctx_tall);
    let frag_tall = builder_tall.build();

    // Wide
    let mut builder_wide = FragmentBuilder::<BranchOp>::new();
    let ct_wide = builder_wide.add_input(MockKey::User("ct".to_string()));
    let mut ctx_wide = BranchContext { is_tall: false, guards: vec![] };
    op.transpose_rule(&mut builder_wide, &[Some(ct_wide)], &inputs, &OpMode::Primal, &mut ctx_wide);
    let frag_wide = builder_wide.build();

    assert_eq!(frag_tall.ops()[0].op, BranchOp::Decompose);
    assert_eq!(frag_wide.ops()[0].op, BranchOp::Add);
    assert_eq!(ctx_tall.guards, vec![true]);
    assert_eq!(ctx_wide.guards, vec![false]);
}
```

- [ ] **Step 6: Run all tests**

Run: `cd /home/shinaoka/tensor4all/chainrules-rs && cargo test --release`
Expected: 14 tests pass (7 old + 2 recording + 4 branching + 1 comparison) + 2 doc-tests

- [ ] **Step 7: Run formatting and clippy**

Run: `cd /home/shinaoka/tensor4all/chainrules-rs && cargo fmt --all && cargo clippy --all-targets --release`
Expected: Clean

- [ ] **Step 8: Commit**

```bash
cd /home/shinaoka/tensor4all/chainrules-rs
git add -A
git commit -m "test: ADContext branching tests simulating shape-dependent linalg AD

BranchOp emits different graph structure depending on BranchContext.is_tall,
simulating the SVD linearize_svd 'if m > n' pattern. Tests verify:
- tall context → Decompose op emitted
- wide context → Add op emitted
- same op with different context → different fragments
- transpose also respects context

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
