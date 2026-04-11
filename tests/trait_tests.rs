use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::{GraphOp, LocalValId};

/// Mock input key implementing ADKey for testing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MockKey {
    User(String),
    Tangent { of: Box<MockKey>, pass: DiffPassId },
}

impl ADKey for MockKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        MockKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

/// Mock operation implementing both GraphOp and PrimitiveOp.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MockOp {
    Add,
    Scale,
}

impl GraphOp for MockOp {
    type Operand = f64;
    type Context = ();
    type InputKey = MockKey;

    fn n_inputs(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

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

// === ADKey tests ===

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

// === PrimitiveOp tests ===

#[test]
fn primitive_op_linearize_add() {
    let mut builder = FragmentBuilder::<MockOp>::new();
    let mut ctx = ();
    let dx = builder.add_input(MockKey::User("dx".to_string()));
    let dy = builder.add_input(MockKey::User("dy".to_string()));

    let primal_in = vec![
        GlobalValKey::Input(MockKey::User("x".to_string())),
        GlobalValKey::Input(MockKey::User("y".to_string())),
    ];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("sum".to_string()))];
    let tangent_in = vec![Some(dx), Some(dy)];

    let result =
        MockOp::Add.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
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
    let mut ctx = ();
    let dx = builder.add_input(MockKey::User("dx".to_string()));

    let primal_in = vec![
        GlobalValKey::Input(MockKey::User("x".to_string())),
        GlobalValKey::Input(MockKey::User("y".to_string())),
    ];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("sum".to_string()))];
    let tangent_in = vec![Some(dx), None];

    let result =
        MockOp::Add.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 0);
}

#[test]
fn primitive_op_transpose_add() {
    let mut builder = FragmentBuilder::<MockOp>::new();
    let mut ctx = ();
    let ct = builder.add_input(MockKey::User("ct".to_string()));

    let inputs = vec![
        ValRef::External(GlobalValKey::Input(MockKey::User("x".to_string()))),
        ValRef::External(GlobalValKey::Input(MockKey::User("y".to_string()))),
    ];
    let cotangent_out = vec![Some(ct)];

    let result = MockOp::Add.transpose_rule(
        &mut builder,
        &cotangent_out,
        &inputs,
        &OpMode::Primal,
        &mut ctx,
    );

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], Some(ct));
    assert_eq!(result[1], Some(ct));
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 0);
}

#[test]
fn primitive_op_transpose_scale() {
    let mut builder = FragmentBuilder::<MockOp>::new();
    let mut ctx = ();
    let ct = builder.add_input(MockKey::User("ct".to_string()));

    let inputs = vec![
        ValRef::External(GlobalValKey::Input(MockKey::User("a".to_string()))),
        ValRef::External(GlobalValKey::Input(MockKey::User("x".to_string()))),
    ];
    let cotangent_out = vec![Some(ct)];
    let mode = OpMode::Linear {
        active_mask: vec![false, true],
    };

    let result =
        MockOp::Scale.transpose_rule(&mut builder, &cotangent_out, &inputs, &mode, &mut ctx);

    assert_eq!(result.len(), 2);
    assert!(result[0].is_none());
    assert!(result[1].is_some());
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, MockOp::Scale);
}

#[derive(Default)]
struct RecordingContext {
    linearize_calls: Vec<String>,
    transpose_calls: Vec<String>,
}

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
        ctx.linearize_calls.push(
            match self {
                RecordingOp::Add => "Add",
                RecordingOp::Foo => "Foo",
            }
            .to_string(),
        );

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
            RecordingOp::Foo => vec![tangent_in.first().copied().flatten()],
        }
    }

    fn transpose_rule(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        ctx: &mut RecordingContext,
    ) -> Vec<Option<LocalValId>> {
        ctx.transpose_calls.push(
            match self {
                RecordingOp::Add => "Add",
                RecordingOp::Foo => "Foo",
            }
            .to_string(),
        );

        match self {
            RecordingOp::Add => vec![cotangent_out[0], cotangent_out[0]],
            RecordingOp::Foo => vec![cotangent_out[0]],
        }
    }
}

#[derive(Default)]
struct BranchContext {
    is_tall: bool,
    guards: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum BranchOp {
    Add,
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
        _primal_in: &[GlobalValKey<Self>],
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
                ctx.guards.push(ctx.is_tall);
                match tangent_in[0] {
                    Some(dx) => {
                        let (op, inputs, active_mask) = if ctx.is_tall {
                            (BranchOp::Decompose, vec![ValRef::Local(dx)], vec![true])
                        } else {
                            (
                                BranchOp::Add,
                                vec![ValRef::Local(dx), ValRef::Local(dx)],
                                vec![true, true],
                            )
                        };
                        let out = builder.add_op(op, inputs, OpMode::Linear { active_mask });
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
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        ctx: &mut BranchContext,
    ) -> Vec<Option<LocalValId>> {
        match self {
            BranchOp::Add => match cotangent_out[0] {
                Some(ct) => vec![Some(ct), Some(ct)],
                None => vec![None, None],
            },
            BranchOp::Decompose => {
                ctx.guards.push(ctx.is_tall);
                match cotangent_out[0] {
                    Some(ct) => {
                        let (op, inputs, active_mask) = if ctx.is_tall {
                            (BranchOp::Decompose, vec![ValRef::Local(ct)], vec![true])
                        } else {
                            (
                                BranchOp::Add,
                                vec![ValRef::Local(ct), ValRef::Local(ct)],
                                vec![true, true],
                            )
                        };
                        let out = builder.add_op(op, inputs, OpMode::Linear { active_mask });
                        vec![Some(out[0])]
                    }
                    None => vec![None],
                }
            }
        }
    }
}

#[test]
fn adcontext_linearize_records_calls() {
    let mut builder = FragmentBuilder::<RecordingOp>::new();
    let mut ctx = RecordingContext::default();
    let dx = builder.add_input(MockKey::User("dx".to_string()));

    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];
    let tangent_in = vec![Some(dx)];

    let result =
        RecordingOp::Foo.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx);

    assert_eq!(result, vec![Some(dx)]);
    assert_eq!(ctx.linearize_calls, vec!["Foo".to_string()]);
    assert!(ctx.transpose_calls.is_empty());
}

#[test]
fn adcontext_transpose_records_calls() {
    let mut builder = FragmentBuilder::<RecordingOp>::new();
    let mut ctx = RecordingContext::default();
    let ct = builder.add_input(MockKey::User("ct".to_string()));

    let inputs = vec![ValRef::External(GlobalValKey::Input(MockKey::User(
        "x".to_string(),
    )))];
    let cotangent_out = vec![Some(ct)];

    let result = RecordingOp::Foo.transpose_rule(
        &mut builder,
        &cotangent_out,
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true],
        },
        &mut ctx,
    );

    assert_eq!(result, vec![Some(ct)]);
    assert_eq!(ctx.transpose_calls, vec!["Foo".to_string()]);
    assert!(ctx.linearize_calls.is_empty());
}

#[test]
fn adcontext_recording_context_accumulates_across_calls() {
    let mut builder = FragmentBuilder::<RecordingOp>::new();
    let mut ctx = RecordingContext::default();
    let dx = builder.add_input(MockKey::User("dx".to_string()));
    let ct = builder.add_input(MockKey::User("ct".to_string()));

    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];
    let tangent_in = vec![Some(dx)];
    let inputs = vec![ValRef::External(GlobalValKey::Input(MockKey::User(
        "x".to_string(),
    )))];
    let cotangent_out = vec![Some(ct)];

    let linearized =
        RecordingOp::Foo.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx);
    let transposed = RecordingOp::Foo.transpose_rule(
        &mut builder,
        &cotangent_out,
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true],
        },
        &mut ctx,
    );

    assert_eq!(linearized, vec![Some(dx)]);
    assert_eq!(transposed, vec![Some(ct)]);
    assert_eq!(ctx.linearize_calls, vec!["Foo".to_string()]);
    assert_eq!(ctx.transpose_calls, vec!["Foo".to_string()]);
}

#[test]
fn adcontext_branch_tall_emits_decompose() {
    let mut builder = FragmentBuilder::<BranchOp>::new();
    let mut ctx = BranchContext {
        is_tall: true,
        ..Default::default()
    };
    let dx = builder.add_input(MockKey::User("dx".to_string()));

    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];
    let tangent_in = vec![Some(dx)];

    let result =
        BranchOp::Decompose.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    assert_eq!(ctx.guards, vec![true]);
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, BranchOp::Decompose);
}

#[test]
fn adcontext_branch_wide_emits_add() {
    let mut builder = FragmentBuilder::<BranchOp>::new();
    let mut ctx = BranchContext {
        is_tall: false,
        ..Default::default()
    };
    let dx = builder.add_input(MockKey::User("dx".to_string()));

    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];
    let tangent_in = vec![Some(dx)];

    let result =
        BranchOp::Decompose.linearize(&mut builder, &primal_in, &primal_out, &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    assert_eq!(ctx.guards, vec![false]);
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, BranchOp::Add);
}

#[test]
fn adcontext_same_op_different_context_different_graph() {
    let mut tall_builder = FragmentBuilder::<BranchOp>::new();
    let mut wide_builder = FragmentBuilder::<BranchOp>::new();
    let mut tall_ctx = BranchContext {
        is_tall: true,
        ..Default::default()
    };
    let mut wide_ctx = BranchContext {
        is_tall: false,
        ..Default::default()
    };
    let tall_dx = tall_builder.add_input(MockKey::User("dx_tall".to_string()));
    let wide_dx = wide_builder.add_input(MockKey::User("dx_wide".to_string()));

    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];

    BranchOp::Decompose.linearize(
        &mut tall_builder,
        &primal_in,
        &primal_out,
        &[Some(tall_dx)],
        &mut tall_ctx,
    );
    BranchOp::Decompose.linearize(
        &mut wide_builder,
        &primal_in,
        &primal_out,
        &[Some(wide_dx)],
        &mut wide_ctx,
    );

    let tall_frag = tall_builder.build();
    let wide_frag = wide_builder.build();
    assert_eq!(tall_ctx.guards, vec![true]);
    assert_eq!(wide_ctx.guards, vec![false]);
    assert_eq!(tall_frag.ops().len(), 1);
    assert_eq!(wide_frag.ops().len(), 1);
    assert_eq!(tall_frag.ops()[0].op, BranchOp::Decompose);
    assert_eq!(wide_frag.ops()[0].op, BranchOp::Add);
    assert_ne!(tall_frag.ops()[0].op, wide_frag.ops()[0].op);
}

#[test]
fn adcontext_transpose_branches_on_context() {
    let mut tall_builder = FragmentBuilder::<BranchOp>::new();
    let mut wide_builder = FragmentBuilder::<BranchOp>::new();
    let mut tall_ctx = BranchContext {
        is_tall: true,
        ..Default::default()
    };
    let mut wide_ctx = BranchContext {
        is_tall: false,
        ..Default::default()
    };
    let tall_ct = tall_builder.add_input(MockKey::User("ct_tall".to_string()));
    let wide_ct = wide_builder.add_input(MockKey::User("ct_wide".to_string()));

    let inputs = vec![ValRef::External(GlobalValKey::Input(MockKey::User(
        "x".to_string(),
    )))];
    let mode = OpMode::Linear {
        active_mask: vec![true],
    };

    let tall_result = BranchOp::Decompose.transpose_rule(
        &mut tall_builder,
        &[Some(tall_ct)],
        &inputs,
        &mode,
        &mut tall_ctx,
    );
    let wide_result = BranchOp::Decompose.transpose_rule(
        &mut wide_builder,
        &[Some(wide_ct)],
        &inputs,
        &mode,
        &mut wide_ctx,
    );

    assert_eq!(tall_result.len(), 1);
    assert!(tall_result[0].is_some());
    assert_eq!(wide_result.len(), 1);
    assert!(wide_result[0].is_some());
    assert_eq!(tall_ctx.guards, vec![true]);
    assert_eq!(wide_ctx.guards, vec![false]);
    let tall_frag = tall_builder.build();
    let wide_frag = wide_builder.build();
    assert_eq!(tall_frag.ops()[0].op, BranchOp::Decompose);
    assert_eq!(wide_frag.ops()[0].op, BranchOp::Add);
}

#[test]
fn adcontext_multiple_branch_points() {
    let primal_in = vec![GlobalValKey::Input(MockKey::User("x".to_string()))];
    let primal_out = vec![GlobalValKey::Input(MockKey::User("y".to_string()))];

    let mut tall_builder = FragmentBuilder::<BranchOp>::new();
    let mut tall_ctx = BranchContext {
        is_tall: true,
        ..Default::default()
    };
    let tall_dx = tall_builder.add_input(MockKey::User("dx_tall".to_string()));
    let tall_first = BranchOp::Decompose.linearize(
        &mut tall_builder,
        &primal_in,
        &primal_out,
        &[Some(tall_dx)],
        &mut tall_ctx,
    );
    let tall_second = BranchOp::Decompose.linearize(
        &mut tall_builder,
        &primal_in,
        &primal_out,
        &[tall_first[0]],
        &mut tall_ctx,
    );

    assert_eq!(tall_ctx.guards, vec![true, true]);
    assert!(tall_second[0].is_some());
    let tall_frag = tall_builder.build();
    assert_eq!(tall_frag.ops().len(), 2);
    assert_eq!(tall_frag.ops()[0].op, BranchOp::Decompose);
    assert_eq!(tall_frag.ops()[1].op, BranchOp::Decompose);

    let mut wide_builder = FragmentBuilder::<BranchOp>::new();
    let mut wide_ctx = BranchContext {
        is_tall: false,
        ..Default::default()
    };
    let wide_dx = wide_builder.add_input(MockKey::User("dx_wide".to_string()));
    let wide_first = BranchOp::Decompose.linearize(
        &mut wide_builder,
        &primal_in,
        &primal_out,
        &[Some(wide_dx)],
        &mut wide_ctx,
    );
    let wide_second = BranchOp::Decompose.linearize(
        &mut wide_builder,
        &primal_in,
        &primal_out,
        &[wide_first[0]],
        &mut wide_ctx,
    );

    assert_eq!(wide_ctx.guards, vec![false, false]);
    assert!(wide_second[0].is_some());
    let wide_frag = wide_builder.build();
    assert_eq!(wide_frag.ops().len(), 2);
    assert_eq!(wide_frag.ops()[0].op, BranchOp::Add);
    assert_eq!(wide_frag.ops()[1].op, BranchOp::Add);
}
