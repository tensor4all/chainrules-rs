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
        primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
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
    let tangent_in = vec![Some(dx), None];

    let result = MockOp::Add.linearize(&mut builder, &primal_in, &primal_out, &tangent_in);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
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

    let result = MockOp::Add.transpose_rule(&mut builder, &cotangent_out, &inputs, &OpMode::Primal);

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

    assert_eq!(result.len(), 2);
    assert!(result[0].is_none());
    assert!(result[1].is_some());
    let frag = builder.build();
    assert_eq!(frag.ops().len(), 1);
    assert_eq!(frag.ops()[0].op, MockOp::Scale);
}
