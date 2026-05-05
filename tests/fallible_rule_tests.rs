use chainrules::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, PrimitiveOp};
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::{GraphOp, LocalValId, OpEmitter};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    Base(&'static str),
    Tangent(Box<Key>, DiffPassId),
}

impl ADKey for Key {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        Key::Tangent(Box::new(self.clone()), pass)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Op {
    Add,
    Missing,
}

impl GraphOp for Op {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn n_inputs(&self) -> usize {
        match self {
            Op::Add => 2,
            Op::Missing => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl PrimitiveOp for Op {
    type ADContext = ();

    fn add() -> Self {
        Op::Add
    }

    fn linearize(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            Op::Add => vec![tangent_in[0].or(tangent_in[1])],
            Op::Missing => panic!("missing AD rule should be reported through try_linearize"),
        }
    }

    fn try_linearize(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        match self {
            Op::Add => Ok(vec![tangent_in[0].or(tangent_in[1])]),
            Op::Missing => Err(ADRuleError::unsupported(
                "Op::Missing",
                ADRuleKind::Linearize,
            )),
        }
    }

    fn transpose_rule(
        &self,
        _emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            Op::Add => vec![cotangent_out[0], cotangent_out[0]],
            Op::Missing => panic!("missing AD rule should be reported through try_transpose_rule"),
        }
    }

    fn try_transpose_rule(
        &self,
        _emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        match self {
            Op::Add => Ok(vec![cotangent_out[0], cotangent_out[0]]),
            Op::Missing => Err(ADRuleError::unsupported(
                "Op::Missing",
                ADRuleKind::Transpose,
            )),
        }
    }
}

#[test]
fn primitive_op_can_report_missing_linearize_rule() {
    let mut builder = FragmentBuilder::<Op>::new();
    let mut ctx = ();
    let dx = builder.add_input(Key::Base("dx"));
    let err = Op::Missing
        .try_linearize(
            &mut builder,
            &[GlobalValKey::Input(Key::Base("x"))],
            &[GlobalValKey::Input(Key::Base("y"))],
            &[Some(dx)],
            &mut ctx,
        )
        .unwrap_err();

    assert_eq!(err.rule(), ADRuleKind::Linearize);
    assert!(err.to_string().contains("Op::Missing"));
}

#[test]
fn primitive_op_can_report_missing_transpose_rule() {
    let mut builder = FragmentBuilder::<Op>::new();
    let mut ctx = ();
    let ct = builder.add_input(Key::Base("ct"));
    let err = Op::Missing
        .try_transpose_rule(
            &mut builder,
            &[Some(ct)],
            &[ValRef::External(GlobalValKey::Input(Key::Base("x")))],
            &OpMode::Linear {
                active_mask: vec![true],
            },
            &mut ctx,
        )
        .unwrap_err();

    assert_eq!(err.rule(), ADRuleKind::Transpose);
    assert!(err.to_string().contains("Op::Missing"));
}
