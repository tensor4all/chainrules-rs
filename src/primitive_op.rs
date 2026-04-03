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
///
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
///     fn eval(&self, _: &mut (), i: &[&f64]) -> Vec<f64> { vec![i[0] + i[1]] }
/// }
///
/// impl PrimitiveOp for AddOp {
///     fn linearize(
///         &self, _b: &mut FragmentBuilder<Self>,
///         _pi: &[GlobalValKey<Self>], _po: &[GlobalValKey<Self>],
///         t: &[Option<LocalValId>],
///     ) -> Vec<Option<LocalValId>> {
///         vec![t[0].or(t[1])]
///     }
///     fn transpose_rule(
///         &self, _b: &mut FragmentBuilder<Self>,
///         ct: &[Option<LocalValId>], _i: &[ValRef<Self>], _m: &OpMode,
///     ) -> Vec<Option<LocalValId>> {
///         vec![ct[0], ct[0]]
///     }
/// }
/// ```
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
