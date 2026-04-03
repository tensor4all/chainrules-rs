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
