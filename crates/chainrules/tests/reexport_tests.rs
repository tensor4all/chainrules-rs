//! Verify that `chainrules` re-exports all public items from `chainrules-core`,
//! so downstream crates only need `chainrules` as a dependency.

use chainrules::{
    AdResult, AutodiffError, Differentiable, ForwardRule, NodeId, PullbackEntry,
    PullbackWithTangentsEntry, ReverseRule, SavePolicy,
};

#[test]
fn reexported_node_id_is_usable() {
    let id = NodeId::new(42);
    assert_eq!(id.index(), 42);
}

#[test]
fn reexported_save_policy_variants_are_accessible() {
    assert_ne!(SavePolicy::SaveForPullback, SavePolicy::RecomputeOnPullback);
}

#[test]
fn reexported_error_variants_are_constructible() {
    let err = AutodiffError::NonScalarLoss { num_elements: 5 };
    assert!(err.to_string().contains("scalar"));
}

#[test]
fn reexported_ad_result_alias_works() {
    let ok: AdResult<i32> = Ok(1);
    assert!(ok.is_ok());
}

#[test]
fn reexported_differentiable_trait_is_usable() {
    let x = 3.0_f64;
    assert_eq!(x.zero_tangent(), 0.0);
    assert_eq!(f64::accumulate_tangent(1.0, &2.0), 3.0);
    assert_eq!(x.num_elements(), 1);
    assert_eq!(x.seed_cotangent(), 1.0);
}

#[test]
fn reexported_pullback_entry_type_alias_compiles() {
    let entry: PullbackEntry<f64> = (NodeId::new(0), 2.5);
    assert_eq!(entry.0.index(), 0);
    assert_eq!(entry.1, 2.5);
}

#[test]
fn reexported_pullback_with_tangents_entry_type_alias_compiles() {
    let entry: PullbackWithTangentsEntry<f64> = (NodeId::new(1), 1.0, 0.5);
    assert_eq!(entry.0.index(), 1);
    assert_eq!(entry.1, 1.0);
    assert_eq!(entry.2, 0.5);
}

/// Minimal impl to verify `ReverseRule` and `ForwardRule` are importable and implementable.
struct IdentityRule;

impl ReverseRule<f64> for IdentityRule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<PullbackEntry<f64>>> {
        Ok(vec![(NodeId::new(0), *cotangent)])
    }
    fn inputs(&self) -> Vec<NodeId> {
        vec![NodeId::new(0)]
    }
}

impl ForwardRule<f64> for IdentityRule {
    fn pushforward(&self, tangents: &[Option<&f64>]) -> AdResult<f64> {
        Ok(*tangents[0].unwrap_or(&0.0))
    }
}

#[test]
fn reexported_reverse_rule_trait_is_implementable() {
    let rule = IdentityRule;
    let grads = rule.pullback(&1.0).unwrap();
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].1, 1.0);
}

#[test]
fn reexported_forward_rule_trait_is_implementable() {
    let rule = IdentityRule;
    let dy = rule.pushforward(&[Some(&3.0)]).unwrap();
    assert_eq!(dy, 3.0);
}
