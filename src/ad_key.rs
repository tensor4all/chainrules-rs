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
///         MyKey::Tangent {
///             of: Box::new(self.clone()),
///             pass,
///         }
///     }
/// }
/// ```
pub trait ADKey: Clone + std::fmt::Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `differentiate` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
