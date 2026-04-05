//! AD trait definitions for the tensor4all v2 stack.
//!
//! This crate defines [`PrimitiveOp`] (extends [`computegraph::GraphOp`] with
//! linearization and transpose rules) and [`ADKey`] (tangent input key
//! generation). It contains no concrete primitives and no graph infrastructure.

pub mod ad_key;
pub mod primitive_op;

pub use ad_key::{ADKey, DiffPassId};
pub use primitive_op::PrimitiveOp;
