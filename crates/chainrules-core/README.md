# chainrules-core

`chainrules-core` defines the engine-independent AD protocol used by
`chainrules-rs`.

It is intentionally small and does not provide function rules. The crate exists
to define the traits and error types that downstream AD engines and rule
libraries build on.

## What It Provides

- `Differentiable`
- `ReverseRule`
- `ForwardRule`
- `AutodiffError`
- `NodeId`
- `SavePolicy`

## Example

```rust
use chainrules_core::NodeId;

let id = NodeId::new(7);
assert_eq!(id.index(), 7);
```

## Notes

This crate is protocol-only. Shared scalar `frule` and `rrule` helpers live in
[`chainrules`](../chainrules).
