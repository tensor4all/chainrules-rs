# chainrules-rs

AD trait definitions for the tensor4all v2 stack.

This crate defines:

- **`PrimitiveOp`** — extends `computegraph::GraphOp` with `linearize`
  (JVP rule) and `transpose_rule` (reverse-mode rule)
- **`ADKey`** — trait on `GraphOp::InputKey` for generating tangent input
  keys during `differentiate`

It contains no concrete primitives and no graph infrastructure.

## Part of the tensor4all v2 stack

```text
computegraph-rs    graph engine (GraphOp, Fragment, compile, eval)
    |
chainrules-rs  <-- this crate (PrimitiveOp, ADKey)
    |
tidu-rs            AD transforms (differentiate, transpose)
    |
tenferro-rs        concrete tensor primitives + backends
```

## Usage

```toml
[dependencies]
chainrules = { git = "https://github.com/tensor4all/chainrules-rs", branch = "feat/v2" }
```

## Testing

```bash
cargo test --release
```
