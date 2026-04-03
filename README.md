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

## Complex number convention (JAX-compatible)

This crate follows the same complex AD convention as JAX.

- **`linearize`** computes the full R-linear JVP:
  `df = (∂f/∂z)·dz + (∂f/∂z̄)·conj(dz)`
- **`transpose_rule`** computes the adjoint w.r.t. the real inner product
  `⟨a, b⟩ = Re(conj(a)·b)`.

For the R-linear map `dz → a·dz`, the adjoint is `ct → conj(a)·ct`.
This is why `transpose_rule` implementations emit `Conj` nodes when
transposing through complex multiplication.

For a real-valued loss L, the VJP cotangent satisfies:

```text
ct_z = 2·(∂L/∂z̄)      (= 2·conj(∂L/∂z))
```

This differs from PyTorch, which returns `∂L/∂z̄` directly (factor of 2
difference). The steepest-descent direction is the same in both conventions.

## Usage

```toml
[dependencies]
chainrules = { git = "https://github.com/tensor4all/chainrules-rs", branch = "feat/v2" }
```

## Testing

```bash
cargo test --release
```
