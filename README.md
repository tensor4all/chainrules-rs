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

For a general function f: C → C, the VJP cotangent relates to Wirtinger
derivatives as:

```text
ct_z = ct_y · conj(∂f/∂z) + conj(ct_y) · (∂f/∂z̄)
```

Special cases:

| Case | Result |
|------|--------|
| Real loss (L: C→R), ct_y=1 | `ct_z = 2·(∂L/∂z̄)` |
| Holomorphic f, ∂f/∂z̄=0 | `ct_z = ct_y · conj(f'(z))` |
| conj(z), ∂f/∂z=0 | `ct_z = conj(ct_y)` |

For real-valued losses, this differs from PyTorch (which returns `∂L/∂z̄`
directly) by a factor of 2. The steepest-descent direction is the same.

## Usage

```toml
[dependencies]
chainrules = { git = "https://github.com/tensor4all/chainrules-rs", branch = "feat/v2" }
```

## Testing

```bash
cargo test --release
```
