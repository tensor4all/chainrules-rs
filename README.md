# chainrules-rs

`chainrules-rs` is the engine-independent part of the tensor4all automatic
differentiation stack.

It contains:

- `chainrules-core`: core AD traits and error types
- `chainrules`: reusable scalar `frule`/`rrule` helpers and related utilities

It intentionally does **not** ship a tape, traced value type, or any other AD
engine runtime. Those live in separate engine crates such as
[`tidu-rs`](https://github.com/tensor4all/tidu-rs).

## Design Goals

- Keep differentiation rules reusable across projects and AD engines
- Keep layering strict: rules do not depend on a specific engine
- Stay DRY and KISS by defining scalar calculus once at the lowest sensible
  layer

## Repository Layout

- [`crates/chainrules-core`](crates/chainrules-core): `Differentiable`,
  `ReverseRule`, `ForwardRule`, `AutodiffError`, and related core types
- [`crates/chainrules`](crates/chainrules): scalar rule implementations such as
  `exp`, `log1p`, `sin`, `atanh`, `powf`, and `atan2`
- [`third_party/tensor-ad-oracles`](third_party/tensor-ad-oracles): vendored
  oracle data used to validate scalar rules against published references

## Oracle Data

`third_party/tensor-ad-oracles` is vendored from
[`tensor4all/tensor-ad-oracles`](https://github.com/tensor4all/tensor-ad-oracles).
The copy is kept in-tree on purpose so this repository stays self-contained for
CI, local development, and downstream Git dependencies.

## Relationship To `tidu`

`chainrules-rs` provides traits and rule implementations. `tidu-rs` provides an
engine that executes those rules over a tape. The boundary is deliberate:

- `chainrules-rs` stays generic and reusable
- `tidu-rs` can evolve independently as an engine
- downstream tensor libraries can swap engines without rewriting scalar rules

## Testing

```bash
cargo test --workspace --release
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

## Solve-Bug Entrypoints

Use `bash ai/run-codex-solve-bug.sh` or `bash ai/run-claude-solve-bug.sh` when
you want a headless agent to pick one actionable bug or bug-like issue, fix it,
and drive the repository-local PR workflow.
