# chainrules-rs

`chainrules-rs` is an engine-independent automatic-differentiation traits and
rules library.

It originated in the tensor4all stack, but it is designed to be reusable from
any Rust project that wants differentiation rules without committing to a
specific AD engine.

It contains:

- `chainrules-core`: engine-independent AD protocol
- `chainrules`: shared scalar rule basis

It intentionally does **not** ship a tape, traced value type, or any other AD
engine runtime. Those live in separate engine crates such as
[`tidu-rs`](https://github.com/tensor4all/tidu-rs).

## Getting Started

Add `chainrules` (or `chainrules-core` alone if you only need the traits) as a
git dependency:

```toml
[dependencies]
chainrules = { git = "https://github.com/tensor4all/chainrules-rs" }

# For complex scalar support, also add:
num-complex = "0.4"
```

### Using scalar rules

```rust
use chainrules::{exp_frule, sin_rrule, powf_frule};

// Forward rule: returns (primal, tangent)
let (y, dy) = exp_frule(1.0_f64, 1.0);
assert!((y - 1.0_f64.exp()).abs() < 1e-12);

// Reverse rule: returns input cotangent
let dx = sin_rrule(0.5_f64, 1.0);
assert!((dx - 0.5_f64.cos()).abs() < 1e-12);
```

### Implementing custom AD types

```rust
use chainrules::{Differentiable, ReverseRule, AdResult, NodeId};

#[derive(Clone)]
struct MyVec(Vec<f64>);

impl Differentiable for MyVec {
    type Tangent = MyVec;
    fn zero_tangent(&self) -> MyVec { MyVec(vec![0.0; self.0.len()]) }
    fn accumulate_tangent(a: MyVec, b: &MyVec) -> MyVec {
        MyVec(a.0.iter().zip(&b.0).map(|(x, y)| x + y).collect())
    }
    fn num_elements(&self) -> usize { self.0.len() }
    fn seed_cotangent(&self) -> MyVec { MyVec(vec![1.0; self.0.len()]) }
}
```

## Design Goals

- Keep differentiation rules reusable across projects and AD engines
- Keep layering strict: rules do not depend on a specific engine
- Stay DRY and KISS by defining scalar calculus once at the lowest sensible
  layer

## Repository Layout

- [`crates/chainrules-core`](crates/chainrules-core): protocol-only crate for
  `Differentiable`, `ReverseRule`, `ForwardRule`, and `AutodiffError`
- [`crates/chainrules`](crates/chainrules): shared scalar rules such as
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

## Crate Roles

`chainrules-core` does not provide function rules.
`chainrules` provides stateless scalar `foo`, `foo_frule`, and `foo_rrule`
helpers.

`chainrules` is a landing zone for scalar rules ported or adapted from Julia's
`ChainRules.jl` where they fit this repository boundary, but `chainrules-rs` is
not a full port of `ChainRules.jl`.

See the crate READMEs for the supported scalar function inventory and examples.

## Testing

Scalar rules are checked in complementary ways:

- formula and behavior tests in `crates/chainrules/tests/scalarops_tests.rs`
- compatibility and edge-case tests such as
  `crates/chainrules/tests/julia_compat_trig_tests.rs` and
  `crates/chainrules/tests/complex_helper_tests.rs`
- oracle replay tests in `crates/chainrules/tests/oracle_scalar_rules.rs`
  against vendored published cases from `third_party/tensor-ad-oracles`,
  including direct float64 replay and selected direct Complex64 replay for
  `tan`, `exp2`, and `log2`; complex
  forward-mode checks use the standard JVP convention on `C ~= R^2`, while
  complex reverse-mode checks remain conjugate-Wirtinger for real-valued losses

```bash
cargo test --workspace --release
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

## Solve-Bug Entrypoints

Use `bash ai/run-codex-solve-bug.sh` or `bash ai/run-claude-solve-bug.sh` when
you want a headless agent to pick one actionable bug or bug-like issue, fix it,
and drive the repository-local PR workflow.
