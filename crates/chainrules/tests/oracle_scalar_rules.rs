use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use chainrules::{
    acos_frule, acos_rrule, acosh_frule, acosh_rrule, asin_frule, asin_rrule, asinh_frule,
    asinh_rrule, atan_frule, atan_rrule, atanh_frule, atanh_rrule, cos_frule, cos_rrule,
    cosh_frule, cosh_rrule, exp_frule, exp_rrule, expm1_frule, expm1_rrule, log1p_frule,
    log1p_rrule, log_frule, log_rrule, sin_frule, sin_rrule, sinh_frule, sinh_rrule, sqrt_frule,
    sqrt_rrule, tanh_frule, tanh_rrule,
};
use serde_json::Value;

struct UnaryRuleCase {
    op: &'static str,
    frule: fn(f64, f64) -> (f64, f64),
    rrule: fn(f64, f64, f64) -> f64,
}

fn oracle_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("third_party")
        .join("tensor-ad-oracles")
}

fn first_successful_float64_case(op: &str) -> Value {
    let path = oracle_root().join("cases").join(op).join("identity.jsonl");
    let file = File::open(&path).unwrap_or_else(|err| panic!("open {}: {err}", path.display()));
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let value: Value = serde_json::from_str(&line)
            .unwrap_or_else(|err| panic!("parse {}: {err}", path.display()));
        let dtype = value["dtype"].as_str();
        let behavior = value["expected_behavior"].as_str();
        if dtype == Some("float64") && behavior == Some("success") {
            return value;
        }
    }

    panic!("no successful float64 case found in {}", path.display());
}

fn scalar(value: &Value, path: &str) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("expected float64 at {path}, got {value}"))
}

fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, label: &str) {
    let tol = atol + rtol * expected.abs().max(actual.abs());
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: actual={actual}, expected={expected}, atol={atol}, rtol={rtol}",
    );
}

#[test]
fn published_float64_oracles_match_unary_rule_entrypoints() {
    let cases = [
        UnaryRuleCase {
            op: "sqrt",
            frule: sqrt_frule,
            rrule: |x, _result, cotangent| sqrt_rrule(x.sqrt(), cotangent),
        },
        UnaryRuleCase {
            op: "exp",
            frule: exp_frule,
            rrule: |_x, result, cotangent| exp_rrule(result, cotangent),
        },
        UnaryRuleCase {
            op: "expm1",
            frule: expm1_frule,
            rrule: |_x, result, cotangent| expm1_rrule(result, cotangent),
        },
        UnaryRuleCase {
            op: "log",
            frule: log_frule,
            rrule: |x, _result, cotangent| log_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "log1p",
            frule: log1p_frule,
            rrule: |x, _result, cotangent| log1p_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "sin",
            frule: sin_frule,
            rrule: |x, _result, cotangent| sin_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "cos",
            frule: cos_frule,
            rrule: |x, _result, cotangent| cos_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "tanh",
            frule: tanh_frule,
            rrule: |_x, result, cotangent| tanh_rrule(result, cotangent),
        },
        UnaryRuleCase {
            op: "asin",
            frule: asin_frule,
            rrule: |x, _result, cotangent| asin_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "acos",
            frule: acos_frule,
            rrule: |x, _result, cotangent| acos_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "atan",
            frule: atan_frule,
            rrule: |x, _result, cotangent| atan_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "sinh",
            frule: sinh_frule,
            rrule: |x, _result, cotangent| sinh_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "cosh",
            frule: cosh_frule,
            rrule: |x, _result, cotangent| cosh_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "asinh",
            frule: asinh_frule,
            rrule: |x, _result, cotangent| asinh_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "acosh",
            frule: acosh_frule,
            rrule: |x, _result, cotangent| acosh_rrule(x, cotangent),
        },
        UnaryRuleCase {
            op: "atanh",
            frule: atanh_frule,
            rrule: |x, _result, cotangent| atanh_rrule(x, cotangent),
        },
    ];

    for case in cases {
        let oracle = first_successful_float64_case(case.op);
        let input = scalar(&oracle["inputs"]["a"]["data"][0], "inputs.a.data[0]");
        let probe = &oracle["probes"][0];
        let tangent = scalar(
            &probe["direction"]["a"]["data"][0],
            "probes[0].direction.a.data[0]",
        );
        let cotangent = scalar(
            &probe["cotangent"]["value"]["data"][0],
            "probes[0].cotangent.value.data[0]",
        );
        let expected_jvp = scalar(
            &probe["pytorch_ref"]["jvp"]["value"]["data"][0],
            "probes[0].pytorch_ref.jvp.value.data[0]",
        );
        let expected_vjp = scalar(
            &probe["pytorch_ref"]["vjp"]["a"]["data"][0],
            "probes[0].pytorch_ref.vjp.a.data[0]",
        );
        let atol = scalar(
            &oracle["comparison"]["first_order"]["atol"],
            "comparison.first_order.atol",
        );
        let rtol = scalar(
            &oracle["comparison"]["first_order"]["rtol"],
            "comparison.first_order.rtol",
        );

        let (result, actual_jvp) = (case.frule)(input, tangent);
        let actual_vjp = (case.rrule)(input, result, cotangent);

        assert_close(actual_jvp, expected_jvp, atol, rtol, case.op);
        assert_close(actual_vjp, expected_vjp, atol, rtol, case.op);
    }
}
