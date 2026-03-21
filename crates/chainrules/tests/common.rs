#![allow(dead_code)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use num_complex::Complex64;
use serde_json::Value;

pub struct UnaryOracleCase<T> {
    pub op: &'static str,
    pub frule: fn(T, T) -> (T, T),
    pub rrule: fn(T, T, T) -> T,
}

pub trait OracleScalar: Copy + core::fmt::Debug {
    fn dtype() -> &'static str;
    fn is_scalar_value(value: &Value) -> bool;
    fn from_json(value: &Value, path: &str) -> Self;
    fn assert_close(actual: Self, expected: Self, atol: f64, rtol: f64, label: &str);
}

impl OracleScalar for f64 {
    fn dtype() -> &'static str {
        "float64"
    }

    fn is_scalar_value(value: &Value) -> bool {
        value.is_number()
    }

    fn from_json(value: &Value, path: &str) -> Self {
        scalar_f64(value, path)
    }

    fn assert_close(actual: Self, expected: Self, atol: f64, rtol: f64, label: &str) {
        assert_close_f64(actual, expected, atol, rtol, label);
    }
}

impl OracleScalar for Complex64 {
    fn dtype() -> &'static str {
        "complex128"
    }

    fn is_scalar_value(value: &Value) -> bool {
        value
            .as_array()
            .is_some_and(|items| items.len() == 2 && items.iter().all(Value::is_number))
    }

    fn from_json(value: &Value, path: &str) -> Self {
        scalar_complex64(value, path)
    }

    fn assert_close(actual: Self, expected: Self, atol: f64, rtol: f64, label: &str) {
        assert_close_complex64(actual, expected, atol, rtol, label);
    }
}

fn oracle_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("third_party")
        .join("tensor-ad-oracles")
}

pub fn successful_cases(op: &str, dtype: &str) -> Vec<Value> {
    let path = oracle_root().join("cases").join(op).join("identity.jsonl");
    let file = File::open(&path).unwrap_or_else(|err| panic!("open {}: {err}", path.display()));
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let value: Value = serde_json::from_str(&line)
            .unwrap_or_else(|err| panic!("parse {}: {err}", path.display()));
        let case_dtype = value["dtype"].as_str();
        let behavior = value["expected_behavior"].as_str();
        if case_dtype == Some(dtype) && behavior == Some("success") {
            cases.push(value);
        }
    }

    assert!(
        !cases.is_empty(),
        "no successful {dtype} cases found in {}",
        path.display()
    );
    cases
}

pub fn scalar_f64(value: &Value, path: &str) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("expected float64 at {path}, got {value}"))
}

pub fn scalar_complex64(value: &Value, path: &str) -> Complex64 {
    let data = value
        .as_array()
        .unwrap_or_else(|| panic!("expected complex128 pair at {path}, got {value}"));
    assert!(
        data.len() == 2,
        "expected complex128 pair at {path}, got {value}"
    );
    Complex64::new(scalar_f64(&data[0], path), scalar_f64(&data[1], path))
}

pub fn assert_close_f64(actual: f64, expected: f64, atol: f64, rtol: f64, label: &str) {
    let tol = atol + rtol * expected.abs().max(actual.abs());
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: actual={actual}, expected={expected}, atol={atol}, rtol={rtol}",
    );
}

pub fn assert_close_complex64(
    actual: Complex64,
    expected: Complex64,
    atol: f64,
    rtol: f64,
    label: &str,
) {
    let tol = atol + rtol * expected.norm().max(actual.norm());
    assert!(
        (actual - expected).norm() <= tol,
        "{label}: actual={actual:?}, expected={expected:?}, atol={atol}, rtol={rtol}",
    );
}

fn collect_scalar_values<T: OracleScalar>(value: &Value, path: &str, out: &mut Vec<T>) {
    if T::is_scalar_value(value) {
        out.push(T::from_json(value, path));
        return;
    }

    let items = value
        .as_array()
        .unwrap_or_else(|| panic!("expected array at {path}, got {value}"));
    for (index, item) in items.iter().enumerate() {
        collect_scalar_values::<T>(item, &format!("{path}[{index}]"), out);
    }
}

fn scalar_values<T: OracleScalar>(value: &Value, path: &str) -> Vec<T> {
    let mut out = Vec::new();
    collect_scalar_values(value, path, &mut out);
    out
}

pub fn run_unary_oracle_cases<T: OracleScalar>(cases: &[UnaryOracleCase<T>]) {
    for case in cases {
        for (case_index, oracle) in successful_cases(case.op, T::dtype())
            .into_iter()
            .enumerate()
        {
            let inputs = scalar_values::<T>(&oracle["inputs"]["a"]["data"], "inputs.a.data");
            let probes = oracle["probes"]
                .as_array()
                .unwrap_or_else(|| panic!("expected probes array for {}", case.op));
            let atol = scalar_f64(
                &oracle["comparison"]["first_order"]["atol"],
                "comparison.first_order.atol",
            );
            let rtol = scalar_f64(
                &oracle["comparison"]["first_order"]["rtol"],
                "comparison.first_order.rtol",
            );

            for (probe_index, probe) in probes.iter().enumerate() {
                let tangents = scalar_values::<T>(
                    &probe["direction"]["a"]["data"],
                    &format!("probes[{probe_index}].direction.a.data"),
                );
                let cotangents = scalar_values::<T>(
                    &probe["cotangent"]["value"]["data"],
                    &format!("probes[{probe_index}].cotangent.value.data"),
                );
                let expected_jvps = scalar_values::<T>(
                    &probe["pytorch_ref"]["jvp"]["value"]["data"],
                    &format!("probes[{probe_index}].pytorch_ref.jvp.value.data"),
                );
                let expected_vjps = scalar_values::<T>(
                    &probe["pytorch_ref"]["vjp"]["a"]["data"],
                    &format!("probes[{probe_index}].pytorch_ref.vjp.a.data"),
                );

                assert_eq!(
                    inputs.len(),
                    tangents.len(),
                    "{} case {case_index} probe {probe_index}: input and tangent lengths differ",
                    case.op
                );
                assert_eq!(
                    inputs.len(),
                    cotangents.len(),
                    "{} case {case_index} probe {probe_index}: input and cotangent lengths differ",
                    case.op
                );
                assert_eq!(
                    inputs.len(),
                    expected_jvps.len(),
                    "{} case {case_index} probe {probe_index}: input and expected jvp lengths differ",
                    case.op
                );
                assert_eq!(
                    inputs.len(),
                    expected_vjps.len(),
                    "{} case {case_index} probe {probe_index}: input and expected vjp lengths differ",
                    case.op
                );

                for index in 0..inputs.len() {
                    let (result, actual_jvp) = (case.frule)(inputs[index], tangents[index]);
                    let actual_vjp = (case.rrule)(inputs[index], result, cotangents[index]);
                    let label = format!(
                        "{} case {case_index} probe {probe_index} element {index}",
                        case.op
                    );
                    T::assert_close(actual_jvp, expected_jvps[index], atol, rtol, &label);
                    T::assert_close(actual_vjp, expected_vjps[index], atol, rtol, &label);
                }
            }
        }
    }
}
