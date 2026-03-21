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
    fn from_json(value: &Value, path: &str) -> Self;
    fn assert_close(actual: Self, expected: Self, atol: f64, rtol: f64, label: &str);
}

impl OracleScalar for f64 {
    fn dtype() -> &'static str {
        "float64"
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

pub fn first_successful_case(op: &str, dtype: &str) -> Value {
    let path = oracle_root().join("cases").join(op).join("identity.jsonl");
    let file = File::open(&path).unwrap_or_else(|err| panic!("open {}: {err}", path.display()));
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let value: Value = serde_json::from_str(&line)
            .unwrap_or_else(|err| panic!("parse {}: {err}", path.display()));
        let case_dtype = value["dtype"].as_str();
        let behavior = value["expected_behavior"].as_str();
        if case_dtype == Some(dtype) && behavior == Some("success") {
            return value;
        }
    }

    panic!("no successful {dtype} case found in {}", path.display());
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

pub fn run_unary_oracle_cases<T: OracleScalar>(cases: &[UnaryOracleCase<T>]) {
    for case in cases {
        let oracle = first_successful_case(case.op, T::dtype());
        let input = T::from_json(&oracle["inputs"]["a"]["data"][0], "inputs.a.data[0]");
        let probe = &oracle["probes"][0];
        let tangent = T::from_json(
            &probe["direction"]["a"]["data"][0],
            "probes[0].direction.a.data[0]",
        );
        let cotangent = T::from_json(
            &probe["cotangent"]["value"]["data"][0],
            "probes[0].cotangent.value.data[0]",
        );
        let expected_jvp = T::from_json(
            &probe["pytorch_ref"]["jvp"]["value"]["data"][0],
            "probes[0].pytorch_ref.jvp.value.data[0]",
        );
        let expected_vjp = T::from_json(
            &probe["pytorch_ref"]["vjp"]["a"]["data"][0],
            "probes[0].pytorch_ref.vjp.a.data[0]",
        );
        let atol = scalar_f64(
            &oracle["comparison"]["first_order"]["atol"],
            "comparison.first_order.atol",
        );
        let rtol = scalar_f64(
            &oracle["comparison"]["first_order"]["rtol"],
            "comparison.first_order.rtol",
        );

        let (result, actual_jvp) = (case.frule)(input, tangent);
        let actual_vjp = (case.rrule)(input, result, cotangent);

        T::assert_close(actual_jvp, expected_jvp, atol, rtol, case.op);
        T::assert_close(actual_vjp, expected_vjp, atol, rtol, case.op);
    }
}

pub fn run_unary_oracle_reverse_cases_complex64(cases: &[UnaryOracleCase<Complex64>]) {
    for case in cases {
        let oracle = first_successful_case(case.op, Complex64::dtype());
        let input = scalar_complex64(&oracle["inputs"]["a"]["data"][0], "inputs.a.data[0]");
        let probe = &oracle["probes"][0];
        let cotangent = scalar_complex64(
            &probe["cotangent"]["value"]["data"][0],
            "probes[0].cotangent.value.data[0]",
        );
        let expected_vjp = scalar_complex64(
            &probe["pytorch_ref"]["vjp"]["a"]["data"][0],
            "probes[0].pytorch_ref.vjp.a.data[0]",
        );
        let atol = scalar_f64(
            &oracle["comparison"]["first_order"]["atol"],
            "comparison.first_order.atol",
        );
        let rtol = scalar_f64(
            &oracle["comparison"]["first_order"]["rtol"],
            "comparison.first_order.rtol",
        );

        let (result, _) = (case.frule)(input, cotangent);
        let actual_vjp = (case.rrule)(input, result, cotangent);
        Complex64::assert_close(actual_vjp, expected_vjp, atol, rtol, case.op);
    }
}
