mod common;

use chainrules::{
    acos_frule, acos_rrule, acosh_frule, acosh_rrule, asin_frule, asin_rrule, asinh_frule,
    asinh_rrule, atan_frule, atan_rrule, atanh_frule, atanh_rrule, cos_frule, cos_rrule,
    cosh_frule, cosh_rrule, exp2_frule, exp2_rrule, exp_frule, exp_rrule, expm1_frule, expm1_rrule,
    log1p_frule, log1p_rrule, log2_frule, log2_rrule, log_frule, log_rrule, sin_frule, sin_rrule,
    sinh_frule, sinh_rrule, sqrt_frule, sqrt_rrule, tan_frule, tan_rrule, tanh_frule, tanh_rrule,
};
use num_complex::Complex64;

use common::{run_unary_oracle_cases, UnaryOracleCase};

#[test]
fn published_float64_oracles_match_unary_rule_entrypoints() {
    let cases: [UnaryOracleCase<f64>; 19] = [
        UnaryOracleCase {
            op: "sqrt",
            frule: sqrt_frule,
            rrule: |x: f64, _result, cotangent| sqrt_rrule(x.sqrt(), cotangent),
        },
        UnaryOracleCase {
            op: "exp",
            frule: exp_frule,
            rrule: |_x: f64, result, cotangent| exp_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "expm1",
            frule: expm1_frule,
            rrule: |_x: f64, result, cotangent| expm1_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "log",
            frule: log_frule,
            rrule: |x: f64, _result, cotangent| log_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "log1p",
            frule: log1p_frule,
            rrule: |x: f64, _result, cotangent| log1p_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "sin",
            frule: sin_frule,
            rrule: |x: f64, _result, cotangent| sin_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "cos",
            frule: cos_frule,
            rrule: |x: f64, _result, cotangent| cos_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "tanh",
            frule: tanh_frule,
            rrule: |_x: f64, result, cotangent| tanh_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "asin",
            frule: asin_frule,
            rrule: |x: f64, _result, cotangent| asin_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "acos",
            frule: acos_frule,
            rrule: |x: f64, _result, cotangent| acos_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "atan",
            frule: atan_frule,
            rrule: |x: f64, _result, cotangent| atan_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "sinh",
            frule: sinh_frule,
            rrule: |x: f64, _result, cotangent| sinh_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "cosh",
            frule: cosh_frule,
            rrule: |x: f64, _result, cotangent| cosh_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "asinh",
            frule: asinh_frule,
            rrule: |x: f64, _result, cotangent| asinh_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "acosh",
            frule: acosh_frule,
            rrule: |x: f64, _result, cotangent| acosh_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "atanh",
            frule: atanh_frule,
            rrule: |x: f64, _result, cotangent| atanh_rrule(x, cotangent),
        },
        UnaryOracleCase {
            op: "tan",
            frule: tan_frule,
            rrule: |_x: f64, result, cotangent| tan_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "exp2",
            frule: exp2_frule,
            rrule: |_x: f64, result, cotangent| exp2_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "log2",
            frule: log2_frule,
            rrule: |x: f64, _result, cotangent| log2_rrule(x, cotangent),
        },
    ];

    run_unary_oracle_cases(&cases);
}

#[test]
fn published_complex128_oracles_match_unary_rule_entrypoints() {
    let cases: [UnaryOracleCase<Complex64>; 3] = [
        UnaryOracleCase {
            op: "tan",
            frule: tan_frule,
            rrule: |_x: Complex64, result, cotangent| tan_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "exp2",
            frule: exp2_frule,
            rrule: |_x: Complex64, result, cotangent| exp2_rrule(result, cotangent),
        },
        UnaryOracleCase {
            op: "log2",
            frule: log2_frule,
            rrule: |x: Complex64, _result, cotangent| log2_rrule(x, cotangent),
        },
    ];

    run_unary_oracle_cases(&cases);
}
