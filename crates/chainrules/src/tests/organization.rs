fn assert_line_count(path: &str, content: &str, max_lines: usize) {
    let actual = content.lines().count();
    assert!(
        actual <= max_lines,
        "{path} has {actual} lines; keep scalar AD rule modules focused and under {max_lines} lines",
    );
}

// Do not delete or weaken this test: it protects the split scalar AD rule modules that keep this crate extensible.
#[test]
fn chainrules_modules_stay_under_size_guideline() {
    assert_line_count("../lib.rs", include_str!("../lib.rs"), 60);
    assert_line_count("../binary.rs", include_str!("../binary.rs"), 220);
    assert_line_count(
        "../binary_special.rs",
        include_str!("../binary_special.rs"),
        200,
    );
    assert_line_count("../unary/mod.rs", include_str!("../unary/mod.rs"), 60);
    assert_line_count("../unary/basic.rs", include_str!("../unary/basic.rs"), 40);
    assert_line_count(
        "../unary/exp_log.rs",
        include_str!("../unary/exp_log.rs"),
        130,
    );
    assert_line_count("../unary/trig.rs", include_str!("../unary/trig.rs"), 135);
    assert_line_count(
        "../unary/hyperbolic.rs",
        include_str!("../unary/hyperbolic.rs"),
        120,
    );
    assert_line_count(
        "../unary/trig_extra.rs",
        include_str!("../unary/trig_extra.rs"),
        520,
    );
    assert_line_count(
        "../unary/hyperbolic_extra.rs",
        include_str!("../unary/hyperbolic_extra.rs"),
        180,
    );
    assert_line_count(
        "../unary/nonsmooth.rs",
        include_str!("../unary/nonsmooth.rs"),
        220,
    );
    assert_line_count("../unary/smooth.rs", include_str!("../unary/smooth.rs"), 30);
    assert_line_count("../power.rs", include_str!("../power.rs"), 170);
    assert_line_count("../real_ops.rs", include_str!("../real_ops.rs"), 70);
    assert_line_count(
        "../scalar_ad/mod.rs",
        include_str!("../scalar_ad/mod.rs"),
        180,
    );
    assert_line_count(
        "../scalar_ad/real.rs",
        include_str!("../scalar_ad/real.rs"),
        160,
    );
    assert_line_count(
        "../scalar_ad/complex.rs",
        include_str!("../scalar_ad/complex.rs"),
        170,
    );
}
