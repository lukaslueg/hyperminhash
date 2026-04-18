use hyperminhash::Sketch;

fn collect_ranges(ranges: &[std::ops::Range<i64>]) -> Vec<Sketch> {
    ranges.iter().map(|r| r.clone().collect()).collect()
}

fn assert_bit_exact(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len(), "length mismatch");
    for (i, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "mismatch at index {i}: {a} vs {b}"
        );
    }
}

fn check_all_four(a: &Sketch, others: &[Sketch]) {
    let want_sim: Vec<f64> = others.iter().map(|o| a.similarity(o)).collect();
    let got_sim: Vec<f64> = a.similarity_many(others.iter()).collect();
    assert_bit_exact(&got_sim, &want_sim);

    let want_simf: Vec<f64> = others.iter().map(|o| a.similarity_fast(o)).collect();
    let got_simf: Vec<f64> = a.similarity_many_fast(others.iter()).collect();
    assert_bit_exact(&got_simf, &want_simf);

    let want_int: Vec<f64> = others.iter().map(|o| a.intersection(o)).collect();
    let got_int: Vec<f64> = a.intersection_many(others.iter()).collect();
    assert_bit_exact(&got_int, &want_int);

    let want_intf: Vec<f64> = others.iter().map(|o| a.intersection_fast(o)).collect();
    let got_intf: Vec<f64> = a.intersection_many_fast(others.iter()).collect();
    assert_bit_exact(&got_intf, &want_intf);
}

#[test]
fn small_cardinality_hits_slow_path() {
    let a: Sketch = (0..100i64).collect();
    let others = collect_ranges(&[0..100, 50..150, 100..200, 0..50]);
    check_all_four(&a, &others);
}

#[test]
fn medium_cardinality() {
    let a: Sketch = (0..10_000i64).collect();
    let others = collect_ranges(&[0..10_000, 5_000..15_000, 10_000..20_000, 0..1_000]);
    check_all_four(&a, &others);
}

#[test]
fn large_cardinality_skips_a_diff() {
    // All pairs exceed 2^(P+5) = 524_288, so the slow exp-loop branch never
    // fires. `a_diff` should stay None internally (we can't inspect it, but
    // bit-exact parity is the invariant we care about).
    let a: Sketch = (0..1_000_000i64).collect();
    let others = collect_ranges(&[0..1_000_000, 500_000..1_500_000, 600_000..800_000]);
    check_all_four(&a, &others);
}

#[test]
fn mixed_regimes_in_one_call() {
    // Mix small (slow-path) and large (closed-form) others. Exercises lazy
    // `a_diff` init being triggered partway through the iterator.
    let a: Sketch = (0..1_000i64).collect();
    let others = collect_ranges(&[
        0..1_000_000, // large → closed-form branch
        0..500,       // small → slow branch, triggers a_diff build
        100_000..200_000,
        500..1_500,
    ]);
    check_all_four(&a, &others);
}

#[test]
fn empty_iterator() {
    let a: Sketch = (0..100i64).collect();
    let empty: Vec<Sketch> = Vec::new();
    assert!(a.similarity_many(empty.iter()).next().is_none());
    assert!(a.similarity_many_fast(empty.iter()).next().is_none());
    assert!(a.intersection_many(empty.iter()).next().is_none());
    assert!(a.intersection_many_fast(empty.iter()).next().is_none());
}

#[test]
fn single_element() {
    let a: Sketch = (0..1_000i64).collect();
    let b: Sketch = (500..1_500i64).collect();
    let others = vec![b];
    check_all_four(&a, &others);
}

#[test]
fn self_comparison_shortcut() {
    let a: Sketch = (0..1_000i64).collect();
    let others = [a.clone()];

    let sim: Vec<f64> = a.similarity_many(others.iter()).collect();
    assert_eq!(sim, vec![1.0]);

    let sim_fast: Vec<f64> = a.similarity_many_fast(others.iter()).collect();
    assert_eq!(sim_fast, vec![1.0]);

    let inter: Vec<f64> = a.intersection_many(others.iter()).collect();
    assert_eq!(inter, vec![a.cardinality()]);

    let inter_fast: Vec<f64> = a.intersection_many_fast(others.iter()).collect();
    assert_eq!(inter_fast, vec![a.cardinality()]);
}

#[test]
fn disjoint_pairs_are_zero() {
    let a: Sketch = (0..1_000i64).collect();
    let others = collect_ranges(&[10_000..11_000, 20_000..21_000]);
    let sim: Vec<f64> = a.similarity_many(others.iter()).collect();
    let inter: Vec<f64> = a.intersection_many(others.iter()).collect();
    assert!(sim.iter().all(|&x| x == 0.0));
    assert!(inter.iter().all(|&x| x == 0.0));
}

#[test]
fn accepts_various_iterable_shapes() {
    let a: Sketch = (0..100i64).collect();
    let b: Sketch = (50..150i64).collect();
    let c: Sketch = (100..200i64).collect();

    // Slice of references
    let slice = [&b, &c];
    let _: Vec<f64> = a.similarity_many(slice.iter().copied()).collect();

    // Vec<Sketch> via .iter()
    let v: Vec<Sketch> = vec![b.clone(), c.clone()];
    let _: Vec<f64> = a.similarity_many(&v).collect();

    // Array literal of references
    let _: Vec<f64> = a.similarity_many([&b, &c]).collect();
}
