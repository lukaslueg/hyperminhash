use hyperminhash::{Entry, Sketch};

#[derive(Clone, Copy)]
struct ExpectedPair {
    similarity: u64,
    similarity_fast: u64,
    intersection: u64,
    intersection_fast: u64,
}

fn assert_bits(label: &str, actual: f64, expected: u64) {
    assert_eq!(
        actual.to_bits(),
        expected,
        "{label}: expected {expected:#018x}, got {:#018x} ({actual})",
        actual.to_bits(),
    );
}

fn assert_pair(label: &str, left: &Sketch, right: &Sketch, expected: ExpectedPair) {
    assert_bits(
        &format!("{label} similarity"),
        left.similarity(right),
        expected.similarity,
    );
    assert_bits(
        &format!("{label} similarity_fast"),
        left.similarity_fast(right),
        expected.similarity_fast,
    );
    assert_bits(
        &format!("{label} intersection"),
        left.intersection(right),
        expected.intersection,
    );
    assert_bits(
        &format!("{label} intersection_fast"),
        left.intersection_fast(right),
        expected.intersection_fast,
    );

    assert_bits(
        &format!("{label} reverse similarity"),
        right.similarity(left),
        expected.similarity,
    );
    assert_bits(
        &format!("{label} reverse similarity_fast"),
        right.similarity_fast(left),
        expected.similarity_fast,
    );
    assert_bits(
        &format!("{label} reverse intersection"),
        right.intersection(left),
        expected.intersection,
    );
    assert_bits(
        &format!("{label} reverse intersection_fast"),
        right.intersection_fast(left),
        expected.intersection_fast,
    );
}

fn from_registers(registers: impl IntoIterator<Item = u16>) -> Sketch {
    let mut bytes = Vec::with_capacity(32_768);
    for register in registers {
        bytes.extend_from_slice(&register.to_le_bytes());
    }
    bytes.resize(32_768, 0);
    Sketch::load(bytes.as_slice()).unwrap()
}

fn assert_round_trip(sketch: &Sketch) {
    let mut first = Vec::new();
    sketch.save(&mut first).unwrap();
    assert_eq!(first.len(), 32_768);

    let loaded = Sketch::load(first.as_slice()).unwrap();
    assert_eq!(&loaded, sketch);
    assert_eq!(
        loaded.cardinality().to_bits(),
        sketch.cardinality().to_bits()
    );

    let mut second = Vec::new();
    loaded.save(&mut second).unwrap();
    assert_eq!(second, first);
}

#[test]
fn public_numerical_results_are_stable() {
    let empty = Sketch::new();
    let singleton: Sketch = [0i64].into_iter().collect();
    let small_a: Sketch = (0..100i64).collect();
    let small_b: Sketch = (50..150i64).collect();
    let small_high_overlap: Sketch = (1..101i64).collect();
    let small_minimal_overlap: Sketch = (99..199i64).collect();
    let disjoint: Sketch = (10_000..11_000i64).collect();
    let asymmetric: Sketch = (50..10_050i64).collect();
    let medium_a: Sketch = (0..10_000i64).collect();
    let medium_b: Sketch = (5_000..15_000i64).collect();
    let large_a: Sketch = (0..1_000_000i64).collect();
    let large_b: Sketch = (500_000..1_500_000i64).collect();
    let sparse = from_registers([1 << 10]);
    let saturated = from_registers(std::iter::repeat_n(u16::MAX, 16_384));

    let mut seeded_entry = Sketch::new();
    seeded_entry.add_with_seed("same", 1);
    seeded_entry.add_with_seed("same", 2);
    let mut entry = Entry::new();
    entry.add("prefix");
    entry.add_bytes(b"payload");
    seeded_entry.add_entry(&entry);

    for (label, sketch, expected) in [
        ("empty", &empty, 0x0000_0000_0000_0000),
        ("singleton", &singleton, 0x3fef_fd8c_52b8_9a77),
        ("small", &small_a, 0x4059_11e6_e2fc_9d96),
        ("asymmetric", &asymmetric, 0x40c3_7234_3eac_914e),
        ("medium", &medium_a, 0x40c3_775b_36b7_63fe),
        ("large", &large_a, 0x412e_c8c3_961e_a44d),
        ("sparse registers", &sparse, 0x3fef_fd8c_52b8_9a77),
        ("saturated registers", &saturated, 0x44b7_1480_1fbc_cd72),
        ("seeded Entry", &seeded_entry, 0x4007_fe9f_9a0e_593b),
    ] {
        assert_bits(label, sketch.cardinality(), expected);
        assert_round_trip(sketch);
    }

    assert_pair(
        "empty/empty",
        &empty,
        &empty,
        ExpectedPair {
            similarity: 0x3ff0_0000_0000_0000,
            similarity_fast: 0x3ff0_0000_0000_0000,
            intersection: 0,
            intersection_fast: 0,
        },
    );
    assert_pair(
        "empty/singleton",
        &empty,
        &singleton,
        ExpectedPair {
            similarity: 0,
            similarity_fast: 0,
            intersection: 0,
            intersection_fast: 0,
        },
    );
    assert_pair(
        "small overlap",
        &small_a,
        &small_b,
        ExpectedPair {
            similarity: 0x3fd5_79fc_903c_26f3,
            similarity_fast: 0x3fd5_79fc_9052_7845,
            intersection: 0x4049_1acb_88f5_7555,
            intersection_fast: 0x4049_1acb_890f_8bd6,
        },
    );
    assert_pair(
        "small high overlap",
        &small_a,
        &small_high_overlap,
        ExpectedPair {
            similarity: 0x3fef_5dc8_3cc4_71fc,
            similarity_fast: 0x3fef_5dc8_3cd4_e930,
            intersection: 0x4058_d1f6_7b79_ce2d,
            intersection_fast: 0x4058_d1f6_7b86_d5aa,
        },
    );
    assert_pair(
        "small minimal overlap",
        &small_a,
        &small_minimal_overlap,
        ExpectedPair {
            similarity: 0x3f74_cab8_82ed_3520,
            similarity_fast: 0x3f74_cab8_8725_af6e,
            intersection: 0x3ff0_17d1_f899_4362,
            intersection_fast: 0x3ff0_17d1_fbdd_8f5a,
        },
    );
    assert_pair(
        "small disjoint",
        &small_a,
        &disjoint,
        ExpectedPair {
            similarity: 0,
            similarity_fast: 0,
            intersection: 0,
            intersection_fast: 0,
        },
    );
    assert_pair(
        "asymmetric",
        &small_a,
        &asymmetric,
        ExpectedPair {
            similarity: 0x3f74_42ce_69c7_7e0f,
            similarity_fast: 0x3f74_42ce_715a_1443,
            intersection: 0x4048_c003_f95d_dc9d,
            intersection_fast: 0x4048_c004_029d_f82b,
        },
    );
    assert_pair(
        "medium overlap",
        &medium_a,
        &medium_b,
        ExpectedPair {
            similarity: 0x3fd5_2617_01a9_92c1,
            similarity_fast: 0x3fd5_2617_07f8_e9db,
            intersection: 0x40b3_64f1_7f4d_fc3a,
            intersection_fast: 0x40b3_64f1_8517_519c,
        },
    );
    assert_pair(
        "large overlap",
        &large_a,
        &large_b,
        ExpectedPair {
            similarity: 0x3fd5_0e28_e2dd_a4e1,
            similarity_fast: 0x3fd5_1900_0000_0000,
            intersection: 0x411e_6836_1329_e3d8,
            intersection_fast: 0x411e_77dd_c2fd_80ea,
        },
    );
    assert_pair(
        "sparse/saturated registers",
        &sparse,
        &saturated,
        ExpectedPair {
            similarity: 0,
            similarity_fast: 0,
            intersection: 0,
            intersection_fast: 0,
        },
    );
    assert_pair(
        "seeded Entry/small",
        &seeded_entry,
        &small_a,
        ExpectedPair {
            similarity: 0,
            similarity_fast: 0,
            intersection: 0,
            intersection_fast: 0,
        },
    );
}

#[test]
fn batch_results_match_pairwise_results_bit_for_bit() {
    let left: Sketch = (0..1_000i64).collect();
    let others: Vec<Sketch> = [0..1_000, 500..1_500, 10_000..11_000, 0..500, 0..1_000_000]
        .into_iter()
        .map(|range| range.collect())
        .collect();

    let pairwise: Vec<_> = others
        .iter()
        .map(|right| left.similarity(right).to_bits())
        .collect();
    let batch: Vec<_> = left
        .similarity_many(others.iter())
        .map(f64::to_bits)
        .collect();
    assert_eq!(batch, pairwise);

    let pairwise: Vec<_> = others
        .iter()
        .map(|right| left.similarity_fast(right).to_bits())
        .collect();
    let batch: Vec<_> = left
        .similarity_many_fast(others.iter())
        .map(f64::to_bits)
        .collect();
    assert_eq!(batch, pairwise);

    let pairwise: Vec<_> = others
        .iter()
        .map(|right| left.intersection(right).to_bits())
        .collect();
    let batch: Vec<_> = left
        .intersection_many(others.iter())
        .map(f64::to_bits)
        .collect();
    assert_eq!(batch, pairwise);

    let pairwise: Vec<_> = others
        .iter()
        .map(|right| left.intersection_fast(right).to_bits())
        .collect();
    let batch: Vec<_> = left
        .intersection_many_fast(others.iter())
        .map(f64::to_bits)
        .collect();
    assert_eq!(batch, pairwise);
}

#[test]
fn union_registers_and_serialization_are_exact() {
    fn assert_union(mut left: Sketch, right: &Sketch, expected: &Sketch) {
        left.union(right);
        assert_eq!(&left, expected);

        let mut actual_bytes = Vec::new();
        let mut expected_bytes = Vec::new();
        left.save(&mut actual_bytes).unwrap();
        expected.save(&mut expected_bytes).unwrap();
        assert_eq!(actual_bytes, expected_bytes);
    }

    let empty = Sketch::new();
    let left: Sketch = (0..1_000_000i64).collect();
    let identical = left.clone();
    let disjoint: Sketch = (1_000_000..2_000_000i64).collect();
    let overlapping: Sketch = (500_000..1_500_000i64).collect();
    let disjoint_union: Sketch = (0..2_000_000i64).collect();
    let overlapping_union: Sketch = (0..1_500_000i64).collect();

    assert_union(empty, &left, &left);
    assert_union(left.clone(), &identical, &left);
    assert_union(left.clone(), &disjoint, &disjoint_union);
    assert_union(left, &overlapping, &overlapping_union);
}
