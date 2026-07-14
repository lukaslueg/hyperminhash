use hyperminhash::{SERIALIZED_SIZE, Sketch};

fn from_registers(registers: impl IntoIterator<Item = u16>) -> Sketch {
    let mut bytes = Vec::with_capacity(SERIALIZED_SIZE);
    for register in registers {
        bytes.extend_from_slice(&register.to_le_bytes());
    }
    bytes.resize(SERIALIZED_SIZE, 0);
    Sketch::load(bytes.as_slice()).unwrap()
}

fn assert_round_trip(sketch: &Sketch) {
    let mut first = Vec::new();
    sketch.save(&mut first).unwrap();
    assert_eq!(first.len(), SERIALIZED_SIZE);

    let loaded = Sketch::load(first.as_slice()).unwrap();
    assert_eq!(&loaded, sketch);

    let mut second = Vec::new();
    loaded.save(&mut second).unwrap();
    assert_eq!(second, first);
}

#[test]
fn serialization_round_trips_are_exact() {
    let empty = Sketch::new();
    let sparse = from_registers([1 << 10]);
    let filled = from_registers(std::iter::repeat_n(
        u16::MAX,
        SERIALIZED_SIZE / size_of::<u16>(),
    ));

    for sketch in [&empty, &sparse, &filled] {
        assert_round_trip(sketch);
    }
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
