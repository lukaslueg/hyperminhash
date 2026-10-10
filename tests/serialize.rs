#[test]
fn serialization_is_stable() {
    let serialized = include_bytes!("serialized.bin");
    assert_eq!(serialized.len(), hyperminhash::SERIALIZED_SIZE);

    // Ensure that a Sketch serialized once yields the same result forever...
    let sketch = hyperminhash::Sketch::load(&serialized[..]).unwrap();
    assert_eq!(sketch.cardinality(), 9931.106244547593);

    let mut saved = Vec::new();
    sketch.save(&mut saved).unwrap();
    assert_eq!(saved.as_slice(), serialized);
}
