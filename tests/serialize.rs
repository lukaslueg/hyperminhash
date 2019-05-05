#[test]
#[cfg(feature = "serialize")]
fn deserialize_is_stable() {
    // Ensure that a Sketch serialized once yields the same result forever...
    let sk = hyperminhash::Sketch::load(&include_bytes!("serialized.bin")[..]).unwrap();
    assert_eq!(sk.cardinality(), 9931.106244547593);
}
