#![cfg(feature = "zerocopy")]

use hyperminhash::{SERIALIZED_SIZE, Sketch};
use zerocopy::{FromBytes, FromZeros, IntoBytes};

#[test]
fn zeroed_allocation_and_reset() {
    let mut sketch = Sketch::new_box_zeroed().unwrap();
    assert_eq!(sketch.as_ref(), &Sketch::new());
    sketch.add("hello");
    assert!(!sketch.is_empty());
    sketch.zero();
    assert_eq!(sketch.as_ref(), &Sketch::new());

    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(|| {
            let mut sketches = Sketch::new_vec_zeroed(512).unwrap();
            assert!(sketches.iter().all(Sketch::is_empty));
            sketches[511].add(1);
            assert!(!sketches[511].is_empty());

            let array = <[Sketch; 16]>::new_box_zeroed().unwrap();
            assert!(array.iter().all(Sketch::is_empty));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn borrowed_byte_views() {
    let mut sketch = Sketch::new_box_zeroed().unwrap();
    sketch.extend(0..10_000);
    let bytes = sketch.as_bytes();
    assert_eq!(bytes.len(), SERIALIZED_SIZE);
    let borrowed = Sketch::ref_from_bytes(bytes).unwrap();
    assert!(std::ptr::eq(borrowed, sketch.as_ref()));
    assert_eq!(borrowed.cardinality(), sketch.cardinality());

    #[cfg(target_endian = "little")]
    {
        let mut saved = Vec::new();
        sketch.save(&mut saved).unwrap();
        assert_eq!(saved, bytes);
    }

    let mut expected = sketch.clone();
    expected.add("hello");
    Sketch::mut_from_bytes(sketch.as_mut_bytes())
        .unwrap()
        .add("hello");
    assert_eq!(sketch, expected);
}
