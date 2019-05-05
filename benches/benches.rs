#![feature(test)]
extern crate test;
use test::Bencher;

fn inp(max: usize) -> Vec<impl std::hash::Hash + Eq> {
    (0..max).map(|i| format!("Foo{}", i)).collect()
}

macro_rules! make_bench {
    ($name:ident, $count:expr) => {
        mod $name {
            #[bench]
            fn count_hyperminhash(b: &mut super::Bencher) {
                let inp = super::inp($count);
                b.iter(|| {
                    let sk = inp.iter().collect::<hyperminhash::Sketch>();
                    test::black_box(sk.cardinality());
                });
            }

            #[bench]
            fn count_hashset(b: &mut super::Bencher) {
                let inp = super::inp($count);
                b.iter(|| {
                    let s = inp.iter().collect::<std::collections::HashSet<_>>();
                    test::black_box(s.len());
                });
            }
        }
    };
}

make_bench!(b01_ten, 10);
make_bench!(b03_thousand, 1_000);
make_bench!(b04_hundredthousand, 100_000);
make_bench!(b05_million, 1_000_000);
