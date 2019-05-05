use hyperminhash;
use rand;

use std::collections;

fn main() {
    for k in 3..8 {
        let k = 10u64.pow(k);
        println!(
            r#"

#### Max cardinality {}
Set1|HLL1|Set2|HLL2|S1 ∪ S2|HLL1 ∪ HLL2|S1 ∩ S2|HLL1 ∩ HLL2|
----|----|----|----|-------|-----------|-------|-----------|"#,
            k
        );

        for _ in 1..5 {
            let size1 = rand::random::<u64>() % k;
            let size2 = rand::random::<u64>() % k;
            let mut sk1 = hyperminhash::Sketch::default();
            let mut sk2 = hyperminhash::Sketch::default();
            let cols = rand::random::<u64>() % size1.min(size2);
            let mut intersections = 0;
            let mut set =
                collections::HashMap::<u64, u64>::with_capacity(size1.max(size2) as usize);
            for i in 0..size1 {
                *set.entry(i).or_default() += 1;
                sk1.add(i);
            }
            for i in (size1 - cols)..(size1 - cols + size2) {
                set.entry(i)
                    .and_modify(|e| {
                        *e += 1;
                        intersections += 1
                    })
                    .or_insert(1);
                sk2.add(i);
            }

            let card1 = sk1.cardinality();
            let card2 = sk2.cardinality();
            let ints1 = sk1.intersection(&sk2);
            let mcard = sk1.union(&sk2).cardinality();
            println!(
                "{0}|{1:.0}|{2}|{3:.0}|{4}|{5:.0}|{6}|{7:.0}",
                size1,
                card1,
                size2,
                card2,
                set.len(),
                mcard,
                cols,
                ints1
            );
        }
    }
}
