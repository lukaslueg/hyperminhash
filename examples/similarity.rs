use std::time;

fn main() {
    let t1 = time::Instant::now();
    let x: Vec<hyperminhash::Sketch> = (0..100000)
        .map(|_| (0..10000).map(|_| rand::random::<u16>()).collect())
        .collect();
    println!("{:?}", t1.elapsed());
    let t2 = time::Instant::now();
    let sk1 = &x[0];
    let avg = sk1.similarity_many_fast(x.iter().skip(1)).sum::<f64>() / x.len() as f64;
    println!("{:?}", t2.elapsed());
    println!("{}, {}", sk1.cardinality(), avg);
}
