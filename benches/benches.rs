use std::hint::black_box;

fn bench_new(c: &mut criterion::Criterion) {
    c.bench_function("Construct new sketch", |b| {
        b.iter(|| {
            let sk = hyperminhash::Sketch::default();
            black_box(&sk);
        })
    });
}

fn bench_add(c: &mut criterion::Criterion) {
    let plot_config =
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Adding small objects");
    group.plot_config(plot_config);
    for i in [10, 1_000, 100_000, 10_000_000] {
        group.throughput(criterion::Throughput::Elements(i));
        group.bench_with_input(
            criterion::BenchmarkId::new("Add hashable object", i),
            &i,
            |b, i| {
                b.iter(|| {
                    let mut sk = hyperminhash::Sketch::default();
                    for value in 0..*black_box(i) {
                        sk.add(value);
                    }
                    black_box(&sk);
                })
            },
        );
    }
    group.finish();
}

fn bench_bytes(c: &mut criterion::Criterion) {
    let plot_config =
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Adding buffers");
    group.plot_config(plot_config);
    for i in [16usize, 128, 1024, 1024 * 1024, 10 * 1024 * 1024] {
        group.throughput(criterion::Throughput::Bytes(i as u64));
        let buf = vec![0; i];
        let mut sk = hyperminhash::Sketch::new();
        group.bench_with_input(
            criterion::BenchmarkId::new("Add hashable object", i),
            &i,
            |b, _| b.iter(|| sk.add_bytes(black_box(&buf))),
        );
    }
    group.finish();
}

fn bench_cardinality(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("Determine cardinality");
    let mut b = |max: u64, name: &str| {
        group.throughput(criterion::Throughput::Elements(max));
        let sk = (0..max).collect::<hyperminhash::Sketch>();
        group.bench_function(name, |b| b.iter(|| black_box(&sk).cardinality()));
    };
    b(0, "Empty");
    b(100, "Mostly empty");
    b(10_000_000, "Filled");
    group.finish();
}

fn bench_is_empty(c: &mut criterion::Criterion) {
    fn sketch_with_nonzero_register(index: usize) -> hyperminhash::Sketch {
        let mut bytes = vec![0; hyperminhash::SERIALIZED_SIZE];
        let offset = index * size_of::<u16>();
        bytes[offset..offset + size_of::<u16>()].copy_from_slice(&1u16.to_le_bytes());
        hyperminhash::Sketch::load(bytes.as_slice()).expect("valid in-memory sketch")
    }

    let empty = hyperminhash::Sketch::new();
    let first = sketch_with_nonzero_register(0);
    let last = sketch_with_nonzero_register(hyperminhash::SERIALIZED_SIZE / size_of::<u16>() - 1);

    let mut group = c.benchmark_group("Check whether sketch is empty");
    for (name, sketch) in [
        ("empty", &empty),
        ("first register nonzero", &first),
        ("last register nonzero", &last),
    ] {
        group.bench_function(name, |b| b.iter(|| black_box(black_box(sketch).is_empty())));
    }
    group.finish();
}

fn bench_union(c: &mut criterion::Criterion) {
    let empty = hyperminhash::Sketch::new();
    let filled = (0..1_000_000).collect::<hyperminhash::Sketch>();
    let disjoint = (1_000_000..2_000_000).collect::<hyperminhash::Sketch>();
    let overlapping = (500_000..1_500_000).collect::<hyperminhash::Sketch>();

    let mut group = c.benchmark_group("Union sketches");
    for (name, left, right) in [
        ("empty/filled", &empty, &filled),
        ("identical", &filled, &filled),
        ("disjoint", &filled, &disjoint),
        ("overlapping", &filled, &overlapping),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || left.clone(),
                |work| {
                    black_box(&mut *work).union(black_box(right));
                    black_box(&*work);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_similarity(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("Determine similarity index");

    let small_a = (0..100).collect::<hyperminhash::Sketch>();
    let small_b = (50..150).collect::<hyperminhash::Sketch>();
    group.bench_function("small/high-precision", |b| {
        b.iter(|| black_box(&small_a).similarity(black_box(&small_b)))
    });
    group.bench_function("small/fast", |b| {
        b.iter(|| black_box(&small_a).similarity_fast(black_box(&small_b)))
    });
    group.bench_function("small/intersection/high-precision", |b| {
        b.iter(|| black_box(&small_a).intersection(black_box(&small_b)))
    });
    group.bench_function("small/intersection/fast", |b| {
        b.iter(|| black_box(&small_a).intersection_fast(black_box(&small_b)))
    });

    let large_a = (0..1_000_000).collect::<hyperminhash::Sketch>();
    let large_b = (500_000..1_500_000).collect::<hyperminhash::Sketch>();
    group.bench_function("large/high-precision", |b| {
        b.iter(|| black_box(&large_a).similarity(black_box(&large_b)))
    });
    group.bench_function("large/fast", |b| {
        b.iter(|| black_box(&large_a).similarity_fast(black_box(&large_b)))
    });
    group.bench_function("large/intersection/high-precision", |b| {
        b.iter(|| black_box(&large_a).intersection(black_box(&large_b)))
    });
    group.bench_function("large/intersection/fast", |b| {
        b.iter(|| black_box(&large_a).intersection_fast(black_box(&large_b)))
    });

    group.finish();
}

fn bench_disjoint_similarity(c: &mut criterion::Criterion) {
    fn sketch_with_registers(start: usize, count: usize, value: u16) -> hyperminhash::Sketch {
        const REGISTERS: usize = 1 << 14;
        let mut bytes = vec![0; REGISTERS * size_of::<u16>()];
        for register in bytes
            .chunks_exact_mut(size_of::<u16>())
            .skip(start)
            .take(count)
        {
            register.copy_from_slice(&value.to_le_bytes());
        }
        hyperminhash::Sketch::load(bytes.as_slice()).expect("valid in-memory sketch")
    }

    let cases = [
        (
            "one/one",
            sketch_with_registers(0, 1, 1 << 10),
            sketch_with_registers(1, 1, 1 << 10),
        ),
        (
            "balanced",
            sketch_with_registers(0, 1_000, 1 << 10),
            sketch_with_registers(1_000, 1_000, 1 << 10),
        ),
        (
            "asymmetric",
            sketch_with_registers(0, 100, 1 << 10),
            sketch_with_registers(100, 8_000, 1 << 10),
        ),
    ];

    let mut group = c.benchmark_group("Determine disjoint similarity index");
    for (name, left, right) in &cases {
        group.bench_function(*name, |b| {
            b.iter(|| black_box(left).similarity(black_box(right)))
        });
    }
    group.finish();
}

fn bench_similarity_many(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("Batch similarity");

    let small_a = (0..1_000).collect::<hyperminhash::Sketch>();
    let small_others: Vec<hyperminhash::Sketch> = (0..16)
        .map(|k| ((k * 250)..(k * 250 + 1_000)).collect())
        .collect();

    let large_a = (0..1_000_000).collect::<hyperminhash::Sketch>();
    let large_others: Vec<hyperminhash::Sketch> = (0..16)
        .map(|k| ((k * 250_000)..(k * 250_000 + 1_000_000)).collect())
        .collect();

    // All four method pairs, both cardinality regimes. `loop` runs the
    // per-pair method through `.map().collect()`; `batch` runs the new
    // `*_many` iterator through `.collect()`.
    macro_rules! pair {
        ($name:expr, $a:expr, $others:expr, $pair:ident, $many:ident) => {{
            group.bench_function(concat!($name, "/loop"), |b| {
                b.iter(|| {
                    let v: Vec<f64> = $others.iter().map(|o| $a.$pair(o)).collect();
                    black_box(v)
                })
            });
            group.bench_function(concat!($name, "/batch"), |b| {
                b.iter(|| {
                    let v: Vec<f64> = $a.$many($others.iter()).collect();
                    black_box(v)
                })
            });
        }};
    }

    pair!(
        "small/similarity",
        small_a,
        small_others,
        similarity,
        similarity_many
    );
    pair!(
        "small/similarity_fast",
        small_a,
        small_others,
        similarity_fast,
        similarity_many_fast
    );
    pair!(
        "small/intersection",
        small_a,
        small_others,
        intersection,
        intersection_many
    );
    pair!(
        "small/intersection_fast",
        small_a,
        small_others,
        intersection_fast,
        intersection_many_fast
    );

    pair!(
        "large/similarity",
        large_a,
        large_others,
        similarity,
        similarity_many
    );
    pair!(
        "large/similarity_fast",
        large_a,
        large_others,
        similarity_fast,
        similarity_many_fast
    );
    pair!(
        "large/intersection",
        large_a,
        large_others,
        intersection,
        intersection_many
    );
    pair!(
        "large/intersection_fast",
        large_a,
        large_others,
        intersection_fast,
        intersection_many_fast
    );

    group.finish();
}

fn bench_unique_integers(c: &mut criterion::Criterion) {
    let plot_config =
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Counting unique integers");
    group.plot_config(plot_config);
    for i in [10, 1_000, 100_000, 10_000_000] {
        if i > 1_000 {
            group.sampling_mode(criterion::SamplingMode::Flat);
            group.measurement_time(std::time::Duration::from_secs(10));
        } else {
            group.sampling_mode(criterion::SamplingMode::Linear);
            group.measurement_time(std::time::Duration::from_secs(5));
        }
        group.throughput(criterion::Throughput::Elements(i));
        group.bench_with_input(
            criterion::BenchmarkId::new("Hyperminhash", i),
            &i,
            |b, i| {
                b.iter(|| {
                    let sk = (0..*i).collect::<hyperminhash::Sketch>();
                    black_box(sk.cardinality())
                })
            },
        );
        group.bench_with_input(criterion::BenchmarkId::new("HashSet", i), &i, |b, i| {
            b.iter(|| {
                let sk = (0..*i).collect::<std::collections::HashSet<_>>();
                black_box(sk.len())
            })
        });
    }
    group.finish();
}

criterion::criterion_group!(
    benches,
    bench_new,
    bench_cardinality,
    bench_is_empty,
    bench_union,
    bench_similarity,
    bench_disjoint_similarity,
    bench_similarity_many,
    bench_unique_integers,
    bench_add,
    bench_bytes
);
criterion::criterion_main!(benches);
