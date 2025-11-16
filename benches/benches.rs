use std::hint::black_box;

fn bench_new(c: &mut criterion::Criterion) {
    c.bench_function("Construct new sketch", |b| {
        b.iter(|| black_box(hyperminhash::Sketch::default()))
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
            |b, i| b.iter(|| black_box((0..*i).collect::<hyperminhash::Sketch>())),
        );
    }
    group.finish();
}

fn bench_cardinality(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("Determine cardinality");
    let mut b = |max: u64, name: &str| {
        group.throughput(criterion::Throughput::Elements(max));
        let sk = (0..max).collect::<hyperminhash::Sketch>();
        group.bench_function(name, |b| b.iter(|| black_box(sk.cardinality())));
    };
    b(0, "Empty");
    b(100, "Mostly empty");
    b(10_000_000, "Filled");
    group.finish();
}

fn bench_similarity(c: &mut criterion::Criterion) {
    let sk1 = (0..100).collect::<hyperminhash::Sketch>();
    let sk2 = (50..150).collect::<hyperminhash::Sketch>();
    c.bench_function("Determine similarity index", |b| {
        b.iter(|| black_box(sk1.similarity(&sk2)))
    });
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
    bench_similarity,
    bench_unique_integers,
    bench_add
);
criterion::criterion_main!(benches);
