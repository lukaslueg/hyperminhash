use crossbeam::thread;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{collections::BTreeMap, env, process};

const DEFAULT_TRIALS: usize = 20_480;
const DEFAULT_MAX_CARDINALITY: u32 = 100_000;
const DEFAULT_LEVELS: usize = 24;
const DEFAULT_SIMILARITY_BUCKETS: u32 = 200;
const DEFAULT_SEED: u64 = 0xD1CE_5EED_5EED_1234;
const ERROR_CAP_PERCENT: f64 = 0.025;

#[derive(Clone, Copy, Debug)]
struct Config {
    trials: usize,
    max_cardinality: u32,
    levels: usize,
    similarity_buckets: u32,
    workers: usize,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            trials: DEFAULT_TRIALS,
            max_cardinality: DEFAULT_MAX_CARDINALITY,
            levels: DEFAULT_LEVELS,
            similarity_buckets: DEFAULT_SIMILARITY_BUCKETS,
            workers: 8,
            seed: DEFAULT_SEED,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    count: u64,
    sum: f64,
    sum_sq: f64,
}

impl Stats {
    fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
    }

    fn mean(self) -> f64 {
        self.sum / self.count as f64
    }

    fn stddev(self) -> f64 {
        let mean = self.mean();
        ((self.sum_sq / self.count as f64) - mean * mean)
            .max(0.0)
            .sqrt()
    }
}

#[derive(Debug, Default)]
struct Aggregates {
    cardinality: BTreeMap<u64, Stats>,
    intersection: BTreeMap<u64, Stats>,
    intersection_fast: BTreeMap<u64, Stats>,
    intersection_fast_error: BTreeMap<u64, Stats>,
    similarity: BTreeMap<u32, Stats>,
    similarity_fast: BTreeMap<u32, Stats>,
    similarity_fast_error: BTreeMap<u32, Stats>,
}

impl Aggregates {
    fn merge(&mut self, other: Self) {
        merge_map(&mut self.cardinality, other.cardinality);
        merge_map(&mut self.intersection, other.intersection);
        merge_map(&mut self.intersection_fast, other.intersection_fast);
        merge_map(
            &mut self.intersection_fast_error,
            other.intersection_fast_error,
        );
        merge_map(&mut self.similarity, other.similarity);
        merge_map(&mut self.similarity_fast, other.similarity_fast);
        merge_map(&mut self.similarity_fast_error, other.similarity_fast_error);
    }
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    mean: f64,
    stddev: f64,
    samples: u64,
}

#[derive(Clone, Copy, Debug)]
enum AxisScale {
    Linear,
    Log1p,
}

#[derive(Clone, Copy, Debug)]
struct PlotSpec<'a> {
    id: &'a str,
    title: &'a str,
    x_label: &'a str,
    y_label: &'a str,
    points: &'a [Point],
    x_max: f64,
    y_max: f64,
    x_scale: AxisScale,
    y_scale: AxisScale,
    reference_line: ReferenceLine,
}

#[derive(Clone, Copy, Debug)]
enum ReferenceLine {
    Identity,
    Zero,
}

fn main() {
    let config = parse_args();
    let sizes = levels(config.max_cardinality, config.levels);
    let aggregates = run_simulation(config, &sizes);
    let cardinality = points_from_u64(&aggregates.cardinality);
    let intersection = points_from_u64(&aggregates.intersection);
    let intersection_fast = points_from_u64(&aggregates.intersection_fast);
    let intersection_fast_error = points_from_u64(&aggregates.intersection_fast_error);
    let similarity = points_from_similarity(&aggregates.similarity, config.similarity_buckets);
    let similarity_fast =
        points_from_similarity(&aggregates.similarity_fast, config.similarity_buckets);
    let similarity_fast_error =
        points_from_similarity(&aggregates.similarity_fast_error, config.similarity_buckets);

    print_instructions(config);
    print_dot(
        config,
        &[
            PlotSpec {
                id: "cardinality",
                title: "cardinality()",
                x_label: "actual cardinality",
                y_label: "estimated cardinality",
                points: &cardinality,
                x_max: config.max_cardinality as f64,
                y_max: max_estimate(&cardinality, config.max_cardinality as f64),
                x_scale: AxisScale::Log1p,
                y_scale: AxisScale::Log1p,
                reference_line: ReferenceLine::Identity,
            },
            PlotSpec {
                id: "intersection",
                title: "intersection()",
                x_label: "actual intersection",
                y_label: "estimated intersection",
                points: &intersection,
                x_max: config.max_cardinality as f64,
                y_max: max_estimate(&intersection, config.max_cardinality as f64),
                x_scale: AxisScale::Log1p,
                y_scale: AxisScale::Log1p,
                reference_line: ReferenceLine::Identity,
            },
            PlotSpec {
                id: "intersection_fast",
                title: "intersection_fast()",
                x_label: "actual intersection",
                y_label: "estimated intersection",
                points: &intersection_fast,
                x_max: config.max_cardinality as f64,
                y_max: max_estimate(&intersection_fast, config.max_cardinality as f64),
                x_scale: AxisScale::Log1p,
                y_scale: AxisScale::Log1p,
                reference_line: ReferenceLine::Identity,
            },
            PlotSpec {
                id: "similarity",
                title: "similarity()",
                x_label: "actual similarity",
                y_label: "estimated similarity",
                points: &similarity,
                x_max: 1.0,
                y_max: 1.0,
                x_scale: AxisScale::Linear,
                y_scale: AxisScale::Linear,
                reference_line: ReferenceLine::Identity,
            },
            PlotSpec {
                id: "similarity_fast",
                title: "similarity_fast()",
                x_label: "actual similarity",
                y_label: "estimated similarity",
                points: &similarity_fast,
                x_max: 1.0,
                y_max: 1.0,
                x_scale: AxisScale::Linear,
                y_scale: AxisScale::Linear,
                reference_line: ReferenceLine::Identity,
            },
            PlotSpec {
                id: "intersection_fast_error",
                title: "intersection_fast() error",
                x_label: "intersection()",
                y_label: "relative error (%)",
                points: &intersection_fast_error,
                x_max: max_x(&intersection_fast_error, config.max_cardinality as f64),
                y_max: ERROR_CAP_PERCENT,
                x_scale: AxisScale::Log1p,
                y_scale: AxisScale::Linear,
                reference_line: ReferenceLine::Zero,
            },
            PlotSpec {
                id: "similarity_fast_error",
                title: "similarity_fast() error",
                x_label: "similarity()",
                y_label: "relative error (%)",
                points: &similarity_fast_error,
                x_max: max_x(&similarity_fast_error, 1.0),
                y_max: ERROR_CAP_PERCENT,
                x_scale: AxisScale::Linear,
                y_scale: AxisScale::Linear,
                reference_line: ReferenceLine::Zero,
            },
        ],
    );
}

fn parse_args() -> Config {
    let mut config = Config::default();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trials" => config.trials = parse_value(&arg, args.next()),
            "--max-cardinality" => config.max_cardinality = parse_value(&arg, args.next()),
            "--levels" => config.levels = parse_value(&arg, args.next()),
            "--workers" => config.workers = parse_value(&arg, args.next()),
            "--seed" => config.seed = parse_value(&arg, args.next()),
            "--help" | "-h" => usage(0),
            _ => {
                eprintln!("Unknown argument: {arg}");
                usage(1);
            }
        }
    }

    config.levels = config.levels.max(2);
    config.workers = config.workers.max(1);
    config.max_cardinality = config.max_cardinality.max(1);
    config
}

fn parse_value<T>(flag: &str, value: Option<String>) -> T
where
    T: std::str::FromStr,
{
    match value {
        Some(value) => value
            .parse::<T>()
            .unwrap_or_else(|_| panic!("Failed to parse value for {flag}: {value}")),
        None => panic!("Missing value for {flag}"),
    }
}

fn usage(exit_code: i32) -> ! {
    eprintln!(
        "Usage: cargo run --release --example drift -- [--trials N] [--max-cardinality N] [--levels N] [--workers N] [--seed N]\n\
         \n\
         Options:\n\
         \t--trials N           Number of Monte Carlo trials to run (default: {DEFAULT_TRIALS}).\n\
         \t                     Higher values reduce noise but take longer.\n\
         \t--max-cardinality N  Largest exact set size to sample (default: {DEFAULT_MAX_CARDINALITY}).\n\
         \t                     The count plots cover values from 1 up to this limit.\n\
         \t--levels N           Number of log-spaced cardinality levels to probe (default: {DEFAULT_LEVELS}).\n\
         \t                     More levels produce smoother lines with more distinct x positions.\n\
         \t--workers N          Number of worker threads used for the simulation (default: 8).\n\
         \t                     Trials are independent, so this scales nearly linearly.\n\
         \t--seed N             Seed for reproducible random trials (default: {DEFAULT_SEED}).\n\
         \n\
         The example prints Graphviz DOT to stdout along with comment lines showing how to render it."
    );
    process::exit(exit_code);
}

fn levels(max_cardinality: u32, levels: usize) -> Vec<u32> {
    let mut values = Vec::with_capacity(levels + 1);
    for idx in 0..levels {
        let t = if levels == 1 {
            1.0
        } else {
            idx as f64 / (levels - 1) as f64
        };
        let value = (max_cardinality as f64).powf(t).round() as u32;
        values.push(value.max(1));
    }
    values.push(max_cardinality);
    values.sort_unstable();
    values.dedup();
    values
}

fn run_simulation(config: Config, sizes: &[u32]) -> Aggregates {
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(config.workers);
        let base_trials = config.trials / config.workers;
        let extra_trials = config.trials % config.workers;

        for worker_id in 0..config.workers {
            let worker_trials = base_trials + usize::from(worker_id < extra_trials);
            handles.push(scope.spawn(move |_| {
                let seed = config
                    .seed
                    .wrapping_add((worker_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut rng = StdRng::seed_from_u64(seed);
                let mut aggregates = Aggregates::default();

                for _ in 0..worker_trials {
                    run_trial(&mut rng, sizes, config.similarity_buckets, &mut aggregates);
                }

                aggregates
            }));
        }

        let mut merged = Aggregates::default();
        for handle in handles {
            merged.merge(handle.join().expect("worker thread panicked"));
        }
        merged
    })
    .expect("simulation scope failed")
}

fn run_trial(
    rng: &mut StdRng,
    sizes: &[u32],
    similarity_buckets: u32,
    aggregates: &mut Aggregates,
) {
    let left_true = sizes[rng.random_range(0..sizes.len())] as u64;
    let right_true = sizes[rng.random_range(0..sizes.len())] as u64;
    let shared = choose_shared(rng, sizes, left_true.min(right_true));
    let left_unique = left_true - shared;
    let right_unique = right_true - shared;
    let base = rng.random::<u64>();

    let mut left = hyperminhash::Sketch::default();
    let mut right = hyperminhash::Sketch::default();

    for idx in 0..shared {
        let value = base.wrapping_add(idx);
        left.add(value);
        right.add(value);
    }
    for idx in 0..left_unique {
        left.add(base.wrapping_add(shared + idx));
    }
    for idx in 0..right_unique {
        right.add(base.wrapping_add(left_true + idx));
    }

    aggregates
        .cardinality
        .entry(left_true)
        .or_default()
        .add(left.cardinality());
    aggregates
        .cardinality
        .entry(right_true)
        .or_default()
        .add(right.cardinality());

    let exact_intersection = shared;
    let precise_intersection = left.intersection(&right);
    let fast_intersection = left.intersection_fast(&right);
    aggregates
        .intersection
        .entry(exact_intersection)
        .or_default()
        .add(precise_intersection);
    aggregates
        .intersection_fast
        .entry(exact_intersection)
        .or_default()
        .add(fast_intersection);
    aggregates
        .intersection_fast_error
        .entry(precise_intersection.round() as u64)
        .or_default()
        .add(capped_relative_error(
            fast_intersection,
            precise_intersection,
            ERROR_CAP_PERCENT,
        ));

    let exact_union = left_true + right_unique;
    let exact_similarity = exact_intersection as f64 / exact_union as f64;
    let sim_bucket = (exact_similarity * similarity_buckets as f64).round() as u32;
    let precise_similarity = left.similarity(&right);
    let fast_similarity = left.similarity_fast(&right);
    aggregates
        .similarity
        .entry(sim_bucket)
        .or_default()
        .add(precise_similarity);
    aggregates
        .similarity_fast
        .entry(sim_bucket)
        .or_default()
        .add(fast_similarity);
    let precise_similarity_bucket = (precise_similarity * similarity_buckets as f64).round() as u32;
    aggregates
        .similarity_fast_error
        .entry(precise_similarity_bucket)
        .or_default()
        .add(capped_relative_error(
            fast_similarity,
            precise_similarity,
            ERROR_CAP_PERCENT,
        ));
}

fn choose_shared(rng: &mut StdRng, sizes: &[u32], max_shared: u64) -> u64 {
    let upper = sizes.partition_point(|value| u64::from(*value) <= max_shared);
    if upper == 0 {
        0
    } else {
        let idx = rng.random_range(0..=upper);
        if idx == 0 {
            0
        } else {
            u64::from(sizes[idx - 1])
        }
    }
}

fn merge_map<K>(left: &mut BTreeMap<K, Stats>, right: BTreeMap<K, Stats>)
where
    K: Ord,
{
    for (key, stats) in right {
        left.entry(key).or_default().merge(stats);
    }
}

fn points_from_u64(map: &BTreeMap<u64, Stats>) -> Vec<Point> {
    map.iter()
        .filter(|(_, stats)| stats.count > 0)
        .map(|(key, stats)| Point {
            x: *key as f64,
            mean: stats.mean(),
            stddev: stats.stddev(),
            samples: stats.count,
        })
        .collect()
}

fn points_from_similarity(map: &BTreeMap<u32, Stats>, buckets: u32) -> Vec<Point> {
    map.iter()
        .filter(|(_, stats)| stats.count > 0)
        .map(|(key, stats)| Point {
            x: *key as f64 / buckets as f64,
            mean: stats.mean(),
            stddev: stats.stddev(),
            samples: stats.count,
        })
        .collect()
}

fn max_estimate(points: &[Point], ideal_max: f64) -> f64 {
    let max_estimate = points
        .iter()
        .map(|point| point.mean + point.stddev)
        .fold(ideal_max, f64::max);
    (max_estimate * 1.05).max(ideal_max)
}

fn max_x(points: &[Point], ideal_max: f64) -> f64 {
    points.iter().map(|point| point.x).fold(ideal_max, f64::max)
}

fn capped_relative_error(estimate: f64, reference: f64, cap_percent: f64) -> f64 {
    let diff = (estimate - reference).abs();
    if reference.abs() <= f64::EPSILON {
        if diff <= f64::EPSILON {
            0.0
        } else {
            cap_percent
        }
    } else {
        (diff / reference.abs() * 100.0).min(cap_percent)
    }
}

fn print_instructions(config: Config) {
    println!("// Monte Carlo drift plots for hyperminhash estimators.");
    println!("// This output is valid Graphviz DOT.");
    println!(
        "// Save it with: cargo run --release --example drift -- --trials {} --max-cardinality {} --levels {} --workers {} --seed {} > drift.dot",
        config.trials, config.max_cardinality, config.levels, config.workers, config.seed
    );
    println!("// Render SVG with: neato -n2 -Tsvg drift.dot -o drift.svg");
    println!("// Render PNG with: neato -n2 -Tpng drift.dot -o drift.png");
    println!("// Count plots use log(1 + x) axes so low-cardinality drift is easier to see.");
    println!(
        "// Fast-path residual plots show capped relative error percentages, limited to {:.3}%.",
        ERROR_CAP_PERCENT
    );
    println!(
        "// Trials derive exact truth from deterministic set construction, so the simulation avoids materializing large exact collections on the heap."
    );
    println!();
}

fn print_dot(config: Config, plots: &[PlotSpec<'_>]) {
    println!("graph drift {{");
    println!("  layout=neato;");
    println!("  splines=polyline;");
    println!("  overlap=false;");
    println!("  outputorder=edgesfirst;");
    println!("  bgcolor=\"white\";");
    println!("  graph [fontname=\"Helvetica\"];");
    println!("  node [fontname=\"Helvetica\"];");
    println!("  edge [fontname=\"Helvetica\"];");

    print_legend(config);

    let plot_positions = [
        (0.0, 440.0),
        (400.0, 440.0),
        (800.0, 440.0),
        (200.0, 190.0),
        (600.0, 190.0),
        (200.0, -60.0),
        (600.0, -60.0),
    ];
    for (plot, (x, y)) in plots.iter().zip(plot_positions) {
        draw_plot(*plot, x, y);
    }

    println!("}}");
}

fn print_legend(config: Config) {
    println!(
        "  legend_title [shape=plaintext, pos=\"480,840!\", label=\"hyperminhash Monte Carlo drift\"];"
    );
    println!(
        "  legend_actual_a [shape=point, width=0.03, pos=\"365,813!\", color=\"#7f7f7f\", label=\"\"];"
    );
    println!(
        "  legend_actual_b [shape=point, width=0.03, pos=\"395,813!\", color=\"#7f7f7f\", label=\"\"];"
    );
    println!("  legend_actual_a -- legend_actual_b [color=\"#7f7f7f\", style=dotted, penwidth=2];");
    println!(
        "  legend_actual_label [shape=plaintext, pos=\"490,813!\", label=\"actual = estimated\"];"
    );
    println!(
        "  legend_mean_a [shape=point, width=0.03, pos=\"365,795!\", color=\"#1f77b4\", label=\"\"];"
    );
    println!(
        "  legend_mean_b [shape=point, width=0.03, pos=\"395,795!\", color=\"#1f77b4\", label=\"\"];"
    );
    println!("  legend_mean_a -- legend_mean_b [color=\"#1f77b4\", penwidth=2.5];");
    println!("  legend_mean_label [shape=plaintext, pos=\"475,795!\", label=\"estimated mean\"];");
    println!(
        "  legend_std_a [shape=point, width=0.03, pos=\"365,777!\", color=\"#d62728\", label=\"\"];"
    );
    println!(
        "  legend_std_b [shape=point, width=0.03, pos=\"395,777!\", color=\"#d62728\", label=\"\"];"
    );
    println!("  legend_std_a -- legend_std_b [color=\"#d62728\", style=dashed, penwidth=1.5];");
    println!("  legend_std_label [shape=plaintext, pos=\"480,777!\", label=\"mean ± stddev\"];");
    println!(
        "  legend_cfg_title [shape=plaintext, pos=\"910,840!\", label=\"simulation config\"];"
    );
    println!(
        "  legend_cfg_trials [shape=plaintext, pos=\"910,820!\", label=\"trials: {}\"];",
        config.trials
    );
    println!(
        "  legend_cfg_max [shape=plaintext, pos=\"910,805!\", label=\"max cardinality: {}\"];",
        config.max_cardinality
    );
    println!(
        "  legend_cfg_levels [shape=plaintext, pos=\"910,790!\", label=\"levels: {}\"];",
        config.levels
    );
    println!(
        "  legend_cfg_workers [shape=plaintext, pos=\"910,775!\", label=\"workers: {}\"];",
        config.workers
    );
    println!(
        "  legend_cfg_seed [shape=plaintext, pos=\"910,760!\", label=\"seed: {}\"];",
        config.seed
    );
    println!(
        "  legend_cfg_buckets [shape=plaintext, pos=\"910,745!\", label=\"similarity buckets: {}\"];",
        config.similarity_buckets
    );
    println!(
        "  legend_cfg_error [shape=plaintext, pos=\"910,730!\", label=\"error cap: {:.3}%\"];",
        ERROR_CAP_PERCENT
    );
}

fn draw_plot(plot: PlotSpec<'_>, off_x: f64, off_y: f64) {
    let left = off_x + 58.0;
    let bottom = off_y + 38.0;
    let width = 208.0;
    let height = 158.0;
    let right = left + width;
    let top = bottom + height;
    let y_max = match plot.reference_line {
        ReferenceLine::Identity => plot.y_max.max(plot.x_max).max(1.0),
        ReferenceLine::Zero => plot.y_max.max(f64::EPSILON),
    };

    println!(
        "  {}_title [shape=plaintext, pos=\"{:.2},{:.2}!\", label=\"{}\"];",
        plot.id,
        (left + right) / 2.0,
        top + 18.0,
        plot.title
    );
    println!(
        "  {}_xlabel [shape=plaintext, pos=\"{:.2},{:.2}!\", label=\"{}\"];",
        plot.id,
        (left + right) / 2.0,
        bottom - 26.0,
        plot.x_label
    );
    println!(
        "  {}_ylabel [shape=plaintext, pos=\"{:.2},{:.2}!\", label=\"{}\"];",
        plot.id,
        left - 76.0,
        (bottom + top) / 2.0,
        plot.y_label.replace(' ', "\\n")
    );

    println!(
        "  {}_axis_x0 [shape=point, width=0.01, pos=\"{:.2},{:.2}!\", label=\"\"];",
        plot.id, left, bottom
    );
    println!(
        "  {}_axis_x1 [shape=point, width=0.01, pos=\"{:.2},{:.2}!\", label=\"\"];",
        plot.id, right, bottom
    );
    println!(
        "  {}_axis_y1 [shape=point, width=0.01, pos=\"{:.2},{:.2}!\", label=\"\"];",
        plot.id, left, top
    );
    println!(
        "  {}_axis_x0 -- {}_axis_x1 [color=\"#222222\", penwidth=1.1];",
        plot.id, plot.id
    );
    println!(
        "  {}_axis_x0 -- {}_axis_y1 [color=\"#222222\", penwidth=1.1];",
        plot.id, plot.id
    );

    for (tick, x_value) in axis_ticks(plot.x_scale, plot.x_max).into_iter().enumerate() {
        let x = left + width * axis_fraction(x_value, plot.x_max, plot.x_scale);
        println!(
            "  {}_xtick_{}a [shape=point, width=0.01, pos=\"{:.2},{:.2}!\", label=\"\"];",
            plot.id, tick, x, bottom
        );
        println!(
            "  {}_xtick_{}b [shape=point, width=0.01, pos=\"{:.2},{:.2}!\", label=\"\"];",
            plot.id,
            tick,
            x,
            bottom - 4.0
        );
        println!(
            "  {}_xtick_{}a -- {}_xtick_{}b [color=\"#666666\", penwidth=1];",
            plot.id, tick, plot.id, tick
        );
        println!(
            "  {}_xtick_label_{} [shape=plaintext, pos=\"{:.2},{:.2}!\", label=\"{}\"];",
            plot.id,
            tick,
            x,
            bottom - 12.0,
            format_axis_value(x_value, plot.x_scale)
        );
    }
    for (tick, y_value) in axis_ticks(plot.y_scale, y_max).into_iter().enumerate() {
        let y = bottom + height * axis_fraction(y_value, y_max, plot.y_scale);
        println!(
            "  {}_ytick_{}a [shape=point, width=0.01, pos=\"{:.2},{:.2}!\", label=\"\"];",
            plot.id, tick, left, y
        );
        println!(
            "  {}_ytick_{}b [shape=point, width=0.01, pos=\"{:.2},{:.2}!\", label=\"\"];",
            plot.id,
            tick,
            left - 4.0,
            y
        );
        println!(
            "  {}_ytick_{}a -- {}_ytick_{}b [color=\"#666666\", penwidth=1];",
            plot.id, tick, plot.id, tick
        );
        println!(
            "  {}_ytick_label_{} [shape=plaintext, pos=\"{:.2},{:.2}!\", label=\"{}\"];",
            plot.id,
            tick,
            left - 24.0,
            y,
            format_axis_value(y_value, plot.y_scale)
        );
    }

    match plot.reference_line {
        ReferenceLine::Identity => {
            let actual_start = plot_point(
                plot, "actual", 0, 0.0, 0.0, left, bottom, width, height, y_max,
            );
            let actual_end = plot_point(
                plot,
                "actual",
                1,
                plot.x_max,
                plot.x_max.min(y_max),
                left,
                bottom,
                width,
                height,
                y_max,
            );
            println!(
                "  {actual_start} -- {actual_end} [color=\"#7f7f7f\", style=dotted, penwidth=2];"
            );
        }
        ReferenceLine::Zero => {
            let zero_start = plot_point(
                plot, "zero", 0, 0.0, 0.0, left, bottom, width, height, y_max,
            );
            let zero_end = plot_point(
                plot, "zero", 1, plot.x_max, 0.0, left, bottom, width, height, y_max,
            );
            println!("  {zero_start} -- {zero_end} [color=\"#7f7f7f\", style=dotted, penwidth=2];");
        }
    }

    draw_series(
        plot,
        "mean",
        "#1f77b4",
        "solid",
        2.5,
        |point| point.mean,
        left,
        bottom,
        width,
        height,
        y_max,
    );
    draw_series(
        plot,
        "upper",
        "#d62728",
        "dashed",
        1.5,
        |point| point.mean + point.stddev,
        left,
        bottom,
        width,
        height,
        y_max,
    );
    draw_series(
        plot,
        "lower",
        "#d62728",
        "dashed",
        1.5,
        |point| (point.mean - point.stddev).max(0.0),
        left,
        bottom,
        width,
        height,
        y_max,
    );
}

fn draw_series<F>(
    plot: PlotSpec<'_>,
    name: &str,
    color: &str,
    style: &str,
    penwidth: f64,
    y_value: F,
    left: f64,
    bottom: f64,
    width: f64,
    height: f64,
    y_max: f64,
) where
    F: Fn(Point) -> f64,
{
    let mut previous = None;
    for (idx, point) in plot.points.iter().copied().enumerate() {
        let y = y_value(point).min(y_max);
        let node = plot_point(
            plot, name, idx, point.x, y, left, bottom, width, height, y_max,
        );
        println!(
            "  {} [tooltip=\"samples={} mean={:.6} stddev={:.6}\"];",
            node, point.samples, point.mean, point.stddev
        );
        if let Some(previous) = previous {
            println!(
                "  {previous} -- {node} [color=\"{color}\", style=\"{style}\", penwidth={penwidth:.1}];"
            );
        }
        previous = Some(node);
    }
}

#[allow(clippy::too_many_arguments)]
fn plot_point(
    plot: PlotSpec<'_>,
    series: &str,
    idx: usize,
    x_value: f64,
    y_value: f64,
    left: f64,
    bottom: f64,
    width: f64,
    height: f64,
    y_max: f64,
) -> String {
    let x = left + width * axis_fraction(x_value, plot.x_max, plot.x_scale);
    let y = bottom + height * axis_fraction(y_value, y_max, plot.y_scale);
    let name = format!("{}_{}_{}", plot.id, series, idx);
    println!(
        "  {name} [shape=point, width=0.03, height=0.03, pos=\"{x:.2},{y:.2}!\", label=\"\"];"
    );
    name
}

fn axis_fraction(value: f64, max: f64, scale: AxisScale) -> f64 {
    match scale {
        AxisScale::Linear => value / max.max(f64::EPSILON),
        AxisScale::Log1p => value.ln_1p() / max.max(f64::EPSILON).ln_1p(),
    }
}

fn axis_ticks(scale: AxisScale, max: f64) -> Vec<f64> {
    match scale {
        AxisScale::Linear => vec![0.0, 0.25 * max, 0.5 * max, 0.75 * max, max],
        AxisScale::Log1p => {
            let mut ticks = vec![0.0];
            let mut value = 1.0;
            while value < max {
                ticks.push(value);
                value *= 10.0;
            }
            if ticks.last().copied() != Some(max) {
                ticks.push(max);
            }
            ticks
        }
    }
}

fn format_axis_value(value: f64, scale: AxisScale) -> String {
    if matches!(scale, AxisScale::Linear) {
        format!("{value:.2}")
    } else if value >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.1}k", value / 1_000.0)
    } else {
        format!("{value:.0}")
    }
}
