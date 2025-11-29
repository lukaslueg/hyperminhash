/// Counts the number of unique files in up to two paths
use std::{env, fs, process, thread, time};

fn fmt_bytes(s: u64) -> String {
    if s > 1024 * 1024 * 1024 {
        return format!("{:.2} GB", s as f32 / 1024.0 / 1024.0 / 1024.0);
    } else if s > 1024 * 1024 {
        return format!("{:.2} MB", s as f32 / 1024.0 / 1024.0);
    } else if s > 1024 {
        return format!("{:.2} KB", s as f32 / 1024.0);
    } else {
        return format!("{} bytes", s);
    }
}

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        eprintln!("Usage: cargo run --release --example unique_files <PATH> [PATH...]");
        process::exit(1);
    }

    let (send, recv) = crossbeam::channel::bounded(100_000);
    let workers = (0..12)
        .map(|_| {
            let recv: crossbeam::channel::Receiver<(usize, walkdir::DirEntry)> = recv.clone();
            thread::spawn(move || {
                let mut sk: Vec<(Box<hyperminhash::Sketch>, u64, u64)> = Vec::new();
                while let Ok((root_idx, entry)) = recv.recv() {
                    if sk.len() <= root_idx {
                        sk.resize(root_idx + 1, Default::default());
                    }
                    sk[root_idx].1 += 1;
                    let f = match fs::File::open(&entry.path()) {
                        Ok(f) => f,
                        Err(e) => {
                            eprintln!("Failed to open file: `{:?}`", e);
                            continue;
                        }
                    };
                    match sk[root_idx].0.add_reader(f) {
                        Ok(s) => sk[root_idx].2 += s,
                        Err(e) => eprintln!("Failed to read file: `{:?}`", e),
                    }
                }
                sk
            })
        })
        .collect::<Vec<_>>();

    let reading_t = time::Instant::now();
    for (root_idx, root) in args.iter().enumerate() {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| {
                !e.file_name()
                    .to_str()
                    .map(|s| s.starts_with("."))
                    .unwrap_or(false)
            })
            .filter_map(|e| {
                e.ok()
                    .and_then(|e| e.metadata().ok().and_then(|m| m.is_file().then_some(e)))
            })
        {
            send.send((root_idx, entry))
                .expect("At least one worker is alive");
        }
    }

    drop(send);
    let workers_res = workers
        .into_iter()
        .map(|w| w.join().expect("The worker did not panic"))
        .collect::<Vec<_>>();
    println!(
        "Read {} files ({}) in {:.2}s",
        workers_res
            .iter()
            .fold(0, |acc, r| acc + r.iter().map(|(_, c, _)| c).sum::<u64>()),
        fmt_bytes(
            workers_res
                .iter()
                .fold(0, |acc, r| acc + r.iter().map(|(_, _, fs)| fs).sum::<u64>())
        ),
        reading_t.elapsed().as_millis() as f32 / 1000.0
    );

    let get_res = |root_idx: usize| {
        workers_res.iter().flat_map(|w| w.get(root_idx)).fold(
            (Box::new(hyperminhash::Sketch::default()), 0, 0),
            |(mut sk, total_count, total_size), (sk2, count, size)| {
                sk.union(&sk2);
                (sk, total_count + count, total_size + size)
            },
        )
    };
    let (sk, count, size) = get_res(0);
    println!(
        "{}: {} files, {}, approx. {:.0} unique files",
        args.first()
            .expect("Already checked args is not empty")
            .display(),
        count,
        fmt_bytes(size),
        sk.cardinality()
    );
    for (root_idx, root) in args.iter().enumerate().skip(1) {
        let (sk1, count, size) = get_res(root_idx);
        println!(
            "{}: {} files, {}, approx. {:.0} unique files; approx. {:.0} files are the same, {:.1}% overlap",
            root.display(),
            count,
            fmt_bytes(size),
            sk1.cardinality(),
            sk.intersection(&sk1),
            sk.similarity(&sk1) * 100.0
        )
    }
}
