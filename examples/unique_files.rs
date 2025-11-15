/// Counts the number of unique files in up to two paths
use std::{env, ffi, fs, io, process};

fn add_file_to_sketch(
    entry: &walkdir::DirEntry,
    sk: &mut hyperminhash::Sketch,
) -> io::Result<bool> {
    if !entry.metadata()?.is_file() {
        return Ok(false);
    }

    let f = fs::File::open(&entry.path())?;
    sk.add_reader(f)?;
    Ok(true)
}

fn read_dir(root: &ffi::OsString) -> (u64, hyperminhash::Sketch) {
    let mut sk = hyperminhash::Sketch::default();
    let mut i: u64 = 0;
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            !e.file_name()
                .to_str()
                .map(|s| s.starts_with("."))
                .unwrap_or(false)
        })
        .filter_map(|e| e.ok())
    {
        match add_file_to_sketch(&entry, &mut sk) {
            Ok(true) => {
                i += 1;
            }
            Ok(false) => {}
            Err(_) => eprintln!("Skipped {}", entry.path().display()),
        }
    }
    (i, sk)
}

fn main() {
    let (root1, root2) = match (env::args_os().nth(1), env::args_os().nth(2)) {
        (Some(r1), r2) => (r1, r2),
        _ => {
            eprintln!("Usage: cargo run --release --example unique_files [ROOT_PATH]");
            process::exit(1);
        }
    };

    let (i, sk1) = read_dir(&root1);
    println!(
        "{}: {} files, approx. {:.0} unique files",
        root1.display(),
        i,
        sk1.cardinality()
    );

    if let Some(root2) = root2 {
        let (j, sk2) = read_dir(&root2);
        println!(
            "{}: {} files, approx. {:.0} unique files",
            root2.display(),
            j,
            sk2.cardinality()
        );
        println!(
            "Approx. {:.0} files are the same, {:.1}% overlap",
            sk1.intersection(&sk2),
            sk1.similarity(&sk2) * 100.0
        );
    }
}
