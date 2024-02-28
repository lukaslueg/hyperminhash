use hyperminhash::Sketch;
use std::{fs, io, io::BufRead, process, thread};

fn read(fname: std::ffi::OsString) -> thread::JoinHandle<io::Result<Sketch>> {
    std::thread::spawn(move || {
        let mut sk = Sketch::default();
        let reader = io::BufReader::new(fs::File::open(fname)?).lines();
        for line in reader {
            sk.add_bytes(line?.as_ref());
        }
        Ok(sk)
    })
}

fn main() -> io::Result<()> {
    let mut fnames = std::env::args_os().skip(1);
    match (fnames.next(), fnames.next()) {
        (None, _) => {
            eprintln!("Usage: cargo run FILE1 [FILE2]");
            process::exit(1);
        }
        (Some(fname1), None) => {
            println!("{}", read(fname1).join().unwrap()?.cardinality());
        }
        (Some(fname1), Some(fname2)) => {
            let sk1 = read(fname1);
            let sk2 = read(fname2);
            let mut sk1 = sk1.join().unwrap()?;
            let sk2 = sk2.join().unwrap()?;
            println!("Intersection: {}", sk1.intersection(&sk2));
            println!("Union: {}", sk1.union(&sk2).cardinality());
        }
    }

    Ok(())
}
