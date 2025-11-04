//! A slightly overengineered example on how to count unique lines in a file

use crossbeam::channel;
use hyperminhash::Sketch;
use std::{
    env, io, process,
    sync::{Arc, Mutex, atomic},
    thread,
};

type Synced<T> = Arc<Mutex<T>>;
type AsyncResult<T, E> = Result<Synced<T>, E>;

#[derive(Debug)]
struct AsyncSink<T, E> {
    t: Vec<(Synced<T>, thread::JoinHandle<AsyncResult<T, E>>)>,
}

impl<T, E> AsyncSink<T, E>
where
    T: Send + 'static,
    E: Send + 'static,
{
    /// Construct a threadpool with the given number of threads.
    /// Each thread constructs it's initial state from the `init`-parameter.
    /// Items sent to the sink are folded into the state using `f`.
    /// The final states, one per thread, are returned via `.join()`.
    fn new<U, V, I>(num_threads: usize, init: U, f: V) -> (channel::Sender<I>, Self)
    where
        U: Send + Fn() -> T + 'static,
        V: Send + Clone + Fn(&mut T, I) -> Result<(), E> + 'static,
        I: Send + 'static,
    {
        assert!(num_threads > 0);
        let (s, recv) = channel::bounded(num_threads + 1);
        let t = (0..num_threads)
            .map(|_| {
                let state = Arc::new(Mutex::new(init()));
                let t_state = state.clone();
                let t_recv = recv.clone();
                let t_f = f.clone();
                (
                    state,
                    thread::spawn(move || {
                        for item in t_recv {
                            t_f(&mut t_state.lock().unwrap(), item)?;
                        }
                        Ok(t_state)
                    }),
                )
            })
            .collect();
        (s, AsyncSink { t })
    }

    pub fn with_default<V, I>(f: V) -> (channel::Sender<I>, Self)
    where
        V: Send + Clone + Fn(&mut T, I) -> Result<(), E> + 'static,
        T: Default,
        I: Send + 'static,
    {
        Self::new(8, Default::default, f)
    }

    /// Wait for all threads and return their final states.
    pub fn join(self) -> thread::Result<Result<Vec<T>, E>> {
        self.t
            .into_iter()
            .map(|t| t.1.join())
            .map(|r| {
                r.map(|sk| match sk {
                    Ok(sk) => match Arc::try_unwrap(sk) {
                        Ok(sk) => Ok(sk.into_inner().unwrap()),
                        // This is unreachable because all threads have been joined
                        // and we take self by value, so .inspect() can't see this
                        Err(_) => unreachable!(),
                    },
                    Err(e) => Err(e),
                })
            })
            .collect()
    }

    /// Folds the given closure over all states.
    pub fn inspect<A, B>(&self, state: B, f: A) -> B
    where
        A: Fn(B, &T) -> B,
    {
        self.t
            .iter()
            .fold(state, |st, t| f(st, &t.0.lock().unwrap()))
    }
}

#[derive(Debug)]
struct LineBuf<B> {
    inner: B,
}

impl<B: LineBuffered> Iterator for LineBuf<B> {
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = Vec::new();
        match self.inner.read_lines(&mut buf) {
            Ok(0) => None,
            Ok(_) => Some(Ok(buf)),
            Err(e) => Some(Err(e)),
        }
    }
}

trait LineBuffered: io::BufRead {
    fn read_lines(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let b = self.fill_buf()?;
        if b.is_empty() {
            return Ok(0);
        }
        let mut len = b.len();
        buf.reserve(len + 128);
        buf.extend_from_slice(b);
        self.consume(len);
        len += self.read_until(b'\n', buf)?;
        Ok(len)
    }

    fn line_buffered(self) -> LineBuf<Self>
    where
        Self: Sized,
    {
        LineBuf { inner: self }
    }
}

impl<T> LineBuffered for T where T: io::BufRead {}

pub fn byte_lines(inp: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut inp = inp;
    std::iter::from_fn(move || {
        if inp.is_empty() {
            return None;
        }
        let ending = memchr::memchr(b'\n', inp).unwrap_or(inp.len() - 1) + 1;
        let (mut line, rest) = inp.split_at(ending);
        inp = rest;
        if let Some(b'\n') = line.last() {
            line = &line[..line.len() - 1];
            if let Some(b'\r') = line.last() {
                line = &line[..line.len() - 1];
            }
        }
        Some(line)
    })
}

pub fn lines(inp: &str) -> impl Iterator<Item = &str> {
    byte_lines(inp.as_bytes()).map(|sl| unsafe { std::str::from_utf8_unchecked(sl) })
}

fn main() -> io::Result<()> {
    let fname = match env::args_os().nth(1) {
        Some(fname) => fname,
        None => {
            eprintln!("Usage: cargo run --release --example parallel [FILENAME]");
            process::exit(1);
        }
    };

    // Create a Sink which will receive chunks of data, do the utf8-decoding, split
    // the lines and feed each line into a Sketch
    let (sender, sink) = AsyncSink::with_default(|sk: &mut Sketch, items: Vec<u8>| {
        String::from_utf8(items).map(|s| lines(&s).for_each(|l| sk.add_bytes(l.as_bytes())))
    });

    // A seperate thread to print intermediate results, so we don't block i/o
    let shall_stop = Arc::new(atomic::AtomicBool::new(false));
    let shall_stop_c = shall_stop.clone();
    let printer = thread::spawn(move || {
        use io::Write;
        let mut stdout = io::stdout();
        while !shall_stop_c.load(atomic::Ordering::Relaxed) {
            let now = std::time::Instant::now();
            if let Some(sk) = sink.inspect(None, |sk1, sk2| match sk1 {
                None => Some(sk2.clone()),
                Some(mut sk) => {
                    sk.union(&sk2);
                    Some(sk)
                }
            }) {
                write!(stdout, "\rCurrent: {:.0}", sk.cardinality()).unwrap();
                stdout.flush().unwrap();
            }
            if let Some(elapsed) = std::time::Duration::from_millis(100).checked_sub(now.elapsed())
            {
                thread::sleep(elapsed);
            }
        }
        sink
    });

    // Main thread does i/o and feeds the sink
    let reader = io::BufReader::with_capacity(512 * 1024, std::fs::File::open(fname)?);
    for chunk in reader.line_buffered() {
        sender.send(chunk?).unwrap();
    }

    // Dropping the sender stops the sink; signal the printer to stop so we get the sink back
    drop(sender);
    shall_stop.store(true, atomic::Ordering::Relaxed);
    let sink = printer.join().unwrap();

    // Compute the total via union of all Sketches
    let final_sketch = sink
        .join()
        .expect("Another thread panicked")
        .expect("utf decoding failed")
        .into_iter()
        .fold(None, |sk1, sk2| match sk1 {
            None => Some(sk2),
            Some(mut sk) => {
                sk.union(&sk2);
                Some(sk)
            }
        })
        .unwrap();
    println!("\nTotal: {:.0}", final_sketch.cardinality());

    Ok(())
}
