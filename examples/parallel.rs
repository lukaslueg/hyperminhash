//! A slightly overengineered example of how to count unique lines on stdout
//! `cargo run --release --example parallel < file.txt`

use crossbeam::channel;
use hyperminhash::Sketch;
use std::{
    io, iter,
    sync::{Arc, Mutex},
    thread, time,
};

type Synced<T> = Arc<Mutex<T>>;
type AsyncResult<T, E> = Result<Synced<T>, E>;

#[derive(Debug)]
struct AsyncSink<T, I, E> {
    t: Vec<(Synced<T>, thread::JoinHandle<AsyncResult<T, E>>)>,
    s: channel::Sender<I>,
}

impl<T, I, E> AsyncSink<T, I, E>
where
    T: Send + 'static,
    I: Send + 'static,
    E: Send + 'static,
{
    /// Construct a threadpool with the given number of threads.
    /// Each thread constructs it's initial state from the `init`-parameter.
    /// Items sent to the sink are folded into the state using `f`.
    /// The final states, one per thread, are returned via `.join()`.
    fn new<U, V>(num_threads: usize, init: U, f: V) -> Self
    where
        U: Send + Fn() -> T + 'static,
        V: Send + Clone + Fn(&mut T, I) -> Result<(), E> + 'static,
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
        AsyncSink { t, s }
    }

    pub fn with_default<V>(f: V) -> Self
    where
        V: Send + Clone + Fn(&mut T, I) -> Result<(), E> + 'static,
        T: Default,
    {
        Self::new(8, Default::default, f)
    }

    /// Send an item to a random thread in the pool.
    pub fn send(&mut self, item: I) -> Result<(), channel::SendError<I>> {
        self.s.send(item)
    }

    /// Wait for all threads and return their final states.
    pub fn join(self) -> thread::Result<Result<Vec<T>, E>> {
        drop(self.s);
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

/// Read chunks from the given reader, split by `b'\n'`
fn lines(mut inp: impl io::BufRead) -> impl Iterator<Item = io::Result<Vec<u8>>> {
    iter::from_fn(move || {
        (|| {
            let b = inp.fill_buf()?;
            if b.is_empty() {
                return Ok(None);
            }
            let mut buf = Vec::with_capacity(b.len() + 128);
            buf.extend_from_slice(b);
            inp.consume(buf.len());
            inp.read_until(b'\n', &mut buf)?;
            Ok(Some(buf))
        })()
        .transpose()
    })
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();

    // Create a Sink which will receive chunks of data, to the utf8-decoding, split
    // the lines and feed each line into a Sketch
    let mut sink = AsyncSink::with_default(|sk: &mut Sketch, items: Vec<u8>| {
        String::from_utf8(items).map(|s| s.lines().for_each(|l| sk.add(l)))
    });

    let mut now = time::Instant::now();
    for chunk in lines(stdin.lock()) {
        sink.send(chunk?).expect("the sink stopped listening");
        // Every once in a while print a current estimate - it's cheap but not free
        if now.elapsed() > time::Duration::from_millis(5) {
            if let Some(sk) = sink.inspect(None, |sk1, sk2| match sk1 {
                None => Some(sk2.clone()),
                Some(mut sk) => {
                    sk.union(&sk2);
                    Some(sk)
                }
            }) {
                print!("\rCurrent: {:.0}", sk.cardinality());
            }
            now = time::Instant::now();
        }
    }

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
