# Hyperminhash for Rust

[![Crates.io Version](https://img.shields.io/crates/v/hyperminhash.svg)](https://crates.io/crates/hyperminhash)
[![Docs](https://docs.rs/hyperminhash/badge.svg)](https://docs.rs/hyperminhash)

A straight port of [Hyperminhash](https://github.com/axiomhq/hyperminhash) for Rust.
Very fast, constant memory-footprint cardinality approximation,
including intersection and union operation.

```rust
use std::{io, io::Bufread, fs};

let reader = io::BufReader::new(fs::File::open(fname)?).lines();
let sketch = reader.collect::<io::Result<hyperminhash::Sketch>>()?;
println!("{}", sketch.cardinality());
```

#### Two files of 10,000,000 random strings each:

Operation | Runtime | Result
----------|----------------|-------
Cardinality via `sort strings1.txt \| uniq \| wc -l` | 13.57 secs | 9,774,970
Union via `cat strings1.txt string2.txt \| sort \| uniq \| wc -l` | 84.4 secs | 19,122,087
Intersection via `comm -12 <(sort string1.txt) <(sort strings2.txt) \| wc -l` | 25.3 secs | 428,370
Cardinality via Hyperminhash | 0.69 secs | 9,861,113
Cardinality via Hyperminhash ([multithreaded](https://github.com/lukaslueg/hyperminhash/blob/master/examples/parallel.rs)) | 0.15 secs | 9,971,928
Union via Hyperminhash | 1.59 secs | 19,042,941
Intersection via Hyperminhash | 1.52 secs | 430,977
