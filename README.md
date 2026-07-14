# Hyperminhash for Rust

[![Crates.io Version](https://img.shields.io/crates/v/hyperminhash.svg)](https://crates.io/crates/hyperminhash)
[![Docs](https://docs.rs/hyperminhash/badge.svg)](https://docs.rs/hyperminhash)
[![PyPI](https://badge.fury.io/py/pyhyperminhash.svg)](https://pypi.org/project/pyhyperminhash/)

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
Cardinality via `sort strings1.txt \| uniq \| wc -l` | 7.01 secs | 9,779,544
Union via `cat strings1.txt string2.txt \| sort \| uniq \| wc -l` | 16.19 secs | 19,130,942
Intersection via `comm -12 <(sort string1.txt) <(sort strings2.txt) \| wc -l` | 6.67 secs | 428,568
Cardinality via Hyperminhash | 0.45 secs | 9,792,822
Cardinality via Hyperminhash ([multithreaded](https://github.com/lukaslueg/hyperminhash/blob/master/examples/parallel.rs)) | 0.29 secs | 9,792,822
Union via Hyperminhash | 0.44 secs | 19,268,781
Intersection via Hyperminhash | 0.44 secs | 434,141
