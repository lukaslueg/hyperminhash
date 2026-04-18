//! Very fast, constant memory-footprint cardinality approximation, including intersection and union operation.
//! A straight port of [Hyperminhash](https://github.com/axiomhq/hyperminhash).
//!
//! As with other cardinality estimators, `Hyperminhash` has two advantages when counting very large
//! sets or streams of elements:
//! * It uses a single data structure that never grows while counting elements. The structure
//!   consumes 32kb of memory, allocated on the stack.
//! * The amount of work done for counting a marginal element stays approximately constant.
//!
//! For sets smaller than roughly 10^3 unique elements, a `std::collections::HashSet` is usually faster,
//! albeit using much more memory. When counting streams of millions of elements, `Hyperminhash` is much
//! faster and uses much less memory. The same applies when repeatedly comparing large sets: once
//! two `Sketch` values have been built, estimating their overlap is very fast and does not revisit
//! the original elements.
//!
//! ```rust
//! use hyperminhash::Sketch;
//!
//! // A `Sketch` can approximate the unique count of elements it has seen over it's lifetime.
//! let mut sk = Sketch::default();
//!
//! // After initialization, a `Sketch` will never allocate.
//!
//! // Elements added to Sketch need to implement `std::hash::Hash`
//! sk.add("foobar");
//! sk.add(1);
//! sk.add(2);
//! sk.add(vec![3, 4, 5]);
//! for i in 0..5000 {
//!     sk.add(i);
//! }
//! // There will be some error in the count
//! assert!(sk.cardinality() > 4_900.0 && sk.cardinality() < 5_100.0);
//!
//!
//! // Using `std::iter::FromIterator`
//! let sketch1 = (0..10_000).collect::<Sketch>();
//! let sketch2 = (5_000..15_000).collect::<Sketch>();
//! assert!(sketch1.cardinality() > 9_800.0 && sketch1.cardinality() < 10_200.0);
//! assert!(sketch2.cardinality() > 9_800.0 && sketch2.cardinality() < 10_200.0);
//!
//! // The intersection of two sketches yields the approximate number of unique
//! // elements that are in both sets.
//! let i = sketch1.intersection(&sketch2);
//! assert!(i > 4_800.0 && i < 5_200.0);
//!
//! // Comparing both sets, the number of elements that are in both sets is roughly 33%
//! let j = sketch1.similarity(&sketch2);
//! assert!(j > 0.33 && j < 0.34);
//!
//! // Merge sketch1 with sketch2
//! let mut sketch1 = sketch1;
//! sketch1.union(&sketch2);
//! assert!(sketch1.cardinality() > 14_800.0 && sketch1.cardinality() < 15_200.0);
//!
//!
//! // Sketches can be serialized/deserialized from any `io::Read`/`io::Write`.
//! let mut buffer = Vec::new();
//! sketch1.save(&mut buffer).expect("Failed to write");
//!
//! let sketch2 = Sketch::load(&buffer[..]).expect("Failed to read");
//! assert_eq!(sketch1.cardinality(), sketch2.cardinality());
//!
//!
//! // Use custom seed-values to distinguish elements that ordinarily
//! // hash equally.
//! const WHITE_SEED: u64 = 0xD1CE_5EED_1234_5678;
//! const BLUE_SEED: u64 = 0xDEAD_BEEF_1234_5678;
//! let mut sk_seeded = Sketch::default();
//! sk_seeded.add_with_seed("foo", WHITE_SEED);
//! sk_seeded.add_with_seed("foo", BLUE_SEED);
//! assert!(sk_seeded.cardinality() > 1.0);
//! ```

use std::{hash, io};

const EMPTY_HASH: u128 = 0x99aa06d3014798d86001c324468d497f;

const P: u32 = 14;
const M: u32 = 1 << P;
const MAX: u32 = 64 - P;
const MAXX: u64 = u64::MAX >> MAX;
const ALPHA: f64 = 0.7213 / (1f64 + 1.079 / (M as f64));
const Q: u8 = 6;
const R: u8 = 10;
const TQ: u32 = 1 << Q;
const TR: u32 = 1 << R;
const C: f64 = 0.169_919_487_159_739_1;

type Regs = [u16; M as usize];

// Lookup table of 2^-i for i in 0..64, used to accumulate register contributions
// during cardinality / similarity estimation. Built at compile time via
// `f64::from_bits` (const since 1.83): for i in 0..=63 the biased exponent is
// 1023 - i (all normal, no subnormals) and the mantissa is 0, so each entry is
// an exact IEEE-754 power of two on every target.
//
// Note: L[0] == 1.0, which lets the hot loops accumulate the "leading-zeros
// is zero" case branchlessly as `sum += L[lz]` rather than a conditional.
static L: [f64; 64] = {
    let mut l = [0.0f64; 64];
    let mut i = 0u64;
    while i < 64 {
        l[i as usize] = f64::from_bits((1023 - i) << 52);
        i += 1;
    }
    l
};

fn beta(ez: u16) -> f64 {
    let zl = (f64::from(ez) + 1.0).ln();
    -0.370_393_911 * f64::from(ez)
        + 0.070_471_823 * zl
        + 0.173_936_86 * zl.powi(2)
        + 0.163_398_39 * zl.powi(3)
        + -0.092_377_45 * zl.powi(4)
        + 0.037_380_27 * zl.powi(5)
        + -0.005_384_159 * zl.powi(6)
        + 0.000_424_19 * zl.powi(7)
}

static TBL: std::sync::LazyLock<EcTable> = std::sync::LazyLock::new(EcTable::new);

struct EcTable {
    ln1p_neg_b1: Box<[[f64; TR as usize]; TQ as usize]>,
    ln1p_neg_b2: Box<[[f64; TR as usize]; TQ as usize]>,
}

impl EcTable {
    fn new() -> Self {
        let mut ln1p_neg_b1 = Box::new([[0.0; TR as usize]; TQ as usize]);
        let mut ln1p_neg_b2 = Box::new([[0.0; TR as usize]; TQ as usize]);

        // Store the inclusive i in 1..=TQ and j in 1..=TR ranges using
        // zero-based row/column indexes.
        for i in 1..=TQ {
            let row = (i as usize) - 1;
            if i != TQ {
                let den = 2f64.powf(f64::from(P) + f64::from(R) + f64::from(i));
                for j1 in 1..=TR {
                    let col = (j1 as usize) - 1;
                    let j = f64::from(j1);
                    let b1 = (f64::from(TR) + j) / den;
                    let b2 = (f64::from(TR) + j + 1.0) / den;
                    ln1p_neg_b1[row][col] = f64::ln_1p(-b1);
                    ln1p_neg_b2[row][col] = f64::ln_1p(-b2);
                }
            } else {
                let den = 2f64.powf(f64::from(P) + f64::from(R) + f64::from(i) - 1.0);
                for j1 in 1..=TR {
                    let col = (j1 as usize) - 1;
                    let j = f64::from(j1);
                    let b1 = j / den;
                    let b2 = (j + 1.0) / den;
                    ln1p_neg_b1[row][col] = f64::ln_1p(-b1);
                    ln1p_neg_b2[row][col] = f64::ln_1p(-b2);
                }
            }
        }

        EcTable {
            ln1p_neg_b1,
            ln1p_neg_b2,
        }
    }
}

/// A single (possibly unique) object yet-to-be-added to a `Sketch`.
///
/// Use `Entry` if what should constitute a single objects in a `Sketch` is comprised of multiple
/// parts. For example, if the object can't fit in memory, you can create an `Entry`-object and
/// add multiple parts one by one.
///
/// One can use the `Clone`-implementation to effectively fork a larger object, which saves the
/// cost to re-hash common prefixes.
///
/// ```rust
/// let mut sk = hyperminhash::Sketch::new();
///
/// let mut user = hyperminhash::Entry::new();
/// user.add("User1");
///
/// let mut page1 = user.clone();
/// page1.add("Page1");
/// sk.add_entry(&page1);
///
/// let mut page2 = user;
/// page2.add("Page2");
/// sk.add_entry(&page2);
///
/// // `sk` now effectively contains `("User1", "Page1")` and `("User1", "Page2")`
///
/// assert!(sk.cardinality() > 1.0)
/// ```
#[derive(Clone, Default)]
pub struct Entry {
    hasher: xxhash_rust::xxh3::Xxh3Default,
}

impl Entry {
    /// Construct a new `Entry`.
    ///
    /// Notice that a new (empty) `Entry` is considered a unique object in a `Sketch`. All empty
    /// `Entry` are the same, though.
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    /// assert!(sk.is_empty());
    ///
    /// let e = hyperminhash::Entry::new();
    /// assert!(e.is_empty());
    ///
    /// sk.add_entry(&e);
    /// assert!(!sk.is_empty());
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Add an element to this `Entry` using the element's Hash-implementation.
    ///
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// let mut e = hyperminhash::Entry::new();
    /// e.add(42);
    /// e.add("The answer");
    ///
    /// sk.add_entry(&e);
    /// assert!(!sk.is_empty());
    /// assert!(sk.cardinality() < 2.0);
    /// ```
    pub fn add(&mut self, v: impl hash::Hash) {
        v.hash(&mut self.hasher);
    }

    /// Add data to this `Entry` in the form of raw bytes.
    ///
    /// This is different from using `.add::<&[u8]>()`: Most implementations of `std::hash::Hash` are
    /// guaranteed to be prefix-collision-free, which means that all variations of `(a, b, c)`,
    /// `(ab, c)`, `(a, bc)`, `...` are considered unique. This may not be desired when
    /// constructing what should be considered a single object from multiple parts.
    /// ```rust
    /// let a = "a".as_bytes();
    /// let bc = "bc".as_bytes();
    /// let abc = "abc".as_bytes();
    ///
    /// // Two elements: `a` and `bc`
    /// let mut e0 = hyperminhash::Entry::new();
    /// e0.add(a);
    /// e0.add(bc);
    ///
    /// // Two elements: `a` and `bc`, but as raw bytes
    /// let mut e1 = hyperminhash::Entry::new();
    /// e1.add_bytes(a);
    /// e1.add_bytes(bc);
    ///
    /// // One element: `abc` as raw bytes
    /// let mut e2 = hyperminhash::Entry::new();
    /// e2.add_bytes(abc);
    ///
    /// assert_ne!(e0, e1); // `a bc` is not the same as `a bc`
    /// assert_ne!(e0, e2); // `a bc` is not the same as `abc`
    /// assert_eq!(e1, e2); // `a bc` is the same as `abc`
    /// ```
    pub fn add_bytes(&mut self, v: &[u8]) {
        self.hasher.update(v);
    }

    /// Add an element using the content of the given `io::Read`
    ///
    /// # Errors
    /// Returns I/O errors that occured while reading
    ///
    /// ```rust
    /// let mut e = hyperminhash::Entry::new();
    ///
    /// e.add_reader(std::io::empty());
    /// ```
    pub fn add_reader(&mut self, mut r: impl io::Read) -> io::Result<u64> {
        io::copy(&mut r, &mut self.hasher)
    }

    /// Returns `true` if this `Entry` has a cardinality of exactly zero
    /// ```rust
    /// let mut e = hyperminhash::Entry::new();
    /// assert!(e.is_empty());
    /// e.add(42);
    /// assert!(!e.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.hasher.digest128() == EMPTY_HASH
    }

    #[doc(hidden)]
    pub fn digest(&self) -> u128 {
        self.hasher.digest128()
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.hasher.digest128() == other.hasher.digest128()
    }
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("digest", &self.hasher.digest128())
            .finish()
    }
}

/// Records the approximate number of unique elements it has seen over it's lifetime.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Sketch {
    regs: Regs,
}

impl std::fmt::Debug for Sketch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Sketch")
            .field("cardinality", &self.cardinality())
            .finish()
    }
}

impl Default for Sketch {
    fn default() -> Self {
        Self {
            regs: [0; M as usize],
        }
    }
}

impl<T: hash::Hash> std::iter::FromIterator<T> for Sketch {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut sk = Self::default();
        iter.into_iter().for_each(|v| sk.add(v));
        sk
    }
}

impl PartialOrd for Sketch {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cardinality().partial_cmp(&other.cardinality())
    }
}

impl Sketch {
    /// Construct an empty `Sketch`
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if this Sketch has a cardinality of exactly zero
    ///
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// assert!(sk.is_empty());
    ///
    /// sk.add(0);
    /// assert!(!sk.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regs.iter().all(|r| *r == 0)
    }

    fn new_reg(lz: u8, sig: u16) -> u16 {
        (u16::from(lz) << R) | sig
    }

    fn lz(reg: u16) -> u8 {
        (reg >> (16 - Q)) as u8
    }

    fn add_hash(&mut self, h: u128) {
        let x: u64 = h as u64;
        let y: u64 = (h >> 64) as u64;
        let k = x >> MAX;
        let lz = ((x << P) ^ MAXX).leading_zeros() as u8 + 1;
        let sig = (y << (64 - R) >> (64 - R)) as u16;
        let reg = Self::new_reg(lz, sig);
        if self.regs[k as usize] < reg {
            self.regs[k as usize] = reg;
        }
    }

    /// Add an element to this Sketch using the element's Hash-implementation
    ///
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// sk.add(42);
    /// sk.add("The answer");
    /// sk.add(vec![1, 2, 3]);
    /// ```
    pub fn add(&mut self, v: impl hash::Hash) {
        let mut hasher = xxhash_rust::xxh3::Xxh3Default::new();
        v.hash(&mut hasher);
        self.add_hash(hasher.digest128());
    }

    /// Add a single element using the content of the given `io::Read`
    ///
    /// # Errors
    /// Returns I/O errors that occured while reading
    ///
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// sk.add_reader(std::io::empty());
    /// ```
    pub fn add_reader(&mut self, mut r: impl io::Read) -> io::Result<u64> {
        let mut hasher = xxhash_rust::xxh3::Xxh3Default::new();
        let read = io::copy(&mut r, &mut hasher)?;
        self.add_hash(hasher.digest128());
        Ok(read)
    }

    /// Add a single element given by raw bytes to this Sketch
    ///
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// let buf: [u8; _] = [1, 2, 3];
    /// sk.add_bytes(&buf);
    /// ```
    pub fn add_bytes(&mut self, v: &[u8]) {
        self.add_hash(xxhash_rust::xxh3::xxh3_128(v));
    }

    /// Add a single element to this `Sketch`
    ///
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// let mut e = hyperminhash::Entry::new();
    /// e.add(42);
    /// e.add("The answer");
    /// sk.add_entry(&e);
    ///
    /// e.add(b"Even more");
    /// sk.add_entry(&e);
    ///
    /// // The `Sketch` now contains `(42, "The answer")` and `(42, "The answer", b"Even more")`
    /// assert!(sk.cardinality() > 1.0);
    /// ```
    pub fn add_entry(&mut self, entry: &Entry) {
        self.add_hash(entry.hasher.digest128())
    }

    /// Add an element to this Sketch using the element's Hash-implementation and a seed-value
    /// Elements that hash equally but use different seed values are seen as unique elements.
    ///
    /// ```rust
    /// const KILOGRAM: u64 = 1;
    /// const POUNDS: u64 = 2;
    ///
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// sk.add_with_seed(100, KILOGRAM);
    /// sk.add_with_seed(100, POUNDS);
    ///
    /// assert!(sk.cardinality() > 1.0);
    /// ```
    pub fn add_with_seed(&mut self, v: impl hash::Hash, seed: u64) {
        // Streaming hasher seeded:
        let mut hasher = xxhash_rust::xxh3::Xxh3::with_seed(seed);
        v.hash(&mut hasher);
        self.add_hash(hasher.digest128());
    }

    /// Add a single element given by raw bytes and a see value to this Sketch
    /// Elements that hash equally but use different seed values are seen as unique elements.
    pub fn add_bytes_with_seed(&mut self, v: &[u8], seed: u64) {
        self.add_hash(xxhash_rust::xxh3::xxh3_128_with_seed(v, seed));
    }

    /// Add a single element using the content of the given `io::Read` and a seed value
    ///
    /// # Errors
    /// Returns I/O errors that occured while reading
    ///
    pub fn add_reader_with_seed(&mut self, mut r: impl io::Read, seed: u64) -> io::Result<u64> {
        let mut hasher = xxhash_rust::xxh3::Xxh3::with_seed(seed);
        let read = io::copy(&mut r, &mut hasher)?;
        self.add_hash(hasher.digest128());
        Ok(read)
    }

    fn sum_and_zeros(&self) -> (f64, u16) {
        // Accumulate across independent lanes to break the serial f64-add
        // dependency chain. f64 addition is non-associative so the compiler
        // cannot do this on its own. M = 16384 is a multiple of LANES,
        // so no remainder handling is needed. L[0] == 1.0 makes the
        // `lz == 0` case a branchless `sum += L[0]`.
        const LANES: usize = 8;
        let mut sums = [0.0f64; LANES];
        let mut ezs = [0u16; LANES];
        for chunk in self.regs.chunks_exact(LANES) {
            for i in 0..LANES {
                let lz = Self::lz(chunk[i]);
                sums[i] += L[lz as usize];
                ezs[i] += u16::from(lz == 0);
            }
        }
        let sum: f64 = sums.iter().sum();
        let ez: u16 = ezs.iter().sum();
        (sum, ez)
    }

    /// The approximate number of unique elements in the set.
    ///
    /// ```rust
    /// let mut sk = hyperminhash::Sketch::new();
    ///
    /// assert_eq!(sk.cardinality(), 0.0);
    ///
    /// for e in [1, 2, 3, 4, 5] {
    ///     sk.add(e);
    /// }
    /// assert!(sk.cardinality() > 4.0);
    /// assert!(sk.cardinality() < 6.0);
    /// ```
    #[must_use]
    pub fn cardinality(&self) -> f64 {
        let (sum, ez) = self.sum_and_zeros();
        ALPHA * (f64::from(M)) * ((f64::from(M)) - f64::from(ez)) / (beta(ez) + sum)
    }

    /// Merge two sets, resulting in this set becoming the union-set.
    ///
    /// ```rust
    /// let mut sk1 = hyperminhash::Sketch::new();
    /// sk1.add(1);
    /// sk1.add(2);
    ///
    /// let mut sk2 = hyperminhash::Sketch::new();
    /// sk2.add(3);
    /// sk2.add(4);
    ///
    /// sk1.union(&sk2);
    /// assert_eq!(sk1, (1..=4).collect::<hyperminhash::Sketch>());
    /// ```
    pub fn union<'a>(&'a mut self, other: &Self) -> &'a Self {
        for (r, rr) in self.regs.iter_mut().zip(other.regs.iter()) {
            if *r < *rr {
                *r = *rr;
            }
        }
        self
    }

    fn cardinality_from_parts(sum: f64, ez: u16) -> f64 {
        ALPHA * (f64::from(M)) * ((f64::from(M)) - f64::from(ez)) / (beta(ez) + sum)
    }

    fn overlap_counts(&self, other: &Self) -> (u32, u32) {
        // Branchless, fixed-chunk formulation to encourage LLVM to emit
        // u16-wide SIMD (NEON `cmeq`/`sub`, SSE2 `pcmpeqw`/`psubw`).
        // Per-lane u16 accumulators cap at `M / LANES = 1024` which fits
        // in u16 with headroom.
        const LANES: usize = 16;
        let mut cc_lanes = [0u16; LANES];
        let mut cn_lanes = [0u16; LANES];
        for (lc, rc) in self
            .regs
            .chunks_exact(LANES)
            .zip(other.regs.chunks_exact(LANES))
        {
            for i in 0..LANES {
                let l = lc[i];
                let r = rc[i];
                cn_lanes[i] += u16::from((l | r) != 0);
                cc_lanes[i] += u16::from((l == r) & (l != 0));
            }
        }
        let cc: u32 = cc_lanes.iter().map(|&x| u32::from(x)).sum();
        let cn: u32 = cn_lanes.iter().map(|&x| u32::from(x)).sum();
        (cc, cn)
    }

    fn comparison_stats(&self, other: &Self) -> ComparisonStats {
        // Two-phase per chunk:
        //   Phase A (u16-only, auto-vectorizable): compute `union_reg = max`,
        //   extract the three lz values, and accumulate `ez`, `cn`, `cc`.
        //   All ops are element-wise u16 arithmetic / compares, which LLVM
        //   can lower to NEON `umax`/`cmeq`/`sub` or SSE2 `pmaxuw`/`pcmpeqw`.
        //
        //   Phase B (scalar FP gather + multi-lane fadd): index `L[]` with
        //   the lz values staged in Phase A and accumulate into independent
        //   lanes to hide fadd latency (non-associative, compiler can't do
        //   this on its own).
        //
        // M is a multiple of LANES, so no remainder handling is needed.
        const LANES: usize = 8;

        let mut left_sums = [0.0f64; LANES];
        let mut right_sums = [0.0f64; LANES];
        let mut union_sums = [0.0f64; LANES];
        let mut left_ez_lanes = [0u16; LANES];
        let mut right_ez_lanes = [0u16; LANES];
        let mut union_ez_lanes = [0u16; LANES];
        let mut cc_lanes = [0u16; LANES];
        let mut cn_lanes = [0u16; LANES];

        for (lc, rc) in self
            .regs
            .chunks_exact(LANES)
            .zip(other.regs.chunks_exact(LANES))
        {
            // Phase A: pure u16 work.
            let mut left_lz = [0u8; LANES];
            let mut right_lz = [0u8; LANES];
            let mut union_lz = [0u8; LANES];
            for i in 0..LANES {
                let l = lc[i];
                let r = rc[i];
                let u = l.max(r);
                left_lz[i] = Self::lz(l);
                right_lz[i] = Self::lz(r);
                union_lz[i] = Self::lz(u);
                left_ez_lanes[i] += u16::from(l < (1u16 << R));
                right_ez_lanes[i] += u16::from(r < (1u16 << R));
                union_ez_lanes[i] += u16::from(u < (1u16 << R));
                cn_lanes[i] += u16::from((l | r) != 0);
                cc_lanes[i] += u16::from((l == r) & (l != 0));
            }

            // Phase B: scalar gather + fadd per lane.
            for i in 0..LANES {
                left_sums[i] += L[left_lz[i] as usize];
                right_sums[i] += L[right_lz[i] as usize];
                union_sums[i] += L[union_lz[i] as usize];
            }
        }

        ComparisonStats {
            left_sum: left_sums.iter().sum(),
            right_sum: right_sums.iter().sum(),
            union_sum: union_sums.iter().sum(),
            left_ez: left_ez_lanes.iter().sum(),
            right_ez: right_ez_lanes.iter().sum(),
            union_ez: union_ez_lanes.iter().sum(),
            cc: cc_lanes.iter().map(|&x| u32::from(x)).sum(),
            cn: cn_lanes.iter().map(|&x| u32::from(x)).sum(),
        }
    }

    fn approximate_expected_collisions(n: f64, m: f64) -> f64 {
        let (n, m) = (n.max(m), n.min(m));
        if n > 2f64.powf(2f64.powf(f64::from(Q)) + f64::from(R)) {
            f64::INFINITY
        } else if n > 2f64.powf(f64::from(P) + 5.0) {
            let d = (4.0 * n / m) / ((1.0 + n) / m).powi(2);
            C * 2f64.powf(f64::from(P) - f64::from(R)) * d
        } else {
            Self::expected_collisions(n, m) / f64::from(P)
        }
    }

    fn expected_collisions(n: f64, m: f64) -> f64 {
        let tbl = &*TBL;

        let mut x = 0.0;
        for i in 1..=TQ {
            let row = (i as usize) - 1;
            let l1 = &tbl.ln1p_neg_b1[row];
            let l2 = &tbl.ln1p_neg_b2[row];

            for j1 in 1..=TR {
                let col = (j1 as usize) - 1;
                // (1 - b)^n = exp(n * ln1p(-b))
                let a1 = (n * l2[col]).exp();
                let a0 = (n * l1[col]).exp();
                let b1 = (m * l2[col]).exp();
                let b0 = (m * l1[col]).exp();
                x += (a1 - a0) * (b1 - b0);
            }
        }

        x * f64::from(P)
    }

    // Precompute `(n * ln1p_neg_b2).exp() - (n * ln1p_neg_b1).exp()` for every
    // cell of the `EcTable`. Bit-identical to the `a1 - a0` subexpression in
    // `expected_collisions`. Cost: ~131K `exp()` calls, 512 KB heap.
    fn precompute_a_diff(n: f64) -> Box<ADiff> {
        let tbl = &*TBL;
        let mut out: Box<ADiff> = Box::new([[0.0; TR as usize]; TQ as usize]);
        for i in 1..=TQ {
            let row = (i as usize) - 1;
            let l1 = &tbl.ln1p_neg_b1[row];
            let l2 = &tbl.ln1p_neg_b2[row];
            let o = &mut out[row];
            for j1 in 1..=TR {
                let col = (j1 as usize) - 1;
                let a1 = (n * l2[col]).exp();
                let a0 = (n * l1[col]).exp();
                o[col] = a1 - a0;
            }
        }
        out
    }

    // `expected_collisions` variant that consumes a precomputed `a_diff` for
    // one side (self) and only does the `exp()` work for the other side.
    // Preserves row-major `(i, j1)` accumulation order and per-cell multiply
    // shape of `expected_collisions`, so the resulting `f64` is bit-identical
    // regardless of which side (self/other) is the larger cardinality (IEEE
    // float multiplication is commutative).
    fn expected_collisions_with_a_diff(a_diff: &ADiff, m: f64) -> f64 {
        let tbl = &*TBL;
        let mut x = 0.0;
        for i in 1..=TQ {
            let row = (i as usize) - 1;
            let l1 = &tbl.ln1p_neg_b1[row];
            let l2 = &tbl.ln1p_neg_b2[row];
            let ad = &a_diff[row];
            for j1 in 1..=TR {
                let col = (j1 as usize) - 1;
                let b1 = (m * l2[col]).exp();
                let b0 = (m * l1[col]).exp();
                x += ad[col] * (b1 - b0);
            }
        }
        x * f64::from(P)
    }

    fn approximate_expected_collisions_with_a_diff(
        a_diff: &ADiff,
        n_self: f64,
        m_other: f64,
    ) -> f64 {
        let (big, small) = (n_self.max(m_other), n_self.min(m_other));
        if big > 2f64.powf(2f64.powf(f64::from(Q)) + f64::from(R)) {
            f64::INFINITY
        } else if big > 2f64.powf(f64::from(P) + 5.0) {
            let d = (4.0 * big / small) / ((1.0 + big) / small).powi(2);
            C * 2f64.powf(f64::from(P) - f64::from(R)) * d
        } else {
            Self::expected_collisions_with_a_diff(a_diff, m_other) / f64::from(P)
        }
    }

    // Right-side + union analogue of `comparison_stats` — omits `left_sum` /
    // `left_ez` since the caller already has them from `sum_and_zeros(self)`.
    // Preserves per-lane accumulation shape of `comparison_stats` so per-field
    // values match bit-for-bit.
    #[inline(always)]
    fn right_and_union_stats(&self, other: &Self) -> PartialStats {
        const LANES: usize = 8;

        let mut right_sums = [0.0f64; LANES];
        let mut union_sums = [0.0f64; LANES];
        let mut right_ez_lanes = [0u16; LANES];
        let mut union_ez_lanes = [0u16; LANES];
        let mut cc_lanes = [0u16; LANES];
        let mut cn_lanes = [0u16; LANES];

        for (lc, rc) in self
            .regs
            .chunks_exact(LANES)
            .zip(other.regs.chunks_exact(LANES))
        {
            let mut right_lz = [0u8; LANES];
            let mut union_lz = [0u8; LANES];
            for i in 0..LANES {
                let l = lc[i];
                let r = rc[i];
                let u = l.max(r);
                right_lz[i] = Self::lz(r);
                union_lz[i] = Self::lz(u);
                right_ez_lanes[i] += u16::from(r < (1u16 << R));
                union_ez_lanes[i] += u16::from(u < (1u16 << R));
                cn_lanes[i] += u16::from((l | r) != 0);
                cc_lanes[i] += u16::from((l == r) & (l != 0));
            }

            for i in 0..LANES {
                right_sums[i] += L[right_lz[i] as usize];
                union_sums[i] += L[union_lz[i] as usize];
            }
        }

        PartialStats {
            right_sum: right_sums.iter().sum(),
            right_ez: right_ez_lanes.iter().sum(),
            union_sum: union_sums.iter().sum(),
            union_ez: union_ez_lanes.iter().sum(),
            cc: cc_lanes.iter().map(|&x| u32::from(x)).sum(),
            cn: cn_lanes.iter().map(|&x| u32::from(x)).sum(),
        }
    }

    fn similarity_impl(&self, other: &Self, high_precision: bool) -> f64 {
        let (cc, cn, ec) = if high_precision {
            let stats = self.comparison_stats(other);
            let n = Self::cardinality_from_parts(stats.left_sum, stats.left_ez);
            let m = Self::cardinality_from_parts(stats.right_sum, stats.right_ez);
            (
                stats.cc,
                stats.cn,
                Self::approximate_expected_collisions(n, m),
            )
        } else {
            let (cc, cn) = self.overlap_counts(other);
            (cc, cn, 0.0)
        };

        if cc == 0 {
            return 0.0;
        }

        if (cc as f64) < ec {
            return 0.0;
        }
        (cc as f64 - ec) / cn as f64
    }

    /// The Jaccard Index similarity estimation
    ///
    /// For pairs whose larger estimated cardinality exceeds `2^(P + 5) =
    /// 524,288`, this uses a cheaper approximation than it does for smaller
    /// sketches. It still does more correction work than
    /// [`Sketch::similarity_fast`], though, so the two methods are not
    /// identical above that threshold.
    ///
    /// ```rust
    /// let sk1 = (0..=75).collect::<hyperminhash::Sketch>();
    /// let sk2 = (50..=125).collect::<hyperminhash::Sketch>();
    /// assert!((sk1.similarity(&sk2) - (25.0 / 125.0)).abs() < 1e-2);
    /// ```
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        if self == other {
            return 1.0;
        }
        self.similarity_impl(other, true)
    }

    /// A faster Jaccard Index estimate with a slightly looser correction model.
    ///
    /// In sampled inputs across the small-cardinality range where this differs
    /// from [`Sketch::similarity`], the absolute drift stayed below `1.3e-5`.
    #[must_use]
    pub fn similarity_fast(&self, other: &Self) -> f64 {
        if self == other {
            return 1.0;
        }
        self.similarity_impl(other, false)
    }

    /// The approximate number of elements in both sets
    ///
    /// This follows the same correction strategy as [`Sketch::similarity`]:
    /// when the larger estimated cardinality exceeds `2^(P + 5) = 524,288`,
    /// it switches away from the expensive small-sketch correction, but still
    /// does more work than [`Sketch::intersection_fast`].
    ///
    /// ```rust
    /// let sk1 = (0..=750).collect::<hyperminhash::Sketch>();
    /// let sk2 = (500..=1250).collect::<hyperminhash::Sketch>();
    /// assert!((sk1.intersection(&sk2) - 250.0).abs() < 1.0);
    /// ```
    #[must_use]
    pub fn intersection(&self, other: &Self) -> f64 {
        if self == other {
            return self.cardinality();
        }
        let stats = self.comparison_stats(other);
        if stats.cc == 0 {
            return 0.0;
        }

        let n = Self::cardinality_from_parts(stats.left_sum, stats.left_ez);
        let m = Self::cardinality_from_parts(stats.right_sum, stats.right_ez);
        let ec = Self::approximate_expected_collisions(n, m);
        if (stats.cc as f64) < ec {
            return 0.0;
        }

        let similarity = (stats.cc as f64 - ec) / stats.cn as f64;
        let union_card = Self::cardinality_from_parts(stats.union_sum, stats.union_ez);
        similarity * union_card
    }

    /// A faster intersection estimate with the same looser correction model as
    /// [`Sketch::similarity_fast`].
    ///
    /// In sampled inputs across the small-cardinality range, the underlying
    /// similarity drift stayed below `1.3e-5`, so the resulting intersection
    /// drift is typically tiny as well.
    #[must_use]
    pub fn intersection_fast(&self, other: &Self) -> f64 {
        if self == other {
            return self.cardinality();
        }
        let stats = self.comparison_stats(other);
        if stats.cc == 0 {
            return 0.0;
        }

        let similarity = stats.cc as f64 / stats.cn as f64;
        let union_card = Self::cardinality_from_parts(stats.union_sum, stats.union_ez);
        similarity * union_card
    }

    /// Compare `self` against each `Sketch` in `others`, yielding
    /// [`Sketch::similarity`] values in order.
    ///
    /// Results are bit-identical to calling [`Sketch::similarity`] on each
    /// pair. Using this method may amortize some of the cost when compareing
    /// one `Sketch` against many others.
    ///
    /// ```rust
    /// let a: hyperminhash::Sketch = (0..1_000).collect();
    /// let others: Vec<hyperminhash::Sketch> = (0..4)
    ///     .map(|k| ((k * 250)..(k * 250 + 1_000)).collect())
    ///     .collect();
    /// let got: Vec<f64> = a.similarity_many(others.iter()).collect();
    /// let want: Vec<f64> = others.iter().map(|o| a.similarity(o)).collect();
    /// assert_eq!(got, want);
    /// ```
    pub fn similarity_many<'a, I>(&'a self, others: I) -> BatchIter<'a, I::IntoIter>
    where
        I: IntoIterator<Item = &'a Sketch>,
        I::IntoIter: 'a,
    {
        BatchIter::new(self, others.into_iter(), BatchKind::Similarity, true)
    }

    /// Batched [`Sketch::similarity_fast`].
    ///
    /// Provided for API symmetry with [`Sketch::similarity_many`] — results
    /// are bit-identical to calling [`Sketch::similarity_fast`] on each pair,
    /// **but this method is not faster than the per-pair loop**.
    pub fn similarity_many_fast<'a, I>(&'a self, others: I) -> BatchIter<'a, I::IntoIter>
    where
        I: IntoIterator<Item = &'a Sketch>,
        I::IntoIter: 'a,
    {
        BatchIter::new(self, others.into_iter(), BatchKind::Similarity, false)
    }

    /// Batched [`Sketch::intersection`]. Bit-identical to the per-pair call.
    ///
    /// See [`Sketch::similarity_many`] for details.
    pub fn intersection_many<'a, I>(&'a self, others: I) -> BatchIter<'a, I::IntoIter>
    where
        I: IntoIterator<Item = &'a Sketch>,
        I::IntoIter: 'a,
    {
        BatchIter::new(self, others.into_iter(), BatchKind::Intersection, true)
    }

    /// Batched [`Sketch::intersection_fast`].
    ///
    /// Provided for API symmetry — results are bit-identical to calling
    /// [`Sketch::intersection_fast`] on each pair, **but this method is not
    /// faster than the per-pair loop**.
    pub fn intersection_many_fast<'a, I>(&'a self, others: I) -> BatchIter<'a, I::IntoIter>
    where
        I: IntoIterator<Item = &'a Sketch>,
        I::IntoIter: 'a,
    {
        BatchIter::new(self, others.into_iter(), BatchKind::Intersection, false)
    }

    /// Serialize this Sketch to the given writer
    ///
    /// # Errors
    /// Returns I/O errors that occured while writing
    ///
    /// ```rust
    /// let sk: hyperminhash::Sketch = (0..100).collect();
    ///
    /// let mut buffer = Vec::new();
    /// sk.save(&mut buffer).expect("Failed to write");
    /// ```
    pub fn save(&self, mut writer: impl std::io::Write) -> std::io::Result<()> {
        for r in self.regs {
            writer.write_all(&r.to_le_bytes())?;
        }
        Ok(())
    }

    /// Deserialize a Sketch from the given reader
    ///
    /// # Errors
    /// Returns I/O errors that occured while reading
    ///
    /// ```rust
    /// let reader = std::io::repeat(0);
    ///
    /// let sk = hyperminhash::Sketch::load(reader).expect("Failed to load");
    /// assert!(sk.is_empty());
    /// ```
    pub fn load(mut reader: impl std::io::Read) -> std::io::Result<Self> {
        let mut regs = [0; M as usize];
        let mut buf = [0u8; 2];
        for r in &mut regs {
            reader.read_exact(&mut buf)?;
            *r = u16::from_le_bytes(buf);
        }
        Ok(Self { regs })
    }
}

type ADiff = [[f64; TR as usize]; TQ as usize];

struct PartialStats {
    cc: u32,
    cn: u32,
    right_sum: f64,
    right_ez: u16,
    union_sum: f64,
    union_ez: u16,
}

#[derive(Copy, Clone)]
enum BatchKind {
    Similarity,
    Intersection,
}

struct LeftCache {
    sum: f64,
    ez: u16,
    n: f64,
}

/// Iterator returned by [`Sketch::similarity_many`],
/// [`Sketch::similarity_many_fast`], [`Sketch::intersection_many`], and
/// [`Sketch::intersection_many_fast`].
pub struct BatchIter<'a, I: Iterator<Item = &'a Sketch>> {
    left: &'a Sketch,
    others: I,
    kind: BatchKind,
    high_precision: bool,
    left_cache: Option<LeftCache>,
    a_diff: Option<Box<ADiff>>,
}

impl<'a, I: Iterator<Item = &'a Sketch>> BatchIter<'a, I> {
    fn new(left: &'a Sketch, others: I, kind: BatchKind, high_precision: bool) -> Self {
        Self {
            left,
            others,
            kind,
            high_precision,
            left_cache: None,
            a_diff: None,
        }
    }
}

impl<'a, I: Iterator<Item = &'a Sketch>> Iterator for BatchIter<'a, I> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let other = self.others.next()?;

        // The batch path only materially wins when a pair hits the slow
        // exp-loop branch of `approximate_expected_collisions` — that's where
        // the ~131K `exp()` hoist lives. For fast-path calls (no `ec`
        // correction) and for pairs that clearly land in the closed-form
        // branch, the pairwise methods are already tightly optimized.
        // Delegate to them and skip the cache init entirely. This keeps
        // `_fast` variants and large-sketch `_many` calls close to per-pair
        // performance.
        if !self.high_precision {
            return Some(match self.kind {
                BatchKind::Similarity => self.left.similarity_fast(other),
                BatchKind::Intersection => self.left.intersection_fast(other),
            });
        }

        let cache = self.left_cache.get_or_insert_with(|| {
            let (sum, ez) = self.left.sum_and_zeros();
            let n = Sketch::cardinality_from_parts(sum, ez);
            LeftCache { sum, ez, n }
        });

        if self.left == other {
            return Some(match self.kind {
                BatchKind::Similarity => 1.0,
                BatchKind::Intersection => Sketch::cardinality_from_parts(cache.sum, cache.ez),
            });
        }

        // If self is already above the slow-path threshold, every pair will
        // take the closed-form branch — no amortization to extract.
        let slow_thresh = 2f64.powf(f64::from(P) + 5.0);
        if cache.n > slow_thresh {
            return Some(match self.kind {
                BatchKind::Similarity => self.left.similarity(other),
                BatchKind::Intersection => self.left.intersection(other),
            });
        }

        let stats = self.left.right_and_union_stats(other);

        if stats.cc == 0 {
            return Some(0.0);
        }

        let ec = if self.high_precision {
            let n = cache.n;
            let m = Sketch::cardinality_from_parts(stats.right_sum, stats.right_ez);

            // Does this pair land in the slow exp-loop branch of
            // `approximate_expected_collisions`? If not, we can skip the
            // `a_diff` precomputation entirely.
            let big = n.max(m);
            let cap = 2f64.powf(2f64.powf(f64::from(Q)) + f64::from(R));
            let slow_thresh = 2f64.powf(f64::from(P) + 5.0);

            if big <= cap && big <= slow_thresh {
                let a_diff = self
                    .a_diff
                    .get_or_insert_with(|| Sketch::precompute_a_diff(n));
                Sketch::approximate_expected_collisions_with_a_diff(a_diff, n, m)
            } else {
                Sketch::approximate_expected_collisions(n, m)
            }
        } else {
            0.0
        };

        if (stats.cc as f64) < ec {
            return Some(0.0);
        }

        let similarity = (stats.cc as f64 - ec) / stats.cn as f64;
        Some(match self.kind {
            BatchKind::Similarity => similarity,
            BatchKind::Intersection => {
                let union_card = Sketch::cardinality_from_parts(stats.union_sum, stats.union_ez);
                similarity * union_card
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.others.size_hint()
    }
}

#[derive(Default)]
struct ComparisonStats {
    cc: u32,
    cn: u32,
    left_sum: f64,
    left_ez: u16,
    right_sum: f64,
    right_ez: u16,
    union_sum: f64,
    union_ez: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: u128 = 0xa96faf705af16834e6c632b61e964e1f;
    const HASH_AA: u128 = 0xb9fe94d346d39b20369242a646a19333;

    #[test]
    fn empty() {
        let mut sk = Sketch::new();
        assert!(sk.is_empty());
        assert_eq!(sk.cardinality(), 0.0);
        sk.add(0);
        assert!(!sk.is_empty());
        assert_ne!(sk.cardinality(), 0.0);
    }

    #[test]
    fn eq() {
        let mut sk1 = Sketch::new();
        let mut sk2 = Sketch::new();
        assert_eq!(sk1, sk2);
        sk1.add("foo");
        assert_ne!(sk1, sk2);
        assert!(sk1 > sk2);
        assert!(sk2 <= sk1);
        sk2.add("foo");
        assert_eq!(sk1, sk2);
        assert!(sk1 >= sk2);
        assert!(sk1 <= sk2);
    }

    #[test]
    fn approx_count() {
        let sk1: Sketch = (0..50_000).collect();
        let exp1: f64 = 50_000.0;
        let sk2: Sketch = (5_000..75_000).collect();
        let exp2: f64 = 70_000.0;
        assert!((sk1.cardinality() - exp1).abs() / exp1 < 0.01);
        assert!((sk2.cardinality() - exp2).abs() / exp2 < 0.01);
        let exp_is = 45_000.0;
        assert!((sk1.intersection(&sk2) - exp_is).abs() / exp_is < 0.01);
    }

    #[test]
    fn intersection_of_disjoint_sketches_is_zero() {
        let sk1: Sketch = (0..1_000).collect();
        let sk2: Sketch = (1_000..2_000).collect();
        assert_eq!(sk1.similarity(&sk2), 0.0);
        assert_eq!(sk1.intersection(&sk2), 0.0);
    }

    #[test]
    fn similarity_of_identical_single_register_sketches_is_one() {
        // Construct a serialized sketch with exactly one non-empty register and
        // deserialize it twice. The sketches are bit-identical and therefore
        // must have Jaccard similarity 1.0. This guards against introducing
        // bias in the expected-collision correction for exact matches.
        let mut serialized = vec![0u8; (M as usize) * std::mem::size_of::<u16>()];
        serialized[..2].copy_from_slice(&Sketch::new_reg(1, 0).to_le_bytes());

        let sk1 = Sketch::load(&serialized[..]).unwrap();
        let sk2 = Sketch::load(&serialized[..]).unwrap();

        assert!((sk1.cardinality() - 1.0).abs() < 1e-3);
        assert!((sk1.similarity(&sk2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_partially_overlapping() {
        let vamax = 300000;
        let va = (0..vamax).collect();
        let vbmin = 290000;
        let vbmax = 2 * vamax;
        let vb = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let sketch1: Sketch = va;
        let sketch2: Sketch = vb;
        let actual_similarity = sketch1.similarity(&sketch2);
        let sigma = (actual_similarity - jexact).abs() / jexact;
        println!(
            " jaccard estimate : {:.5e}  exact value : {:.5e} , error : {:.5e}",
            actual_similarity, jexact, sigma
        );
        assert!((actual_similarity - jexact).abs() / jexact < 0.1);
    }

    #[test]
    fn seeded_hash() {
        // Same value, different seeds, registers differ, sketches not equal.

        let mut s = Sketch::default();
        s.add_with_seed("foo", 0);
        assert!(s.cardinality() >= 0.0);
        s.add_with_seed("foo", 1);
        assert!(s.cardinality() >= 1.0);

        let mut s0 = Sketch::default();
        s0.add_with_seed("foo", 0);
        let mut s1 = Sketch::default();
        s1.add_with_seed("foo", 1);

        assert_ne!(s0, s1, "different seeds should yield different registers");

        // Cardinality should be nearly identical for one element;
        // tiny numerical differences are possible due to lz contribution.
        let c0 = s0.cardinality();
        let c1 = s1.cardinality();
        let rel = (c0 - c1).abs() / c0.max(c1).max(1.0);
        assert!(
            rel < 1e-3,
            "cardinality should be nearly identical with different seeds: c0={c0}, c1={c1}, rel={rel}"
        );
    }

    #[test]
    fn add_io() {
        let elemens = [&[b'a'; 640][..], &[b'b'; 30], &[b'c'; 1]];
        let mut s1 = Sketch::default();
        let mut s2 = Sketch::default();
        for e in elemens {
            s1.add_reader(e).unwrap();
            s2.add_bytes(e);
        }
        assert_eq!(s1, s2);
        let c = s1.cardinality();
        assert!((2.9..=3.1).contains(&c));
    }

    #[test]
    fn entry_consistency() {
        let mut e = Entry::new();
        assert_eq!(e.hasher.digest128(), EMPTY_HASH);
        e.add_bytes(b"a");
        assert_eq!(e.hasher.digest128(), HASH_A);
        e.add_bytes(b"a");
        assert_eq!(e.hasher.digest128(), HASH_AA);

        let mut e2 = Entry::new();
        e2.add_bytes(b"aa");
        assert_eq!(e2.hasher.digest128(), HASH_AA);

        assert_eq!(e, e2);
    }
}
