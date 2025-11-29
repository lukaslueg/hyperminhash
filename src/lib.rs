//! Very fast, constant memory-footprint cardinality approximation, including intersection and union operation.
//! A straight port of [Hyperminhash](https://github.com/axiomhq/hyperminhash).
//!
//! As with other cardinality estimators, `Hyperminhash` has two advantages when counting very large
//! sets or streams of elements:
//! * It uses a single data structure that never grows while counting elements. The structure
//!   consumes 32kb of memory, allocated on the stack.
//! * The amount of work done for counting a marginal element stays approximately constant.
//!
//! For sets smaller than roughly 10^4 unique elements, a `std::collections::HashSet` is usually faster,
//! albeit using much more memory. When counting streams of millions of elements, `Hyperminhash` is much
//! faster and uses much less memory.
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

struct EcTable {
    ln1p_neg_b1: Box<[[f64; TR as usize]; TQ as usize]>,
    ln1p_neg_b2: Box<[[f64; TR as usize]; TQ as usize]>,
}

impl EcTable {
    fn new() -> Self {
        let mut ln1p_neg_b1 = Box::new([[0.0; TR as usize]; TQ as usize]);
        let mut ln1p_neg_b2 = Box::new([[0.0; TR as usize]; TQ as usize]);

        // i in 1..TQ (exclusive upper bound), j in 1..TR (skip j=0)
        for i in 1..TQ {
            let row = (i as usize) - 1;
            if i != TQ {
                let den = 2f64.powf(f64::from(P) + f64::from(R) + f64::from(i));
                for j1 in 1..TR {
                    let j = f64::from(j1);
                    let b1 = (f64::from(TR) + j) / den;
                    let b2 = (f64::from(TR) + j + 1.0) / den;
                    ln1p_neg_b1[row][j1 as usize] = f64::ln_1p(-b1);
                    ln1p_neg_b2[row][j1 as usize] = f64::ln_1p(-b2);
                }
            } else {
                let den = 2f64.powf(f64::from(P) + f64::from(R) + f64::from(i) - 1.0);
                for j1 in 1..TR {
                    let j = f64::from(j1);
                    let b1 = j / den;
                    let b2 = (j + 1.0) / den;
                    ln1p_neg_b1[row][j1 as usize] = f64::ln_1p(-b1);
                    ln1p_neg_b2[row][j1 as usize] = f64::ln_1p(-b2);
                }
            }
        }

        EcTable {
            ln1p_neg_b1,
            ln1p_neg_b2,
        }
    }
}

/// Records the approximate number of unique elements it has seen over it's lifetime.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Sketch {
    regs: Regs,
}

impl std::fmt::Debug for Sketch {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "Sketch {{ {} }}", self.cardinality())
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
        static L: std::sync::LazyLock<[f64; 64]> = std::sync::LazyLock::new(|| {
            let mut l: [f64; 64] = [0.0; 64];
            for (i, v) in l.iter_mut().enumerate() {
                *v = 1.0 / (2f64).powi(i32::try_from(i).unwrap());
            }
            l
        });
        let l = &*L;
        let mut sum = 0.0;
        let mut ez: u16 = 0;
        for reg in self.regs {
            let lz = Self::lz(reg);
            if lz == 0 {
                ez += 1;
            } else {
                sum += l[lz as usize];
            }
        }
        (sum + f64::from(ez), ez)
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

    fn approximate_expected_collisions(n: f64, m: f64) -> f64 {
        let (n, m) = (n.max(m), n.min(m));
        if n > 2f64.powf(2f64.powf(f64::from(Q)) + f64::from(R)) {
            f64::INFINITY
        } else if n > 2f64.powf(f64::from(P) + 5.0) {
            let d = (4.0 * n / m) / ((1.0 + n) / m).powi(2);
            C * 2f64.powf(f64::from(P) - f64::from(R)) * d + 0.5
        } else {
            Self::expected_collisions(n, m) / f64::from(P)
        }
    }

    fn expected_collisions(n: f64, m: f64) -> f64 {
        static TBL: std::sync::LazyLock<EcTable> = std::sync::LazyLock::new(EcTable::new);
        let tbl = &*TBL;

        let mut x = 0.0;
        for i in 1..TQ {
            let row = (i as usize) - 1;
            let l1 = &tbl.ln1p_neg_b1[row];
            let l2 = &tbl.ln1p_neg_b2[row];

            for j1 in 1..TR {
                // (1 - b)^n = exp(n * ln1p(-b))
                let a1 = (n * l2[j1 as usize]).exp();
                let a0 = (n * l1[j1 as usize]).exp();
                let b1 = (m * l2[j1 as usize]).exp();
                let b0 = (m * l1[j1 as usize]).exp();
                x += (a1 - a0) * (b1 - b0);
            }
        }

        (x * f64::from(P)) + 0.5
    }

    /// The Jaccard Index similarity estimation
    ///
    /// ```rust
    /// let sk1 = (0..=75).collect::<hyperminhash::Sketch>();
    /// let sk2 = (50..=125).collect::<hyperminhash::Sketch>();
    /// assert!((sk1.similarity(&sk2) - (25.0 / 125.0)).abs() < 1e-2);
    /// ```
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        let cc = self
            .regs
            .iter()
            .zip(other.regs.iter())
            .filter(|(r, rr)| **r != 0 && r == rr)
            .count();
        let cn = self
            .regs
            .iter()
            .zip(other.regs.iter())
            .filter(|(r, rr)| **r != 0 || **rr != 0)
            .count();
        if cc == 0 {
            return 0.0;
        }

        let n = self.cardinality();
        let m = other.cardinality();
        let ec = Self::approximate_expected_collisions(n, m);
        if (cc as f64) < ec {
            return 0.0;
        }
        (cc as f64 - ec) / cn as f64
    }

    /// The approximate number of elements in both sets
    ///
    /// ```rust
    /// let sk1 = (0..=750).collect::<hyperminhash::Sketch>();
    /// let sk2 = (500..=1250).collect::<hyperminhash::Sketch>();
    /// assert!((sk1.intersection(&sk2) - 250.0).abs() < 1.0);
    /// ```
    #[must_use]
    pub fn intersection(&self, other: &Self) -> f64 {
        let sim = self.similarity(other);
        sim * self.clone().union(other).cardinality() + 0.5
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
