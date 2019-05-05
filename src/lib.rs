use std::hash;

const P: u32 = 14;
const M: u32 = 1 << P;
const MAX: u32 = 64 - P;
const MAXX: u64 = u64::max_value() >> MAX;
const ALPHA: f64 = 0.7213 / (1f64 + 1.079 / (M as f64));
const Q: u8 = 6;
const R: u8 = 10;
const TQ: u32 = 1 << Q;
const TR: u32 = 1 << R;
const C: f64 = 0.169_919_487_159_739_1;

fn beta(ez: f64) -> f64 {
    let zl = (ez + 1.0).ln();
    -0.370_393_911 * ez
        + 0.070_471_823 * zl
        + 0.173_936_86 * zl.powi(2)
        + 0.163_398_39 * zl.powi(3)
        + -0.092_377_45 * zl.powi(4)
        + 0.037_380_27 * zl.powi(5)
        + -0.005_384_159 * zl.powi(6)
        + 0.000_424_19 * zl.powi(7)
}

#[derive(Clone)]
pub struct Sketch {
    regs: Vec<u16>,
}

impl Default for Sketch {
    fn default() -> Self {
        Self {
            regs: vec![0; M as usize],
        }
    }
}

impl<T: hash::Hash> std::iter::FromIterator<T> for Sketch {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut sk = Self::default();
        for v in iter.into_iter() {
            sk.add(v)
        }
        sk
    }
}

impl Sketch {
    fn new_reg(lz: u8, sig: u16) -> u16 {
        (u16::from(lz) << R) | sig
    }

    fn lz(reg: u16) -> u8 {
        (reg >> (16 - Q)) as u8
    }

    fn add_hash(&mut self, h: (u64, u64)) {
        let (x, y) = h;
        let k = x >> MAX;
        let lz = ((x << P) ^ MAXX).leading_zeros() as u8 + 1;
        let sig = (y << (64 - R) >> (64 - R)) as u16;
        let reg = Self::new_reg(lz, sig);
        if self.regs[k as usize] < reg {
            self.regs[k as usize] = reg;
        }
    }

    /// Add a value to this set
    pub fn add(&mut self, v: impl hash::Hash) {
        let mut mh = metrohash::MetroHash128::default();
        v.hash(&mut mh);
        self.add_hash(mh.finish128());
    }

    fn sum_and_zeros(&self) -> (f64, f64) {
        let mut sum = 0.0;
        let mut ez = 0.0;
        for reg in self.regs.iter() {
            let lz = Self::lz(*reg);
            if lz == 0 {
                ez += 1.0;
            }
            sum += 1.0 / (2f64).powi(i32::from(lz));
        }
        (sum, ez)
    }

    /// The approximate number of unique elements in the set.
    pub fn cardinality(&self) -> f64 {
        let (sum, ez) = self.sum_and_zeros();
        ALPHA * (f64::from(M)) * ((f64::from(M)) - ez) / (beta(ez) + sum)
    }

    /// Merge two sets, resulting in this set becoming the union-set.
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
            std::f64::INFINITY
        } else if n > 2f64.powf(f64::from(P) + 5.0) {
            let d = (4.0 * n / m) / ((1.0 + n) / m).powi(2);
            C * 2f64.powf(f64::from(P) - f64::from(R)) * d + 0.5
        } else {
            Self::expected_collisions(n, m) / f64::from(P)
        }
    }

    fn expected_collisions(n: f64, m: f64) -> f64 {
        let mut x = 0.0;
        let mut b1: f64;
        let mut b2: f64;
        for i in 1..TQ {
            for j in 1..TR {
                let j = f64::from(j);
                if i != TQ {
                    let den = 2f64.powf(f64::from(P) + f64::from(R) + f64::from(i));
                    b1 = (f64::from(TR) + j) / den;
                    b2 = (f64::from(TR) + j + 1.0) / den;
                } else {
                    let den = 2f64.powf(f64::from(P) + f64::from(R) + f64::from(i) - 1.0);
                    b1 = j / den;
                    b2 = (j + 1.0) / den;
                }
                let prx = (1.0 - b2).powf(n) - (1.0 - b1).powf(n);
                let pry = (1.0 - b2).powf(m) - (1.0 - b1).powf(m);
                x += prx * pry;
            }
        }
        (x * f64::from(P)) + 0.5
    }

    /// The Jaccard Index similarity estimation
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
    pub fn intersection(&self, other: &Self) -> f64 {
        let sim = self.similarity(other);
        sim * self.clone().union(other).cardinality() + 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
