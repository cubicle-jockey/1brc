extern crate core;
use memchr::memchr_iter;
use memmap2::MmapOptions;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;
use std::fmt::Display;
use std::ops::{Add, AddAssign};

fn main() {
    // let start = std::time::Instant::now();
    let cpu_count: usize = std::thread::available_parallelism().unwrap().into();
    let path = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| "measurements.txt".to_owned());
    let file = std::fs::File::open(path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;

    let mut blocks = Vec::with_capacity(cpu_count);
    let chunk_size = file_len / cpu_count;
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };

    let mut current_pos = 0;
    for _ in 0..cpu_count {
        let end = (current_pos + chunk_size).min(file_len);

        // split at clean boundaries
        let next_new_line = match memchr::memchr(b'\n', &mmap[end..]) {
            Some(v) => v,
            None => {
                assert_eq!(end, mmap.len());
                0
            }
        };
        let end = end + next_new_line;
        blocks.push((current_pos, end));
        current_pos = end + 1;
    }

    let results = blocks
        .par_iter()
        .map(|(start, end)| process_block(&mmap, *start, *end))
        .collect::<Vec<HashMap<Vec<u8>, Tally>>>();

    // estimate the total number of unique keys to avoid rehashing
    let total_keys = results.iter().map(|m| m.len()).sum::<usize>();

    // create the map with sufficient initial capacity
    let mut map = HashMap::<Vec<u8>, Tally>::with_capacity_and_hasher(
        total_keys,
        Default::default(),
    );

    // reuse a vector for collecting keys to avoid extra allocations
    let mut keys = Vec::with_capacity(total_keys);

    // process any empty results
    for result in results {
        for (key, tally) in result {
            // track keys for sorting (only first occurrence)
            if !map.contains_key(&key) {
                keys.push(key.clone());
            }

            // more efficient update without redundant checks
            match map.entry(key) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    e.get_mut().merge(tally);
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(tally);
                }
            }
        }
    }

    keys.sort_unstable();

    let key_list = keys;

    let mut is_first = true;
    print!("{{");
    for key in key_list {
        if is_first {
            is_first = false;
            print!(
                "{}={}",
                String::from_utf8_lossy(&key),
                map[&key].to_string()
            );
        } else {
            print!(
                ", {}={}",
                String::from_utf8_lossy(&key),
                map[&key].to_string()
            );
        }
    }
    println!("}}");

    // let elapsed = start.elapsed();
    // println!("Elapsed time: {:?}", elapsed);
}

fn process_block(mem: &[u8], start: usize, end: usize) -> HashMap<Vec<u8>, Tally> {
    let mut line_rec = LineRec::new();
    // estimate the size based on the block size and average key/value size
    let estimated_size = ((end - start) / 256).max(8);
    let mut map = HashMap::with_capacity_and_hasher(estimated_size, Default::default());

    let mut inx = start;

    if end <= start {
        return map;
    }

    for x in memchr_iter(b'\n', &mem[start..end]) {
        let inx2 = inx + x;
        if inx2 > end || line_rec.load_from_slice(&mem[inx..inx2]).is_err() {
            break;
        };

        if let Some(tally) = map.get_mut(line_rec.key()) {
            tally.add(line_rec.val());
        } else {
            let tally = Tally::new_add(line_rec.val());
            map.insert(line_rec.key_clone(), tally);
        }

        inx = start + x + 1;
        if inx >= end {
            break;
        }
    }

    map
}

struct Tally {
    min: MiniDec,
    max: MiniDec,
    sum: MiniDec,
    count: u32,
}

impl Tally {
    #[allow(unused)] // saving for future use
    #[inline(always)]
    pub const fn new() -> Tally {
        Tally {
            min: MiniDec::new(true, 0),
            max: MiniDec::new(true, 0),
            sum: MiniDec::new(true, 0),
            count: 0,
        }
    }

    #[inline(always)]
    pub fn new_add(val: &MiniDec) -> Tally {
        let tally = Tally {
            min: MiniDec::new(val.is_positive, val.v),
            max: MiniDec::new(val.is_positive, val.v),
            sum: MiniDec::new(val.is_positive, val.v),
            count: 1,
        };
        tally
    }

    #[inline(always)]
    pub fn add(&mut self, val: &MiniDec) {
        self.sum += val;
        self.count += 1;
        if self.min > *val {
            self.min.is_positive = val.is_positive;
            self.min.v = val.v;
        }
        if self.max < *val {
            self.max.is_positive = val.is_positive;
            self.max.v = val.v;
        }
    }

    #[inline(always)]
    pub fn merge(&mut self, other: Tally) {
        self.sum += other.sum;
        self.count += other.count;
        if self.min > other.min {
            self.min.is_positive = other.min.is_positive;
            self.min.v = other.min.v;
        }
        if self.max < other.max {
            self.max.is_positive = other.max.is_positive;
            self.max.v = other.max.v;
        }
    }
}

impl Display for Tally {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let avg = self.sum.v as f32 * 0.1;
        let mut avg = avg / self.count as f32;
        if !self.sum.is_positive {
            avg = -avg;
        }
        // write!(
        //     f,
        //     "min: {}, max: {}, sum: {}, count: {}, avg: {:1.1}",
        //     self.min, self.max, self.sum, self.count, avg
        // )
        // I guess this is the format they want
        write!(f, "{:.1}/{:.1}/{:.1}", self.min, avg, self.max)
    }
}

#[derive(Debug, Clone)]
struct LineRec {
    key: Vec<u8>,
    val: MiniDec,
}

impl LineRec {
    pub fn new() -> LineRec {
        LineRec {
            key: Vec::with_capacity(30), // slightly larger to reduce reallocations for longer city names
            val: MiniDec::new(true, 0),
        }
    }

    #[inline(always)]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    #[inline(always)]
    pub fn key_clone(&self) -> Vec<u8> {
        // optimized clone for byte vectors - using with_capacity and extend
        // is usually faster than Vec::clone() for short vectors
        let mut new_key = Vec::with_capacity(self.key.len());
        unsafe {
            // Fast copy for small vectors
            std::ptr::copy_nonoverlapping(
                self.key.as_ptr(),
                new_key.as_mut_ptr(),
                self.key.len(),
            );
            new_key.set_len(self.key.len());
        }
        new_key
    }

    #[inline(always)]
    pub fn val(&self) -> &MiniDec {
        &self.val
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.key.clear();
    }

    #[allow(unused)] // saving for future use
    fn split(self) -> (Vec<u8>, MiniDec) {
        (self.key, self.val)
    }

    #[inline(always)]
    pub fn load_from_slice(&mut self, line: &[u8]) -> Result<(), ()> {
        self.clear();

        let sep_pos = memchr::memchr(b';', line).ok_or(())?;

        if self.key.capacity() < sep_pos {
            self.key.reserve(sep_pos - self.key.len());
        }
        self.key.extend_from_slice(&line[..sep_pos]);
        self.val.load_from_slice(&line[sep_pos + 1..])?;

        Ok(())
    }
}

impl AddAssign<LineRec> for LineRec {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.val += other.val;
    }
}

// using u32 "should" be faster than f32 arithmetic
#[derive(Debug)]
struct MiniDec {
    is_positive: bool,
    v: u32,
}

impl MiniDec {
    #[inline(always)]
    pub const fn new(is_positive: bool, v: u32) -> MiniDec {
        MiniDec { is_positive, v }
    }

    #[inline]
    pub const fn clear(&mut self) {
        self.is_positive = true;
        self.v = 0;
    }

    #[inline]
    pub fn load_from_str(&mut self, line: &str) -> Result<(), ()> {
        self.clear();
        let mut is_positive = true;
        let mut v = 0;
        let mut is_decimal = false;

        for c in line.as_bytes() {
            match c {
                b'-' => is_positive = false,
                b'.' => is_decimal = true,
                b'\n' => break,
                b'0'..=b'9' => {
                    v = fast_10(v) + (c - b'0') as u32;
                    if is_decimal {
                        break;
                    }
                }
                _ => {}
            }
        }
        if !is_decimal {
            v = fast_10(v);
        }

        self.is_positive = is_positive;
        self.v = v;

        Ok(())
    }

    #[inline(always)]
    pub fn load_from_slice(&mut self, slice: &[u8]) -> Result<(), ()> {
        let mut is_positive = true;
        let mut v = 0;
        let mut is_decimal = false;

        let len = slice.len();
        let mut i = 0;

        // Manual loop unrolling for better performance
        while i < len {
            let b = unsafe { *slice.get_unchecked(i) };
            match b {
                b'-' => is_positive = false,
                b'.' => is_decimal = true,
                b'\n' => break,
                b'0'..=b'9' => {
                    v = fast_10(v) + (b - b'0') as u32;
                    if is_decimal {
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        if !is_decimal {
            v = fast_10(v);
        }

        self.is_positive = is_positive;
        self.v = v;

        Ok(())
    }
}

impl Clone for MiniDec {
    #[inline(always)]
    fn clone(&self) -> MiniDec {
        MiniDec {
            is_positive: self.is_positive,
            v: self.v,
        }
    }
}

#[inline(always)]
const fn fast_10(i: u32) -> u32 {
    (i << 3) + (i << 1)
}

impl Display for MiniDec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let float = self.v as f32 * 0.1;
        write!(
            f,
            "{}{:1.1}",
            if self.is_positive { "" } else { "-" },
            float
        )
    }
}

impl From<&str> for MiniDec {
    #[inline]
    fn from(value: &str) -> Self {
        let mut md = MiniDec::new(true, 0);
        md.load_from_str(value).unwrap();
        md
    }
}

impl Add<MiniDec> for MiniDec {
    type Output = MiniDec;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self::Output {
        self.add_assign(&other);
        self
    }
}

impl Add<&MiniDec> for MiniDec {
    type Output = MiniDec;
    #[inline(always)]
    fn add(mut self, other: &Self) -> Self::Output {
        self.add_assign(other);
        self
    }
}

impl AddAssign<MiniDec> for MiniDec {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign(&other);
    }
}

impl AddAssign<&MiniDec> for MiniDec {
    #[inline(always)]
    fn add_assign(&mut self, other: &Self) {
        if other.v == 0 {
            return;
        }

        if self.is_positive == other.is_positive {
            self.v += other.v;
        } else {
            if self.v >= other.v {
                self.v -= other.v;
            } else {
                self.v = other.v - self.v;
                self.is_positive = !self.is_positive;
            }
        }
    }
}

impl PartialEq for MiniDec {
    fn eq(&self, other: &Self) -> bool {
        self.is_positive == other.is_positive && self.v == other.v
    }
}

impl PartialOrd for MiniDec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_positive == other.is_positive {
            if self.is_positive {
                self.v.partial_cmp(&other.v)
            } else {
                if self.v == other.v {
                    Some(Ordering::Equal)
                } else if self.v > other.v {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            }
        } else {
            if self.is_positive {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Less)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = MiniDec::new(true, 10);
        let b = MiniDec::new(true, 30);
        let c = a + b;
        assert_eq!(c.to_string(), "4.0");
    }

    #[test]
    fn test_add_negative() {
        let a = MiniDec::new(true, 12);
        let b = MiniDec::new(false, 34);
        let c = a + b;
        assert_eq!(c.to_string(), "-2.2");
    }

    #[test]
    fn test_add_negative2() {
        let a = MiniDec::new(true, 16);
        let b = MiniDec::new(false, 34);
        let c = a + b;
        assert_eq!(c.to_string(), "-1.8");
    }

    #[test]
    fn test_add_negative3() {
        let a = MiniDec::new(false, 16);
        let b = MiniDec::new(false, 34);
        let c = a + b;
        assert_eq!(c.to_string(), "-5.0");
    }

    #[test]
    fn test_add_negative4() {
        let a = MiniDec::new(false, 16);
        let b = MiniDec::new(true, 34);
        let c = a + b;
        assert_eq!(c.to_string(), "1.8");
    }

    #[test]
    fn test_add_assign() {
        let mut a = MiniDec::new(true, 12);
        let b = MiniDec::new(true, 34);
        a += b;
        assert_eq!(a.to_string(), "4.6");
    }

    #[test]
    fn test_from_str() {
        let a = MiniDec::from("1.2");
        assert_eq!(a.to_string(), "1.2");
    }

    #[test]
    fn test_from_str2() {
        let a = MiniDec::from("12");
        assert_eq!(a.to_string(), "12.0");
    }

    #[test]
    fn test_from_str_neg() {
        let a = MiniDec::from("-1.2");
        assert_eq!(a.to_string(), "-1.2");
    }

    #[test]
    fn test_from_str_neg2() {
        let a = MiniDec::from("-12");
        assert_eq!(a.to_string(), "-12.0");
    }

    #[test]
    fn test_from_str3() {
        let a = MiniDec::from("12.0");
        assert_eq!(a.to_string(), "12.0");
    }

    #[test]
    fn test_from_str_neg3() {
        let a = MiniDec::from("-12");
        assert_eq!(a.to_string(), "-12.0");
    }

    #[test]
    fn test_line_rec() {
        let mut lr = LineRec::new();
        lr.load_from_slice(b"key;12.0").unwrap();
        assert_eq!(lr.key, b"key");
        assert_eq!(lr.val.to_string(), "12.0");
    }
}
