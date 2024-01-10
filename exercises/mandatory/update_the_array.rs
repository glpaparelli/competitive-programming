// SPHERE ONLINE JUDGE (SPOJ): Update the array

#[derive(Debug)]
pub struct FenwickTree {
    tree: Vec<i64>,
}
impl FenwickTree {
    pub fn with_len(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }
    pub fn len(&self) -> usize {
        self.tree.len() - 1
    }
    /// Indexing is 0-based, even if internally we use 1-based indexing
    pub fn add(&mut self, i: usize, delta: i64) {
        let mut i = i + 1; 
        assert!(i < self.tree.len());

        while i < self.tree.len() {
            self.tree[i] += delta;
            i = Self::next_sibling(i);
        }
    }
    /// Indexing is 0-based, even if internally we use 1-based indexing
    pub fn sum(&self, i: usize) -> i64 {
        let mut i = i + 1;  

        assert!(i < self.tree.len());
        let mut sum = 0;
        while i != 0 {
            sum += self.tree[i];
            i = Self::parent(i);
        }
        sum
    }
    pub fn range_sum(&self, l: usize, r: usize) -> i64 {
        self.sum(r) - if l == 0 { 0 } else { self.sum(l - 1) }
    }
    fn isolate_trailing_one(i: usize) -> usize {
        if i == 0 {
            0
        } else {
            1 << i.trailing_zeros()
        }
    }
    fn parent(i: usize) -> usize {
        i - Self::isolate_trailing_one(i)
    }
    fn next_sibling(i: usize) -> usize {
        i + Self::isolate_trailing_one(i)
    }
}

#[derive(Debug)]
struct UpdateArray {
    ft: FenwickTree,
}
impl UpdateArray {
    pub fn with_len(n: usize) -> Self {
        Self {
            ft: FenwickTree::with_len(n),
        }
    }

    pub fn len(&self) -> usize {
        self.ft.len()
    }

    pub fn access(&self, i: usize) -> i64 {
        self.ft.sum(i)
    }

    pub fn range_update(&mut self, l: usize, r: usize, v: i64) {
        assert!(l <= r);
        assert!(r < self.ft.len());

        self.ft.add(l, v);
        if r + 1 < self.ft.len() {
            self.ft.add(r + 1, -v);
        }
    }
}

/*
    1) from array we build the Fenwick Tree of length n 
        1.2) the operation `access(i)` is a wrapper of the operation `sum(i)` 
    2) the operation `range_update(l,r,v)` exploit the operation `add(i, v)` 
       of the implementation of the Fenwick Tree: 
          2.1) first we check that `l` is `<=` than `r`, aka that the interval 
               of entries to update is well formed
          2.2) then we check that `r <= n`, aka that the interval of entries to update 
               is actually in the array
          2.3) we perform `add(l,v)`: this trigger the addition of the value `v` to each 
               node whose range include the position `l` in the Fenwick Tree
          2.4) we perform `add(r, -v)`: this trigger the subtraction of the value `v` 
               to each node whose range include the position `r` in the Fenwick Tree
          2.5) we have added and subtracted the same same quantity `v` in the Fenwick tree, 
               this means that prefix sum are coherent and the elements in `[l,r]` are increased by `v` 
*/
fn main() {
    // create the array
    let mut array = UpdateArray::with_len(5);

    // println!("Fenwick Tree State: {:?}", array);

    // 3 range updates
    array.range_update(0, 1, 7);
    array.range_update(2, 4, 6);
    array.range_update(1, 3, 2);

    // 3 query
    assert_eq!(array.access(0), 7); 
    assert_eq!(array.access(3), 8); 
    assert_eq!(array.access(4), 6);

    // println!("Fenwick Tree State: {:?}", array); 
}
