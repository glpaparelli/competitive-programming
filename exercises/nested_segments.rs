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
fn fenwick_tree_nested_segments(input_segments: &[(i32, i32)]) -> Vec<(i64, usize)> {
    let n = input_segments.len();
    let mut events: Vec<(i32, i32, usize)> = Vec::with_capacity(n);
    for (i, &(l, r)) in input_segments.iter().enumerate() {
        events.push((l, r, i));
    }

    // sort by starting endpoint
    events.sort_by(|a, b| a.0.cmp(&b.0));

    let mut tree = FenwickTree::with_len(input_segments.len()*2 + 1);
    for i in 0..n {
        tree.add(events[i].1 as usize, 1);
    }

    let mut sol: Vec<(i64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        sol.push((tree.sum(events[i].1 as usize) - 1, events[i].2));
        tree.add(events[i].1 as usize, -1);
    }
    
    // restore so that the solution are paired with the input ordering
    sol.sort_by(|a, b| a.1.cmp(&b.1));

    sol
}

struct Node {
    sum: i32,                       
    left_edge: usize,               // Left edge of the segment 
    right_edge: usize,              // Right edge of the segment
    left_child: Option<Box<Node>>,  // Left subtree
    right_child: Option<Box<Node>>, // Right subtree
}
impl Node {
    fn new(
        sum: i32,
        left_edge: usize,
        right_edge: usize,
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
    ) -> Self {
        Self {
            sum,
            left_edge,
            right_edge,
            left_child,
            right_child,
        }
    }
}

struct SegmentTree {
    root: Node,
}

impl SegmentTree {
    pub fn new(a: &[i32]) -> SegmentTree {
        let n: usize = a.len();

        let mut root = Node::new(
            0,
            0,
            n - 1, 
            None,
            None
        );

        SegmentTree::build(&mut root, a, 0, n - 1);
        SegmentTree { root }
    }

    fn build(curr_node: &mut Node, a: &[i32], l: usize, r: usize) {
        if l == r {
            // Leaf node, store the value from the array
            curr_node.sum = a[l];
        } else {
            // Non-leaf node, divide the segment and build children
            let m = l + (r - l)/2;

            // Create left child node
            let mut left_child = Node::new(
                0, 
                l, 
                m, 
                None, 
                None
            );
            // Recursively build the left subtree
            SegmentTree::build(&mut left_child, a, l, m);

            // Create right child node
            let mut right_child = Node::new(
                0,
                m + 1, 
                r, 
                None, 
                None
            );
            // Recursively build the right subtree
            SegmentTree::build(&mut right_child, a, m + 1, r);

            // Update the current node's sum to the sum of its children
            curr_node.sum = left_child.sum + right_child.sum;

            // Store the left and right children
            curr_node.left_child = Some(Box::new(left_child));
            curr_node.right_child = Some(Box::new(right_child));
        }
    }

    /// Compute the sum of the range [l, r]
    pub fn range_sum(&self, l: usize, r: usize) -> i32 {
        SegmentTree::range_sum_rec(&self.root, l, r)
    }

    fn range_sum_rec(node: &Node, l: usize, r: usize) -> i32 {
        // If the current node's range is completely outside the query range, return 0
        if r < node.left_edge || l > node.right_edge {
            return 0;
        }

        // If the current node's range is completely inside the query range, return the node's sum
        if l <= node.left_edge && r >= node.right_edge {
            return node.sum;
        }

        // Otherwise, the query range partially overlaps with the current node's range
        // Recursively compute the sum for the left and right subtrees
        let left_sum = if let Some(ref left_child) = node.left_child {
            SegmentTree::range_sum_rec(&left_child, l, r)
        } else {
            0
        };

        let right_sum = if let Some(ref right_child) = node.right_child {
            SegmentTree::range_sum_rec(&right_child, l, r)
        } else {
            0
        };

        // Return the sum of the left and right subtree sums
        left_sum + right_sum
    }

    pub fn increment_by_one(&mut self, i: usize) {
        let mut val = self.range_sum(i, i);
        val += 1;
        SegmentTree::update_rec(&mut self.root, i, i, val);
    }
    
    /// Update the values in the range [i, j] with the new value t
    pub fn range_update(&mut self, l: usize, r: usize, t: i32) {
        SegmentTree::update_rec(&mut self.root, l, r, t)
    }
    fn update_rec(curr_node: &mut Node, l: usize, r: usize, t: i32) {
        // Check if the current node's segment is outside the update range
        if curr_node.right_edge < l || curr_node.left_edge > r {
            return;
        }

        // Check if the current node's segment is completely inside the update range
        if curr_node.left_edge >= l && curr_node.right_edge <= r {
            // Update the current node's sum
            curr_node.sum = t;

            // Propagate the update to the children
            if let Some(left_child) = &mut curr_node.left_child {
                SegmentTree::update_rec(left_child, l, r, t);
            }
            if let Some(right_child) = &mut curr_node.right_child {
                SegmentTree::update_rec(right_child, l, r, t);
            }

            return;
        }

        // Recur for both left and right children
        if let Some(left_child) = &mut curr_node.left_child {
            SegmentTree::update_rec(left_child, l, r, t);
        }
        if let Some(right_child) = &mut curr_node.right_child {
            SegmentTree::update_rec(right_child, l, r, t);
        }

        if let Some(left_child) = &mut curr_node.left_child {
    
            if let Some(right_child) = &mut curr_node.right_child {
                curr_node.sum = left_child.sum + right_child.sum;
            }   
        } 
    }

    /// Print the segment tree (for debugging purposes)
    pub fn print_tree(&self) {
        SegmentTree::print_tree_rec(&self.root, 0);
    }
    fn print_tree_rec(node: &Node, depth: usize) {
        let indentation = "  ".repeat(depth);
        println!(
            "{}Node: sum: {}, Range: [{}, {}]",
            indentation, node.sum, node.left_edge, node.right_edge
        );

        if node.left_child.is_some() {
            let left_child = node.left_child.as_ref().unwrap();
            SegmentTree::print_tree_rec(&left_child, depth + 1);
        }
        if node.right_child.is_some() {
            let right_child = node.right_child.as_ref().unwrap();
            SegmentTree::print_tree_rec(&right_child, depth + 1);
        }
    }
}
fn segment_tree_nested_segments(input_segments: &[(i32, i32)]) -> Vec<(usize, i32)> {
    let n = input_segments.iter().max_by_key(|&x| x.1).unwrap();
    
    let a = vec![0; (n.1) as usize];
    let mut seg_tree = SegmentTree::new(&a);

    let mut res = Vec::<(usize, i32)>::new();

    // axis[i] = (segment endpoint, index in segment, is_start)
    let mut axis: Vec<(i32, usize, bool)> = Vec::<(i32, usize, bool)>::new();

    for i in 0..input_segments.len() {
        axis.push((input_segments[i].0, i, true));
        axis.push((input_segments[i].1, i, false));
    }

    // sort axis by the first element (the segment endpoint)
    axis.sort_by(|a, b| a.0.cmp(&b.0));

    for i in 0..axis.len() {
        // the end of a segment
        if axis[i].2 == false {
            let index_of_segment = axis[i].1;

            // the number of segments is the range sum [start, end of this segment]
            let l_of_segment = input_segments[index_of_segment].0 as usize;
            let r_of_segment = axis[i].0 as usize;

            // nested segment
            let ns = seg_tree.range_sum(l_of_segment, r_of_segment);

            res.push((axis[i].1, ns));
            
            seg_tree.increment_by_one(input_segments[axis[i].1].0 as usize);
        }
    }

    // restore the order of the input
    res.sort_by(|a, b| a.0.cmp(&b.0));
    res
}

fn main() {
    let input_segments1 = vec![(1, 8), (2, 3), (4, 7), (5, 6)];
    let sol1 = segment_tree_nested_segments(&input_segments1);
    sol1.iter().for_each(|&(_, ns)| println!("{}", ns));

    println!("");

    let input_segments2 = vec![(3, 4), (1, 5), (2, 6)];
    let sol2 = segment_tree_nested_segments(&input_segments2);
    sol2.iter().for_each(|&(_, ns)| println!("{}", ns));

    // let input_segments1 = vec![(1, 8), (2, 3), (4, 7), (5, 6)];
    // let mut sol1 = fenwick_tree_nested_segments(&input_segments1);
    // sol1.iter().for_each(|&(first, _)| println!("{}", first));
    // println!("");
    // let input_segments2 = vec![(3, 4), (1, 5), (2, 6)];
    // let mut sol2 = fenwick_tree_nested_segments(&input_segments2);
    // sol2.iter().for_each(|&(first, _)| println!("{}", first));

    // println!("");
    // println!("");
}
