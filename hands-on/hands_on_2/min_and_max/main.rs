use std::cmp;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::env;

struct Node {
    key: u32,                       // Store the max
    left_edge: usize,               // Left edge of the segment 
    right_edge: usize,              // Right edge of the segment
    left_child: Option<Box<Node>>,  // Left subtree
    right_child: Option<Box<Node>>, // Right subtree
}
impl Node {
    fn new(
        key: u32,
        left_edge: usize,
        right_edge: usize,
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
    ) -> Self {
        Self {
            key,
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
    pub fn new(a: &[u32]) -> SegmentTree {
        let n: usize = a.len();

        let mut root = Node::new(
            u32::MIN,
            0,
            n - 1, 
            None,
             None
        );

        SegmentTree::build(&mut root, a, 0, n - 1);
        SegmentTree { root }
    }

    fn build(curr_node: &mut Node, a: &[u32], l: usize, r: usize) {
        if l == r {
            // Leaf node, store the value from the array
            curr_node.key = a[l];
        } else {
            // Non-leaf node, divide the segment and build children
            let m = l + (r - l)/2;

            // Create left child node
            let mut left_child = Node::new(
                u32::MIN, 
                l, 
                m, 
                None, 
                None
            );
            // Recursively build the left subtree
            SegmentTree::build(&mut left_child, a, l, m);

            // Create right child node
            let mut right_child = Node::new(
                u32::MIN,
                m + 1, 
                r, 
                None, 
                None
            );
            // Recursively build the right subtree
            SegmentTree::build(&mut right_child, a, m + 1, r);

            // Update the current node's key to the maximum of its children
            curr_node.key = cmp::max(left_child.key, right_child.key);

            // Store the left and right children
            curr_node.left_child = Some(Box::new(left_child));
            curr_node.right_child = Some(Box::new(right_child));
        }
    }

    /// Return the maximum element in range [l, r]
    pub fn max(&self, l: usize, r: usize) -> u32 {
        let root: &Node = &self.root;
        SegmentTree::max_rec(root, l, r)
    }
    /// Query the max in a range [l, r]
    fn max_rec(curr_node: &Node, l: usize, r: usize) -> u32 {
        // The query range must be valid
        assert!(l <= r);

        // Base case, we have reached a leaf
        if curr_node.left_edge == curr_node.right_edge {
            return curr_node.key;
        }
        // The segment of the current node is exactly the query range
        if l == curr_node.left_edge && r == curr_node.right_edge {
            return curr_node.key;
        }

        // The query segment is fully contained in the left subtree
        if curr_node.left_child.is_some() {
            let left_child = curr_node.left_child.as_ref().unwrap();
            if l >= left_child.left_edge && r <= left_child.right_edge {
                return SegmentTree::max_rec(left_child, l, r);
            }
        }
        // The query segment is fully contained in the rigth subtree
        if curr_node.right_child.is_some() {
            let right_child = curr_node.right_child.as_ref().unwrap();
            if l >= right_child.left_edge && r <= right_child.right_edge {
                return SegmentTree::max_rec(right_child, l, r);
            }
        }

        // The query range intersects both left and right subtree
        // compute the max by recurring on both
        let m: usize = (curr_node.left_edge + curr_node.right_edge) / 2;
        let left_max = SegmentTree::max_rec(
            curr_node.left_child.as_ref().unwrap(), 
            l, 
            m
        );
        let right_max = SegmentTree::max_rec(
            curr_node.right_child.as_ref().unwrap(), 
            m + 1, 
            r
        );
        cmp::max(left_max, right_max)
    }

    /// Update the values in the range [i, j] with min(a[k], T)
    pub fn update(&mut self, l: usize, r: usize, t: u32) {
        SegmentTree::update_rec(&mut self.root, l, r, t)
    }
    fn update_rec(curr_node: &mut Node, l: usize, r: usize, t: u32) {
        // Check if the current node's segment is outside the update range
        if curr_node.right_edge < l || curr_node.left_edge > r {
            return;
        }

        // Check if the current node's segment is completely inside the update range
        if curr_node.left_edge >= l && curr_node.right_edge <= r {
            // Update the current node's key
            curr_node.key = cmp::min(curr_node.key, t);

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

        // Update the current node's key based on the updated left and right children
        curr_node.key = cmp::max(
            curr_node
                .left_child
                .as_ref()
                .map_or(u32::MAX, |node| node.key),
            curr_node
                .right_child
                .as_ref()
                .map_or(u32::MAX, |node| node.key),
        );
    }

    /// Print the segment tree (for debugging purposes)
    pub fn print_tree(&self) {
        SegmentTree::print_tree_rec(&self.root, 0);
    }
    fn print_tree_rec(node: &Node, depth: usize) {
        let indentation = "  ".repeat(depth);
        println!(
            "{}Node: Key: {}, Range: [{}, {}]",
            indentation, node.key, node.left_edge, node.right_edge
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

struct Input {
    array: Vec<u32>,
    queries: Vec<(u32, u32, u32, Option<u32>)>,
}

fn main() {
    // produce_outputs();
    let input = read_input();
    println!();
    let mut tree = SegmentTree::new(&input.array);

    for query in input.queries {
        match query.0 {
            0 => tree.update(
                    query.1 as usize, 
                    query.2 as usize, 
                    query.3.unwrap()
                ),
            1 => println!("{}", tree.max(query.1 as usize, query.2 as usize)),
            _ => println!("input error"),
        }
    }
}

fn read_input() -> Input {
    // Read the first line with n and m
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .expect("Failed to read line");
    let mut iter = line.split_whitespace();
    let n: usize = iter.next().unwrap().parse().expect("Failed to parse n");
    let m: usize = iter.next().unwrap().parse().expect("Failed to parse m");

    // Read the second line with n positive integers forming the array
    line.clear();
    io::stdin()
        .read_line(&mut line)
        .expect("Failed to read line");

    let array: Vec<u32> = line
        .split_whitespace()
        .map(|x| x.parse().expect("Failed to parse line"))
        .collect();
    assert_eq!(array.len(), n);

    // Read the next m lines with quadruples forming the queries
    let mut queries = Vec::new();
    for _ in 0..m {
        line.clear();
        io::stdin().read_line(&mut line).expect("Failed to read line");
        let mut iter = line.split_whitespace();
        let mut query: (u32, u32, u32, Option<u32>) = (
            iter.next()
                .unwrap()
                .parse()
                .expect("Failed to read element"),
            iter.next()
                .unwrap()
                .parse()
                .expect("Failed to read element"),
            iter.next()
                .unwrap()
                .parse()
                .expect("Failed to read element"),
            iter.next()
                .map(|x| x.parse().expect("Failed to parse query")),
        );
        // Input is 1-indexed, we use 0-indexing
        query.1 = query.1 - 1;
        query.2 = query.2 - 1;
        queries.push(query);
    }

    Input { array, queries }
}
fn read_input_from_file(file_path: String) -> Input {
    let file = File::open(file_path).expect("Failed to open file");
    let mut reader = BufReader::new(file);

    // Read the first line with n and m
    let mut line = String::new();
    reader.read_line(&mut line).expect("Failed to read line");

    let mut iter = line.split_whitespace();
    let n: usize = iter.next().unwrap().parse().expect("Failed to parse n");
    let m: usize = iter.next().unwrap().parse().expect("Failed to parse m");

    // Read the second line with n positive integers forming the array
    line.clear();
    reader.read_line(&mut line).expect("Failed to read line");

    let array: Vec<u32> = line
        .split_whitespace()
        .map(|x| x.parse().expect("Failed to parse line"))
        .collect();
    assert_eq!(array.len(), n);

    // Read the next m lines with quadruples forming the queries
    let mut queries = Vec::new();
    for _ in 0..m {
        line.clear();
        reader.read_line(&mut line).expect("Failed to read line");
        let mut iter = line.split_whitespace();
        let mut query: (u32, u32, u32, Option<u32>) = (
            iter.next()
                .unwrap()
                .parse()
                .expect("Failed to read element"),
            iter.next()
                .unwrap()
                .parse()
                .expect("Failed to read element"),
            iter.next()
                .unwrap()
                .parse()
                .expect("Failed to read element"),
            iter.next()
                .map(|x| x.parse().expect("Failed to parse query")),
        );
        // Input is 1-indexed, we use 0-indexing
        query.1 = query.1 - 1;
        query.2 = query.2 - 1;
        queries.push(query);
    }
    Input { array, queries }
}
fn produce_outputs() -> io::Result<()> {
    // To use create directory inputs and my_outputs
    // In inputs place the inputs file, call this from main and run
    for i in 0..11 {
        if let Ok(current_dir) = env::current_dir() {
            // Build the full path to the file in the current working directory
            let input_file_name = format!("/inputs/input{}.txt", i);
            let input_file_path = 
                current_dir.to_string_lossy().to_string() + &input_file_name;

            let input = read_input_from_file(input_file_path);
            let mut tree = SegmentTree::new(&input.array);

            // Create output file
            let output_file_name = format!("/my_outputs/output{}.txt", i);
            let output_file_path =
                current_dir.to_string_lossy().to_string() + &output_file_name;
            let mut output_file = File::create(output_file_path)?;

            // Fill output file
            for query in input.queries {
                match query.0 {
                    0 => tree.update(
                        query.1 as usize, 
                        query.2 as usize,
                        query.3.unwrap()
                        ),
                    1 => writeln!(
                        output_file,
                        "{}",
                        tree.max(query.1 as usize, query.2 as usize)
                    )?,
                    _ => println!("input error"),
                }
            }
        }
    }
    Ok(())
}