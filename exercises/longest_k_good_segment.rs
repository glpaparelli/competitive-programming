// CODEFORCES: Longest k-Good Segment

use std::io;
use std::collections::HashSet;

fn main() {
    let test1 = vec![1,2,3,4,5];
    assert_eq!(longest_kgood_segment(&test1, 5), Some((0, 4)));

    let test2 = vec![6, 5, 1, 2, 3, 2, 1, 4, 5];
    assert_eq!(longest_kgood_segment(&test2, 3), Some((2, 6)));

    let test3 = vec![1, 2, 3];
    assert_eq!(longest_kgood_segment(&test3, 1), Some((0, 0)));
}

fn longest_kgood_segment(array: &Vec<i32>, k: i32) -> Option<(usize, usize)> {
    use std::collections::HashSet;

fn main() {
    // Example input array and k value
    let arr = vec![1, 2, 1, 2, 3, 4, 5, 4];
    let k = 3;

    // Call the find_max_segment function
    let result = find_max_segment(&arr, k);

    // Print the result
    println!("{} {}", result.0 + 1, result.1 + 1);
}

fn find_max_segment(arr: &[i32], k: usize) -> (usize, usize) {
    let mut l = 0;
    let mut r = 0;
    let mut max_l = 0;
    let mut max_r = 0;
    let mut unique_set = HashSet::new();

    while r < arr.len() {
        if unique_set.len() <= k {
            unique_set.insert(arr[r]);
            if unique_set.len() <= k && r - l > max_r - max_l {
                max_l = l;
                max_r = r;
            }
            r += 1;
        } else {
            unique_set.remove(&arr[l]);
            l += 1;
        }
    }

    // Return the left and right indices of the maximum segment
    (max_l, max_r)
}

