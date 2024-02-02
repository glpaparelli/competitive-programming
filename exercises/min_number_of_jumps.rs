use std::cmp::min;
use std::i32;

fn min_num_of_jumps(arr: &[i32]) -> i32 {
    let n = arr.len();
    let mut jumps = vec![i32::MAX; n];
    jumps[0] = 0;

    if n == 0 || arr[0] == 0 {
        return -1;
    }

    for i in 1..n {
        for j in 0..i {
            if i as i32 <= j as i32 + arr[j] && jumps[j] != i32::MAX {
                jumps[i] = min(jumps[i], jumps[j] + 1);
                break;
            }
        }
    }

    jumps[n - 1]
}

fn main() {
    let arr1 = [1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9];
    let test1 = min_num_of_jumps(&arr1);
    assert_eq!(test1, 3);

    let arr2 = [1, 4, 3, 2, 6, 7];
    let test2 = min_num_of_jumps(&arr2);
    assert_eq!(test2, 2);
}

