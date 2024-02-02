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
    if k == 0 {
        return None;
    }
    if k == 1 {
        return Some((0, 0));
    }
    
    let mut w_size = 1; // store the current window size
    let mut left = 0; // left delimiter of the window
    let mut right = 0; // right delimiter of the window
    let mut pointer = 1; // points always to element just outside the window

    // store the distinct elements in the window right now
    let mut distincts: HashSet<i32> = HashSet::new();
    
    // the window starts being of just one element, the first
    distincts.insert(array[0]);
    let mut w_max_size = 1;

    // to save the left and right edge of the longest k-good segment (window)
    let mut left_res;
    let mut right_res;

    // we proceed until the pointer is outside of the array
    while pointer != array.len() {

        // the next element is to be included in the window, 
        // since it is a consecutive number of the last element
        if (array[right] - array[pointer]).abs() == 1 {
            // we insert in distincts
            distincts.insert(array[right+1]);
            // expand the window, the window is now bigger as the right edge has shifted
            right += 1;
            w_size += 1;
            // we move the pointer, so it point the right next element outside the window
            pointer += 1;
            
            // if the current window is bigger than the maximum window so far
            if w_size >= w_max_size {
                // update the new max size
                w_max_size = w_size;
                // store the indices of the window
                left_res = left;
                right_res = right;
                
                // if the window has exactly k distinct elements we are done
                if distincts.len() == k as usize {
                    return Some((left_res, right_res));
                }
            }
        } else {
            // we need to shift and reset the window
            // the start of the window becomes the pointer
            left = pointer;
            // the window is made of one element
            right = left;
            w_size = 0;
            // points to the element right next to the window
            pointer += 1;

            // reset the distinct elements in the window
            distincts.clear();
        }
    }
    None
}

