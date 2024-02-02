// LEETCODE 523 (Medium): Continuous Subarray Sum

use std::collections::HashMap;

fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
    // create the prefix sum array
	let prefix_sums = nums
    .iter()
    .scan(0, |sum, e| {
    *sum += e;
    Some(*sum)
})
.collect::<Vec<_>>();

let mut mods_to_indices: HashMap<i32, usize> = HashMap::new();
    for (i, &prefix_sum) in prefix_sums.iter().enumerate() {
        let modulo = prefix_sum % k;
        // if the current prefix sum has modulo 0 we are done
        if modulo == 0 && i != 0 {
            return true;
        }
        // if the mod has never been seen we add it
        if !mods_to_indices.contains_key(&modulo) {
            mods_to_indices.insert(modulo, i);
        } else {
            // if the mod has been seen we check that is not 
            // the immediate predecessor prefix sum, and return true
            let prev = mods_to_indices[&modulo];
            if prev < i - 1 {
                return true;
            }
        }
    }
    false
}

pub fn check_subarray_sum_optimized(nums: Vec<i32>, k: i32) -> bool {
	let mut mods_to_indices: HashMap<i32, usize> = HashMap::new();
	let mut sum = 0;

	for (i, &num) in nums.iter().enumerate() {
		sum += num;
		sum %= k;

		if sum == 0 && i > 0 {
			return true;
		}	

		// if the mod has never been seen we add it
		if !mods_to_indices.contains_key(&sum) {
			mods_to_indices.insert(sum, i);
		} else {
			// if the mod has been seen we check that is not 
			// the immediate predecessor prefix sum, and return true
			let prev = mods_to_indices[&sum];
			if prev < i - 1 {
				return true;
			}
		}
	}
	false
}


fn main() {
    let test1 = vec![23,2,4,6,7];
	let res1 = check_subarray_sum_optimized(test1, 6);
	assert_eq!(res1, true);

	let test2 = vec![23,2,6,4,7];
	let res2 = check_subarray_sum_optimized(test2, 6);
	assert_eq!(res2, true);

	let test3 = vec![23,2,6,4,7];
	let res3 = check_subarray_sum_optimized(test3, 13);
	assert_eq!(res3, false);
}
