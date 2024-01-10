// LEETCODE 162 (Medium): Find Peak Element

fn find_peak_element(nums: &[i32]) -> i32 { 
	let mut left = 0; 
	let mut right = nums.len() - 1; 
	
	while left < right { 
		let mid = left + (right - left) / 2; 
		if nums[mid] < nums[mid + 1] { 
			left = mid + 1; 
		} else { 
			right = mid; 
		} 
	} 
	left as i32
}

fn main(){
	let test1 = [1,2,3,1];
	let res1 = find_peak_element(&test1);
	assert_eq!(res1, 2);

	let test2 = [1,2,1,3,5,6,4];
	let res2 = find_peak_element(&test2);
	assert_eq!(res2, 5);    
}