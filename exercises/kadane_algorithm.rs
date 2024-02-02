// LEETCODE 53 (Medium): Maximum Subarray

fn subarray_max_sum(array: &[i32]) -> i32 {
	let mut max = 0;
	let mut sum = 0;

	for i in 0..array.len() {
		if sum > 0 {
			sum = sum + array[i]
		} else {
			// restart the subarrayay from here
			sum = array[i]
		} 
		// the considered subarrayay has maximum sum
		if sum > max {
			max = sum;
		}
	}
	// println!("{}", max);
	max
}

fn main(){
	let test1 = [-2,1,-3,4,-1,2,1,-5,4];
	let res1 = subarray_max_sum(&test1);
	assert_eq!(res1, 6);

	let test2 = [1];
	let res2 = subarray_max_sum(&test2);
	assert_eq!(res2, 1);

	let test3 = [5,4,-1,7,8];
	let res3 = subarray_max_sum(&test3);
	assert_eq!(res3, 23);
}
