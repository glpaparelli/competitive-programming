// LEETCODE 42 (Hard): Trapping Rain Water

fn max_water(heights: &[i32]) -> i32 {
    let mut result = 0;

    // initialize left and right pointers
    let mut left = 0; 
    let mut right = heights.len() as i32 - 1;
    
    // maximum height "seen by" the left pointer so far
    let mut left_max = 0; 
    // maximum height "seen by" the right pointer so far
    let mut right_max = 0;

    while left < right {
        // we shift the pointer that has the smaller height value:
        // the smaller bar is what keeps the water at bay. 
        // if we are at a given index `left` and `left_max' is currently 4, 
        // we know that we can store w(left) = left_max - heights[left]. 

        if heights[left as usize] < heights[right as usize] {
            if heights[left as usize] > left_max {
                left_max = heights[left as usize];
            } else {
                result += left_max - heights[left as usize];
            }
            left += 1;
        } else {
            if heights[right as usize] > right_max {
                right_max = heights[right as usize];
            } else {
                result += right_max - heights[right as usize];
            }
            right -= 1;
        }
    }
    // println!("{}", result);
    result
}


fn main(){
	let test1 = [0,1,0,2,1,0,1,3,2,1,2,1];
	let res1 = max_water(&test1);
	assert_eq!(res1, 6);

	let test2 = [4,2,0,3,2,5];
	let res2 = max_water(&test2);
	assert_eq!(res2, 9);
}