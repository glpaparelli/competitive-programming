fn wilbur_and_array(arr: &[i32]) -> i32 {
    let mut result = arr[0].abs();
    for i in 1..arr.len() {
        result += (arr[i] - arr[i-1]).abs();
    }

    result
}

fn main() {
    let arr1 = [1, 2, 3, 4, 5];
    assert_eq!(wilbur_and_array(&arr1), 5);

    let arr2 = [1, 2, 2, 1];
    assert_eq!(wilbur_and_array(&arr2), 3);
}