fn longest_bitonic_subsequence(arr: &[i32]) -> i32 {
    let n = arr.len();

    // exactly as LIS
    let mut lis = vec![1; n];
    for i in 1..n {
        for j in 0..i {
            if arr[i] > arr[j] && lis[i] < lis[j] + 1 {
                lis[i] = lis[j] + 1;
            }
        }
    }

    // a decreasing sequence visited in revers
    // is an increasing sequence
    let mut lds = vec![1; n];
    for i in (0..n - 1).rev() {
        for j in (i + 1..n).rev() {
            if arr[i] > arr[j] && lds[i] < lds[j] + 1 {
                lds[i] = lds[j] + 1;
            }
        }
    }

    // return the maximum value of lis[i] + lds[i] - 1
    lis.iter().zip(&lds).map(|(l, d)| l + d - 1).max().unwrap_or(0)
}

fn main() {
    let arr1 = [1, 2, 5, 3, 2];
    let test1 = longest_bitonic_subsequence(&arr1);
    assert_eq!(test1, 5);

    let arr2 = [1, 11, 2, 10, 4, 5, 2, 1];
    let test2 = longest_bitonic_subsequence(&arr2);
    assert_eq!(test2, 6);
}
