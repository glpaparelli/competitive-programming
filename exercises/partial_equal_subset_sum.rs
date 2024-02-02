fn partial_equal_subset_sum(arr: &[i32]) -> bool {
    let n = arr.len();

    // if the sum of arr cannot be divided by two
    // there can't be two paritions that summed give sum
    let sum: i32 = arr.iter().sum();
    if sum % 2 != 0 {
        return false;
    }
    let sum = sum / 2;
    
    let mut sol = vec![vec![false; (sum + 1) as usize]; n + 1];

    for i in 0..=n {
        sol[i][0] = true;
    }

    for j in 1..=(sum as usize) {
        sol[0][j] = false;
    }

    for i in 1..=n {
        for j in 1..=(sum as usize) {
            // rust handy syntax
            // if arr[i - 1] is greater than j, it means that including
            // the current element i in the subset would make the sum 
            // exceed the current target sum j, hence we do not include i
            sol[i][j] = if arr[i - 1] > (j as i32) {
                sol[i - 1][j]
            } else {
                sol[i - 1][j] || sol[i - 1][j - arr[i - 1] as usize]
            };
        }
    }

    sol[n][sum as usize]
}

fn main() {
    let arr1 = [1, 5, 11, 5];
    let test1 = partial_equal_subset_sum(&arr1);
    assert_eq!(test1, true);

    let arr2 = [1, 3, 5];
    let test2 = partial_equal_subset_sum(&arr2);
    assert_eq!(test2, false);
}
