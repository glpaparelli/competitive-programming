fn longest_common_subsequence(s1: &str, s2: &str) -> usize {
    let n = s1.len() + 1;
    let m = s2.len() + 1;
    let mut lcs = vec![vec![0; m]; n];

    for i in 0..n {
        lcs[i][0] = 0;
    }
    for j in 0..m {
        lcs[0][j] = 0;
    }

    for i in 1..n {
        for j in 1..m {
            if s1.chars().nth(i - 1) == s2.chars().nth(j - 1) {
                lcs[i][j] = 1 + lcs[i - 1][j - 1];
            } else {
                lcs[i][j] = lcs[i - 1][j].max(lcs[i][j - 1]);
            }
        }
    }

    lcs[n - 1][m - 1]
}

fn main() {
    // test_1
    let mut s1 = String::from("ABCDGH");
    let mut s2 = String::from("AEDFHR");
    assert_eq!(longest_common_subsequence(&s1, &s2), 3); 

    // test_2
    s1 = String::from("ABC");
    s2 = String::from("AC");
    assert_eq!(longest_common_subsequence(&s1, &s2), 2); 
}
