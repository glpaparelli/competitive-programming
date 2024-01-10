// LEETCODE 1893 (Easy): Check if all Integers in a Range are Covered

// TODO with Line Sweep

fn is_covered(intervals: Vec<Vec<i32>>, left: i32, right: i32) -> bool {
    let mut covered = false;

    for i in left..=right {
        for interval in intervals.iter() {
            if i >= interval[0] && i <= interval[1] {
                covered = true;
                break;
            }
        }

        if covered == false {
            return false;
        }

        covered = false;
    }

    true
}

fn main() {
    let ranges_1 = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let left_1 = 2;
    let right_1 = 5;
    let test_1 = is_covered(ranges_1, left_1, right_1);
    assert_eq!(test_1, true);

    let ranges_2 = vec![vec![1, 10], vec![10, 20]];
    let left_2 = 21;
    let right_2 = 21;
    let test_2 = is_covered(ranges_2, left_2, right_2);
    assert_eq!(test_2, false);
}
