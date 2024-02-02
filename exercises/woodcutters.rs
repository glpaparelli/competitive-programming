fn woodcutters(trees: &mut [(i32, i32)]) -> i32 {
    let n = trees.len();
    // we always cut the first tree to the left 
    // and the right tree to the right
    let mut cutted = 2;

    for i in 1..n-1 {
        // left fall: if the previous tree is at coordinate x_i-1
        // and the current trees can fall to the left then we cut
        if trees[i-1].0 < trees[i].0 - trees[i].1 {
            cutted += 1;
            continue;
        }

        // can't fall to the left, fall to the right
        if trees[i].0 + trees[i].1 < trees[i + 1].0 {
            cutted += 1;
            // cutted to the right!
            trees[i].0 += trees[i].1;
        }
    }
    cutted
}

fn main() {
    // Set values for trees (x[i], h[i])
    let mut test1 = [(1, 2), (2, 1), (5, 10), (10, 9), (19, 1)];
    assert_eq!(woodcutters(&mut test1), 3);

    let mut test2 = [(1, 2), (2, 1), (5, 10), (10, 9), (20, 1)];
    assert_eq!(woodcutters(&mut test2), 4);
}
