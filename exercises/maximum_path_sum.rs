// LEETCODE 124 (Hard): Bianry Tree Maxium Path Sum

use std::rc::Rc;
use std::cell::RefCell;

// definition for a binary tree node.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
  pub val: i32,
  pub left: Option<Rc<RefCell<TreeNode>>>,
  pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
  #[inline]
  pub fn new(val: i32) -> Self {
    TreeNode {
      val,
      left: None,
      right: None
    }
  }
}
   
fn find_max_path_sum(node: Option<Rc<RefCell<TreeNode>>>, max: &mut i32) -> i32 {
    match node {
        Some(n) => {
            let l = i32::max(0, find_max_path_sum(n.borrow().left.clone(), max));
            let r = i32::max(0, find_max_path_sum(n.borrow().right.clone(), max));
            *max = i32::max(*max, l + r + n.borrow().val);
            i32::max(l, r) + n.borrow().val
        }
        None => 0,
    }
}

pub fn max_path_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut max = i32::MIN;  
    find_max_path_sum(root, &mut max);
    max
}

fn main() {
    //tree: root = [1,2,3]
    let tree1 = Some(Rc::new(RefCell::new(TreeNode {
        val: 1,
        left: Some(Rc::new(RefCell::new(TreeNode::new(2)))),
        right: Some(Rc::new(RefCell::new(TreeNode::new(3)))),
    })));
    let test1 = max_path_sum(tree1);
    assert_eq!(test1, 6);

    // Test tree: root = [-10,9,20,null,null,15,7]
    let tree2 = Some(Rc::new(RefCell::new(TreeNode {
        val: -10,
        left: Some(Rc::new(RefCell::new(TreeNode::new(9)))),
        right: Some(Rc::new(RefCell::new(TreeNode {
            val: 20,
            left: Some(Rc::new(RefCell::new(TreeNode::new(15)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(7)))),
        }))),
    })));
    let test2 = max_path_sum(tree2);
    assert_eq!(test2, 42);
}
