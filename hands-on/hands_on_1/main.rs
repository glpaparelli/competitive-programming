use std::collections::VecDeque;

struct Node {
    key: u32,
    id_left: Option<usize>,
    id_right: Option<usize>,
}
impl Node {
    fn new(key: u32) -> Self {
        Self {
            key,
            id_left: None,
            id_right: None,
        }
    }
}

struct Tree {
    nodes: Vec<Node>,
}
impl Tree {
    pub fn with_root(key: u32) -> Self {
        Self {
            nodes: vec![Node::new(key)],
        }
    }

    /// Adds a child to the node with `parent_id` and returns the id of the new node.
    /// The new node has the specified `key`. The new node is the left  child of the  
    /// node `parent_id` iff `is_left` is `true`, the right child otherwise.
    pub fn add_node(&mut self, parent_id: usize, key: u32, is_left: bool) -> usize {
        assert!(
            parent_id < self.nodes.len(),
            "Parent node id does not exist"
        );
        if is_left {
            assert!(
                self.nodes[parent_id].id_left.is_none(),
                "Parent node has the left child already set"
            );
        } else {
            assert!(
                self.nodes[parent_id].id_right.is_none(),
                "Parent node has the right child already set"
            );
        }

        let child_id = self.nodes.len();
        self.nodes.push(Node::new(key));

        let child = if is_left {
            &mut self.nodes[parent_id].id_left
        } else {
            &mut self.nodes[parent_id].id_right
        };

        *child = Some(child_id);

        child_id
    }

    /// Returns the sum of all the keys in the tree
    pub fn sum(&self) -> u32 {
        self.rec_sum(Some(0))
    }

    /// A private recursive function that computes the sum of
    /// nodes in the subtree rooted at `node_id`.
    fn rec_sum(&self, node_id: Option<usize>) -> u32 {
        if let Some(id) = node_id {
            assert!(id < self.nodes.len(), "Node id is out of range");
            let node = &self.nodes[id];

            let sum_left = self.rec_sum(node.id_left);
            let sum_right = self.rec_sum(node.id_right);

            return sum_left + sum_right + node.key;
        }
        0
    }

    /// Checks if the tree is a binary search tree (BST).
    pub fn is_bst(&self) -> bool {
        self.rec_is_bst(Some(0), u32::MIN, u32::MAX)
    }
    /// Checks if the tree rooted in the given node is a binary search tree (BST).
    fn subtree_is_bst(&self, node_id: usize) -> bool {
        self.rec_is_bst(Some(node_id), u32::MIN, u32::MAX)
    }
    /// A private helper function for checking if the tree is a binary search tree (BST).
    fn rec_is_bst(&self, node_id: Option<usize>, min: u32, max: u32) -> bool {
        if let Some(id) = node_id {
            assert!(id < self.nodes.len(), "Node id is out of range");
            let node: &Node = &self.nodes[id];

            // Check if the node's key is not within the valid return false
            if min > node.key || max < node.key {
                return false
            }

            // Recursively check the left and right subtrees
            let left_is_bst = self.rec_is_bst(node.id_left, min, node.key);
            let right_is_bst = self.rec_is_bst(node.id_right, node.key, max);
            left_is_bst && right_is_bst
        } else {
            true // Empty subtree is a BST
        }
    }

    /// Checks if the tree is balanced.
    pub fn is_balanced(&self) -> bool {
        self.rec_is_balanced(Some(0)) != -1
    }
    /// Checks if the tree rooted in the given node is balanced.
    fn subtree_is_balanced(&self, node_id: usize) -> bool {
        self.rec_is_balanced(Some(node_id)) != -1
    }
    /// A private helper function to compute the height of the tree 
    /// rooted at `node_id` and check if it is balanced.
    /// Returns `None` if the subtree is not balanced, otherwise returns`Some(height)`.
    fn rec_is_balanced(&self, node_id: Option<usize>) -> i32 {
        if let Some(id) = node_id {
            assert!(id < self.nodes.len(), "Node id is out of range");
            let node = &self.nodes[id];

            // Recursively get the heights of the left and right subtrees
            let left_height = self.rec_is_balanced(node.id_left);
            let right_height = self.rec_is_balanced(node.id_right);

            // Check if the left and right subtrees are balanced (height difference <= 1)
            if left_height != -1 && right_height != -1 && (left_height - right_height).abs() <= 1 {
                1 + left_height.max(right_height)
            } else {
                -1 // Unbalanced subtree
            }
        } else {
            0 // Empty subtree is balanced
        }
    }

    /// Checks if the tree is a max-heap.
    pub fn is_max_heap(&self) -> bool {
        self.iter_is_max_heap(Some(0))
    }
    /// Checks if the tree rooted in the given node is a max-heap.
    fn subtree_is_max_heap(&self, node_id: usize) -> bool {
        self.iter_is_max_heap(Some(node_id))
    }
    /// A private helper function to check if the subtree rooted
    /// at `node_id` satisfies the max-heap property.
    fn iter_is_max_heap(&self, node_id: Option<usize>) -> bool {
        if let Some(id) = node_id {
            assert!(id < self.nodes.len(), "Node id is out of range");
            
            let mut queue: VecDeque<&Node> = VecDeque::new();
            let mut last_level: bool = false;

            let node: &Node = &self.nodes[id];
            queue.push_back(node);

            // While there are nodes in the frontier
            while let Some(curr_node) = queue.pop_front() {
                // If a leaf node has been found previously, and the current 
                // node has a child, it violates the completeness property
                if last_level && (curr_node.id_left.is_some() || curr_node.id_right.is_some()) {
                    return false;
                }
                
                // If the current node has no left child, it should have no right child.
                if curr_node.id_left.is_none() && curr_node.id_right.is_some() {
                    return false;
                }

                // Check left child
                if let Some(id_left) = curr_node.id_left {
                    let left_child = &self.nodes[id_left];

                    // the current node has a left child but no right child,
                    // it should be the last level, and all subsequent nodes
                    // should be leaves (left-aligned).
                    if curr_node.id_right.is_none() {
                        last_level = true;  
                    }  

                    // Check max-heap property for the left child
                    if curr_node.key < left_child.key {
                        return false;
                    }
                    queue.push_back(left_child);            
                } 

                // Check if the current node has a right child
                if let Some(id_right) = curr_node.id_right {
                    let right_child = &self.nodes[id_right];

                    // Check max-heap property for the right child
                    if curr_node.key < right_child.key {
                        return false
                    }
                    queue.push_back(right_child);
                } 

                // if the current node is a leaf all the next nodes should be leaves
                if curr_node.id_left.is_none() && curr_node.id_right.is_none() {
                    last_level = true;
                } 
            }
            true // If the traversal completes without violating any condition, it's a max-heap.
        } else {
            // the empty tree is a max heap
            true
        }
    }
}

fn main() {
    // just to snooze warnings of unused code
    let mut tree: Tree = Tree::with_root(10);
    tree.add_node(0, 1, true); // id 1
    tree.sum();
    tree.is_balanced();
    tree.is_bst();
    tree.subtree_is_balanced(1);
    tree.subtree_is_bst(1);
    tree.is_max_heap();
    tree.subtree_is_max_heap(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sum() {
        let mut tree = Tree::with_root(10);

        assert_eq!(tree.sum(), 10);

        tree.add_node(0, 5, true); // id 1
        tree.add_node(0, 22, false); // id 2

        assert_eq!(tree.sum(), 37);

        tree.add_node(1, 7, false); // id 3
        tree.add_node(2, 20, true); // id 4

        assert_eq!(tree.sum(), 64);
    }

    #[test]
    fn test_is_bst() {
        let mut tree: Tree = Tree::with_root(10);
        // with one node is a bst
        assert!(tree.is_bst());

        /*
                10
               /  \
              8    12
        */
        tree.add_node(0, 8, true); // id 1
        tree.add_node(0, 12, false); // id 2
        assert!(tree.is_bst());

        /*
                 10
               /    \
              8      12
             / \    /  \
            5   9  11  15
        */
        tree.add_node(1, 5, true); // id 3
        tree.add_node(1, 9, false); // id 4
        tree.add_node(2, 11, true); // id 5
        tree.add_node(2, 15, false); // id 6
        assert!(tree.is_bst());

        /*
                 10
               /    \
              8      12
             / \    /  \
            5   9  11  15
                         \
                          4
        */
        tree.add_node(6, 4, false); // id 7
        assert!(!tree.is_bst());

        // subtree rooted in 8 is a BST
        assert!(tree.subtree_is_bst(1));
        // subtree rooted in 12 is not a BST
        assert!(!tree.subtree_is_bst(2));
        // subtree rooted in 4 is a BST
        assert!(tree.subtree_is_balanced(7));

        /* 
                    10
                   /
                  8 
                 /
                6 
               /
              4  
        */
        let mut tree1 = Tree::with_root(10); // id 0
        tree1.add_node(0, 8, true); // id 1
        tree1.add_node(1, 6, true); // id 2
        tree1.add_node(2, 4, true); // id 2
        assert!(tree1.is_bst()) // is BST
    }

    #[test]
    fn test_is_balanced() {
        let mut tree: Tree = Tree::with_root(10);
        // with one node is balanced
        assert!(tree.is_balanced());

        /*
                    10
                   /
                  3
        */
        tree.add_node(0, 3, true); // id 1
        assert!(tree.is_balanced());

        /*
                    10
                   /
                  3
                 /
                4
        */
        tree.add_node(1, 4, true); // id 2
        assert!(!tree.is_balanced()); // not balanced

        /*
                    10
                   /
                  3
                 / \
                4   5
        */
        tree.add_node(1, 5, false); // id 3
        assert!(!tree.is_balanced()); // not balanced
        assert!(tree.subtree_is_balanced(1)); // subtree rooted in 3 is balanced

        /*
                    10
                   /  \
                  3    8
                 / \
                4   5
        */
        tree.add_node(0, 8, false); // id 4
        assert!(tree.is_balanced()); // balanced

        /*
                  10
               /       \
              3         8
             / \       /  \
            4   5     3    9
                            \
                             2
                              \
                               3
        */
        tree.add_node(4, 3, true); // id 5
        tree.add_node(4, 9, false); // id 6
        tree.add_node(6, 2, false); // id 7
        tree.add_node(7, 3, false); // id 8
        assert!(!tree.is_balanced());
    }

    #[test]
    fn test_is_max_heap() {
        let mut tree: Tree = Tree::with_root(10);
        // a one-node tree is a max-heap
        assert!(tree.is_max_heap());
        /*
                10
                  \
                   9
        */
        tree.add_node(0, 9, false); // id 1
        assert!(!tree.is_max_heap()); // not a max-heap (completeness)

        /*
                10
               /  \
              7    9
        */
        tree.add_node(0, 7, true); // id 2
        assert!(tree.is_max_heap()); // a max heap

        /*
                 10
               /    \
              7      9
             /     
            5     
        */
        tree.add_node(2, 5, true); // id 3
        assert!(tree.is_max_heap()); // a max heap

        /*
                 10
               /    \
              7      9
             / \      \
            5   6      5
        */
        tree.add_node(2, 6, false); // id 4
        tree.add_node(1, 5, false); // id 5
        assert!(!tree.is_max_heap()); // not a max heap (completeness)

        /*
                 10
               /    \
              7      9
             / \    / \
            5   6  4   5
        */
        tree.add_node(1, 4, true); // id 6
        assert!(tree.is_max_heap()); // a max heap

        /*
                   10
                 /    \
                7      9
               / \    / \
              5   6  4   5
             /
            8    
        */
        tree.add_node(3, 8, true);
        assert!(!tree.is_max_heap()); // not a max heap (max-heap property)

        
        // another tree
        /*
                    10
                      \
                       9
                        \
                         8
        */
        let mut tree1: Tree = Tree::with_root(10); // id 0
        tree1.add_node(0, 9, false); // id 1
        tree1.add_node(1, 8, false); // id 2
        assert!(!tree1.is_max_heap()); // not a max heap (completeness)

        /*
                    10
                      \
                       9
                      / \
                     7   8
        */
        tree1.add_node(1, 7, true); // id 3
        assert!(!tree1.is_max_heap()); // not a max heap

        /*
                    10
                   /  \
                  5    9
                      / \
                     7   8
        */
        tree1.add_node(0, 5, true); // id 4
        assert!(!tree1.is_max_heap()); // not a max heap

        /*
                    10
                   /  \
                  5    9
                 /    / \
                3    7   8
        */
        tree1.add_node(4, 3, true); // id 5
        assert!(!tree1.is_max_heap()); // not a max heap

        /*
                     10
                   /    \
                  5      9
                 / \    / \
                3   6  7   8 
        */
        tree1.add_node(4,6, false); // id 6
        assert!(!tree1.is_max_heap()); // not a max heap (max-heap property)

        // another tree
        /*
                    10
                  /    \
                 6      7
                /      /
               5      6
        */
        let mut tree2: Tree = Tree::with_root(10); // id 0
        tree2.add_node(0, 6, true); // id 1
        tree2.add_node(0, 7, false); // id 2
        tree2.add_node(1, 5, true); // id 3
        tree2.add_node(2, 6, true); // id 4
        assert!(!tree2.is_max_heap()); // not a max heap (completeness)
    }
}
