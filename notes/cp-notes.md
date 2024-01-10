# 1 - Contiguous Sub-Array with Max Sum
![[Screenshot from 2024-01-05 09-29-59.png | center | 700]]
## 1.1 - Trivial Solution, O(n^3)
This is the obvious brute force method, where we compute every single subarray and store the current maximum sum.
```rust
fn subarray_max_sum(arr: &[i32]) {
	let mut max = 0; 
	for i in 0..arr.len() {
		for j in i..arr.len() {
			let mut sum = 0; 
			for k in i..j {
				sum = sum + a[k];
			}
			if sum > max {
				max = sum
			}
		}
	}
	println!("{}", max);
}
```
### 1.1.1 - Optimized Brute Force, O(n^2)
We notice that the most inner for loop is useless, as we can compute the sum of the current subarray, for every subarray, with just the `j` index
```rust
fn subarray_max_sum(arr: &[i32]) {
	let mut max = 0; 
	for i in 0..arr.len() {
		for j in i..arr.len() {
			sum = sum + a[j]; 
			if sum > max {
				max = sum;
			}
		}
	}
	println!("{}", max);
}
```
## 1.2 - Optimal Solution: Kadane's algorithm, O(n)
Kadane's algorithm is based on two properties of the subarray with maximum sum: 
1) sum of values in any prefix of the optimal subarray is positive. By contradiction, **remove a negative prefix to get a subarray with a larger sum**
2) **the value that precede the first value of the optimal subarray is negative.** By contradiction, include this positive value to get a subarray with a larger sum

```rust 
fn subarray_max_sum(arr: &[i32]) {
	let mut max = 0;
	let mut sum = 0;

	for i in 0..arr.len() {
		if sum > 0 {
			sum = sum + arr[i]
		} else {
			sum = arr[i]
		} 

		if sum > max {
			max = sum;
		}
	}
	println!("{}", max);
}
```

![[IMG_0394.png | center | 600]]
# 2 - Trapping Rain Water
Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining. 

![[Screenshot from 2024-01-05 11-17-40.png | center | 400]]
## 2.1 -Trivial Solution, O(n^2)
Traverse every array element and find the highest bars on the left and right sides. 
Take the smaller of two heights. 
The difference between the smaller height and the height of the current element is the amount of water that can be stored in this array element.
## 2.2 - Smart Solution: Precomputation, O(n)
**The point is to think locally, not globally.**
**We can compute the left and right leaders.** 
Given the array $h$ of heights, we compute:
$$
\begin{align}
	\text{LL} &= \max_{j < i} h[j] \\
	\text{RL} &= \max_{j > i} h[j]
\end{align}
$$
Then, how many units of water we can find on top of the cell $i$?
$$w(i) = \min(\max_{j < i}h[j], \max_{j > i}) - h[i]$$
![[Pasted image 20240105111518.png | center | 500]]
## 2.3 - Optimal Solution: Two Pointers, O(N)
The idea is taken from the previous solution, where we only use two variables to store the currently "meaningful" leaders. 

We take two pointers, `left` and `right`. We initialize `left` to `0` and `right` to the last index `N-1`. 
We also create two variables, `left_max` and `right_max`. They represent the maximum left/right height seen so far. 
Since `left` is the first `left_max` is `0`, same for `right_max`. This is intuitive: we can't store water in the first and last column. 

Now we iterate, as long `left <= right`. 
We have to decide which pointer we have to shift: we shift the pointer that has smaller height value: 
- if `heights[left] <= heights[right]` we shift `left`
- we shift `right` otherwise

Now, if we have that `heights[left] > left_max` we can't store any water over the position pointed by `left`, **it would fall as left_max is not high enough**. 
![[Pasted image 20240105122707.png | center | 600]]
Otherwise we compute the amount of water stored in `left`, as always with `left_max - heights[left]`. 
And then we finally shift `left` by `1`. 

The reasoning with the `right` pointer is the same. 

Implementing the solution in rust we obtain: 
```rust
fn max_water(heights: &[i32]) {
	
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
		if heights[left as usize] < heights[right as usize] {
			if heights[left as usize] > left_max {
				left_max = heights[left as usize];
			} else {
				result += left_max - heights[left as usize];
			}
			// shift left
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
	println!("{}", result);
}
```
# 3 - Find Peak Element
![[Screenshot from 2024-01-05 15-00-58.png | center | 700]]
To solve this problem we use **Binary Search**. 
We start by giving the basics of this technique. 
## 3.1 - Binary Search
**An efficient algorithm that searches for a given key within a sorted array of items.**
Binary search repeatedly divides the search range in half until the target element is found or the search range becomes empty, resulting in a time complexity of $O(\log n)$

Binary search is a technique that falls under the umbrella of the **divide-and-conquer** paradigm, that tackles a complex problem by breaking it down into smaller, more manageable subproblems of the same type: 
- **divide:** the initial problem instance is partitioned into smaller subinstances of the same problem
- **solve:** these subinstances are then solved recursively. If a subinstance reaches a certain manageable size, a straightforward approach is employed to solve it directly
- **combine:** the solutions obtained from the subinstances are combined to obtain the final solution for the original, larger instance of the problem

Considering the binary search, we have: 
- **divide:** the array is divided into two roughly equal halves, centering around the middle element of the array
- **solve:** compare the middle element of the array with the searched key. If the middle element is a match, the search stop successfully. If not, we recursively search for the key only in one of the two halves that may contain the key, based on whether the desired key is greater or lesser than the middle element
- **combine:** there is nothing to combine

A generic implementation of binary search is the following: 
```rust 
fn binary_search<T: Ord>(arr: &[T], key: T) -> Option<usize> {
    let mut low = 0;
    let mut high = arr.len();

    while low < high {
	    // never use middle = (low + high)/2, as it can
	    // lead to overflow if (low+high) > usize::MAX
        let middle = low + (high - low)/2;

        match key.cmp(&arr[middle]) {
            std::cmp::Ordering::Equal   => return Some(middle),
            std::cmp::Ordering::Less    => high = middle,
            std::cmp::Ordering::Greater => low = middle + 1,
        }
    }
    None
}
```

It is also important to observe that when there are multiple occurrences of the searched key, the function returns the position of the first encountered occurrence, not necessarily the first occurrence in the vector.
However, **it is often very useful to report the position of the first (or last) occurrence of the searched key**. We can obtain this behavior with the following implementation.

```rust
fn binary_search<T: Ord>(arr: &[T], key: T) -> Option<usize> {
    let mut low = 0;
    let mut high = arr.len(); // note that high is excluded

    let mut ans = None;

    while low < high {
        let middle = low + (high - low) / 2;

        match key.cmp(&arr[middle]) {
            std::cmp::Ordering::Equal => {
                ans = Some(middle);
                high = middle
            }
            std::cmp::Ordering::Less => high = middle,
            std::cmp::Ordering::Greater => low = middle + 1,
        }
    }
    
    ans
}
```
In this implementation, when a match is found, we do not immediately return its position. Instead, we update the `ans` variable and set `high` to the position of this occurrence. 
This way, we continue the search in the first half of the array, seeking additional occurrences of the `key`. 
If there are more matches, `ans` will be further updated with smaller positions.
## 3.2 - Solution
**We can use the binary search philosophy to solve the problem.**
We compare the middle element with its neighbors to determine if it is a peak. 
If the middle element is not a peak and its left neighbor is greater, then the peak must be in the left half; otherwise, it must be in the right half. 

```rust
fn find_peak_element(nums: &[i32]) -> i32 { 
	let mut left = 0; 
	let mut right = nums.len() - 1; 
	
	while left < right { 
		let mid = left + (right - left) / 2; 
		if nums[mid] < nums[mid + 1] { 
			left = mid + 1; 
		} else { 
			right = mid; 
		} 
	} 
	left as i32
}
```

Mind that the definition about the out-of-bound elements 
`nums[-1] = nums[nums.len()] = - infinite` is crucial to the solution.

We can also have the recursive solution: 
```java 
	public int findPeakElement(int[] nums) {
        return peek_bs(nums, 0, nums.length-1);
    }

    private int peek_bs(int[] nums, int start, int end){
        if(start == end)
            return start; 
            
        int middle = start + (end - start)/2; 
        if(nums[middle] > nums[middle+1])
            return peek_bs(nums, start, middle);
        else
            return peek_bs(nums, middle+1, end);
    }
```
# 4 - Maximum Path Sum
Given a binary tree in which each node element contains a number. Find the maximum possible path sum from one special node to another special node.
**Note:** Here special node is a node which is connected to exactly one different node.

**Example:**
**Input:**     
```plaintext
						            3                               
						           / \                          
						          4   5                     
						         / \      
						      -10   4  
```
**Output:** 16
**Explanation:**
Maximum Sum lies between special node 4 and 5.
4 + 4 + 3 + 5 = 16.

Differently said: **Find the maximum possible sum from one leaf node to another.**
![[Pasted image 20240108100401.png | center | 490]]

**To solve this problem we need to use a tree traversal.**
Fist we go back and give the basics. 
## 4.1 - Tree Traversals 
**Tree traversal** (also known as tree search and walking the tree) **is a form of graph traversal and refers to the process of visiting** (e.g. retrieving, updating, or deleting) **each node in a tree data structure, exactly once.** 
**Such traversals are classified by the order in which the nodes are visited.**

Traversing a tree involves iterating over all nodes in some manner. 
Because from a given node there is more than one possible next node some nodes must be deferred, aka stored, in some way for later visiting. 
This is often done via a stack (LIFO) or queue (FIFO). 
As a tree is a self-referential (recursively defined) data structure, traversal can be defined by recursion.
In these cases the deferred nodes are stored implicitly in the call stack.

Depth-first search is easily implemented via a stack, including recursively (via the call stack), while breadth-first search is easily implemented via a queue.
### 4.1.1 - Depth-First Search
In depth-first search (DFS), **the search tree is deepened as much as possible before going to the next sibling.**

To traverse binary trees with depth-first search, perform the following operations at each node:
1) If the current node is empty then return.
2) Execute the following three operations in a certain order:
	1) **N:** Visit the current node.
	2) **L:** Recursively traverse the current node's left subtree.
	3) **R:** Recursively traverse the current node's right subtree.

**There are three methods at which position of the traversal relative to the node (in the figure: red, green, or blue) the visit of the node shall take place.** 
**The choice of exactly one color determines exactly one visit of a node as described below.**
![[Screenshot from 2024-01-08 09-19-39.png | center | 350]]

**Complexity Analysis:**
- **Time Complexity:** O(N) where N is the total number of nodes. 
- **Auxiliary Space:** 
	- **O(1)** if no recursion stack space is considered. 
	- Otherwise, **O(h)** where h is the height of the tree
	- In the worst case, **h** can be the same as **N** (when the tree is a skewed tree)
	- In the best case, **h** can be the same as **log N** (when the tree is a complete tree)
#### 4.1.1.1 - PreOrder
1) visit the current node, in the figure: position red
2) recursively traverse the current node's left subtree
3) recursively traverse the current node's right subtree

The preorder traversal is a topologically sorted one, because the parent node is processed before any of its child nodes is done. 

![[Screenshot from 2024-01-08 09-27-13.png | center | 300]]
#### 4.1.1.2 - PostOrder
1) recursively traverse the current node's left subtree
2) recursively traverse the current node's right subtree
3) visit the current node, in the figure: position blue

```plaintext 
procedure postorder(node)
	if node = null
		return 
	postorder(node.left)
	postorder(node.right)
	visit(node)
```
#### 4.1.1.3 - InOrder
1) recursively traverse the current node's left subtree
2) visit the current node, in the figure: position green
3) recursively traverse the current node's right subtree

In a binary search tree ordered such that in each node the key is greater than all keys in its left subtree and less than all keys in its right subtree, in-order traversal retrieves the keys in ascending sorted order. 

![[Screenshot from 2024-01-08 09-29-11.png | center | 300]]

## 4.2 Trivial Solution, O(n^2)
A simple solution is to traverse the tree and do following for every traversed node X: 

1. Find maximum sum from leaf to root in left subtree of X  $\clubsuit$
3. Find maximum sum from leaf to root in right subtree of X. 
4. Add the above two calculated values and X->data and compare the sum with the maximum value obtained so far and update the maximum value. 
5. Return the maximum value.

---
$\clubsuit$  **Max Sum leaf to root path in a Binary Tree:**
Given a Binary Tree, find the maximum sum path from a leaf to root.
For example, in the following tree, there are three leaf to root paths:
- 8 -> -2 -> 10
- -4 -> -2 -> 10
- 7 -> 10. 

The sums of these three paths are 16, 4 and 17 respectively. 
The maximum of them is 17 and the path for maximum is 7 -> 10.
```
					              10  
				                 /  \  
				               -2    7  
				              /  \       
				             8   -4
```

**Solution:**
1) First find the leaf node that is on the maximum sum path.
2) Once we have the target leaf node, we can print the maximum sum path by traversing the tree. 

Here we provide a Java pseudo-implementation of the problem: 
```java 
	// wrapper used so that max_no can be updated among function calls
	class Maximum {
		int value = Integer.MIN_VALUE;
	}

	Node targetLeaf = null;

	// sets the targetLeaf to refer the leaf node of the maximum path sum.
	// returns the maximum sum using maxSum
	void getTargetLeaf(
		Node node, 
		Maximum maxSum,
		int currentSum
	){
		if (node == null)
			return;

		// update current sum to hold sum of nodes 
		// on path from root to this node
		currentSum = currentSum + node.data;

		// if this is a leaf node and path to this node
		// has maximum sum so far, then make this node
		// targetLeaf
		if (node.left == null && node.right == null) {
			if (currentSum > maxSum.value) {
				maxSum.value = currentSum;
				targetLeaf = node;
			}
		}

		// If this is not a leaf node recur down to find targetLeaf
		getTargetLeaf(node.left, maxSum, currentSum);
		getTargetLeaf(node.right, maxSum, currentSum);
	}
	
	// returns the maximum sum and prints the nodes on max sum path
	int maxSumPath(Maximum maxSum) {
		// base case
		if (root == null)
			return 0;

		// find the target leaf and maximum sum
		getTargetLeaf(root, maxSum, 0);

		// print the path from root to the target leaf
		printPath(root, targetLeaf);
		return maxSum.value; // return maximum sum
	}

	boolean printPath(Node node, Node targetLeaf) {
		// base case
		if (node == null)
			return false;

		// return true if this node is the target leaf or
		// target leaf is present in one of its descendants
		if (
			node == targetLeaf || 
			printPath(node.left, targetLeaf) || 
			printPath(node.right, targetLeaf)
		){
			System.out.print(node.data + " ");
			return true;
		}

		return false;
	}
```
---

So, how the trivial solution works is clear. 
For every node X we compute: 
- the maximum path sum from X.left to a leaf, O(n)
- the maximum path sum from X.right to a leaf, O(n)
- we take the maximum between the two 
- we sum the value of X
- if the sum is greater than the current maximum we update the current maximum

For every node we call twice `maxSumPath`, which costs O(n), hence the total cost is O(n^2)
## 4.3 - Optimal Solution, O(n)
**We can find the maximum sum using single traversal of binary tree**. 
The idea is to maintain two values in recursive calls
1. Maximum root to leaf path sum for the subtree rooted under current node. 
2. The maximum path sum between leaves (desired output).

For every visited node X, we find the maximum root to leaf sum in left and right subtrees of X. We add the two values with X->data, and compare the sum with maximum path sum found so far.

As before, here a pseudo-implementation using Java 

```java
// Maximum is passed around so that the
// same value can be used by multiple recursive calls.
class Maximum {
	int value = Integer.MIN_VALUE;
}

public int maxPathSum(TreeNode root) {
	Maximum max = new Maximum();
	findMaxPathSum(root, max);	
	return max.value;
}

// calculates two things:
// 1) maximum path sum between two leaves, which is stored in max.
// 2) maximum root to leaf path sum, which is returned.
private int findMaxPathSum(TreeNode node, Maximum max) {
	if (node == null) 
		return 0;

	// find maximum root-to-leaf-sum in left and right subtrees
	int left = Math.max(0, findMaxPathSum(node.left, max));
	int right = Math.max(0, findMaxPathSum(node.right, max));
	// update the max sum path from leaf to leaf
	max.value = Math.max(max.value, left + right + node.val);
	
	return Math.max(left, right) + node.val;
}
```

In "one pass" (aka one traversal) we do what we did in the trivial solution.
The reasoning is the following: 
- take a node X
- compute the max sum path from X.left to a leaf k and the max sum path from X.right to a leaf j. 
- the sum path from k to j (passing by X) is greater than the one previously stored? then we update the max
# 5 - Tree Representations
TODO
# 6 - Binary Search Tree
A binary search tree (BST), also called an ordered or sorted binary tree, **is a rooted binary tree data structure with the key of each internal node being greater than all the keys in the respective node's left subtree and less than the ones in its right subtree.** 
The time complexity of operations on the binary search tree is linear with respect to the height of the tree.

Binary search trees allow binary search for fast lookup, addition, and removal of data items. 
Since the nodes in a BST are laid out so that **each comparison skips about half of the remaining tree**, the lookup performance is proportional to that of binary logarithm.

The complexity analysis of BST shows that, on average, the insert, delete and search takes $O(\log n)$ for n nodes. 
In the worst case, they degrade to that of a singly linked list: $O(n)$

![[Pasted image 20240109090705.png | center | 250]]
# 7 - Two Pointers Trick
Two pointers is really an easy and effective technique that is typically used for searching pairs in a sorted array.
## 7.1 - Two Sum in a Sorted Array
Given a sorted array A (sorted in ascending order), having N integers, find if there exists any pair of elements (A[i], A[j]) such that their sum is equal to X.
### 7.1.1 - Naive Solution, O(n^2)
The naive approach is obvious and takes O(n^2), we simply scan the array and we return when we find two numbers that adds up to X. 
```java
boolean isPairSum(int[] A, int X){
	for(int i = 0; i < A.length; i++){
		for(int j = i+1; j < A.length; j++){
			if(A[i] + A[j] == X)
				return true;
			if(A[i] + A[j] > X)
				break; // need a new A[i]
		}
	}
	return false;
} 
```
### 7.1.2 - Smart Solution, O(n)
**We take two pointers, one representing the first element and other representing the last element of the array, and then we add the values kept at both the pointers.** 

If their sum is smaller than X then we shift the left pointer to right or if their sum is greater than X then we shift the right pointer to left, in order to get closer to the sum. 
We keep moving the pointers until we get the sum as X.

```java
public boolean isPairSum(int A[], int X){
	int left = 0;
	int rigth = N - 1;

	while (left < right) {
		// found the pair sum
		if (A[left] + A[rigth] == X)
			return true;
		// smaller than the target, we increase the first
		if (A[left] + A[rigth] < X)
			left++;
		// bigger than the target, we decrease the second
		else
			rigth--;
	}
	return false;
}
```
# 8 - Frogs and Mosquitoes
TODO
# 9 - Maximum Number of Overlapping Intervals
Consider a set of $n$ intervals $[s_i, e_i]$ on a line. 
We say that two intervals $[s_i, e_i]$ and $[s_j, e_j]$ overlaps if and only if their intersection is not empty, i.e., if there exist at least a point $x$ belonging to both intervals. 
Compute the maximum number of overlapping intervals. 

**Example:** 
![[Pasted image 20240109094209.png | center | 550]]
We have a set of 10 intervals, the maximum number of overlapping intervals is 5 (at positions 3 and 4)
## 9.1 - Sweep Line Algorithm
The Sweep Line Algorithm is an algorithmic paradigm used to solve a lot of problems in computational geometry efficiently. 
**The sweep line algorithm can be used to solve problems on a line or on a plane.**

The sweep and line algorithm use an imaginary vertical line **sweeping** over the x-axis. 
As it progresses, we maintain a running solution to the problem at hand. 
The solution is updated when the vertical line reaches a certain key points where some event happen. The type of the event tells us how to update the solution. 
## 9.2 Solution
Let's apply the sweep and line algorithm to the problem above. 
We let the sweep line from left to right and stop at the beginning or at the end of every interval. 
These are the important points at which an event occurs: intervals start or end. 
We also maintain a counter which keeps track of the number of intervals that are currently intersecting the sweep line, along with the maximum value reached by the counter so far. 

For each point we first add to the counter the number of intervals that begin at that point, and then we subtract the number of intervals that end at that point. 
The figure below shows the points touched by the sweep line and the values of the counter:
![[Pasted image 20240109095128.png| center | 550]]
**Observation:** The sweep line only touches points on the x-axis where an event occurs. 
This is important because the number of considered points, and thus the time complexity, is proportional to the number of intervals, and not to the size of the x-axis. 

We provide a Rust implementation: 
- we represent each interesting point as a pair consisting of the point and the kind, which is either `Begin` or `End` 
- we then sort the vector of pairs in increasing order
- we compute every state of the counter and its largest value 

```rust
#[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
enum Event {
    Begin,
    End,
}

pub fn max_overlapping(intervals: &[(usize, usize)]) -> usize {
    let mut pairs: Vec<_> = intervals
        .iter()
        .flat_map(|&(b, e)| [(b, PointKind::Begin), (e, PointKind::End)])
        .collect();

    pairs.sort_unstable();

    pairs
        .into_iter()
        .scan(0, |counter, (_, kind)| {
            if kind == Event::Begin {
                *counter += 1;
            } else {
                *counter -= 1;
            }
            Some(*counter)
        })
        .max()
        .unwrap()
}
```
# 10 - Check if all Integers in a Range are Covered
![[Screenshot from 2024-01-09 10-01-57.png | center | 700]]
## 10.1 - Intuitive Solution 
The following is the intuitive solution.
Its complexity is `O((right - left) * |intervals|)`, and gives the perfect score (0 ms) on leetcode
```rust
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
```
## 10.2 - Sweep Line Solution
TODO
# 11 - Longest k-Good Segment
The array *a* with *n* integers is given. 
Let's call the sequence of one or more *consecutive* elements in a segment. 
Also let's call the segment k-good if it contains no more than _k_ different values.

**Note:** if the distance between two numbers is `abs(1)` then the two numbers are consecutive.

Find any longest k-good segment.
## 11.1 - Sliding Window Solution, O(n)
We use the sliding window approach. 
Simply follow the following implementation
```rust
fn longest_kgood_segment(array: &Vec<i32>, k: i32) -> Option<(usize, usize)> {
	if k == 0 {
		return None;
	}
	if k == 1 {
		return Some((0, 0));
	}
	let mut w_size = 1; // store the current window size
	let mut left = 0; // left delimiter of the window
	let mut right = 0; // right delimiter of the window
	let mut pointer = 1; // points always to element just outside the window

	// store the distinct elements in the window right now
	let mut distincts: HashSet<i32> = HashSet::new();
	// the window starts being of just one element, the first

	distincts.insert(array[0]);
	let mut w_max_size = 1;
	// to save the left and right edge of the longest k-good segment (window)
	let mut left_res;
	let mut right_res;
	
	// we proceed until the pointer is outside of the array
	while pointer != array.len() {
		// the next element is to be included in the window,
		// since it is a consecutive number of the last element
		if (array[right] - array[pointer]).abs() == 1 {
			// we insert in distincts
			distincts.insert(array[right+1]);
			// the window is now bigger as the right edge has shifted
			right += 1;
			w_size += 1;
			// point to the right next element outside the window
			pointer += 1;
			
			// if the current window is bigger than the maximum window so far

			if w_size >= w_max_size {
				// update the new max size
				w_max_size = w_size;
				// store the indices of the window
				left_res = left;
				right_res = right;
				// if the window has exactly k distinct elements we are done
				if distincts.len() == k as usize {
					return Some((left_res, right_res));
				}
			} else {
				// we need to shift and reset the window 
				// the start of the window becomes the pointer
				left = pointer;
				// the window is made of one element
				right = left;
				w_size = 0;
				// points to the element right next to the window
				pointer += 1;
				// reset the distinct elements in the window
				distincts.clear();
			}
		}
	}
	None
}
```
## 11.2 - Sweep Line Solution
TODO
# 12 - Prefix Sums 
**Prefix sums**, also known as cumulative sums or cumulative frequencies, **offer an elegant and efficient way to solve a wide range of problems that involve querying cumulative information about a sequence of values or elements.**

The essence of prefix sums lies in **transforming a given array of values into another array, where each element at a given index represents the cumulative sum of all preceding elements in the original array.**

An example of prefix sum array is shown in the picture: 
![[Pasted image 20240110100959.png | center | 350]]
Where it is clear that $$P[i] = \Sigma_{j=1}^i\ A[k]$$
## 12.1 Prefix Sum using Rust
We can use the combinator `scan` to produce the prefix sums from an iterator. 

`scan` is an **iterator adapter**, it maintains an internal state, initially set to a seed value, which is modified by a closure taking both the current internal state and the current element from the iterator into account.

```rust 
let a = vec![2, 4, 1, 7, 3, 0, 4, 2];

let psums = a
    .iter()
    .scan(0, |sum, e| {
        *sum += e;
        Some(*sum)
    })
    .collect::<Vec<_>>();

assert!(psums.eq(&vec![2, 6, 7, 14, 17, 17, 21, 23]));
```
# 13 - Contiguous Subarray Sum
Given an integer array `nums` and an integer `k`, return `true` if `nums` has a **good subarray** or false otherwise. 
A **good subarray** is a subarray where: 
- its length is **at least** 2 
- the sum of the elements of the subarray is a multiple of `k`

**Note** that:
- A **subarray** is a contiguous part of the array.
- An integer `x` is a multiple of `k` if there exists an integer `n` such that `x = n * k`. `0` is **always** a multiple of `k`.
## 13.1 - Naive Solution, O(n^2)
The obvious brute force approach: from each element we compute every possible subarray starting in that element, check if sum is a multiple of `k` and store the maximum length so far.
## 13.2 - Prefix Sum Solution, O(n)
The key to use the prefix sum technique is the following. 
**Observation:** any two prefix sums that are not next to each other with the same mod k, or a prefix sum with mod k = 0 that is not the first number will yield a valid subarray.

Hence: 
- we compute the prefix sum
- we create a map that goes from the modulo to the index of the array that start the subarray with that modulo
- if the modulo of an element `i` of the prefix sum array is 0 we return true
- if the modulo of an element `i` of the prefix sum array was already found we check that it is not the immediate previous element of `i`, and in case we return true
	- otherwise we add the pair `(modulo, i)` to the map and we loop
- at the end we return false

Here's the solution in Rust: 
```rust
fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
	// create the prefix sum array
	let prefix_sums = nums
		.iter()
		.scan(0, |sum, e| {
		*sum += e;
		Some(*sum)
	})
	.collect::<Vec<_>>();
	
	let mut mods_to_indices: HashMap<i32, usize> = HashMap::new();
	for (i, &prefix_sum) in prefix_sums.iter().enumerate() {
		let modulo = prefix_sum % k;
		// if the current prefix sum has modulo 0 we are done
		if modulo == 0 && i != 0 {
			return true;
		}
		// if the mod has never been seen we add it
		if !mods_to_indices.contains_key(&modulo) {
			mods_to_indices.insert(modulo, i);
		} else {
			// if the mod has been seen we check that is not 
			// the immediate predecessor prefix sum, and return true
			let prev = mods_to_indices[&modulo];
			if prev < i - 1 {
				return true;
			}
		}
	}
	false
}
```
## 13.2 - Optimized Solution, O(n)
Turn's out we really do not need a full prefix sum array. 
We can just compute `sum` as we go and use it also as the mod result. 
This results in a better space efficiency, from $O(n)$ to $O(1)$

```rust
pub fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
	let mut mods_to_indices: HashMap<i32, usize> = HashMap::new();
	let mut sum = 0;

	for (i, &num) in nums.iter().enumerate() {
		// current prefix sum at index i
		sum += num;
		// we directly consider the modulo of the prefix sum
		sum %= k;

		if sum == 0 && i > 0 {
			return true;
		}	

		// if the mod has never been seen we add it
		if !mods_to_indices.contains_key(&sum) {
			mods_to_indices.insert(sum, i);
		} else {
			// if the mod has been seen we check that is not 
			// the immediate predecessor prefix sum, and return true
			let prev = mods_to_indices[&sum];
			if prev < i - 1 {
				return true;
			}
		}
	}
	false
}
```
# 14 - Fenwick Tree
**The Fenwick Tree, also known as the Binary Indexed Tree (BIT), is a data structure that maintains the prefix sums of a dynamic array.** 
**With this data structure we can update values in the original array and still answer prefix sum queries.** 
Both operations runs in **logarithmic time**.

Consider the following problem: we have an array `A[1..n]` of integers and we would like to support the following operations: 
- `sum(i)` returns the sum of the elements in `A[1...i]`
- `add(i,v)` adds the value `v` to the entry `A[i]`

We have **two trivial solutions:**
- Store `A` as it is. This way, `sum(i)` is solved by scanning the array in $\Theta(n)$ time, and `add(i,v)` is solved in $O(1)$
- Store the prefix sums of `A`. This way `sum(i)` is solved in $O(1)$ and `add(i,v)` is solved modifying all the entries from `i` to `n` in $\Theta(n)$ time. 

The `sum`/`add` query time trade-offs of these solutions are unsatisfactory

**The Fenwick Tree provides better tradeoffs for this problem.** 
The Fenwick tree efficiently handles these queries in $\Theta(\log n)$ while using linear space. 
In fact the Fenwick tree is an *implicit* data structure, which means it requires only $O(1)$ additional space to the space needed to store the input data, in our case the array `A`

In our description, we will gradually introduce this data structure by constructing it level by level.

We use the following 1-indexed array in our example: 
![[Pasted image 20240110113032.png | center | 350]]

We start with a simpler version of the problem: we focus on solving `sum` queries only for positions that are powers of 2, positions 1, 2, 4, 8 in `A`. 
The solution of this simplified version, shown in the picture, will be the **first level** of the Fenwick Tree

![[Pasted image 20240110114748.png | center | 350]]

We notice that: 
- there is a fictitious root node named 0 (the array is 1-indexed, the root is a dummy as it is an out-of-bound node)
- over every node there is the corresponding index in `A`
- the value of every node is the prefix sum up to that index
- under every node we have a range of integers: those are the positions covered by that node

With this solution we have that: 
- `sum(i)` query: straightforward, we simply access the node `i`, provided that `i` is a power of 2
- `add(i,v)` query: we need to add `v` to all the nodes that covers ranges that include the position `i`, for example if we want to do `add(10,3)` we need to add 10 to the nodes 4 (as it has range $[1,4]$ and $3 \in [1,4]$) and 8 (as it has range $[1,8]$ and $3 \in [1,8]$).
	- **general rule** `add(i,v)`: find the smallest power of 2 grater than `i`, let's call it `j`, and we add `v` to the nodes `j, j*2, j*(2^2), ...`
		- `j` is a power of `2`, we are using the obvious power rules

We observe that `sum` takes constant time and `add` takes $\Theta(\log n)$ time. 
This is very good, can we extend this solution to support `sum` queries on more positions? 

**Observation:** we are not currently supporting queries for positions within the ranges between consecutive powers of 2. 
Look at the image above: positions that falls in the range (subarray) `[5, 7]`, which falls between the indices 4 (2^2) and 8 (2^3), are not supported. 
In fact we can't make the query `sum(5)`.
**Enabling queries for this subarray is a smaller instance of our original problem.**

We can apply the **same strategy by adding a new level** to our tree. 
The children of a node stores the partial sums **starting from the next element**

![[Pasted image 20240110121552.png | center | 350]]

If the subarray is `A[l..r]`, the new level will support the `sum(i)` query for any `i` such that `i-l+1` is a power of 2. 

**Lemma:** our two-level tree can now handle `sum(i)` queries for positions that are the sum of two powers of 2. 
**Proof:** 
- Consider a position `i` expressed as $2^{k'} + 2^k$, where $k' > k$. 
- We can decompose the range $[1, i]$ into two subranges: 
	- $[1, 2^{k'}]$
	- $[2^{k'}+1, 2^{k'} + 2^k = i]$
- Both of these two subranges are covered by nodes in our tree
	- the range $[1, 2^{k'}]$ is covered by the node $2^{k'} = 4$ at the first level
	- the range $[2^{k'}+1, 2^{k'} + 2^k = i] = [5,5]$ is covered by the node `i` at the second level
**Example:** *intuition* on how to compute `sum(5)`
- $i = 5$ 
	- $k' = 2$
	- $k = 0$
	- $5 = 2^2 + 2^0$ 
- the range $[1,5]$ is divided in two subranges
	- $[1,4]$, covered by the node 4
	- $[5,5]$, covered by the node 5
- the result, which is 16, is obtained by summing the values of the nodes
	- search the node 4 in the tree: it is at the first level as it is a power of 2
	- take its value, which is 9
	- search the node 5
		- it is not a power of two but a sum of power of 2, hence it is at the second level
		- the biggest power of 2 smaller than 5 is 4, hence the node 5 is a children of the node 4
		- take its value, which is 7
	- 9 + 7 = 16

**Which are the position still not supported?** The positions that are neither a power of 2 or the sum of two powers of 2. 
In our example, we can't compute `sum(7)` as `7` is either of those. 
To support this we add a new level to our tree, so we can support positions that are sum of three powers of 2
![[Pasted image 20240110141449.png | center | 350]]


**Example:** intuition on how to compute `sum(7)`
- the position $7$ can be expressed with $2^{k''} + 2^{k'} + 2^{k}$, with $k'' > k'' > k$, where: 
	- $k'' = 2$
	- $k' = 1$
	- $k = 0$
- in fact we have that $7 = 2^2 + 2^1 + 2^0 = 4 + 2 + 1$ 
- the range $[1,7]$ can be divided in three subranges: 
	- $[1, 2^{k''}] = [1, 4]$
	- $[2^{k''}+1, 2^{k''} + 2^{k'}] = [5, 6]$
	- $[2^{k''} + 2^{k'} + 1, 2^{k''} + 2^{k'}+2^k] = [7, 7]$ 
- and we have that
	- the first subrange is covered by the node 4, which has value = 9
	- the second subrange is covered by the node 6, which has value = 4
	- the third subrange is covered by the node 7, which has value = 2
- the biggest power of 2 smaller than 7 is 4, hence all the useful nodes will be in the subtree rooted in the node 4
	- 4 is a power of 2, first level
	- 6 is the sum of two power of 2, second level
	- 7 is the sum of three power of 2, third level
- 9 + 4 + 5 = 15, which is `sum(7)`

And we are done, this is the Fenwick tree of the array `A`. 
We can make some observations: 
1) While we have represented our solution as a tree, it cal also be represented as an array of size n+1, as shown in the figure above
2) We no longer require the original array `A` because any of its entries `A[i]` can be simply obtained by doing `sum(i) - sum(i-1)`. This is why the Fenwick tree is an **implicit data structure**
3) Let be $h = \lfloor\log(n)+1\rfloor$, which is the length of the binary representation of any position in the range $[1,n]$. Since any position can be expressed as the sum of at most $h$ powers of 2, the tree has no more than $h$ levels. In fact, the number of levels is either $h$ or $h-1$, depending on the value of n

Now, let’s delve into the details of how to solve our `sum` and `add` queries on a Fenwick tree.
## 14.1 - Answering a `sum` query
This query involves beginning at a node `i` and traversing up the tree to reach the node `0`. 
Thus `sum(i)` takes time proportional to the height of the tree, resulting in a time complexity of $\Theta(\log n)$. 

Let's consider the case `sum(7)` more carefully. 
We start at node 7 and move to its parent (node 6), its grandparent (node 4), and stop at its great-grandparent (the dummy root 0), summing their values along the way. 
This works because the ranges of these nodes ($[1,4], [5,6], [7,7]$) collectively cover the queried range $[1,7]$. 

Answering a a `sum` query is straightforward **if we are allowed to store the tree's structure.**
However a significant part of the Fenwick tree's elegance lies in the fact that storing the tree is not actually necessary. 
This is because **we can efficiently navigate from a node to its parent using a few bit-tricks, which is the reason why the Fenwick trees are also called Binary Indexed trees.**
### 14.1.1 - Compute the Parent of a Node
We want to compute the parent of a node, and we want to it quickly and without representing the structure of the tree.

Let's consider the binary representation of the indexes involved in the query `sum(7)`

![[Pasted image 20240110150658.png | center ]]

**Theorem:** the binary representation of a node's parent can be obtained by removing the trailing one (i.e., the rightmost bit set to 1) from the binary representation of the node itself.
**Proof:** TODO
## 14.2 - Performing an `add`
Now we consider the operation `add(i,v)`. 
We need to add the value `v` to each node whose range include the position `i`.

Surely, the node `i` itself is one of these nodes as its range ends in `i`. 
Additionally, the right siblings of node `i` also encompasses the position `i` in their ranges. 
This is because **siblings share the same starting positions, and right siblings have increasing sizes.**
The right siblings of the parent node of `i`, the right siblings of the grandparent, and so on can also contain position `i`. 

It might seem like we have to modify a large number of nodes, however we can show that **the number of nodes to be modified is at most log(n).** 
This is because **each time we move from a node to its right sibling or to the right sibling of its parent, the size of the covered range at least doubles. And a range cannot double more than \log(n) times.**

If we want to perform `add(5, x)` we just need to modify the nodes in red:

![[Pasted image 20240110152607.png | center | 350]]

We now know which are the nodes to modify for `add(i, x)`. 
**Let's discuss how to compute these nodes with bit-tricks.**
### 14.2.1 - Computing the Siblings
Continuing the above example, starting with `i = 5`, the next node to modify is its right sibling, node 6. 
Their binary representation is 
![[Pasted image 20240110152844.png| center]]
We see that if we isolate the trailing one in the binary rep. of 5, which is `0001`, and add it to the binary rep. of 5 itself, we obtain the binary rep of 6.

**Finding the Sibling**
The binary representation of a node and its sibling matches, except for the position of the trailing one.When we move from a node to its right sibling, this trailing one shifts one position to the left. 
Adding this trailing one to a node accomplishes the required shift. 
Now, consider the ID of a node that is the last child of its parent. 
In this case, the rightmost and second trailing one are adjacent. To obtain the right sibling of its parent, we need to remove the trailing one and shift the second trailing one one position to the left.
Thankfully, this effect is one again achieved by adding the trailing one to the node’s ID.

**The time complexity** of `add` is $\Theta(n)$, as we observe that each time we move to the right sibling of the current node or the right sibling of its parent, the trailing one in its binary rep. shifts at lest one position to the left, and this can occur at most $\lfloor\log(n)\rfloor+1$ times.
## 14.3 - Fenwick Tree in Rust
The following is a minimal implementation. 
While we’ve transitioned to 0-based indexing for queries, internally, we still use the 1-based indexing to maintain consistency with the examples above.

```rust
#[derive(Debug)]
pub struct FenwickTree {
    tree: Vec<i64>,
}
impl FenwickTree {
    pub fn with_len(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }
    pub fn len(&self) -> usize {
        self.tree.len() - 1
    }
    /// Indexing is 0-based, even if internally we use 1-based indexing
    pub fn add(&mut self, i: usize, delta: i64) {
        let mut i = i + 1; 
        assert!(i < self.tree.len());

        while i < self.tree.len() {
            self.tree[i] += delta;
            i = Self::next_sibling(i);
        }
    }
    /// Indexing is 0-based, even if internally we use 1-based indexing
    pub fn sum(&self, i: usize) -> i64 {
        let mut i = i + 1;  

        assert!(i < self.tree.len());
        let mut sum = 0;
        while i != 0 {
            sum += self.tree[i];
            i = Self::parent(i);
        }
        sum
    }
    pub fn range_sum(&self, l: usize, r: usize) -> i64 {
        self.sum(r) - if l == 0 { 0 } else { self.sum(l - 1) }
    }
    fn isolate_trailing_one(i: usize) -> usize {
        if i == 0 {
            0
        } else {
            1 << i.trailing_zeros()
        }
    }
    fn parent(i: usize) -> usize {
        i - Self::isolate_trailing_one(i)
    }
    fn next_sibling(i: usize) -> usize {
        i + Self::isolate_trailing_one(i)
    }
}
```

## 14.4 - Fenwick Tree Applications: Update the Array
You have an array containing n elements initially all 0. 
You need to do a number of update operations on it. 
In each update you specify `l`, `r` and `val` which are the starting index, ending index and value to be added. 
After each update, you add the `val` to all elements from index `l` to `r`. 
After `u` updates are over, there will be `q` queries each containing an index for which you have to print the element at that index.

**Observation:** `access(i)` wants the prefix sum of the elements `A[1..i]`

**Solution:**
We are given an array `A[1,n]` initially set to 0. 
We want to support two operations: 
- `access(i)` returns `A[i]`
- `range_update(l, r, v)`, updates the entries in `A[l,r]` adding to them `v`

The following Fenwick solution solve the problem on an array `B[1,n]`. 
1) from `B` we build the Fenwick Tree of length `n` (mind that `B` is initialized with all zeros)
2) the operation `access(i)` is a wrapper of the operation `sum(i)` we have seen before
3) the operation `range_update(l,r,v)` exploit the operation `add(i, v)` of the implementation of the Fenwick Tree: 
	1) first we check that `l` is `<=` than `r`, aka that the interval of entries to update is well formed
	2) then we check that `r <= n`, aka that the interval of entries to update is actually in the array
	3) we perform `add(l,v)`: this trigger the addition of the value `v` to each node whose range include the position `l` in the Fenwick Tree
	4) we perform `add(r, -v)`: this trigger the subtraction of the value `v` to each node whose range include the position `r` in the Fenwick Tree
	5) we have added and subtracted the same same quantity `v` in the Fenwick tree, this means that prefix sum are coherent and the elements in `[l,r]` are increased by `v` 

```rust
#[derive(Debug)]
struct UpdateArray {
    ft: FenwickTree,
}
impl UpdateArray {
    pub fn with_len(n: usize) -> Self {
        Self {
            ft: FenwickTree::with_len(n),
        }
    }
    pub fn len(&self) -> usize {
        self.ft.len()
    }
    pub fn access(&self, i: usize) -> i64 {
        self.ft.sum(i)
    }
    pub fn range_update(&mut self, l: usize, r: usize, v: i64) {
        assert!(l <= r);
        assert!(r < self.ft.len());

        self.ft.add(l, v);
        if r + 1 < self.ft.len() {
            self.ft.add(r + 1, -v);
        }
    }
}
```

## 14.5 - Fenwick Tree Applications: Nested Segments
