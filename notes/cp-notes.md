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
**TODO**
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
**TODO**
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
**TODO**
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
**TODO**
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

# 14 - Update the Array
You have an array containing n elements initially all 0. 
You need to do a number of update operations on it. 
In each update you specify `l`, `r` and `val` which are the starting index, ending index and value to be added. 
After each update, you add the `val` to all elements from index `l` to `r`. 
After `u` updates are over, there will be `q` queries each containing an index for which you have to print the element at that index.
**Observation:** `access(i)` wants the prefix sum of the elements `A[1..i]

To efficiently solve this problem we introduce a new data structure, the **Fenwick Tree**
## 14.1 - Fenwick Tree
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
### 14.1.1 - Answering a `sum` query
This query involves beginning at a node `i` and traversing up the tree to reach the node `0`. 
Thus `sum(i)` takes time proportional to the height of the tree, resulting in a time complexity of $\Theta(\log n)$. 

Let's consider the case `sum(7)` more carefully. 
We start at node 7 and move to its parent (node 6), its grandparent (node 4), and stop at its great-grandparent (the dummy root 0), summing their values along the way. 
This works because the ranges of these nodes ($[1,4], [5,6], [7,7]$) collectively cover the queried range $[1,7]$. 

Answering a a `sum` query is straightforward **if we are allowed to store the tree's structure.**
However a significant part of the Fenwick tree's elegance lies in the fact that storing the tree is not actually necessary. 
This is because **we can efficiently navigate from a node to its parent using a few bit-tricks, which is the reason why the Fenwick trees are also called Binary Indexed trees.**
#### 14.1.1.1 - Compute the Parent of a Node
We want to compute the parent of a node, and we want to it quickly and without representing the structure of the tree.

Let's consider the binary representation of the indexes involved in the query `sum(7)`

![[Pasted image 20240110150658.png | center ]]

**Theorem:** the binary representation of a node's parent can be obtained by removing the trailing one (i.e., the rightmost bit set to 1) from the binary representation of the node itself.
**Proof:** **TODO**
### 14.1.2 - Performing an `add`
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
### 14.1.2.1 - Computing the Siblings
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
## 14.1.2 - Fenwick Tree in Rust
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
## 14.2 - Fenwick Tree Solution
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

# 15 - Nested Segments
We are given $n$ segments: $[l_1, r_1],\dots, [l_n, r_n]$ on a line. There are no coinciding endpoints among the segments. 
The task is to determine and report the number of other segments each segment contains.
**Alternatively said:** for the segment $i$ we want to count the number of segments $j$ such that the following condition hold: $l_i < l_j \land r_j < r_i$. 

We provide two solutions to this problem: 
- with Fenwick Tree
- with **Segment Tree**
## 15.1 - Fenwick Tree Solution
We use a sweep line & Fenwick tree approach. 

First we build the Fenwick tree by adding $1$ in each position that corresponds to the right endpoint of a segment. 
This way, a `sum(r)` reports the number of segments that end in the range $[1,r]$

Next, we let a sweep line process the segments in increasing order of their left endpoints.
When we process the segment $[l_i, r_i]$ we compute `sum(r_i - 1)` as the result for the current segment. 
Before moving to the next segment, we add $-1$ at position $r_i$, to remove the contribution of the right endpoint of the current segment. 
**said easy:** the Fenwick tree acts like the "counter" variable in a sweep line algorithm. 

The claim is that `sum(r_i - 1)` is the number of segments contained in $[l_i, r_i]$. 
This is because all the segments that starts before $l_i$ have already been processed, and their right endpoints have been removed from the Fenwick tree.
Therefore, `sum(r_i - 1)` is the number of segments that starts after $l_i$ and ends before $r_i$

The following snippet implement the solution above, using the Fenwick tree previously defined. 
**TODO**
## 15.2 - Segment Tree
**A Segment Tree is a data structure that stores information about array intervals as a tree.**
This allows answering range queries over an array efficiently, while still being flexible enough to **allow quick modification of the array**.
We can **find the sum of consecutive array elements**`A[l..r]` or **find the minimum element in a segment** in $O(\log(n))$ time. 
Between answering such queries **we can modifying the elements by replacing one element of the array, or even changing the elements of a whole subsegment** (e.g., assigning all elements `a[l..r]` to any value, or adding a value to all element in the subsegment)

**Segment trees can be generalized to larger dimensions.** For instance, with 2-dimensional Segment trees you can answer sum or minimum queries over some subrectangle of a given matrix in $O(\log^2(n))$ time. 

**We consider the simplest form of Segment Trees**. 
**We want to answer sum queries efficiently.** 
The formal definition of our task is the following: given an array $a[0,\dots,n-1]$, the Segment Tree must be able to perform the following operations in $O(\log(n))$ time
1) find the sum of elements between the indices $l$ and $r$: $\Sigma_{i=l}^r\ a[i]$ 
2) change values of elements in the array: $a[i] = x$

**Observation:** even our simple form of Segment Tree is an improvement over the simpler approaches: 
- a naive implementation that uses just the array can update element in O(1) but requires O(n) to compute each sum query
- a precomputed prefix sums can compute the sum queries in O(1) but updating an array element requires O(n) changes to the prefix sums
### 15.2.1 - Structure of the Segment Tree
We can take a divide-and-conquer approach when it comes to array segments.
We compute and store the sum of the elements of the whole array, i.e. the sum of the segment $a[0,\dots,n-1]$. 
Then we split the array into two halves $a[0,\dots,n/2 -1]$ and $a[n/2,\dots, n-1]$, compute the sum of each halves and store it. Each of this halves are split in half, and so on until all segments reach size 1. 

We can view this segment as forming a binary tree: the root is the whole array segment, and each vertex (except leaves) has exactly two children. 
This is why this data structure is called "segment tree". 

**Example:** consider the array $a = [1,3,-2,8,-7]$
![[Pasted image 20240112095556.png | center | 450]]
From the short description we just gave we can conclude that Segment Trees only require a linear number of vertices. 
The first level of our tree contains a single node (the root), the second level will contain two nodes, the third we have four nodes, until the reaches $n$. 
Thus, the number of vertices in the worst case can be estimated by the sum 
$$1+2+4+\dots+ 2^{\lceil\log(n)\rceil+1}<4n$$
Mind that whenever $n$ is not a power of 2, not all levels of the Segment Tree will be completely filled, as shown in the image. 
**The height of a Segment Tree** is $O(\log(n))$, because when going down the root to the leaves the size of the segments decrease approximately by half. 
#### 15.2.1.1 - Construction
Before constructing the segment tree we need to decide: 
- the value that gets stored at each node of the segment tree. In a sum segment tree a node would store the sum of the elements in its range $[l,r]$
- the merge operation that merges two sibling in a segment tree. In a sum segment tree, the two nodes corresponding to the ranges $a[l_1,\dots,r_1]$ and $a[l_2,\dots,r_2]$ would be merged into a node corresponding to the range $a[l_1,\dots,r_2]$ by adding the values of the two nodes

Note that a vertex is a leaf if its corresponding segment covers only one value of the original array. It is present at the lowest level of the tree and its value would be equal to the corresponding element $a[i]$. 

**For construction of the segment tree, we start at the bottom level (the leaves) and assign them their respective values**. 
**On the basis of these values we can compute the values of the previous level**, using the `merge`  function. 
And on the basis of those, we can compute the values of the previous, and so on until we reach the root.
**It is convenient to describe this operation recursively in the other direction, i.e., from the root vertex to the leaf vertices.**

The construction procedure, if called of a non-leaf vertex, does the following:
- recursively construct the values of the two child vertices 
- merge the computed values of these children 

We start the construction at the root vertex, and hence, we are able to compute the entire segment tree. 
The **time complexity of the construction** is O(n), assuming that the merge operation is O(1), as the merge operation gets called n times, which is equal to the number of internal nodes in the segment tree.
#### 15.2.1.2 - Sum Queries
We receive two integers $l$ and $r$, and we have to compute the sum of the segment $a[l,\dots,r]$ in $O(\log(n))$ time. 
To do this we will traverse the tree and use the precomputed sums of the segments. 

Let's assume that we are currently at the vertex that covers the segment $a[tl,\dots,tr]$. 
There are three possible cases: 
1) the segment $a[l,\dots,r]$ is equal to the corresponding segment of the current index, then we are finished and we return the sum that is stored in the vertex
2) the segment of the query can fall completely into the domain of either the left or the right child. In this case we can simply go to the child vertex, which corresponding segment covers the query segment, and execute the algorithm described here with that vertex
3) the query segment intersects with both children. In this case we have no other option as to make two recursive calls, one for each child. First we go to the left child, compute a partial answer for this vertex (i.e. the sum of values of the intersection), then go the right child, compute the partial answer using that vertex, and then combine the answers.

So processing a sum query is a function that recursively calls itself once with either the left or the right child (without changing the query boundaries), or twice, once for the left and once for the right child (by splitting the query into two subqueries). 
And the recursion ends whenever the boundaries of the current query segment coincides with the boundaries of the segment of the current vertex. 
In that case the answer will be the precomputed value of the sum of this segment, which is stored in the tree.

Obviously we will start the traversal from the root vertex of the Segment Tree.

**Example:** consider the array $a = [1,3,-2,8,-7]$, and we want to compute the sum of the segment $a[2,\dots,4] = [-2,8,-7] = -1$ 
![[Pasted image 20240112103329.png | center | 450]]
**Let's now reason about the complexity of the algorithm.**
We have to show that we can compute the sum queries in $O(\log(n))$.
**Theorem:** for each level we only visit no more than four vertices.
**Proof:** **TODO**
And since the height of the tree is $O(\log(n))$, we receive the desired running time. 
#### 15.2.1.3  - Update Queries
Now we want to modify a specific element in the array, let's say we want to do the assignment $a[i] = x$. And we have to rebuild the Segment Tree, such that it corresponds to the new, modified array. 

This query is easier than the sum query. Each level of a segment tree forms a partition of the array. Therefore an element $a[i]$ only contributes to one segment from each level. 
Thus only $O(\log(n))$ vertices need to be updated. 

It is easy to see that the update request can be implemented using a recursive function. The function gets passed the current tree vertex, and it recursively calls itself with one of the two child vertices (the one that contains $a[i]$) and after that recomputes its sum value, similar how it is done in the build method (that is as the sum of its two children). 

**Example:** given the same array as before, we want to perform the update $a[2] = 3$ 
![[Pasted image 20240112112424.png | center | 450]]
### 15.2.2 - Simple Implementation
The main consideration is how to store the Segment Tree. 
Of course we can define a   `Vertex` struct and create objects, that store the boundaries of the segment, its sum and additionally also pointers to its child vertices. 
However, this requires storing a lot of redundant information in the form of pointers. 
We use a simple trick to make this a lot more efficient by **using an implicit data structure**: only storing the sums in an array.

The sum of the root vertex at index 1, the sums of its two child vertices at indices 2 and 3, the sums of the children of those two vertices at indices 4 to 7, and so on. With 1-indexing, conveniently the left child of a vertex at index   $i$  is stored at index   $2i$ , and the right one at index   $2i + 1$ . 
Equivalently, the parent of a vertex at index   $i$  is stored at   $i/2$  (integer division).
**This simplifies the implementation a lot.** 
We don't need to store the structure of the tree in memory. It is defined implicitly. We only need one array which contains the sums of all segments.

As noted before, we need to store at most   $4n$  vertices. 
It might be less, but for convenience we always allocate an array of size  $4n$ . 
There will be some elements in the sum array, that will not correspond to any vertices in the actual tree, but this doesn't complicate the implementation.

**Here's the Rust implementation of what we have seen**
```rust
/// A struct representing a Segment Tree
struct SegmentTree {
	/// The size of the array
	n: usize,
	/// The array to store the segment tree
	t: Vec<i32>,
}

impl SegmentTree {
	/// Constructor to initialize the segment tree
	///
	/// # Arguments
	///
	/// * `a` - The input array
	///
	/// # Returns
	///
	/// A new instance of `SegmentTree`
	fn new(a: &[i32]) -> SegmentTree {
		let n = a.len();
		let mut t = vec![0; 4 * n];
		SegmentTree::build(&mut t, a, 1, 0, n - 1);
		SegmentTree { n, t }
	}

	/// Build the segment tree recursively
	///
	/// # Arguments
	///
	/// * `t` - The segment tree array
	/// * `a` - The input array
	/// * `v` - The current vertex
	/// * `tl` - The left boundary of the current segment
	/// * `tr` - The right boundary of the current segment
	fn build(t: &mut Vec<i32>, a: &[i32], v: usize, tl: usize, tr: usize) {
		if tl == tr {
			// If the segment has a single element, store it in the tree
			t[v] = a[tl];
		} else {
			// Otherwise, recursively build the left and right subtrees
			let tm = (tl + tr) / 2;
			SegmentTree::build(t, a, v * 2, tl, tm);
			SegmentTree::build(t, a, v * 2 + 1, tm + 1, tr);
			// Combine the values of the left and right subtrees
			t[v] = t[v * 2] + t[v * 2 + 1];
		}
	}

	/// Query the sum in a range [l, r]
	///
	/// # Arguments
	///
	/// * `v` - The current vertex
	/// * `tl` - The left boundary of the current segment
	/// * `tr` - The right boundary of the current segment
	/// * `l` - The left boundary of the query range
	/// * `r` - The right boundary of the query range
	///
	/// # Returns
	///
	/// The sum in the specified range
	fn sum(&self, v: usize, tl: usize, tr: usize, l: usize, r: usize) -> i32 {
		if l > r {
			// If the query range is invalid, return 0
			0
		} else if l == tl && r == tr {
			// If the segment is within the query range, return its value
			self.t[v]
		} else {
			// Otherwise, recursively query the left and right subtrees
			let tm = (tl + tr) / 2;
			self.sum(v * 2, tl, tm, l, std::cmp::min(r, tm))
			+ self.sum(v * 2 + 1, tm + 1, tr, std::cmp::max(l, tm + 1), r)
		}
	}
	
	/// Update a value at a specific position
	///
	/// # Arguments
	///
	/// * `v` - The current vertex
	/// * `tl` - The left boundary of the current segment
	/// * `tr` - The right boundary of the current segment
	/// * `pos` - The position to update
	/// * `new_val` - The new value to set at the specified position
	fn update(
		&mut self, 
		v: usize, 
		tl: usize, 
		tr: usize, 
		pos: usize, 
		new_val: i32
	) {
		if tl == tr {
			// If the segment has a single element, update it
			self.t[v] = new_val;
		} else {
			// Otherwise, recursively update the left or right subtree
			let tm = (tl + tr) / 2;
			if pos <= tm {
			self.update(v * 2, tl, tm, pos, new_val);
			} else {
			self.update(v * 2 + 1, tm + 1, tr, pos, new_val);
			}
			// Update the current node
			self.t[v] = self.t[v * 2] + self.t[v * 2 + 1];
		}
	}

	/// Print the segment tree in a meaningful way
	///
	/// # Arguments
	///
	/// * `v` - The current vertex
	/// * `tl` - The left boundary of the current segment
	/// * `tr` - The right boundary of the current segment
	fn print_tree(&self, v: usize, tl: usize, tr: usize, indent: usize) {
		// Print the current node
		println!(
			"{0:width$}Node {1}: [{2}, {3}] = {4}",
			"",
			v,
			tl,
			tr,
			self.t[v],
			width = indent
		);
		// If it's not a leaf node, recursively print left and right subtrees
		if tl < tr {
			let tm = (tl + tr) / 2;
			self.print_tree(v * 2, tl, tm, indent + 2);
			self.print_tree(v * 2 + 1, tm + 1, tr, indent + 2);
		}
	}
}
```
### 15.2.3 Lazy Propagation
Segment Tree allows applying modification queries to an entire segment of contiguous elements, and perform the query in the same time $O(\log(n))$.
When there are many updates and updates are done on a range, we can postpone some updates (avoid recursive calls in update) and do those updates only when required.

**TODO:** https://cp-algorithms.com/data_structures/segment_tree.html
## 15.3 - Segment Trees Solution
**Let's now solve nested segments with a Segment Tree**
**TODO**
# 16 - Powerful Array
An array of positive integers $a_1,\dots,a_n$ is given. Let us consider its arbitrary subarray $a_l, a_{l+1},\dots, a_r$, where $1 \le l \le r \le n$.
For every positive integer $s$ we denote by $K_s$ the number of occurrences of $s$ into the subarray.
We call the **power** of the subarray the sum of products $K_s \cdot K_s \cdot s$ for every positive integer $s$. 
The sum contains only finite number of nonzero summands as the number of different values in the array is indeed finite. 

You should calculate the power of $t$ given subarrays.

**Besides the trivial solutions, we introduce a new algorithmic technique.**
## 16.1 - Mo's Algorithm 
The Mo’s Algorithm is a powerful and efficient technique for solving a wide variety of range query problems. 
It becomes particularly **useful for kind of queries where the use of a Segment Tree or similar data structures is not feasible.** 
**This typically occurs when the query is non-associative, meaning that the result of a query on a range cannot be derived by combining the answers of the subranges that cover the original range.**

Mo’s algorithm typically achieves a time complexity of $O((n+q)\sqrt n)$, where $n$ represents the size of the dataset, and $q$ is the number of queries.

**Let's consider an easier problem than Powerful Array**
We are given an array $A[0,n-1]$ consisting of colors, with each color represented by an integer within $[0,n-1]$. 
Additionally we are given a set of $q$ range queries called `three_or_more`. 
The query `three_or_more(l,r)` aims to count the colors that occur at least three times in the subarray $A[l..r]$. 

A **straightforward solution** for the problem: simply scan the subarray and use an additional array as a counter to keep track of occurrences of each color within the range. 
Whenever a color reaches three the answer is incremented by 1.
Mind that after each query we have to reset the array of counters. 

Indeed, it’s evident that it has a time complexity of $\Theta(qn)$. 
The figure below illustrates an input that showcases the worst-case running time. 
We have $n$ queries. The first query range has a length of $n$ and spans the entire array. Then, the subsequent query ranges are each one unit shorter, until the last one, which has a length of one. 
The total length of these ranges is $\Theta(n^2)$, which is also the time complexity of the solution.
![[Pasted image 20240116092201.png | center | 600]]

**Let's now see the solution using the Mo's Algorithm.**
Suppose we have just answered the query for the range $[l',r']$ and are now addressing the range $[l,r]$. 
Instead of starting from scratch, we can update the previous answer and counters by adding or removing the contributions of colors that are new in the query range but not in the previous one, or vice versa.
Specifically, for left endpoints, we must remove all the colors in $A[l',l-1]$ if $l' < l$, or we need to add all the colors in $A[l,l']$ if $l < l'$. The same applies to right endpoints $r$ and $r'$. 

The rust implementation below uses two closures, `add` and `remove` to keep `answer` and `counters` updated as we adjust the endpoints
```rust
pub fn three_or_more(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    let mut counters: Vec<usize> = vec![0; a.len()];
    let mut answers = Vec::with_capacity(queries.len());

    let mut cur_l = 0;
    let mut cur_r = 0; // here right endpoint is excluded
    let mut answer = 0;

    for &(l, r) in queries {
        let mut add = |i| {
            counters[a[i]] += 1;
            if counters[a[i]] == 3 {
                answer += 1
            }
        };

        while cur_l > l {
            cur_l -= 1;
            add(cur_l);
        }

        while cur_r <= r {
            add(cur_r);
            cur_r += 1;
        }

        let mut remove = |i| {
            counters[a[i]] -= 1;
            if counters[a[i]] == 2 {
                answer -= 1
            }
        };

        while cur_l < l {
            remove(cur_l);
            cur_l += 1;
        }

        while cur_r > r + 1 {
            cur_r -= 1;
            remove(cur_r);
        }

        answers.push(answer);
    }

    answers
}
```

The time complexity of the algorithm remains $\Theta(qn)$. 
However we observe that a query now executes more quickly if its range significantly overlaps with the range of the previous query. 

This implementation is **highly sensitive to the ordering of the queries.**
The previous figure becomes now a best-case for the new implementation as it takes $\Theta(n)$time. Indeed, after spending linear time on the first query, any subsequent query is answered in constant time.
Mind that it is enough to modify the ordering of the above queries to revert to quadratic time (alternate between very short and very long queries).

The above considerations lead to a question: **if we have a sufficient number of queries, can we rearrange them in a way that exploits the overlap between successive queries to gain an asymptotic advantage in the overall running time?**
Mo's Algorithm answers positively to this question by providing a reordering of the queries such that the time complexity is reduces to $\Theta((q+n)\sqrt n)$  

The idea is to conceptually partition the array $A$ into $\sqrt n$ buckets, each of size $\sqrt n$, named $B_1,B_2,\dots,B_{\sqrt n}$. 
A query **belongs** to a bucket $B_k$ if and only if its left endpoint $l$ falls into the $k-\text{th}$ bucket, which can be expressed as $\lfloor l/\sqrt n\rfloor = k$
Initially we group the queries based on their corresponding buckets, and within each bucket the queries are solved in ascending order of their right endpoints.

The figure shows this bucketing approach and the queries of one bucket sorted by their right endpoint.
![[Pasted image 20240116094255.png | center | 600]]

**Let's analyze the complexity of the solution using this ordering**
It is sufficient to count the number of times we move the indexes `cur_l` and `cur_r`. This is because both `add` and `remove` take constant time, and thus the time complexity is proportional to the overall number of moves of these two indexes. 

Let's concentrate on a specific bucket. As we process the queries in ascending order of their right endpoints, the index `cur_r` moves a total of at most $n$ times. 
On the other hand, the index `cur_l` can both increase and decrease but it is limited within the bucket, and thus it cannot move more than $\sqrt n$ times per query. 
Thus, for a bucket with $b$ queries, the overall time to process its queries is $\Theta(b\sqrt n + n)$. 

Summing up over all buckets the time complexity is $\Theta(q\sqrt n + n\sqrt n)$, aka $\Theta((n+q)\sqrt n))$. 

Here's a Rust implementation of the reordering process. 
We sort the queries by buckets, using their left endpoints, and within the same bucket we sort them in ascending order by their right endpoints. 
We also have to compute a `permutation` to keep track of how the queries have been reordered. This permutation is essential for returning the answers to their original ordering. 
```rust
pub fn mos(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    // Sort the queries by bucket, get the permutation induced by this sorting.
    // The latter is needed to permute the answers to the original ordering
    let mut sorted_queries: Vec<_> = queries.iter().cloned().collect();
	    let mut permutation: Vec<usize> = (0..queries.len()).collect();

    let sqrt_n = (a.len() as f64) as usize + 1;
    sorted_queries.sort_by_key(|&(l, r)| (l / sqrt_n, r));
    permutation.sort_by_key(|&i| (queries[i].0 / sqrt_n, queries[i].1));

    let answers = three_or_more(a, &sorted_queries);

    let mut permuted_answers = vec![0; answers.len()];
    for (i, answer) in permutation.into_iter().zip(answers) {
        permuted_answers[i] = answer;
    }

    permuted_answers
}
```

**Final Considerations on Mo's Algorithm**
Mo’s algorithm is an offline approach, which means we cannot use it when we are constrained to a specific order of queries or when update operations are involved.

When implementing Mo’s algorithm, the most challenging aspect is implementing the functions `add` and `remove`. 
There are query types for which these operations are not as straightforward as in previous problems and require the use of more advanced data structures than just an array of counters
## 16.2 - Solution
We can just use Mo's Algorithm and a little bit of attention in updating the answer after a `add` or a `remove`.

The solution is identical to the one seen in the previous problem, with one difference. 
We are not interested anymore in the number of occurrences of $i$, denoted $K_i$, in a given subarray, but we want to compute $$\Sigma_i\ K_i^2\cdot i,\ i\in [l,r]$$
**When we increase the number of an occurrence we have to first remove the number obtained when we thought that there was one less occurrence.** 
Code talks more than words: 
```rust 
let mut add = |i| {
	// we found another occurrence of i, we remove the old "power"
	sum -= counters[a[i]] * counters[a[i]] * a[i];
	counters[a[i]] += 1;
	// we update the power using the right number of occurreces of i
	sum += counters[a[i]] * counters[a[i]] * a[i];
};

let mut remove = |i| {
	sum -= counters[a[i]] * counters[a[i]] * a[i];
	counters[a[i]] -= 1;
	sum += counters[a[i]] * counters[a[i]] * a[i];
};
```
# 17 - Longest Common Subsequence
Given two strings, `S1` and `S2`, the task is to find the length of the longest common subsequence, i.e. longest subsequence present in both strings. 

**There are many ways to attack this problem, we use it to talk about Dynamic Programming.**
## 17.1 - Dynamic Programming
Dynamic Programming, like divide-and-conquer, solves problems by combining solutions of subproblems. 
Divide-and-Conquer algorithms partitions the problem into disjoint subproblems, solve the subproblems and then combine their solutions to solve the original problem. 
In contrast, **dynamic programming applies when subproblems overlap, that is, when sub-problems share sub-sub-problems.**
In this context a divide-and-conquer algorithm does more work than necessary, repeatedly solving the common sub-sub-problems. 
**A dynamic programming algorithm solves each sub-sub-problem just once and then saves its answer in a table, avoiding the work of recomputing the answer every time it solves each sub-sub-problem.** 
### 17.1.2 - A first easy problem: Fibonacci
Fibonacci numbers are defined as 
$$
\begin{align}
	& F_0 = 0 \\ 
	& F_1 = 1 \\ 
	& F_n = F_{n-1} + F_{n-2}
\end{align}
$$
Our goal is to compute the n-th Fibonacci number $F_n$. 

Let's consider the following trivial recursive algorithm: 
```java
int fibonacci(int n) {
	if (n == 0)
		return 0; 
	if (n == 1)
		return 1; 
	else
		return fibonacci(n-1) + fibonacci(n-2);
}
```

In computing `fibonacci(n-1)` we will compute `fibonacci(n-2)` and `fibonacci(n-3)`, 
and in computing `fibonacci(n-2)` we will compute `fibonacci(n-3)` and `fibonacci(n-4)` and so on. 
There are lots of the same Fibonacci numbers that are computed every time from scratch. 

**Memorization is a trick that allows to reduce the time complexity.**
Whenever we compute a Fibonacci number we store it in an array `M`. 
Every time we need a Fibonacci number, we compute it only if the answer is not in the array. 
**This algorithm requires linear time and space.**
```java
int fibonacciM(n) {
	if (n == 0)
		return 0; 
	if (n == 1)
		return 1; 
	if (M[n] == null) 
		M[n] = fibonacciM(n-1) + fibonacciM(n-2);

	return M[n];
}
```
