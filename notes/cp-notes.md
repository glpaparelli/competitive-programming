This notes are problem-centered: we introduce a problem and then the theory to efficiently solve it.
**Use at Your Own Risk.**
# 1 - Contiguous Sub-Array with Max Sum
![[Screenshot from 2024-01-05 09-29-59.png | center | 700]]
### 1.1.1 - Brute Force, O(n^2)
For each possible subarray we compute the sum and store the maximum
```java
brute_force(a[])
	n = a.length()
	max = Integer.MIN_VALUE
	for i = 0 to n
		sum = 0
		for j = i to n
			sum += a[j]
			if sum > max
				max = sum
	return max
```
## 1.2 - Optimal Solution: Kadane's algorithm, O(n)
Kadane's algorithm is based on two properties of the subarray with maximum sum: 
1) **sum of values in any prefix of the optimal subarray is positive.** By contradiction, remove a negative prefix to get a subarray with a larger sum
2) **the value that precede the first value of the optimal subarray is negative.** By contradiction, include this positive value to get a subarray with a larger sum

To solve the problem we use Kadane's algorithm: 
```java 
kadane_algorithm(a[])
	n = a.length()
	max = Integer.MIN_VALUE
	sum = 0

	for i = 0 to n
		if sum > 0 // property 1)
			sum += arr[i]
		else // property 2)
			sum = arr[i]

		max = Math.max(sum, max)

	return max
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

An element of array is a **right leader** if it is greater than or equal to all the elements to its right side. The rightmost element is always a leader.

Given the array $h$ of heights, we use two arrays to store the right and left leaders:
$$
\begin{align}
	\text{LL[$i$]} &= \max_{j < i} h[j] \\
	\text{RL[$i$]} &= \max_{j > i} h[j]
\end{align}
$$
Now, how many units of water we can find on top of the cell $i$?
The minimum height between the left and right leader of the current position minus $h[i]$, the height of the current position: 
$$w(i) = \min(\text{LL[$i$]}, \text{RL[$i$]}) - h[i]$$
![[Pasted image 20240105111518.png | center | 500]]
## 2.3 - Optimal Solution: Two Pointers, O(N)
The idea is taken from the Precomputation solution, where we only use two variables to store the currently "meaningful" leaders. 

We take two pointers, `left` and `right` and we initialize `left` to `0` and `right` to the last index `n-1`. 
We also create two variables, `left_max` and `right_max`, they represent the maximum left/right height seen so far. 

Since `left` is the first `left_max` is `0`, same for `right_max`. 
This is intuitive: we can't store water in the first and last column. 

Now we iterate, as long `left < right`. 
We have to decide which pointer we have to shift: **we shift the pointer that has smaller height value:** 
- if `heights[left] < heights[right]` we shift `left`
- we shift `right` otherwise

Now, if we have that `heights[left] > left_max` we can't store any water over the position pointed by `left`, **it would fall as left_max is not high enough**. 
![[Pasted image 20240105122707.png | center | 600]]
Otherwise we compute the amount of water stored in `left`, as always with `left_max - heights[left]`. Then we finally shift `left` by `1`.
The reasoning with the `right` pointer is the same. 

The following pseudo-code solves the problem
```java
max_water(heights[]) 
	n = heights.length()
	result = 0
	left = 0
	right = n-1

	left_max = 0 // max height seen by "left" so far
	right_max = 0 // max height seen by "right" so far

	while left < right 
		if heights[left] < heights[right] // shift "left"
			if heights[left] > left_max 
				left_max = heigths[left] // cant store water
			else 
				result += left_max - heights[left]
			left += 1
		else // shift right
			if heights[right] > right_max 
				right_max = heights[right] // cant store water
			else 
				result += right_max - heights[right]
			right -= 1

	return result
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

It is also important to observe that when there are multiple occurrences of the searched key, the function returns the position of the first encountered occurrence, not necessarily the first occurrence in the vector.
However, **it is often very useful to report the position of the first (or last) occurrence of the searched key**. 

We can obtain this behavior with the following pseudo-implementation.
```java
binary_search(a[], target) 
	n = a.length()
	low = 0
	high = n - 1
	answer = -1

	while low < high
		// (low + high)/2 might overflow if (high+low) is very big
		middle = low + (high - low) / 2

		if a[middle] == target
			answer = middle
			high = middle
		else if a[middle] < target
			low = middle
		else 
			high = middle

	return answer
```
In this implementation, when a match is found, we do not immediately return its position. Instead, we update the `answer` variable and set `high` to the position of this occurrence. 
This way, we continue the search in the first half of the array, seeking additional occurrences of the `key`. 
If there are more matches, `answer` will be further updated with smaller positions.
### 3.1.1 Applications of Binary Search
#### 3.1.1.1 Binary Search the Answer
Consider a problem where all the possible candidate answers are restricted to a range of values between certain `low` and `high` possible answers. 
In other words, any candidate answer $x$ falls within the range `[low, high]`. 
We also have a boolean predicate `pred` defined on the candidate answers that tells us if an answer is good or bad for our aims.
Our goal is to **find** **the largest good answer.**

When no assumption are made about the predicate, we cannot do better than evaluating the predicate on all possible answers. 
Hence, the number of times we evaluate the predicate is $\Theta(n)$, where $n =\text{high} - \text{low}$ is the number of possible answers. 

On the other hand, **if the predicate is monotone**, we can binary search the answer to find it with $\Theta(\log(n))$ evaluations. 

**A predicate is said to be monotone** if the truth value of a predicate is true for one combination of inputs, it will remain true if any of those inputs are increased (or remain the same). Similarly, if the truth value is false for one combination, it will remain false or become true if any of the inputs are increased (or remain the same).

The following pseudo-code clarifies the concept: 
```java
rangeBS(pred, low, high)
	low = low;
	high = high; 

	answer; 

	while low < high
		mid = low + (high-low) / 2

		if pred(mid) == true
			answer = mid
			low = mid + 1 // we want the largest good answer
		else 
			high = mid

	return answer
```
#### 3.1.1.2 Square Root
We can use the previous problem to compute the square root. 

We are given a non-negative integer $v$ and we want to compute the square root of $v$ down to the nearest integer. 

The possible answers are in $[0,v]$ and for each candidate $x$ we have the boolean monotone predicate $p(x) = x^2= v$. 
We can find the result simply by doing `rangeBS(p, 0, v+1)`
#### 3.1.1.3 Social Distancing
We have a sequence of $n$ mutually-disjoint intervals. The extremes of each interval are non-negative integers. 
We aim to find $c$ integer points within the intervals such that the smallest distance $d$ between consecutive selected point is maximized. 

If a certain distance is feasible (i.e., there exists a selection of points at that distance), then any smaller distance is also feasible. 
Thus the feasibility is a monotone boolean predicate that we can use to binary search the answer. 
As the candidate answers range from $1$ to $l$, where $l$ is the overall length of the intervals, the solution takes $\Theta(\log(l))$ evaluations of the predicate.  

Whats the cost of evaluating the predicate? 
We first sort the intervals. 
Now we can evaluate any candidate distance $d'$ by scanning the sorted intervals from left to right. 
First we select the left extreme of the first interval as the first point. Then, we move over the intervals and we choose greedily the first point, which is at a distance at least $d'$ from the previous one. 
Thus an evaluation of the predicate takes $\Theta(n)$ time. 

The total running time is $\Theta(n\log(n))$ 

The pseudo-implementation greatly clarifies the process: 
```java
pred(intervals, distance, c)
	// the first point is the start of the first interval
	lastSelected = intervals[0].start 
	counter = 1
	// for every interval in intervals
	for interval in intervals
		// we select as many points as we can in every interval
		while Math.max(lastSelected + distance, interval.start) <= interval.end
			lastSelected = Math.max(lastSelected + distance, interval.start)
			counter++

	return counter >= c
```
```java
socialDistancing(intervals, c)
	// l is the maximum length of an interval
	if l < c
		return -1 // there is no solution

	// sort the intervals
	intervals.sort()
	
	rangeBS(1, l+1, pred)
```
## 3.2 - Solution
**We can use the binary search philosophy to solve the problem.**
We compute the middle of the array. 
If the element in the middle is smaller than its right neighbor than the right neighbor could be a leader, and we do `low = mid + 1` to proceed the search in the right side of the array. 
Otherwise we do the opposite. 
```java
peak_elements(a[])
	n = a.length()
	low = 0
	right = n - 1

	while low < high
		mid = low + (high - low) / 2

		if a[mid] < a[mid + 1]
			low = mid + 1
		else 
			right = mid 

	return low 
```
![[Pasted image 20240211104316.png | center | 500]]

This solution works because a leader is guaranteed to be found (as a last resource, in the first or last element of the array).
The beauty of this solution is that it naturally covers all the strange cases. 
It would be tempting to write something like 
```java
if a[mid] > a[mid+1] && a[mid] > a[mid-1]
	return mid
```
but that requires to cover many edge cases (array of size 1, 2, if mid is $0$, if mid is $n-1$, ...)
# 4 - Maximum Path Sum
Find the maximum possible sum from one leaf node to another.

![[Pasted image 20240108100401.png | center | 490]]

**To solve this problem we need to use a tree traversal.**
Fist we go back and give the basics. 
## 4.1 - Tree Traversals 
**Tree traversal** (also known as tree search and walking the tree) **is a form of graph traversal and refers to the process of visiting** (e.g. retrieving, updating, or deleting) **each node in a tree data structure, exactly once.** 
**Such traversals are classified by the order in which the nodes are visited.**

**Traversing a tree involves iterating over all nodes in some manner.** 
Because from a given node there is more than one possible next node some nodes must be deferred, aka stored, in some way for later visiting. 
This is often done via a stack (LIFO) or queue (FIFO). 
**As a tree is a recursively defined data structure, traversal can be defined by recursion.**
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
A simple solution is to traverse the tree and do following for every traversed node $X$: 
1. Find maximum sum from-leaf-to-root in left subtree of $X\ \clubsuit$
3. Find maximum sum from-leaf-to-root in right subtree of $X$. 
4. Add the above two calculated values and $X\rightarrow \text{data}$ and compare the sum with the maximum value obtained so far and update the maximum value. 
5. Return the maximum value.

---
$\clubsuit \text{ }$**Max Sum leaf-to-root path in a Binary Tree:**
**Given a Binary Tree, find the maximum sum path from a leaf to root.**
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
maxValue = 0
targetLeaf = null

maxLeafRootSum(root)
	targetLeaf = findTargetLeaf(root, 0)
	print(maxValue)

// set targetLeaf to the leaf with maximum sum path to root
findTargetLeaf(node, currMaxValue) 
	if root == null
		return 

	// hold the sum of nodes on path from root to this node
	currMaxValue += node.value

	// leaf and the path to this has maximum sum: set targetLeaf
	if node.isLeaf 
		if currMaxValue > maxValue
			maxValue = currMaxValue
			targetLeaf = node

	// this is not a leaf node: recur down to find targetLeaf
	findTargetLeaf(node.left, currMaxValue)
	findTargetLeaf(node.right, currMaxValue)
```
---
So, the trivial solution of **Maximum Path Sum** works as follows: 
- For every node $X$ we compute: 
	- the maximum path sum from $X.\text{left}$ to a leaf, $O(n)$
	- the maximum path sum from $X.\text{right}$ to a leaf, $O(n)$
	- we take the maximum between the two 
	- we sum the value of X
	- if the sum is greater than the current maximum we update the current maximum

For every node we call twice `maxLeafRootSum`, which costs O(n), hence the total cost is O(n^2)
## 4.3 - Optimal Solution, O(n)
**We can find the maximum sum using single traversal of binary tree**. 
The idea is to maintain two values in recursive calls
1. Maximum root-to-leaf path sum for the subtrees rooted under current node. 
2. The maximum path sum between leaves (desired output).

For every visited node $X$, we find the maximum root-to-leaf sum in left and right subtrees of $X$. We add the two values with $X->\text{data}$, and compare the sum with maximum path sum found so far.

As before, here a pseudo-code implementation:
```java
maxValue = 0

// calculates two things:
// 1) maximum path sum between two leaves, which is stored in maxValue.
// 2) maximum root-to-leaf path sum, which is returned.
maxPathSum(root)
	if root == null
		return 0

	left = maxPathSum(root.left)
	right = maxPathSum(root.right)
	
	maxValue = Math.max(maxValue, left + right + root.value)

	return left + right + root.value
```

In "one pass" (aka one traversal) we do what we did in the trivial solution.
The reasoning is the following: 
- take a node $X$
- compute the max sum path from $X.\text{left}$ to a leaf $k$ and the max sum path from $X.\text{right}$ to a leaf $j$. 
- the sum path from $k$ to $j$ (passing by $X$) is greater than the one previously stored? then we update the max

The result is `maxValue` when the program terminates.
# 5 - Two Sum in a Sorted Array
**This Problem is not mandatory: used to present the Two Pointers Trick.**
Given a sorted array $a$ (sorted in ascending order), having $n$ integers, find if there exists any pair of elements ($a[i]$, $a[j]$) such that their sum is equal to $x$.
## 5.1 - Naive Solution, O(n^2)
The naive approach is obvious and takes O(n^2), we simply scan the array and we return when we find two numbers that adds up to X. 
```java
existsPairSum(a[], x)
	n = a.length()
	for i = 0 to n
		for j = i+1 to n
			if a[i] + a[j] == x
				return true
			if a[i] + a[j] > x
				break // a is sorted, need a new "i"
	return false
```
## 5.2 - Two Pointers Trick
Two pointers is really an easy and effective technique that is typically used for searching pairs in a sorted array.
It employs using two pointer, typically one from the start of the array going left-to-right and one from the end of the array going right-to-left, until they meet. 
We already seen a sneak peak of this in Trapping Rain Water. 
## 5.3 - Two Pointers Solution, O(n)
**We take two pointers, one representing the first element and other representing the last element of the array, and then we add the values kept at both the pointers.** 

If their sum is smaller than X then we shift the left pointer to right or if their sum is greater than X then we shift the right pointer to left, in order to get closer to the sum. 
We keep moving the pointers until we get the sum as X.
```java
existsPairSum(a[], x)
	n = a.length()
	left = 0
	right = n-1

	while left < right
		if a[left] + a[right] == x
			return true
		if a[left] + a[right] < x
			left += 1 // smaller than target, we need to increase
		else
			right -= 1 // bigger than target, we need to decrease

	return false
```
# 6 - Frogs and Mosquitoes
There are $n$ frogs sitting on the coordinate axis $Ox$. 
For each frog two values $x_i$, $t_i$ are known - the position and the initial length of the tongue of the $i$-th frog (it is guaranteed that all positions $x_i$ are different).
$m$ mosquitoes one by one are landing to the coordinate axis. 
For each mosquito two values are known $p_j$ - the coordinate of the position where the $j$-th mosquito lands and $b_j$ - the size of the $j$-th mosquito. 
Frogs and mosquitoes are represented as points on the coordinate axis.

The frog can eat mosquito if mosquito is in the same position with the frog or to the right, and the distance between them is not greater than the length of the tongue of the frog.

If at some moment several frogs can eat a mosquito the leftmost frog will eat it (with minimal $x_i$). 
After eating a mosquito the length of the tongue of a frog increases with the value of the size of eaten mosquito. 
It's possible that after it the frog will be able to eat some other mosquitoes (the frog should eat them in this case).

For each frog print two values - the number of eaten mosquitoes and the length of the tongue after landing all mosquitoes and after eating all possible mosquitoes by frogs.

Each mosquito is landing to the coordinate axis only after frogs eat all possible mosquitoes landed before. 
Mosquitoes are given in order of their landing to the coordinate axis.
## 6.1 Binary Search Tree
A binary search tree (BST), also called an ordered or sorted binary tree, is a rooted binary tree data structure with the key of each internal node being greater than all the keys in the respective node's left subtree and less than the ones in its right subtree. 
The time complexity of operations on the binary search tree is linear with respect to the height of the tree.

Binary search trees allow binary search for fast lookup, addition, and removal of data items. 
Since the nodes in a BST are laid out so that each comparison skips about half of the remaining tree, the lookup performance is proportional to that of binary logarithm.

The complexity analysis of BST shows that, on average, the insert, delete and search takes $O(\log n)$ for n nodes. 
In the worst case, they degrade to that of a singly linked list: $O(n)$

![[Pasted image 20240109090705.png | center | 250]]
### 6.1.1 Searching
Searching for a specific key can be programmed recursively or iteratively. 

Searching begins by examining the root node. If the tree is `nil`, the key being searched for does not exist in the tree. 
Otherwise, if the key equals that of the root, the search is successful and the node is returned. 
**If the key is less than that of the root, the search proceeds by examining the left subtree. Similarly, if the key is greater than that of the root, the search proceeds by examining the right subtree.** 
This process is repeated until the key is found or the remaining subtree is `nil`. 
If the searched key is not found after a `nil` subtree is reached, then the key is not present in the tree.
![[Screenshot from 2024-01-19 15-34-31.png | center | 500]]
### 6.1.2 Predecessor and Successor
For certain operations, given a node $x$, finding the successor or predecessor of $x$ is crucial. 
Assuming all the keys of the BST are distinct, **the successor of a node** $x$ in BST **is the node with the smallest key greater** than $x$'s key. 
On the other hand, **the predecessor of a node** $x$ in BST **is the node with the largest key smaller** than $x$'s key. 
Following is pseudo-code for finding the successor and predecessor of a node $x$ in BST.
![[Screenshot from 2024-01-19 15-39-52.png | center ]]
## 6.2 - Solution
We store the position of the frogs in a BST, `frogBST`. 
When a mosquito lands on a position $b$ we check which frog will eat it by simply doing a `frogBST.predecessor(b)` query on the tree. 
It is possible that some mosquito cannot be eaten right away, the uneaten mosquito will be stored in their own BST (`mosquitoBST`), using their landing position as keys. 

**There is no need to sort anything.**

The algorithm behaves as follows: 
- insert the frogs, by their order, in a BST `frogBST` using their position as key
- for each mosquito `m` in `mosquitoes`
	- find the predecessor `f` of `m`  in `frogBST`, **if** `f.position + f.tongue >= m.position`
		- `f` eats the mosquito `m` and its tongue grows by the size of `m`
		- maybe `f` now can eat other mosquitoes that could not be eaten before, **inspect** `mosquitoBST` to see if there is a mosquito that `f` can eat, and in case repeat.
			- **inspect:** find the successor `m'` of `f.position` in `mosquitoBST` and see if `f.position + f.tongue >= m'.position`
		- `f` now may overlap or fully contain other frogs
			- For every successor of `f` in `frogBST`:
				- if it overlaps with `f` remove the overlap by updating their positions
					- position of the successor = `f.position + f.tongue + 1`
				- if it is fully contained by `f` then delete this frog as it will never eat
	- **else**, insert `m` in `mosquitoBST` and continue
![[IMG_0413.png | center | 600]]

**Time Complexity:** 
- Each predecessor query on the frog tree can be answered in $O(\log(n))$ time.
- Each successor query of the mosquito tree can be answered in $O(\log(m))$ time.
- Forcing no overlaps: we can "move/shift" $n$ frogs at most for every mosquito that is eaten, then $O((n+m)\log(n))$ 
# 7 - Maximum Number of Overlapping Intervals
Consider a set of $n$ intervals $[s_i, e_i]$ on a line. 
We say that two intervals $[s_i, e_i]$ and $[s_j, e_j]$ overlaps if and only if their intersection is not empty, i.e., if there exist at least a point $x$ belonging to both intervals. 
Compute the maximum number of overlapping intervals. 

**Example:** 
![[Pasted image 20240109094209.png | center | 550]]
We have a set of 10 intervals, the maximum number of overlapping intervals is 5 (at positions 3 and 4)
## 7.1 - Sweep Line Algorithm
The Sweep Line Algorithm is an algorithmic paradigm used to solve a lot of problems in computational geometry efficiently. 
**The sweep line algorithm can be used to solve problems on a line or on a plane.**

The sweep and line algorithm use an imaginary vertical line **sweeping** over the x-axis. 
As it progresses, we maintain a running solution to the problem at hand. 
The solution is updated when the vertical line reaches a certain key points where some event happen. The type of the event tells us how to update the solution. 
## 7.2 Sweep Line Solution, O(n)
Let's apply the sweep and line algorithm to the problem above. 
We let the sweep line from left to right and stop at the beginning or at the end of every interval. 
These are the important points at which an event occurs: interval start or end. 
We also maintain a counter which keeps track of the number of intervals that are currently intersecting the sweep line, along with the maximum value reached by the counter so far. 

For each point we first add to the counter the number of intervals that begin at that point, and then we subtract the number of intervals that end at that point. 
The figure below shows the points touched by the sweep line and the values of the counter:
![[Pasted image 20240109095128.png| center | 550]]
**Observation:** The sweep line only touches points on the x-axis where an event occurs. 
This is important because the number of considered points, and thus the time complexity, is proportional to the number of intervals, and not to the size of the x-axis. 
# 8 - Check if all Integers in a Range are Covered
![[Screenshot from 2024-01-09 10-01-57.png | center | 700]]
## 8.1 - Intuitive Solution 
The following is the intuitive solution.
It has two nested loops, one that iterates `right-left` times and the inner one that iterates $O(n)$ times, where $n$ is the number of ranges. 
The complexity then is $O(\text{(right-left)}*n)$ time.
```java
isCovered(ranges, left, right) 
	for i = left to right 
		covered = false // i is not covered until proven otherwise
		for range in ranges 
			if i >= range.start && i <= range.end 
				covered = true // i is covered by the current interval
				break;
			
		// if i is not covered then we return false
		if covered == false 
			return false
	
	// each i in [left, right] is covered by at least one range
	return true;
```
## 8.2 - Sweep Line Solution
To solve the problem more efficiently we use sweep line and a map. 
The complexity is linear in the maximum between the number of `ranges` and `right`
```java
isCovered(ranges, left, right)
	// map: point i -> number of open ranges in i
	HashMap openRangesAt; 
	for range in ranges 
		openRangesAt.insert(openRangesAt.getOrDefault(range.start, 0) + 1)
		// range.end+1 as the ranges are inclusive!
		openRangesAt.insert(openRangesAt.getOrDefault(range.end+1, 0) + 1)
	

	openRangesNow = 0
	for i = 0 to left 
		openRangesNow += openRangesNow.getOrDefault(i, 0)

	for point=left to right 
		openRangesNow += openRangesNow.getOrDefault(point, 0)
		if openRangesNow == false
			return false

	return true
```
# 9 - Longest k-Good Segment
The array $a$ with $n$ integers is given. 
Let's call the sequence of one or more consecutive elements in $a$ a segment. 
Also let's call the segment $k$-good if it contains no more than _k_ different values.

**Note:** if the distance between two numbers is `abs(1)` then the two numbers are consecutive.

Find any longest k-good segment.
**Note:** we return the indexes of the longest k-good segment
## 9.1 - Sliding Window Solution, O(n)
We use the sliding window approach: the implementation is self-explanatory
```java
longestKGoodSegment(array, k)
	n = array.length()
	if k == 0 || n == 0
		return (-1,-1)
	if k == 1
		return (0,0)

	wSize = 1 // current window size
	left, right = 0 // left and right delimiter of the window
	pointer = 1 // always points to the element just outside the window

	maxWSize = 1 // store the maximum size of the window so far
	leftResult, rightResult = 0 // delimiter of the maximum window

	HashSet distincts // store distinct elements in the window
	distincts.insert(array[0]) // the windows starts with the first element

	while pointer != n // proceed until the pointer is outside the array
		// the element just outside the window is to be included
		if Math.abs(array[right], array[pointer]) == 1
			distincts.insert(array[pointer]);
			right ++ 1 // stretch the window
			wSize ++ 1 // the size of the window grows
			pointer ++ 1 // shift the pointer to be outside of the window
			
			if wSize > maxWSize // the new size is the maximum?
				maxWSize = wSize
				leftResult = left
				rightResult = right 

				if distincts.size() == k // the window contains k distinct?
					return (leftResult, rightResult)
		else 
			left = pointer // shift the window: the start becomes pointer
			right = left // the window is made of one element
			w_size = 1 
			pointer = right+1 // points to the element outside the window
			distincts.clear() // reset the distinct elements in the window

	return (-1,-1)
```
# 10 - Contiguous Subarray Sum
Given an integer array `nums` and an integer `k`, return `true` if `nums` has a **good subarray** or false otherwise. 
A **good subarray** is a subarray where: 
- its length is **at least** 2 
- the sum of the elements of the subarray is a multiple of `k`

**Note** that:
- A **subarray** is a contiguous part of the array.
- An integer `x` is a multiple of `k` if there exists an integer `n` such that `x = n * k`. `0` is **always** a multiple of `k`.
## 10.1 - Naive Solution, O(n^2)
The obvious brute force approach: from each element we compute every possible subarray starting in that element, check if sum is a multiple of `k` and store the maximum length so far.
## 10.2 - Prefix Sums 
**Prefix sums**, also known as cumulative sums or cumulative frequencies, **offer an elegant and efficient way to solve a wide range of problems that involve querying cumulative information about a sequence of values or elements.**

The essence of prefix sums lies in **transforming a given array of values into another array, where each element at a given index represents the cumulative sum of all preceding elements in the original array.**

An example of prefix sum array is shown in the picture: 
![[Pasted image 20240110100959.png | center | 350]]
Where it is clear that $$P[i] = \Sigma_{j=1}^i\ A[k]$$
### 10.2.1 Prefix Sum using Rust
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
## 10.3 - Prefix Sum Solution, O(n)
The solution is based on the following mathematical property:
**Observation:** any two prefix sums that are not next to each other with the same mod k, or a prefix sum with mod k = 0 that is not the first number will yield a valid subarray.

```java
checkSubarraySum(array, k)
	n = array.length()
	
	prefixSumArray[] // prefix sum array
	pS = 0
	for i = 0 to n 
		pS += array[i]
		prefixSumArray[i] = pS

	// map: modulo -> index i st prefixSumArray[i] % k = modulo
	HashMap modsToIndices
	for (i, prefixSum) in array.enumerate() 
		modulo = prefixSum % k

		if modulo == 0 && i != 0 // if modulo is 0 and not the first ok
			return true

		// first time we see this modulo
		if modsToIndices.contains(modulo) == false
			modsToIndices.insert(modulo, i)
		else 
			// this modulo has been seen
			previousIndex = modsToIndices.get(modulo);
			if previousIndex < i - 1
				return true

	return false
```

**Observation:** 
We really do not need a full prefix sum array, it is enough to compute `sum` as we go and use it also as the mod result (we make `sum = sum%k` and use it as key, sums in modulo works). This is true because we never use the prefix sum of an element before the predecessor, so we can store only the sum up until now. 
# 11 - Update the Array
You have an array containing n elements initially all 0. 
You need to do a number of update operations on it. 
In each update you specify `l`, `r` and `val` which are the starting index, ending index and value to be added. 
After each update, you add the `val` to all elements from index `l` to `r`. 
After `u` updates are over, there will be `q` queries each containing an index for which you have to print the element at that index.
**Observation:** `access(i)` wants the prefix sum of the elements `A[1..i]

To efficiently solve this problem we introduce a new data structure, the **Fenwick Tree**
## 11.1 - Fenwick Tree
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
In fact **the Fenwick tree is an implicit data structure**, which means it requires only $O(1)$ additional space to the space needed to store the input data, in our case the array `A`

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
- `add(i,v)` query: we need to add `v` to all the nodes that covers ranges that include the position `i`, for example if we want to do `add(3,10)` we need to add 10 to the nodes 4 (as it has range $[1,4]$ and $3 \in [1,4]$) and 8 (as it has range $[1,8]$ and $3 \in [1,8]$).
	- **general rule** `add(i,v)`: find the smallest power of 2 grater than `i`, let's call it `j`, and we add `v` to the nodes `j, j*2, j*(2^2), ...` (`j` is the index in the original array, stored over every node in the picture above)

We observe that `sum` takes constant time and `add` takes $\Theta(\log n)$ time. 
This is very good, can we extend this solution to support `sum` queries on more positions? 

**Observation:** currently we are not supporting queries for positions within the ranges between consecutive powers of 2. 
Look at the image above: positions that falls in the range (subarray) `[5, 7]`, which falls between the indices $4$ ($2^2$) and $8$ ($2^3$), are not supported. 
In fact we can't make the query `sum(5)`.
**Enabling queries for this subarray is a smaller instance of our original problem.**

We can apply the **same strategy by adding a new level** to our tree. 
The children of a node stores the partial sums **starting from the next element**. 
The emphasis is in *starting*. 

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
3) Let be $h = \lfloor\log(n)+1\rfloor$, which is the length of the binary representation of any position in the range $[1,n]$. Since any position can be expressed as the sum of at most $h$ powers of $2$, the tree has no more than $h$ levels. In fact, the number of levels is either $h$ or $h-1$, depending on the value of $n$ (**theory_TODO, not clear**)

Now, let’s delve into the details of how to solve our `sum` and `add` queries on a Fenwick tree.
### 11.1.1 - Answering a `sum` query
This query involves beginning at a node `i` and traversing up the tree to reach the node `0`
Thus `sum(i)` takes time proportional to the height of the tree, resulting in a time complexity of $\Theta(\log n)$. 

Let's consider the case `sum(7)` more carefully. 
We start at node 7 and move to its parent (node 6), its grandparent (node 4), and stop at its great-grandparent (the dummy root 0), summing their values along the way. 
This works because the ranges of these nodes ($[1,4], [5,6], [7,7]$) collectively cover the queried range $[1,7]$. 

Answering a `sum` query is straightforward **if we are allowed to store the tree's structure.**
However a significant part of the Fenwick tree's elegance lies in the fact that storing the tree is not actually necessary. 
This is because **we can efficiently navigate from a node to its parent using a few bit-tricks, which is the reason why the Fenwick trees are also called Binary Indexed Trees.**
#### 11.1.1.1 - Compute the Parent of a Node
We want to compute the parent of a node, and we want to do it quickly and without representing the structure of the tree.

Let's consider the binary representation of the indexes involved in the query `sum(7)`

![[Pasted image 20240110150658.png | center ]]

**Theorem:** the binary representation of a node's parent can be obtained by removing the trailing one (i.e., the rightmost bit set to 1) from the binary representation of the node itself.
### 11.1.2 - Performing an `add`
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
### 11.1.2.1 - Computing the Siblings
Continuing the above example, starting with `i = 5`, the next node to modify is its right sibling, node 6. 
Their binary representation is 
![[Pasted image 20240110152844.png| center]]
We see that if we isolate the trailing one in the binary rep. of 5, which is `0001`, and add it to the binary rep. of 5 itself, we obtain the binary rep of 6.

**Finding the Sibling**
**The binary representation of a node and its sibling matches, except for the position of the trailing one**. When we move from a node to its right sibling, this trailing one shifts one position to the left. 
Adding this trailing one to a node accomplishes the required shift. 
Now, consider the ID of a node that is the last child of its parent. 
In this case, the rightmost and second trailing one are adjacent. To obtain the right sibling of its parent, we need to remove the trailing one and shift the second trailing one one position to the left.
Thankfully, this effect is one again achieved by adding the trailing one to the node’s ID.

**The time complexity** of `add` is $\Theta(\log(n))$, as we observe that each time we move to the right sibling of the current node or the right sibling of its parent, the trailing one in its binary rep. shifts at lest one position to the left, and this can occur at most $\lfloor\log(n)\rfloor+1$ times.
## 11.1.2 - Fenwick Tree in Rust
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
## 11.2 - Fenwick Tree Solution
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
# 12 - Nested Segments
We are given $n$ segments: $[l_1, r_1],\dots, [l_n, r_n]$ on a line. 
There are no coinciding endpoints among the segments. 
The task is to determine and report the number of other segments each segment contains.
**Alternatively said:** for the segment $i$ we want to count the number of segments $j$ such that the following condition hold: $l_i < l_j \land r_j < r_i$. 

We provide two solutions to this problem: 
- with Fenwick Tree
- with Segment Tree
## 12.1 - Fenwick Tree Solution
We use a sweep line & Fenwick tree approach. 

For starters we map the segments to the range $[1,2n]$ and sort them by their starting point. 
Then we build the Fenwick tree, we scan each segment $[l_i, r_i]$ and add $1$ in each position $r_i$

Now we scan the segments again. 
When we process the segment $[l_i, r_i]$ we observe that the segments already processed are only the ones that starts before the current one, as they are sorted by their starting points.
Now to find the solution of this problem for the current segment (aka the number of segments contained in the current one) we need to know the number of these segments (the ones that starts before the current one) that also end before the current one, before $r_i$. 
This is computed with a query `sum(r_i)` on the Fenwick Tree.
After computing the solution for the current segment we subtract $1$ to position $r_i$, to remove the contribution of the right endpoint of the current segment. 

The following snippet implement the solution above, using the Fenwick tree previously defined. 
```rust
fn fenwick_nested_segments(input_segments: &[(i32, i32)]) -> Vec<(i64, usize)> {
	let n = input_segments.len();
	// (start, end, index_in_input)
	let mut events: Vec<(i32, i32, usize)> = Vec::with_capacity(n);
	for (i, &(l, r)) in input_segments.iter().enumerate() {
		events.push((l, r, i));
	}
	// sort by starting endpoint
	events.sort_by(|a, b| a.0.cmp(&b.0));
	let mut tree = FenwickTree::with_len(input_segments.len()*2 + 1);
	for i in 0..n {
		tree.add(events[i].1 as usize, 1);
	}
	let mut sol: Vec<(i64, usize)> = Vec::with_capacity(n);
	for i in 0..n {
		sol.push((tree.sum(events[i].1 as usize) - 1, events[i].2));
		tree.add(events[i].1 as usize, -1);
	}
	// restore so that the solution are paired with the input ordering
	sol.sort_by(|a, b| a.1.cmp(&b.1));
	
	sol
}
```
## 12.2 - Segment Tree
**A Segment Tree is a data structure that stores information about array intervals as a tree.**
This allows answering **range queries** over an array efficiently, while still being flexible enough to **allow quick modification of the array**.

The key point here is **range queries**, not only range sums!
We can **find the sum of consecutive array elements**`A[l..r]` or **find the minimum element in a segment** in $O(\log(n))$ time. 
Between answering such queries **we can modifying the elements by replacing one element of the array, or even changing the elements of a whole subsegment** (e.g., assigning all elements `a[l..r]` to any value, or adding a value to all element in the subsegment)

**Segment trees can be generalized to larger dimensions.** For instance, with 2-dimensional Segment trees you can answer sum or minimum queries over some subrectangle of a given matrix in $O(\log^2(n))$ time. 

**We consider the simplest form of Segment Trees**. 
**We want to answer sum queries efficiently.** 
The formal definition of our task is the following: given an array $a[0,\dots,n-1]$, the Segment Tree must be able to perform the following operations in $O(\log(n))$ time
1) find the sum of elements between the indices $l$ and $r$: $\Sigma_{i=l}^r\ a[i]$ 
2) change values of elements in the array: $a[i] = x$

**Observation:** even our simple form of Segment Tree is an improvement over the simpler approaches: 
- a naive implementation that uses just the array can update element in $O(1)$ but requires $O(n)$ to compute each sum query
- a precomputed prefix sums can compute the sum queries in $O(1)$ but updating an array element requires $O(n)$ changes to the prefix sums
### 12.2.1 - Structure of the Segment Tree
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
#### 12.2.1.1 - Construction
Before constructing the segment tree we need to decide: 
- the value that gets stored at each node of the segment tree. In a sum segment tree a node would store the sum of the elements in its range $[l,r]$
- the merge operation that merges two sibling in a segment tree. In a sum segment tree, the two nodes corresponding to the ranges $a[l_1,\dots,r_1]$ and $a[l_2,\dots,r_2]$ would be merged into a node corresponding to the range $a[l_1,\dots,r_2]$ by "concatenating" the values of the two nodes

Note that a vertex is a leaf if its corresponding segment covers only one value of the original array. It is present at the lowest level of the tree and its value would be equal to the corresponding element $a[i]$. 

**For construction of the segment tree, we start at the bottom level (the leaves) and assign them their respective values**. 
**On the basis of these values we can compute the values of the previous level**, using the `merge`  function. 
And on the basis of those, we can compute the values of the previous, and so on until we reach the root.
**It is convenient to describe this operation recursively in the other direction, i.e., from the root vertex to the leaf vertices.**

The construction procedure, if called of a non-leaf vertex, does the following:
- recursively construct the values of the two child vertices 
- merge the computed values of these children 

We start the construction at the root vertex, and hence, we are able to compute the entire segment tree. 
The **time complexity of the construction** is $O(n)$, assuming that the merge operation is $O(1)$, as the merge operation gets called n times, which is equal to the number of internal nodes in the segment tree.
#### 12.2.1.2 - Sum Queries
We receive two integers $l$ and $r$, and we have to compute the sum of the segment $a[l,\dots,r]$ in $O(\log(n))$ time. 
To do this we will traverse the tree and use the precomputed sums of the segments. 

Let's assume that we are currently at the vertex that covers the segment $a[tl,\dots,tr]$. 
There are three possible cases: 
1) the segment $a[l,\dots,r]$ is equal to the corresponding segment of the current index, then we are finished and we return the sum that is stored in the vertex
2) the segment of the query fall completely into the domain of either the left or the right child. In this case we can simply go to that child vertex, which corresponding segment covers the query segment, and execute the algorithm described here with that vertex
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
And since the height of the tree is $O(\log(n))$, we receive the desired running time. 
#### 12.2.1.3  - Update Queries
Now we want to modify a specific element in the array, let's say we want to do the assignment $a[i] = x$. And we have to rebuild the Segment Tree, such that it corresponds to the new, modified array. 

This query is easier than the sum query. Each level of a segment tree forms a partition of the array. Therefore an element $a[i]$ only contributes to one segment from each level. 
Thus only $O(\log(n))$ vertices need to be updated. 

It is easy to see that the update request can be implemented using a recursive function. The function gets passed the current tree vertex, and it recursively calls itself with one of the two child vertices (the one that contains $a[i]$) and after that recomputes its sum value, similar how it is done in the build method (that is as the sum of its two children). 

**Example:** given the same array as before, we want to perform the update $a[2] = 3$ 
![[Pasted image 20240112112424.png | center | 450]]
### 12.2.2 Range Update and Lazy Propagation
Segment Trees allows applying modification queries to an entire segment of contiguous elements and perform the query in the same time $O(\log(n))$. 
When we need to update an interval we will update a node and mark its child that it needs to be updated and update it only when needed. 
To every node we add a field that marks if the current node has a pending update or not. 
Then, when we perform another range update or a sum query, if nodes with a pending update are involved, we first perform the updates and then solve the query. 

**Alternatively said, consider the following segment tree:**


![[Pasted image 20240212165129.png | center | 500]]
**When there are many updates and updates are done on a range, we can postpone some updates (avoid recursive calls in update) and do those updates only when required.**  

Please remember that a node in segment tree stores or represents result of a query for a range of indexes. 
And if this node’s range lies within the update operation range, **then all descendants of the node must also be updated.** 

**Example:** consider the node with value $27$ in above diagram, this node stores sum of values at indexes from $3$ to $5$. 
If our update query is for range $2$ to $5$, then we need to update this node and all descendants of this node. 
With Lazy propagation, we update only node with value $27$ and postpone updates to its children by storing this update information in separate nodes called lazy nodes or values. We create an array `lazy[]` which represents lazy node. 
The size of `lazy[]` is same as array that represents segment tree.
The idea is to initialize all elements of `lazy[]` as 0. A value 0 in `lazy[i]` indicates that there are no pending updates on node `i` in segment tree. 
A non-zero value of `lazy[i]` means that this amount needs to be added to node `i` in segment tree before making any query to the node.
## 12.3 - Segment Trees Solution
**Let's now solve nested segments with a Segment Tree and Sweep Line**

Given the input array `segments` we compute the maximum endpoint between the segments, and call it `n`. 
Then we create a segment tree, based on an array of length $n$, initialized with all zeroes. 
At this point we create the array `axis`, which stores triples where: 
- the first element of the triple is a segment endpoint
- the second element is the index of the endpoint's segment in `segments`
- the third element is a boolean `isStart`, which is set to `true` if the endpoint is the start of its segment, otherwise the end. 

We sort `axis` by the first element, the segments endpoints. 
Finally we "sweep" over axis. 

- `for` `i = 0 to axis.length()
	- if `axis[i].isStart == false`, aka if the current endpoint is the end of its segment
		- the number of nested segments in the segment where `axis[i].endpoint`, that we call `res`, is the range sum on the range `[segments[axis[i].index, axis[i].endpoint` 
		- we push the tuple `(axis[i].index, res)` in the array of results
		- we increase by one the start of the segment indexed with `axis[i].index` in `segments`
	- sort `results` by the indexes and return it

**Why it works?**
![[Pasted image 20240212162551.png | center | 600]]
![[IMG_0416.png | center | 600]]
**In words:**
Consider the segments $[(s_0, e_0), \dots, (s_{n-1}, e_{n-1})]$.

When we find the end of a segment $i$, namely $e_i$ we do the range sum of $(s_i, e_i)$ to get the number of segments contained in the segment $(s_i, e_i)$. 
Then we increase by $1$ the segment $(s_i, s_i)$ in the segment tree. 

This works because we increase by one the start $s_i$ when its segment $i$ has been closed. 
The range sum on $(s_i, e_i)$ will count only segments that starts after $s_i$ and have already been closed (otherwise they would be $0$ in the tree).

**Alternatively said:**
- find the end of the segment $i$, $e_i$. 
- do the range sum $(s_i, e_i)$
	- all the segments $(s_j, e_j)$ that starts after $s_i$ and have already been closed have caused the increment by one of $s_j$
		- starts after $i$ and already been closed, fully contained in $i$
- the segment $i$ has been closed, increase by one $(s_i, s_i)$ in the tree.
# 13 - Powerful Array
An array of positive integers $a_1,\dots,a_n$ is given. 
Let us consider its arbitrary subarray $a_l, a_{l+1},\dots, a_r$, where $1 \le l \le r \le n$.
For every positive integer $s$ we denote with $K_s$ the number of occurrences of $s$ into the subarray.
We call the **power** of the subarray the sum of products $K_s \cdot K_s \cdot s$ for every positive integer $s$
The sum contains only finite number of nonzero summands as the number of different values in the array is indeed finite. 

You should calculate the power of $t$ given subarrays.

**Besides the trivial solutions, we introduce a new algorithmic technique.**
## 13.1 - Mo's Algorithm 
The Mo’s Algorithm is a powerful and efficient technique for **solving a wide variety of range query problems.** 
It becomes particularly **useful for kind of queries where the use of a Segment Tree or similar data structures are not feasible.** 
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
## 13.2 - Solution
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
# 14 - Longest Common Subsequence
Given two strings, `S1` and `S2`, the task is to find the length of the longest common subsequence, i.e. longest subsequence present in both strings. 
**Observation:** subsequence != substring. A subsequence do not have to be contiguous. 

**There are many ways to attack this problem, we use it to talk about Dynamic Programming.**
## 14.1 - Dynamic Programming
Dynamic Programming, like divide-and-conquer, solves problems by combining solutions of subproblems. 
Divide-and-Conquer algorithms partitions the problem into disjoint subproblems, solve the subproblems and then combine their solutions to solve the original problem. 
In contrast, **dynamic programming applies when subproblems overlap, that is, when sub-problems share sub-sub-problems.**
In this context a divide-and-conquer algorithm does more work than necessary, repeatedly solving the common sub-sub-problems. 
**A dynamic programming algorithm solves each sub-sub-problem just once and then saves its answer in a table, avoiding the work of recomputing the answer every time it solves each sub-sub-problem.** 
### 14.1.2 - A first easy problem: Fibonacci
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
fibonacci(n) 
	if (n == 0)
		return 0
	if (n == 1)
		return 1
	
	return fibonacci(n-1) + fibonacci(n-2)
```

In computing `fibonacci(n-1)` we will compute `fibonacci(n-2)` and `fibonacci(n-3)`, 
and in computing `fibonacci(n-2)` we will compute `fibonacci(n-3)` and `fibonacci(n-4)` and so on. 
There are lots of the same Fibonacci numbers that are computed every time from scratch. 

**Memorization is a trick that allows to reduce the time complexity.**
Whenever we compute a Fibonacci number we store it in an array `M`. 
Every time we need a Fibonacci number, we compute it only if the answer is not in the array. 
**This algorithm requires linear time and space.**
```java
fibonacciDP(n)
	if (n == 0)
		return 0 
	if (n == 1)
		return 1
		
	if (M[n] == null) 
		M[n] = fibonacciM(n-1) + fibonacciM(n-2)

	return M[n]
```

**There is a more direct bottom-up approach which uses linear time and constant space.** 
This approach typically depends on some natural notion of "size" of a sub-problem, such that solving any particular sub-problem depends only on solving "smaller" sub-problems. 

We solve the subproblems by size and solve them in size order, smallest first. 
**When solving a particular sub-problem, we have already solved all the smaller subproblems its solution depends upon, and we have saved their solution.** 

In our case this approach corresponds to compute an array `F` which entry `F[i]` requires only on entries `F[i-1]` and `F[i-2`. 
```java
iteFibonacci(n) 
	F[n]
	F[0] = 0
	F[1] = 1

	for i = 2 to n
		F[i] = F[i-1] + F[i-2]

	return F[n-1]
```
### 14.1.3 Memorization vs Tabulation
Tabulation and Memorization are two common techniques used in dynamic programming to optimize the solution of problems by avoiding redundant computations and storing intermediate results.
1. **Tabulation:**
    - **Definition:** **Tabulation involves solving a problem by building a table** **(usually a 2D array or matrix) and filling it in a bottom-up manner. The table is filled iteratively, starting from the base cases and moving towards the final solution.**
    - **Process:**
        - The tabulation approach directly computes the solution for smaller subproblems and uses these solutions to build up to the final solution.
        - It is an iterative approach that avoids recursion and typically uses loops to fill in the table.
        - The final result is usually found in the bottom-right cell of the table.
    - **Advantages:**
        - It is often more intuitive and easier to implement in an iterative manner.
        - It usually has lower memory requirements since it only needs to store the necessary values in the table.
    - **Disadvantages:**
        - It may compute values for all subproblems, even those that are not needed for the final solution, leading to potentially higher time complexity.
2. **Memorization:**
    - **Definition:** **Memorization involves solving a problem by storing the results of expensive function calls and returning the cached result when the same inputs occur again.**
    - **Process:**
        - It is a top-down approach that starts with the original problem and recursively breaks it down into smaller subproblems.
        - The results of subproblems are cached (memorized) in a data structure (like a dictionary or an array).
        - Before computing a subproblem, the algorithm checks whether the result is already stored in the cache. If yes, it returns the cached result; otherwise, it computes the result and stores it for future use.
    - **Advantages:**
        - It avoids redundant computations by storing and reusing previously computed results.
        - It is often more space-efficient because it only stores results for the subproblems encountered during the actual computation.
    - **Disadvantages:**
        - It may introduce overhead due to function calls and the need for a cache, potentially resulting in a slightly slower runtime compared to tabulation.

In summary, both tabulation and memorization are techniques used to optimize dynamic programming solutions by avoiding redundant computations. 
Tabulation builds a table bottom-up, while memorization caches results top-down. 
## 14.2 - Solution
Now that we have refreshed dynamic programming we can go back to our problem. 

The subproblems here ask to compute the longest common subsequence (LCS) of prefixes of the two sequences `S1` and `S2`: given two prefixes `S1[1,i]` and `S[1,j]` our goal is to compute `LCS(S1[1,i], S2[1,j])`

Assume that we already know the solutions to the following three smaller problems
1) `LCS(S1[1,i-1], S2[1,j-1])`
2) `LCS(S1[1,i], S2[1,j-1])`
3) `LCS(S1[1,i-1], S2[1,j])`

Then we have that 
1) if `S1[i] == S2[j]` we can extend a LCS of `S1[1,i-1]` and `S2[1,j-1]` by adding one character `c = S1[i]`
2) if `S1[i] != S2[j]` we can only consider `LCS(S1[1,i], S2[1, j-1])` and `LCS(S1[1,i-1], S2[1,j])`, and we take the longer. 

**Summarizing:**
$$
\text{LCS(S1[1, i], S2[1, j])} = 
	\begin{align}
	\begin{cases}
		0\ &\text{if i = 0 or j = 0} \\
		\text{LCS(S1[1, i-1], S2[1, j-1]) + 1}\ &\text{if S1[i] == S2[j]} \\
		\max(\text{LCS(S1[1, i], S2[1, j-1]), LCS(S1[1, i-1], S2[1, j])})\ &\text{otherwise}
	\end{cases}
	\end{align}
$$

The pseudo-code of the problem is the following 
```java
longestCommonSubsequence(x, y) 
	n = x.length() + 1
	m = y.length() + 1
	dp = [n][m]

	// the first row and the first column are initialized to 0
	// represent the length 0 of the first or second string
	for i = 0 to n
		dp[i][0] = 0
	for j = 0 to m
		dp[0][j] = 0
	
	for i = 1 to n
		for j = 1 to m
			if x[i] == y[j]
				dp[i][j] = dp[i-1][j-1] + 1
			else 
				dp[i][j] = max(dp[i][j-1], dp[i-1][j])

	return C[n-1][m-1]
```
# 15 - Minimum Number of Jumps
Consider an array of `N` integers `arr[]`. 
Each element represents the maximum length of the jump that can be made forward from that element. 
This means if `arr[i] = x`, then we can jump any distance `y` such that `y <= x`.  
Find the minimum number of jumps to reach the end of the array starting from the first element. 
If an element is 0, then you cannot move through that element.  
**Note:** Return -1 if you can't reach the end of the array.
## 15.1 - Solution
**We use Dynamic Programming to solve the problem.** 
More specifically we use **Tabulation** to solve the problem: 
1) create an array `jumps[]` from left to right such that `jumps[i]` indicate the minimum number of jumps needed to reach the position `i` starting from the start
2) we need to fill the array `jumps[]`. we use two nested loops, the outer indexed with `i` and the inner indexed with `j`
	1) outer loop goes from `i = 1` to `n-1` and the inner loop goes from `j = 0` to `i`
	2) if `i` is less than `j+arr[j]` then sets `jumps[i]` to `min(jumps[i], jumps[j]+1`
		1) initially we set `jump[i] = INT_MAX`
3) return `jumps[i-1]`

**The implementation of the solution is the following:** 
```rust
minNumberOfJumps(array)
	n = array.le
	jumps[n]

	for i = 0 to n
		jumps[i] = MAX
	jumps[0] = 0

	if n == 0 || arr[0] == 0
		return -1

	for i = 1 to n
		for j = 0 to i
			// read below
			if i <= j + array[j] && jumps[j] != MAX
				jumps[i] = Math.min(jumps[i], jumps[j]+1)
				break

	return jumps[n-1]
}
```

**The key is the if guard inside the two loops:** 
- `i <= j + arr[j]`: we want to reach `i` and we are in `j`. If `i <= j + array[j]` it means that `i` is reachable from `j` doing the available number of jumps in `j`, which are `array[j`
- `jumps[j] != MAX`: we can actually reach `j` from the start

Then the minimum number of jumps required to reach `i` is the minimum between the current number of jumps to reach `i` and the number of jumps required to reach `j` plus one more jump (to reach `i`)
![[Pasted image 20240116171229.png |center| 600]]
# 16 - Partial Equal Subset Sum
Given an array `array[]` of size `N`, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.

This problem is a well-known NP-Hard problem, which admits a pseudo-polynomial time algorithm. 
The problem has a solution which is almost the same as **0/1 Knapsack Problem.**

--- 

**0/1 Knapsack Problem**
We are given $n$ items. Each item $i$ has a value $v_i$ and a weight $w_i$. We need to put a subset of these items in a knapsack of capacity $C$ to get the maximum total value in the knapsack. 
**It is called 0/1 because each item is either selected or not selected.** 

We can use the following solutions: 
1) if $C$ is small we can use **Weight Dynamic Programming.** The time complexity is $\Theta(Cn)$
2) If $V = \Sigma_i\ v_i$ is small we can use **Profit Dynamic Programming.** The time complexity is $\Theta(Vn)$ 
3) if both $V$ and $C$ are large we can use *branch and bound*, not covered here. 

**Weight Dynamic Programming**
The idea is to fill a $(n+1)\times (C+1)$ matrix $K$. 
Let $K[i,A]$ be the max profit for weight $\le A$ using items from 1 up to $i$.
![[Pasted image 20240116180507.png | center | 600]]
**Profit Dynamic Programming**
The idea is similar. 
We can use a $(n+1)\times (V+1)$ matrix $K$. 
Let $K[V][i]$ be the minimum weight for profit at least $V$ using items from $1$ up to $i$. 
Thus we have:
$$K[V][i] = \min(K[V][i-1], K[V-v[i][i-1])$$
The solution is $\max(a:K[V,n]\le C)$ 

---
## 16.2 - Solution
As in the 0/1 knapsack problem we construct a matrix $W$ with $n+1$ rows and $v+1$ columns. 
Here the matrix contains booleans. 

The entry $W[i][j]$ is `true` if and only if there exists a subset of the first $i$ items with sum $j$, false otherwise. 

The entries of the first row $W[0][]$ are set to false, as with $0$ elements you can not make any sum. 
The entries of the first column $W[][0]$ are set to true, as with the first $j$ elements you can always make a subset that has sum $0$, the empty subset. 

Entry $W[i][j]$ is true either if $W[i-1][j]$ is true or $W[i-1][j - S[i]]$ is true. 
- $W[i-1][j] = \text{T}$, we simply do not take the $i$-th element, and with the elements in $1, i-1$ we already can make a subset which sum is $j$ 
- $W[i-1][j-S[i]] = \text{T}$, as before: if the subset with one element less than $i$ has sum equal to $j - S[i]$ it means that if we take $i$ we reach exactly a subset with sum $j$

**Said easy:**
- we divide the sum of the array by $2$: if the sum is not divisible by $2$ it means that there cannot be two partitions that summed gives the the sum.
- once divided is the same problem above: 
	- exists a subset of the elements that summed gives the half of the sum?
		- if yes then the answer will be true, false otherwise

```java
partialEqualSubsetSum(array)
	n = array.length()
	sum = 0
	for i = 0 to n 
		sum += array[i]
	
	if sum % 2 != 0 
		return false

	sum = sum / 2

	// dp[i][j] = true if exists a subset of the 
	// first i elements in array that summed gives j
	dp[n+1][sum+1]
	
	for i = 0 to n+1
		dp[i][0] = true
	for j = 1 to sum+1
		dp[0][j] = false

	// i is the current number of elements of which we make a subset
	for i = 1 to n+1
		// j is the current "target" sum 
		for j = 1 to sum+1
			// if array[i-1] is alone bigger than j then it cannot be in 
			// the subset of elements that summed gives j
			if array[i-1] > j
				dp[i][j] = dp[i - 1][j]
			else 
				dp[i][j] = 
					dp[i-1][j] || // dont put i in the subset
					dp[i-1][j - array[i - 1]] // put i in the subset

	return dp[n][sum]
```

The **then branch** of the `if` is crucial: if `arr[i - 1]` is greater than `j`, it means that including the current element `i` in the subset would make the sum exceed the current target sum `j`.
Therefore, the solution at `dp[i][j]` would be the same as the solution without including the current element, i.e., `sol[i - 1][j]` (aka, we do not include `i`, as we can't)
# 17 - Longest Increasing Subsequence
Given an array of integers, find the **length** of the **longest (strictly) increasing subsequence**
from the given array.
**observation:** subsequence, as before, it is not a contiguous. 

As an example consider the sequence $S=\{10,22,9,21,33,50,41,60,80\}$.
The length of $LIS$ is $6$ and $\{10,22,33,50,60,80\}$ is a $LIS$. 
In general the $LIS$ is not unique.

Consider the sequence $S[1,n]$ and let $LIS(i)$ be the $LIS$ of the prefix $S[1,i]$ whose last element is $S[i]$. 
$$LIS(i) = \begin{cases}1 + \max(LIS(j)\ |\ 1\le j\le i\ \text{and}\ S[j] < S[i]\\ 
1 \text\ \ \ \ \text{if such $j$ does not exists}
\end{cases}$$
## 17.1 - Solution
Due to optimal substructure and overlapping subproblem property, we can also utilize Dynamic programming to solve the problem. Instead of memoization, we can use the nested loop to implement the recursive relation.

The outer loop will run from `i = 1 to N` and the inner loop will run from `j = 0 to i` and use the recurrence relation to solve the problem.

The reasoning is the following:
- The outer loop (`for i in 1..n`) iterates over each element of the array starting from the second element.
- The inner loop (`for j in 0..i`) iterates over elements before the current element `arr[i]`.
- The if statement checks if the current element `arr[i]` is greater than the element at index `j` and if increasing the length by 1 (`lis[j] + 1`) results in a longer LIS ending at index `i`. If true, it updates `lis[i]` accordingly.

```java
longestIncreasingSubsequence(array) 
	n = array.length()
	
	lis[n]
	for i = 0 to n
		lis[i] = 1

	for i = 1 to n 
		// j is used to compute the LIS of the prefix up to i
		for j = 0 to i
			// if the current element is bigger than the previous element 
			// array[j] and the lis[i] is smaller than lis[j] plus the 
			// current element
			if array[i] > array[j] && lis[i] < lis[j] + 1
				lis[i] = lis[j] + 1

	return lis.max()
```
## 17.2 - Smarter Solution: Speeding up LIS
The main idea of the approach is to simulate the process of finding a subsequence by maintaining a list of “buckets” where each bucket represents a valid subsequence. 
Initially, we start with an empty list and iterate through the input `nums` vector from left to right.

For each number in `nums`
- If the number is greater than the last element of the last bucket (i.e., the largest element in the current subsequence), we append the number to the end of the list. This indicates that we have found a longer subsequence.
- Otherwise, we perform a binary search on the list of buckets to find the smallest element that is greater than or equal to the current number. This step helps us maintain the property of increasing elements in the buckets.
- Once we find the position to update, we replace that element with the current number. This keeps the buckets sorted and ensures that we have the potential for a longer subsequence in the future.

Consider the following pseudo-implementation
```java
smartLIS(nums) 
	n = nums.length()
	List ans

	ans.add(nums[0])

	for i = 1 to n
		if nums[i] > ans.last()
			ans.add(nums[i])
		else
			// low is the index of the smallest element >= nums[i] in `ans`
			low = binarySearch(ans, nums[i])
			ans.set(low, nums[i]);
		
	return ans.length()
```

**theory_TODO**
# 18 - Longest Bitonic Sequence
Given an array `arr[0 … n-1]` containing $n$ positive integers, a subsequence of `arr[]` is called **bitonic** if it is first increasing, then decreasing. 
Write a function that takes an array as argument and returns the length of the longest bitonic subsequence. 
A sequence, sorted in increasing order is considered Bitonic with the decreasing part as empty. Similarly, decreasing order sequence is considered Bitonic with the increasing part as empty. 
## 18.1 - Solution
This problem is a slight variation of the previous problem. 
Let the input array `arr[]` be of length `n`. 
We need to construct two arrays `lis[]` and `lds[]` using the DP solution of the Longest Increasing Subsequence
- `lis[i]` stores the length of the longest increasing subsequence **ending** in `arr[i]`
- `lds[i]` stores the length of the longest decreasing subsequence **starting** in `arr[i]`

We return the max value of `lis[i] + lds[i] -1` where $i\in [0,n-1]$

To compute `lds[i]` we iterate through the array backwards and apply the same reasoning used for `lis[]`, as we are looking for a decreasing sequence but proceeding backwards, is the same as an increasing sequence. 

**Basically the output is the sum of the longest increasing sequence left to right and the longest increasing sequence right to left**

The following is the implementation of the solution:
```rust
longestBitonicSubsequence(array)
	n = array.length()

	// exactly as LIS
	lis[n]
	for i = 0 to n
		lis[i] = 1

	for i = 1 to n 
		for j = 0 to i
			if array[i] > array[j] && lis[i] < lis[j] + 1
				lis[i] = lis[j] + 1

	// LIS but backwards
	lds[n]
	for i = 0 to n
		lds[i] = 1

	for i = n-1 to 0
		for j = n-1 to i
			if array[i] > array[j] && lds[i] < lds[j] + 1 {
				lds[i] = lds[j] + 1
			}

	lisMax = lis.max()
	ldsMax = lds.max()

	return lisMax + ldsMax - 1
```
# 19 - Meetings in One Room
There is **one** meeting room in a firm. 
There are `N` meetings in the form of `(start[i], end[i])` where `start[i]` is start time of meeting `i` and `end[i]` is finish time of meeting `i`.  
Find the maximum number of meetings that can be accommodated in the meeting room, knowing that only one meeting can be held in the meeting room at a particular time.

**Note:** Start time of one chosen meeting can't be equal to the end time of the other chosen meeting.

There are various ways to solve the problem, we use it to present **greedy algorithms.**
## 19.1 - Greedy Algorithms
A **greedy algorithm** is any algorithm that follows the problem-solving heuristic of making the locally optimal choice at each stage. 
In many problems a greedy strategy **does not produce an optimal solution**, but a greedy heuristic **can yield locally optimal solutions** that approximate a globally optimal solution in a reasonable amount of time. 

**Example:** a greedy strategy for the Travelling Salesman Problem (TSP) is the heuristic "at each step of the journey visit the nearest city". 
This heuristic does not intend to find the best solution, but it terminates in a reasonable number of steps; finding an optimal solution to such a complex problem typically requires unreasonably many steps.
**Note:** this greedy approach is not very good on the TSP, as the current optimal choice is based on previous choices, and greedy algorithms never reconsider the choices they have made so far.

Most problems for which greedy algorithms yields good solutions (aka good approximation of the globally optimal solution) have two property: 
1) **greedy choice property:** we can make whatever choice seems best at the moment and then solve the subproblems that arise later. The choice made by a greedy algorithm may depend on choices made so far, but not on future choices or all solutions to the subproblem. It iteratively makes one greedy choice after another, reducing each given problem into a smaller one. In other words **a greedy algorithm never reconsider its choices**. This is **the main difference from dynamic programming**, which is exhaustive and is guaranteed to find the solution. After every stage, dynamic programming makes decisions based on all the decisions made at the previous stage and may reconsider the previous stage's algorithmic path to the solution
2) **optimal substructure:** a problem exhibits optimal substructure if an optimal solution to the problem contains optimal solutions to the sub-problems. 
## 19.2 - Solution
We use the greedy approach to solve the problem: 
- sort all the meetings in increasing order using their finish time
- select the first meeting of the sorted list as the first meeting that will be held, push it to the list `results` of meetings and set a variable `time_limit` to the finish time of the first meeting 
- iterate from the second meeting to the last one: if the starting time of the current meeting is greater that the previously selected meeting's finish time then select the current meeting
	- push it in `results`
	- update the `time_limit` variable to the finishing time of the current meeting

**Observation:** the problem requires only to give the **maximum number** of meetings that we can have in one room. 
We do not care about the meetings that can be held, so `results` is useless. 
Also we can sort the array of meetings that we have received, as there is no problem in messing their order. 
In the end, we need only the variable `time_limit` and a variable `result` to increment every time we find a meeting that can happen. 

The following implementation solves the problem: 
```rust
meetingsInARoom(meetings) 
	// sort meetings by their starting time
	meetings.sort()

	timeLimit = meetings[0].end
	// at least we held the first meeting
	result = 1

	for meeting in meetings 
		if meeting.0 > timeLimit
			result++ 
			timeLimit = meeting.end

	return result; 
```
# 20 - Wilbur and Array
Wilbur the pig is tinkering with arrays again. 
He has the array $a_1,\dots,a_n$ initially consisting on $n$ zeroes.
At one step he can choose any index $i$ and either add $1$ to all elements $a_i,\dots, a_n$ or subtract $1$ from all elements $a_i,\dots,a_n$. 
His goal is to end up with the array $b_1,\dots,b_n$. 
Of course Wilbur wants to achieve this goal in the minimum number of steps and asks you to compute this value. 

**Example:** 
- input: 
	- 5, the size of the array $a$ `= [0,0,0,0,0]`
	- `[1,2,3,4,5]`, the target array $b$
- output: 
	- 5, as we need five `+1` operation, one for every element `i`
		- `i = 0: +1` $\rightarrow$ `a = [1,1,1,1,1]`
		- `i = 1: +1` $\rightarrow$ `a = [1,2,2,2,2]`
		- ...
## 20.1 - Solution
The solution is based upon the observation that the minimum number of operations is equal to the sum of differences (in absolute value) between consecutive elements. 
Given the array $v[1\dots n]$ we have that
$$\text{result} = v[1] + \Sigma_{i=2}^n\ |v_i-v_{i-1}|$$
Why this works is pretty intuitive. 

Here's the rust implementation: 
```java
wilburAndArray(array)
	n = array.length()
	result = Math.abs(array[0])

	for i = 1 to n
		result += Math.abs(array[i] - array[i-1])

	return result
```
# 21 - Woodcutters
Little Susie listens to fairy tales before bed every day. Today's fairy tale was about wood cutters and the little girl immediately started imagining the choppers cutting wood. 
She imagined the situation that is described below.

There are $n$ trees located along the road at points with coordinates $x_1,\dots,x_n$
Each tree has its height $h_i$.
Woodcutters can cut a tree and fell it to the left or to the right. 
After that it occupies one of the segments $[x_i - h_i, x_i]$ or $[x_i, x_i+h_i]$. 
The tree trait that is not cut down occupies a single point with coordinate $x_i$. 
Woodcutters can fell a tree if the segment to be occupied by the fallen tree does not contain any occupied point. 
The woodcutters want to process as many trees as possible. 
**What is the maximum number of trees to fell?**
## 21.1 - Solution
We use a greedy approach to solve the problem. 
1) we always cut the first tree, making it fall to the left
2) we always cut the last tree, making it fall to the right
3) we **prioritize** **left falls**, meaning that if we can make the current tree falling to the left we always will: consider the current tree $i$
	1) if the previous tree is at a point $x_{i-1}$ that is smaller (aka farther) that where the current tree would fall, then we cut the tree and make it fall to the left
	2) else, if the current tree position $x_i$ plus its height $h_i$ do not fall over next tree $i+1$ we cut to the right
		1) in this case we also update $x_i$ as $x_i + h_i$, the tree has fallen to the right and from the standpoint of the next tree its position is $x_i+h_i$
	3) otherwise we do nothing

The code makes it even more clear: 
```java
woodcutters(trees)
	n = trees.length()
	// always cut at least the first and last trees
	cutted = 2

	for i = 1 to n-1
		// left fall: the previous tree is placed before where the 
		// current trees fall when "left-cutted"
		if trees[i-1].x < trees[i].x - trees[i].h
			cutted++
			continue
			
		// cant fall to the left, lets try to the right
		// if the current tree fall in a place that is smaller 
		// than the next tree place, then we cut
		if trees[i].x trees[i].h < trees[i+1].x
			cutted++
			continue

	return result
}
```
# 22 - Bipartite Graph
You are given an adjacency list of a graph **adj**  of V of vertices having 0 based index. Check whether the graph is bipartite or not.

**The Problem in detail:**
A **Bipartite Graph** **is a graph whose vertices can be divided into two independent sets**, $U$ and $V$, such that every edge $(u, v)$ either connects a vertex from $U$ to $V$ or a vertex from $V$ to $U$. 
In other words, for every edge $(u, v)$, either $u$ belongs to $U$ and $v$ to $V$, or $u$ belongs to $V$ and $v$ to $U$. 
We can also say that there is no edge that connects vertices of same set.

**A bipartite graph is possible if the graph coloring is possible using two colors such that vertices in a set are colored with the same color.**
## 25.1 - Graphs 101
**theory_TODO**
## 25.2 - Solution
We use the Breadth-First Search (BFS) to solve the problem: 
- assign `red` to the source vertex
- color with `blue` all the neighbors 
- color with `red` all neighbor's neighbor
- this way, assign color to all vertices such that it satisfies all the constraints of $m$ way coloring, where $m=2$
- **while assigning colors, if we find a neighbor of the same color of the current vertex, we return false as the graph cannot be bipartite.**

```java
isBipartite(graph[][]) 
	nNodes = graph.length()
	// maps node i to the color of i
	HashMap colors

	for i = 0 to nNodes 
		// if the node i has not yet a node
		if colors.containsKey(i) == false
			// assign a color to i, if i has a neighbor 
			// of the same color then we return false
			if colorBFS(graph, i, colors) == false
				return false

	return true

// returns true if a color was correctly assigned
// returns false if it finds a neighbor of node with the same color
colorBFS(graph[][], node, colors)
	nNodes = graph.length()

	// color the node
	color.insert(node, 1)
	// create the frontier
	queue = new Queue()
	queue.add(node)

	while queue.isEmpty() == false
		currentNode = queue.poll()

		// iterate all the nodes
		for i = 0 to nNodes 
		// currentNode and the node `i` are connected: i is a neighbor 
		if graph[node][i] == 1
			// the neighbor i has an assinged color
			if colors.containsKey(i) == true
				// has the neighbor the same color as the current node?
				if colors.get(i) == color.get(currentNode)
					return false // the graph is not bipartite
			else 
				// assign a different color to the neighbor
				colors.add(i, 1 - colors.get(currentNode));
			
	return true
```