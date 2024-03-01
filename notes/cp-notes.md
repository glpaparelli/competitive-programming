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
# 2 - Missing Number in an Array
Given an array of size $n-1$ and given that there are numbers from 1 to $n$ with one missing, find the missing number.
## 2.1 - Trivial Solutions
- for each element we scan the array searching for the next number, if we do not find it we return $-1$, $O(n^2)$
- sort the array then find the missing one, $O(n\log(n))$
## 2.2 - Smart Solution, O(n)
We use the gauss sum to find the missing number: 
- compute the sum of the elements of the array, `actualSum`
- compute the sum of the first $n$ elements: `sum = ` $\frac{n(n+1)}{2}$
- return `sum - acualSum`
# 3 - Trapping Rain Water
Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining. 

![[Screenshot from 2024-01-05 11-17-40.png | center | 400]]
## 3.1 -Trivial Solution, O(n^2)
Traverse every array element and find the highest bars on the left and right sides. 
Take the smaller of two heights. 
The difference between the smaller height and the height of the current element is the amount of water that can be stored in this array element.
## 3.2 - Smart Solution: Precomputation, O(n)
**The point is to think locally, not globally.**
**We compute the left and right leaders.** 

An element of an array is a **right leader** if it is greater than or equal to all the elements to its right side. The rightmost element is always a leader.
**Left leaders are analogous.**

Given the array $h$ of heights, we use two arrays to store the right and left leaders:
$$
\begin{align}
	\text{LL[$i$]} &= \max_{j < i} h[j] \\
	\text{RL[$i$]} &= \max_{j > i} h[j]
\end{align}
$$
**Said easy:**
- $LL[i]$ contains the left leader for the position $i$
- $RL[i]$ contains the right leader for the position $i$

Now, how many units of water we can find on top of the cell $i$?
The minimum height between the left and right leader of the current position minus $h[i]$, the height of the current position: 
$$w(i) = \min(\text{LL[$i$]}, \text{RL[$i$]}) - h[i]$$
![[Pasted image 20240105111518.png | center | 500]]
## 3.2 - Two Pointers Trick
Two pointers is really an easy and effective technique that is typically used for searching pairs in a sorted array.
It employs using two pointer, typically one from the start of the array going left-to-right and one from the end of the array going right-to-left, until they meet. 
We already seen a sneak peak of this in Trapping Rain Water. 
## 3.3 - Solution, O(N)
The idea is taken from the Precomputation solution, but we only use two variables to store the currently "meaningful" leaders. 

**We take two pointers**, `left` and `right` and we initialize `left` to `0` and `right` to the last index `n-1`. 
We also create two variables, `left_max` and `right_max`, they represent the maximum left/right height seen so far. 

Since `left` is the first `left_max` is `0`, same for `right_max`
This is intuitive: `left_max` of `left = 0` falls outside the array, its height is 0, the same goes for `right_max` when `right = n-1`.

Now we iterate, as long as `left < right`. 
We have to decide which pointer we will have to shift: **shift the pointer that has smaller height value:** 
- if `heights[left] < heights[right]` we will shift `left`
- we will shift `right` otherwise

Now, if we have that `heights[left] > left_max` we can't store any water over the position pointed by `left`, **it would fall as left_max is not high enough** to contain any water over `left`
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
				// add to the result the amount of warer
				// that can be kept over the position left
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
# 4 - Find Peak Element
![[Screenshot from 2024-01-05 15-00-58.png | center | 700]]
To solve this problem we use **Binary Search**. 
**We start by giving the basics of this technique.** 
## 4.1 - Binary Search
**An efficient algorithm that searches for a given key within a sorted array of items.**
Binary search repeatedly divides the search range in half until the target element is found or the search range becomes empty, resulting in a time complexity of $O(\log n)$

Binary search is a technique that falls under the umbrella of the **divide-and-conquer** paradigm, that tackles a complex problem by breaking it down into smaller, more manageable subproblems of the same type: 
- **divide:** the initial problem instance is partitioned into smaller subinstances of the same problem
- **solve:** these subinstances are then solved recursively. If a subinstance reaches a certain manageable size, a straightforward approach is employed to solve it directly
- **combine:** the solutions obtained from the subinstances are combined to obtain the final solution for the original, larger instance of the problem

**Considering the binary search**, we have: 
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
**This way, we continue the search in the first half of the array**, seeking additional occurrences of the `key`. 
If there are more matches, `answer` will be further updated with smaller positions.
### 4.1.1 Applications of Binary Search
#### 4.1.1.1 Binary Search the Answer
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
// [left, right] is the range of candidate answers
rangeBS(pred, left, right)
	low = left
	high = right

	answer

	while low < high
		mid = low + (high-low) / 2

		if pred(mid) == true
			answer = mid
			// we want the largest good answer, hence we 
			// continue the search on the second half
			low = mid + 1 
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
We have a sequence of $n$ mutually-disjoint intervals. The extremes of each interval are non-negative integers. We aim to find $c$ integer points within the intervals such that the smallest distance $d$ between consecutive selected point is **maximized**. 

If a certain distance is feasible (i.e., there exists a selection of points at that distance), then any smaller distance is also feasible. 
Thus the feasibility is a monotone boolean predicate that we can use to binary search the answer. 
As the candidate answers range from $1$ to $l$, where $l$ is the overall length of the intervals, the solution takes $\Theta(\log(l))$ evaluations of the predicate.  

**What's the cost of evaluating the predicate?** 
We first sort the intervals. 
Now we can evaluate any candidate distance $d'$ by scanning the sorted intervals from left to right. 
First we select the left extreme of the first interval as the first point. 
Then, we move over the intervals and we choose greedily the first point, which is at a distance at least $d'$ from the previous one. 
Thus an evaluation of the predicate takes $\Theta(n)$ time. 

The total running time is $\Theta(n\log(n))$ 

**Said Easy:**
We want to place $c$ points where the smallest distance $d$ between consecutive points in maximized. 
Consider such a distance $d'$. Any distance bigger than that would be a better answer, if it were to exists. The predicate is monotone. 
The answers range is $[1, l]$, where $l$ is the overall length of the intervals. Our answers lies within $[1,l]$. 
We binary search it using `rangeBS` we have seen before, hence we evaluate the predicate $O(\log n)$ times.
How much it costs to evaluate the predicate? We have to scan all the intervals checking that the current candidate allow us to place at least $c$ points, hence $O(n)$

The **pseudo**-implementation greatly clarifies the process: 
```java
pred(intervals, distance, c)
	// the first point is the start of the first interval
	lastSelected = intervals[0].start 
	counter = 1
	// for every interval in intervals
	for i in intervals
		// we select as many points as we can in every interval
		// we get the max using i.start because if a point falls
		// in the empty space between two intervals we place it 
		// at the start of the current interval
		while Math.max(lastSelected + distance, i.start) <= i.end
			lastSelected = Math.max(lastSelected + distance, i.start)
			counter++
	
	// returns true if we placed at least c points
	return counter >= c
```
```java
socialDistancing(intervals, c)
	// l is the maximum length of an interval
	if l < c
		return -1 // there is no solution

	// sort the intervals by the start
	intervals.sort()
	
	// do a tailored rangeBS on all the candidate answers
	// for each of them compute the predicate
	rangeBS(1, l+1, pred)
```
## 4.2 - Solution
**We can use the binary search philosophy to solve the problem.**
We compute the middle of the array. 
If the element in the middle is smaller than its right neighbor **than the right neighbor could be a peak**, and we do `low = mid + 1` to proceed the search in the right side of the array. 
Otherwise we do the opposite. 
```java
peak_elements(a[])
	n = a.length()
	low = 0
	high = n - 1

	while low < high
		mid = low + (high - low) / 2

		if a[mid] < a[mid + 1]
			low = mid + 1
		else 
			high = mid 

	return low 
```
![[Pasted image 20240211104316.png | center | 500]]

This solution works because a peak is guaranteed to be found (as a last resource, in the first or last element of the array).
The beauty of this solution is that it naturally covers all the strange cases. 
It would be tempting to write something like 
```java
if a[mid] > a[mid+1] && a[mid] > a[mid-1]
	return mid
```
but that requires to cover many edge cases (array of size 1, 2, if mid is $0$, if mid is $n-1$, ...)
# 5 - Maximum Path Sum
Find the maximum possible sum from one leaf node to another.

![[Pasted image 20240108100401.png | center | 490]]

**To solve this problem we need to use a tree traversal.**
Fist we go back and give the basics. 
## 5.1 - Tree Traversals 
**Tree traversal** (also known as tree search and walking the tree) **is a form of graph traversal and refers to the process of visiting** (e.g. *retrieving*, *updating*, or *deleting*) **each node in a tree data structure, exactly once.** 
**Such traversals are classified by the order in which the nodes are visited.**

**Traversing a tree involves iterating over all nodes in some manner.** 
Because from a given node there is more than one possible next node some nodes must be deferred, aka stored, in some way for later visiting. 
This is often done via a stack (LIFO) or queue (FIFO). 
**As a tree is a recursively defined data structure, traversal can be defined by recursion.**
In these cases the deferred nodes are stored implicitly in the call stack.

**Depth-first search is easily implemented via a stack, including recursively (via the call stack), while breadth-first search is easily implemented via a queue.**
### 5.1.1 - Depth-First Search
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
#### 5.1.1.1 - PreOrder
1) visit the current node, in the figure: position red
2) recursively traverse the current node's left subtree
3) recursively traverse the current node's right subtree

The preorder traversal is a topologically sorted one, because the parent node is processed before any of its child nodes is done. 

![[Screenshot from 2024-01-08 09-27-13.png | center | 300]]
#### 5.1.1.2 - PostOrder
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
#### 5.1.1.3 - InOrder
1) recursively traverse the current node's left subtree
2) visit the current node, in the figure: position green
3) recursively traverse the current node's right subtree

In a binary search tree ordered such that in each node the key is greater than all keys in its left subtree and less than all keys in its right subtree, in-order traversal retrieves the keys in ascending sorted order. 

![[Screenshot from 2024-01-08 09-29-11.png | center | 300]]
## 5.2 Trivial Solution, O(n^2)
A simple solution is to traverse the tree and do following for every traversed node $X$: 
1. Find maximum sum from-leaf-to-root in left subtree of $X\ \clubsuit$
3. Find maximum sum from-leaf-to-root in right subtree of $X\ \clubsuit$. 
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

For every node we call twice `maxLeafRootSum`, which costs O(n), hence the total cost is $O(n^2)$
## 5.3 - Optimal Solution, O(n)
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
// 2) root-to-leaf path sum, which is returned.
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

**Said Easy:**
To solve the problem we use a custom **post-order visit** `maxPathSum` that exploits a global variable `maxSum` to store the result. 
Our visit `maxSumPath` receive in input the root of the tree and then
- if the passed root is `null` we return $0$
- `left = maxSumPath(root.leftST)`, we recur on the left subtree
- `right = maxSumPath(root.rightST)`, we recur on the right subtree
	- the function **returns the sum of the path from the root to a leaf**, hence `left` is the sum from the current node to a leaf in the left subtree, analogous for `right`
- `maxSum = max(left+right+root.value, maxValue)`, update the `maxSum` if we found a bigger path from leaf to leaf passing for the current node

When the visit terminates the result is stored in the global variable `maxSum`
# 6 - Two Sum in a Sorted Array
Given a sorted array $a$ (sorted in ascending order), having $n$ integers, find if there exists any pair of elements ($a[i]$, $a[j]$) such that their sum is equal to $x$.
## 6.1 - Naive Solution, O(n^2)
The naive approach is obvious and takes $O(n^2)$, we simply scan the array twice and we return when we find two numbers that adds up to X. 
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
## Two Pointers Solution, O(n)
**We take two pointers, one representing the first element and other representing the last element of the array, and then we add the values kept at both the pointers.** 

If their sum is smaller than X then we shift the left pointer to right, as moving `left` to the right increase the first adding element.
If their sum is greater than X then we shift the right pointer to left, as moving `right` to the left decrease the second adding element.
**We keep moving the pointers until we get the sum as X.**

**The following pseudo-implementation clarifies the process.**
```java
existsPairSum(a[], x)
	n = a.length()
	left = 0
	right = n-1

	while left < right
		if a[left] + a[right] == x
			return true
		if a[left] + a[right] < x
			left++ // smaller than target, we need to increase
		else
			right-- // bigger than target, we need to decrease

	return false
```
# 7 - Frogs and Mosquitoes
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
## 7.1 Binary Search Tree
**A binary search tree (BST), also called an ordered or sorted binary tree, is a rooted binary tree data structure with the key of each internal node being greater than all the keys in the respective node's left subtree and less smaller the ones in its right subtree.** 
The time complexity of operations on the binary search tree is linear with respect to the height of the tree.

Binary search trees allow binary search for fast lookup, addition, and removal of data items. 
Since the nodes in a BST are laid out so that each comparison skips about half of the remaining tree, the lookup performance is proportional to that of binary logarithm.

The complexity analysis of BST shows that, on average, the insert, delete and search takes $O(\log n)$, where $n$ is the number of nodes. 

In the worst case, they degrade to that of a singly linked list: $O(n)$
This is true only if we are not using a Balanced BST, which is always possible: if you have a sorted array to make a BST, select the median element as the "root". 

![[Pasted image 20240109090705.png | center | 250]]
### 7.1.1 Searching
Searching for a specific key can be programmed recursively or iteratively. 

Searching begins by examining the root node. If the tree is `nil`, the key being searched for does not exist in the tree. 
Otherwise, if the key equals that of the root, the search is successful and the node is returned. 
**If the key is less than that of the root, the search proceeds by examining the left subtree. Similarly, if the key is greater than that of the root, the search proceeds by examining the right subtree.** 
This process is repeated until the key is found or the remaining subtree is `nil`. 
If the searched key is not found after a `nil` subtree is reached, then the key is not present in the tree.
![[Screenshot from 2024-01-19 15-34-31.png | center | 500]]
### 7.1.2 Predecessor and Successor
For certain operations, given a node $x$, finding the successor or predecessor of $x$ is crucial. 
Assuming all the keys of the BST are distinct, **the successor of a node** $x$ in BST **is the node with the smallest key greater** than $x$'s key. 
On the other hand, **the predecessor of a node** $x$ in BST **is the node with the largest key smaller** than $x$'s key. 
Following is pseudo-code for finding the successor and predecessor of a node $x$ in BST.
![[Screenshot from 2024-01-19 15-39-52.png | center ]]

**Said easy:**
**Predecessor** and **Successor** takes $O(\log(n))$ and we have that: 
- the **successor of a node** is the node with the smallest key that is bigger than the current node
	- once to the right, then left as long as possible
- the **predecessor of a node** is the node with the greatest key that is smaller than the current node
	- once to the left, then right as long as possible
### 7.1.3 Removing Elements
**Removing an element** from a BST costs $O(h)$
- if it is a leaf just remove it
- if it is a node with a single child: copy the node of the child into the current node and delete the child
- if it is a node with two children: find the **in-order successor** of the node, copy the contents of the in-order successor in the current node and then remove the in-order successor 
## 7.2 - Solution
We store the position of the frogs in a BST, `frogBST`. 
When a mosquito lands on a position $b$ we check which frog will eat it by simply doing a f`rogBST.predecessor(b)` query on the tree. 
It is possible that some mosquito cannot be eaten right away, the uneaten mosquitoes will be stored in their own BST (`mosquitoBST`), using their landing position as keys. 

**We have to handle overlaps**.
If a frog partially overlaps with another then we have to fix the overlap to make sure that they cover different ranges.
If a frog totally overlaps with another then the second frog will never eat and must be removed. 
To solve the overlapping that may arise within the input: 
- Sort the frogs by their position
- If a frog `f1` partially overlaps the frog `f2` move `f2` by adding to its position the needed amount. Mind that `f2` has now a shorter tongue (shorted by the amount of the shift)
- If a frog `f1` fully overlaps `f2`, remove `f2`
- Sort back the frogs by their original ordering

The algorithm behaves as follows: 
- insert the frogs in a BST `frogBST` using their position as key
- for each mosquito `m` in `mosquitoes`
  - find the predecessor `f` of `m`  in `frogBST`, if `f.position + f.tongue >= m.position`
    - `f` eats the mosquito `m` and its tongue grows by the size of `m`
    - maybe `f` now can eat other mosquitoes that could not be eaten before, inspect `mosquitoBST` to see if there is a mosquito that f can eat, and in case repeat.
      - inspect: find the successor `m'` of `f.position` in `mosquitoBST` and see if `f.position + f.tongue >= m'.position`
    - `f` now may overlap or fully contain other frogs
      - For every successor of `f` in `frogBST`:
        - if it overlaps with `f` remove the overlap by updating their positions
          - position of the successor = `f.position + f.tongue + 1`
        - if it is fully contained by `f` then delete this frog as it will never eat
  - else, insert `m` in `mosquitoBST` and continue
![[IMG_0413.png | center | 600]]
**Time Complexity:** 
Removing elements costs $O(\log(n))$ and we need to remove at most $n$ nodes. so the cost of the algorithm is dominated by the worst-case deletion of nodes.
Hence the worst case complexity is $O(n\log(n))$.
# 8 - Sliding Window Maximum
Given an array `A[0, n-1]` and an integer `k`, the goal is to find the maximum of each subarray (window) of `A` of size `k`
## 8.1 - Trivial Solution, O(nk)
The simplest approach is to consider each of the $n-k+1$ windows independently. 
Within each window we compute its maximum by scanning the window through all its elements, which takes $O(k)$ time. 
We then have that this brute force approach takes $O(n\cdot k)$ time. 

**The problem in this approach is that when we compute the maximum of the current window we disregard all the previous computations, done on windows that are partially overlapping to the current one.**
## 8.2 - BST-Based Solution
To solve more efficiently the problem we need a data structure that handle the next window while capitalizing on the progress made in processing the preceding window. 

We consider the two following **observations:**
1) We can represent the elements within a window with a multiset $\mathcal{M}$ of size $k$. In this representation the result for the window is essentially the largest element contained within this multiset. 
2) When we transition from a window to the next, **only two elements change:** the first element of the first window exits from the scene and the last one enters (the window **slides** of one position to the right). Hence, we can derive the multiset of the new window from the multiset of the previous window by adding one element and removing another one. 

We then require a data structure capable of performing three crucial operations on a (multi)set: 
- inserting a new element
- deleting an arbitrary element
- retrieving the maximum element within the multiset

A Balanced Binary Search Tree (BBST) supports any of these operations in $\Theta(\log|\mathcal{M}|)$ where $|\mathcal{M}|$ is the number of elements in the multiset (and it is **optimal in the comparison model**)
This way we can solve the problem in $O(n\log(k))$ time. 

**The algorithm at this point is easy:**
- insert the first $k$ elements of the array (the first window) in the tree and store the maximum
- iterate until we have covered all the windows
	- shift the window
		- remove the element that went outside of the window
		- insert the element that entered the window
	- store the maximum 
- return the maximums

In rust we use `BTreeSet` that offers efficiently the operations `insert`, `remove`, and `last`(find max).
The only hiccup is that it do not store repeated elements, hence we store elements as pair with their index in the array. 
## 8.3 - Heap
A heap is a tree-based data structure that satisfies **the heap property:** 
- **max-heap:** for any given node `C`, if `P` is a parent node of `C`, then the key of `P` is greater than or equal to the key of `C`. 
- **min-heap:** the key of `P` is less than or equal to the key of `C`

**The main operations are:**
- **insertion:** $O(\log(n))$
	- insert the new element at the bottom of the heap (next available position)
	- perform "heapify up", an operation to restore the heap property by swapping the element with its parent until the heap property is satisfied again
- **remove max/min:** $O(\log(n))$
	- remove the root element (which is the min or the max of the heap)
	- replace the root with the last element in the heap
	- perform "heapify down", an operation to restore the heap property by swapping the element with one of its children until the heap property is satisfied again
- **peek:** $O(1)$
	- returning the maximum (or the minimum) without removing takes constant time since it is stored in the root node. 
### 8.3.1 - Heap Solution
The heap-based solution is theoretically less efficient than the BST one ($O(n\log(n))$ vs $O(n\log(k))$) but in practice it often yields better results. 

Since we are talking about the maximum the immediate choice is a priority queue with its most famous manifestation being the (max-)heap. 
A max-heap stores a set of $n$ keys and supports three operations: 
- insert an element in the max-heap, $O(\log(n))$, in rust called `push`
- report the max element, $O(1)$, in rust called `peek`
- extract (remove) the max element, $O(\log(n))$, `pop`

We can solve the sliding window maximum property by employing a `max-heap` and scanning the array from left to right: 
- populate the max-heap with the first $k$ elements, along with their respective positions (indices)
	- this gives us the maximum within the initial window 
- as we move on to process the remaining elements one-by-one, we insert each current element with its position. We request the heap to provide us the current maximum. The current maximum could be outside of the current window: to address this case we continuously extract elements from the heap until the reported maximum is within the constraints of the current window

With this approach there are a total of $n$ insertions and at most $n$ extractions of the maximum in the heap.  
Since the maximum number of elements present in the heap at any given time is up to $n$ each of these operations takes $\Theta(\log(n))$ time. 
Consequently the overall time complexity is $\Theta(n\log(n)).$
```java 
heamSWM(a[])
	n = a.length()
	MaxHeap h
	maxs[n-k+1]

	for i = 0 to k
		h.push((a[i], i))

	for i = k to n
		h.push((a[i], i))

		while !h.isEmpty()
			if h.peek().i < i-(k-1) // max ouside of the window
				h.pop()
			else 
				break
		maxs.push(h.peek())

	return maxs
```
## 8.4 Sliding Window Solution, O(n)
The BST solution can do more than what is strictly necessary, e.g. what is the second largest element in the window?
**The fact that we can do much more than what is requested, it’s an important signal to think that a faster solution could exist.** 

The better solution uses a **deque** (a double-ended queue) which supports constant time insertion, removal and access at the frond and back of the queue. 
There are many ways to implement a deque, the simplest one (but not the fastest) is a bidirectional list. 

The algorithm starts with an empty deque $Q$ and with the window $W$ that covers the positions in the range $\langle -k, -1\rangle$. 
That is, the window starts before the beginning of `a`. 
Then we start sliding the window one position at a time and remove/insert elements from $Q$.
**The element in front of the deque will be the element to report.**

**The algorithm behaves as follows:**
1. We iterate through the array elements one by one.
2. While processing each element, we maintain **the deque stores only elements relevant to the current sliding window**. Specifically, we ensure that **the deque stores only elements that are potentially maximum candidates for the current and future windows.**
3. At each step, we perform the following operations:
    - We remove elements from the front of the deque that are outside the current window. We do this by checking if the indices of those elements are beyond the current window boundary.
    - We remove elements from the back of the deque that are smaller than or equal to the current element. **This is because those elements cannot be the maximum within the current or any future window. We only want to keep elements that might potentially become the maximum.**
    - We append the index of the current element to the back of the deque, as it could potentially be the maximum in the current or future windows.
4. Once we have processed all elements and maintained the deque accordingly, the maximum elements for each sliding window are stored at the front of the deque.
5. We collect these maximum values and return them as the result.

**Note:** the deque stores indices, not actual elements.
```java 
// q = [a1, ..., ak]
//      back     front
//      tail     head
slidingWindowMaximum(a[]) 
	n = a.length()
	Deque q
	maxs[n-k+1]

	// first window
	for i = 0 to k 
		while !q.isEmpty() && a[i] > a[q.back()]
			q.popBack()
			
		q.pushBack(i)

	maxs.push(a[q.front()]);

	for i = k to n 
		// remove elements that dont belongs in the window
		while !q.isEmpty() && q.front() + k <= i
			q.popFront()

		// remove elements that are smaller that the current element
		// as this elements cannot be the maximum for this window nor 
		// for future windows
		while !q.isEmpty() && a[i] > a[q.back()]
			q.popBack()

		q.pushBack(i)
		maxs.push(a[q.front()])

	return maxs;
```
![[Pasted image 20240226104956.png | center | 500]]
### 8.4.1 - Correctness
**Theorem:** The elements in $Q$ are sorted in decreasing order.
**Proof:** we prove the theorem by induction on the number of iterations
- **base case:** $Q$ is empty, true
- **inductive case:** given the queue after $i$ iterations by inductive hypothesis it is sorted. the current iteration will only remove elements (no change in the ordering of the remaining elements) or insert the current element `a[i+1]` as the tail of the queue just below the first element which is larger than it (if any). Thus the queue remains sorted

Now we introduce the definition of **right leaders of the window** to show that the largest element within the current window is at the top of the queue. 
Given a window, an element is called a right leader if and only if the element is larger that any other element of the window at its right.
As an example consider the following image, where the red elements are right leaders: 
![[Pasted image 20240219095000.png | center | 700]]

**Theorem:** At every iteration $Q$ contains all and only the right leaders of the current window
**Proof:**
1. **Removing elements from the front of the deque**: At each step, the algorithm removes elements from the front of the deque that are outside the current window. Since these elements are no longer part of the current window, they cannot be the maximum within the current or any future window. Thus, they are not right leaders and are removed from the deque.
2. **Removing elements from the back of the deque and then Append the current element to the back:** The algorithm removes elements from the back of the deque as long they are $\le$ than the current element. This ensures that the appended element will be a right leader, as by construction is greater than all the element to its right

**Proof, said easy:** 
- removing from the right of the deque do not alter the structure of right leaders
- before appending to the left we remove all the elements that are $\le$ than the element we are going to append, hence we append only right leaders

**We derive the correctness of the algorithm by combining the sortedness of Q with the fact that the largest right leader is the element to report.**
### 8.4.2 Time Complexity
We have a loop that is repeated `n` times. 
The cost of an iteration is dominated by the cost (and thus the number) of pop operations. 
However in a certain iteration we may pop out all the elements in the deque. 
As far as we know there may be up to `n` elements in the deque and thus an iteration costs $O(n)$ time. 
Thus we get that the cost of the algorithm is $O(n^2)$. 
**The problem here list in this kind of complexity analysis.**

There may indeed exists very costly iterations but they are greatly amortized by many very cheap ones. 
Overall, the number of pop operations cannot be larger than $n$ as any element is not considered anymore by the algorithm as soon it is removed from $Q$.
Each of them cost constant time and thus the algorithm runs in linear time. 
# 9 - Next Larger Element
Given an array `A[0..n-1]` having distinct elements, the goal is to find the next greater element for each element of the array in order of their appearance in the array. 
**Adapt the Sliding Window Maximum Solution.**

**TODO**
# 10 - Maximum Number of Overlapping Intervals
Consider a set of $n$ intervals $[s_i, e_i]$ on a line. 
We say that two intervals $[s_i, e_i]$ and $[s_j, e_j]$ overlaps if and only if their intersection is not empty, i.e., if there exist at least a point $x$ belonging to both intervals. 
Compute the maximum number of overlapping intervals. 

**Example:** 
![[Pasted image 20240109094209.png | center | 550]]
We have a set of 10 intervals, the maximum number of overlapping intervals is 5 (at positions 3 and 4)
## 10.1 - Sweep Line Algorithm
The Sweep Line Algorithm is an algorithmic paradigm used to solve a lot of problems in computational geometry efficiently. 
**The sweep line algorithm can be used to solve problems on a line or on a plane.**

The sweep and line algorithm use an imaginary vertical line **sweeping** over the x-axis. 
As it progresses, we maintain a running solution to the problem at hand. 
The solution is updated when the vertical line reaches a certain key points where some event happen. The type of the event tells us how to update the solution. 
## 10.2 - Sweep Line Solution, O(n)
Let's apply the sweep and line algorithm to the problem above. 
We let the sweep line from left to right and stop at the beginning or at the end of every interval. 
These are the important points at which an event occurs: interval start or end. 
We also maintain a counter which keeps track of the number of intervals that are currently intersecting the sweep line, along with the maximum value reached by the counter so far. 

For each point: we first add to the counter the number of intervals that begin at that point, and then we subtract the number of intervals that end at that point. 
The figure below shows the points touched by the sweep line and the values of the counter:
![[Pasted image 20240109095128.png| center | 550]]
**Observation:** The sweep line only touches points on the x-axis where an event occurs. 
This is important because the number of considered points, and thus the time complexity, is proportional to the number of intervals, and not to the size of the x-axis.

**In Practice:**
We create an array `axis` of pairs `(p, isStart)`, where
- `p` is either a $s_i$ or $e_i$ of an interval
- `isStart` is true if `p` is a $s_i$, false otherwise

Then we sort `axis` by the first element of each pair, and we set `counter = 0` and `max = 0`. 
Now we iterate over `axis`: 
- if `axis[i]` is the start of an interval (i.e., `axis[i].isStart == true) 
	- `counter++`
	- `max = Math.max(counter, max)`
- otherwise 
	- `counter--`
# 11 - Closest Pair of Points
Let’s tackle a second problem to apply the **sweep line paradigm** to a **two-dimensional problem.**

We are given a set of $n$ points in the plane. 
**The goal is to find the closest pair of points in the set.** 
The **distance** between two points $(x_1,y_1)$ and $(x_2, y_2)$ is the Euclidian distance $$d((x_1,y_1), (x_2, y_2)) = \sqrt{(x_1-x_2)^2 + (y_1 - y_2)^2}$$A **brute force** algorithm computes the distance between al possible points, resulting in a time of $O(n^2)$. 

**Let's now use the sweep-line technique.** 
We start by sorting the points in increasing order of their $x-$coordinates. 

We keep track of the shortest distance, denoted with $\delta$, seen so far. 
Initially we set $\delta$ to the distance of an arbitrary pair of points. 
Then we use a vertical sweep line to iterate through the points, attempting to improve the current shortest distance $\delta$.
Consider the point $p = (x,y)$ just reached by the vertical sweep line. 
We can improve $\delta$ if the closest point *to the left* of $p$ has a smaller distance than $\delta$. 
If such point exists it must have a $x-$coordinate in the interval $[x-\delta, x]$ as it is to the left of $p$, and a $y-$coordinate in the interval $[y - \delta, y+\delta]$. 

The following figure shows the rectangle within this point must lie.
![[Pasted image 20240219103626.png | center | 250]]
There can be **at most 6 points** within this rectangle. 
The 6 circles within the perimeter of the rectangle represent points that are at distance exactly $\delta$ apart from each other. 

**Four our purpose a slightly weaker result is enough, which states that the rectangle contains at most 8 points.** 
**To understand why consider the 8 squares in the figure above.** 
Each of these squares, including its perimeter can contain **at most one point.** 
Assume, for sake of contradiction, that a square contains two points, denoted as $q$ and $q'$. 
The distance between $q$ and $q'$ is smaller than $\delta$. If point $q'$ exists it would already been processed by the sweep line because it has an $x-$coordinate smaller than that of $p$. 
**However this is not possible, because otherwise the value of $\delta$ would be smaller than its current value.**

**Let's use the above intuition to solve the problem.** 
We maintain a BST with points sorted by their $y-$coordinates. 
When processing a point $p = (x,y)$ we iterate over the points with $y-$coordinates in the interval $[y - \delta, y + \delta]$. 
If the current point has an $x-$coordinate smaller (which means farthest) than $x-\delta$ we remove this point from the set. **It will never be useful anymore.** 
Otherwise we compute its distance with $p$ and $\delta$ if needed. 
Before moving the sweep line to the next point, we insert $p$ in the set. 

**What is the complexity of the algorithm?**
- Identifying the range of points with the required $y-$coordinates takes $\Theta(\log(n))$ time, this is because the points are a $BST$
- Iterating over the points in this range take constant time per point
- Removing a point takes $\Theta(\log(n))$ (we use a balanced BST)
- How many points we do iterate over? There can be at most 6 points that have an $x-$coordinate greater or equal to $x-\delta$ and therefore survive. 
- There can be many points with smaller $x-$coordinates. However, since each point is inserted and subsequently removed from the set at most once during the execution of the algorithm, the cost of dealing with all these points is at most $O(n\log(n))$. 

**Clarification:** To store the solution we use a set. This set stores all the points at the (current) minimum distance $\delta$. 
When the program terminates we extract two random points from the set. 

**In rust:**
- compute the distance squared to avoid dealing with float
- swap the roles of $x$ and $y$ since we cannot insert in `BTreeSet` ordering by the second component
# 12 - Check if all Integers in a Range are Covered
![[Screenshot from 2024-01-09 10-01-57.png | center | 700]]
## 12.1 - Intuitive Solution 
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
## 12.2 - Sweep Line Solution
To solve the problem more efficiently we use sweep line and a map. 
The complexity is linear in the maximum between the number of `ranges` and `right`
```java
isCovered(ranges, left, right)
	// map: point i -> number of open ranges in i
	HashMap openRangesAt
	for range in ranges 
		openRangesAt.insert(openRangesAt.getOrDefault(range.start, 0) + 1)
		// range.end+1 as the ranges are inclusive!
		openRangesAt.insert(openRangesAt.getOrDefault(range.end+1, 0) - 1)

	openRangesNow = 0
	for i = 0 to left 
		openRangesNow += openRangesNow.getOrDefault(i, 0)

	for point=left to right 
		openRangesNow += openRangesNow.getOrDefault(point, 0)
		if openRangesNow == 0
			return false

	return true
```
# 13 - Longest K-Good Segment
The array $a$ with $n$ integers is given. 
Let's call the sequence of one or more **consecutive** elements in $a$ a segment. 
Also let's call the segment $k$-good if it contains **no more** than _k_ different values.

**Note:** if the distance between two numbers is `abs(1)` then the two numbers are **consecutive**.

Find **any** longest k-good segment.
**Note:** we return the indexes of the longest k-good segment
## 13.1 - Solution, O(n)
This problem is somewhat similar to Sliding Window Maximum. 

There you wanted to know the maximum element for every window of size $k$ in the array.
**Here we want to have the longest window with exactly $k$ distinct elements.** 

**We then use the two-pointers trick to basically simulate a dynamic sliding window.** 
**The algorithm behaves as follows:** 
- we start with an empty window and we use an hash set `unique` to store the distinct elements
- the starting window is of size $0$, the two pointers (aka left and right delimiter of the window) `left` and `right` both starts at $0$
	- right basically points the element just outside the right end of the window 
- we iterate until `right` becomes $n$, that is we iterate the whole array using `right`
	- is `unique` of size at most $k$? then we can insert in the window?
		- insert the element pointed by right `a[right]` in `unique`
		- `unique` is still of size at most $k$ (`unique.size() <= k`) (the inserted element either was new but there were less than $k$ distinct elements or it was an element already in `unique`) **AND** the current size of the window is the biggest so far?
			- save `left` and `right` as they might be the delimiters of the result
		- move right by one on the right
	- otherwise we shrink the window from the left
		- `unique.remove(a[left])`
		- `left++`
```java
findMaxSegment(a[], k) 
	n = a.length
	left = 0, right = 0 // delimiter of the current window
	maxLeft = 0, maxRight = 0 // delimiter of the current result
	currentSize = 0
	maxSize = -1
	Set unique

	while right < n 
		if uniqueSet.size() <= k
			unique.add(a[right])
			currentSize++

			if uniqueSet() <= k && currentSize > maxSize
				maxLeft = left
				maxRight = right
				maxSize = currentSize
			
			right++
		else 
			unique.remove(a[left])
			left++
		
	return (maxLeft, maxRight)
```
# 14 - Contiguous Subarray Sum
Given an integer array `nums` and an integer `k`, return `true` if `nums` has a **good subarray** or false otherwise. 
A **good subarray** is a subarray where: 
- its length is **at least** 2 
- the sum of the elements of the subarray is a multiple of `k`

**Note** that:
- A **subarray** is a contiguous part of the array.
- An integer `x` is a multiple of `k` if there exists an integer `n` such that `x = n * k`. `0` is **always** a multiple of `k`.
## 14.1 - Naive Solution, O(n^2)
The obvious brute force approach: from each element we compute every possible subarray starting in that element, check if sum is a multiple of `k` and store the maximum length so far.
## 14.2 - Prefix Sums 
**Prefix sums**, also known as cumulative sums or cumulative frequencies, **offer an elegant and efficient way to solve a wide range of problems that involve querying cumulative information about a sequence of values or elements.**

The essence of prefix sums lies in **transforming a given array of values into another array, where each element at a given index represents the cumulative sum of all preceding elements in the original array.**

An example of prefix sum array is shown in the picture: 
![[Pasted image 20240110100959.png | center | 350]]
Where it is clear that $$P[i] = \Sigma_{j=1}^i\ A[k]$$
### 14.2.1 - Prefix Sum using Rust
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
### 14.2.2 - Applications of Prefix Sum
#### 14.2.2.1 - Ilya and Queries
We have a string $s = s_1s_2\dots s_n$ consisting only of characters $a$ and $b$ and we need to answer $m$ queries.
Each query $q(l,r)$ where $1 \le l \le r \le n$ asks for the number of positions $i \in [l,r]$ such that $s_i = s_{i+1}$ 

Let's consider an example (remember that we use 1-indexing)
- $s = a\ a\ b\ b\ b\ a\ a\ b\ a$ 
- $q(3, 6) = 2$
	- the substring is $b\ b\ b\ a$
	- the position $3$, $4$, $6$ make true $s_i = s_{i+1}$ but $s_6 = s_{6+1}$ do not count as $6+1 = 7 \notin [3,6]$   

The idea is that of computing the binary vector $B[1,n]$ such that $B[i] = 1$ if $s_i == s_{i+1}$, $0$ otherwise. 
This way the answer to the query $q(l,r)$ is $$\Sigma_{i=l}^{r-1}\ B[i]$$Thus each query can be solved in constant time computing the prefix sum of the vector $B$.
#### 14.2.2.2 - Little Girl and Maximum
We are given an array $A[1,n]$ and a set $Q$ of queries. Each query is a range sum query $[i,j]$ which returns the sum of the elements in $A[i..j]$. 
**The goal is to permute the elements** in $A$ in order to maximize the sum of the results of the queries in $Q$.

The key is to observe that if we want to maximize the sum **we have to assign the largest value to the most frequently accessed entries.**
Thus the solution consists of sorting both $A$ by descending values and the indexes of $A$ by descending frequency of access and **pairing them in this order.** 
Therefore, once we have computed the frequencies, the solution takes $\Theta(n\log(n))$ time. 

**We are left with the problem of computing access frequencies.** 
We want to compute an array $F[1,n]$ where $F[i]$ is the number of times the index $i$ belongs to the query $Q$. 
Computing this array by updating every single entry $F$ for each query takes $O(n\cdot q)$ and thus is unfeasible. 

**We use the prefix sum to solve the problem.**
We construct an array $U[1,n]$ such that its prefix sums are equal to our target array $F$. 
Interestingly we need to modify just two entries of $U$ to account for a query in $Q$.

Initially we set all entries of $U$ to $0$. 
Then for a query $\langle l, r \rangle$ we add $1$ to $U[l]$ and subtract $1$ from $U[r+1]$, this way the prefix sums are as follows: 
- unchanged for indexes less than $l$
- increased by one for the indexes in $[l,r]$
- unchanged for indexes greater than $r$

Therefore, **the prefix sum** of $U$ up to $i$ equals to $F[i]$. 
This algorithm takes $O(q+n)$ time ($q$ since we have $q$ queries to go through plus $n$, which is the time that takes to build $U$ and its prefix sum array).

**Said easy:** we build $F$ through the prefix sum of $U$. 

**The following pseudo-implementation clarifies the approach:** 
```java
littleGirlMax(a[], q[]) 
	n = a.length()
	
	u[n]
	for (l, r) in q
		u[l]++
		if r + 1 < n 
			u[r+1]--

	f[n]
	pS = 0
	for i = 0 to n
		pS += u[i]
		f[i] = pS

	a.sortDecreasing()
	f.sortDecreasing()

	res = 0
	for i = 0 to n
		res += a[i] * f[i]

	return res
```
**Basically** **we permute the elements of $A$ so that bigger is the element the more times it is used in the queries.**
#### 14.2.2.3 - Number of Ways
Given an array $A[1,n]$ count the number of ways to split the array into three contiguous parts to that they have the same sum. 
Formally, you need to find the number of such pairs of indices $i$ and $j$ (with $2 \le i \le j \le n-1$) such that: $$\Sigma_{k=1}^{i-1}\ A[k]= \Sigma_{k=i}^j\ A[k] = \Sigma_{k = j+1}^n\ A[k]$$Let be $S$ the sum of all the elements in the array. 
If $S$ cannot be divided by $3$ we can immediately return $0$. 

We compute an array $C$ that stores, at position $i$, **the number of suffixes of the suffix** $A[i..n]$ that sums to $\frac{S}{3}$. 
![[Pasted image 20240226134323.png | center | 600]]

Then we scan $A$ from left to right to compute the prefix sum of $A$. 
Every time the prefix sum at position $i$ is $\frac{S}{3}$ we add $C[i+2]$ to the result.  
This is because the part $A[1..i]$ sums to $S/3$ and can be combined with any **pair** of parts of $A[i+1..n]$ where both parts sums to $S/3$. 
Since the values in $A[i+1..n]$ sums to $\frac{2}{3}S$, the number of such pairs is the number of suffixes that sum to $S/3$ in $A[i+2..n]$.

**Clarification:** $C[i+2]$ is because we want **three partitions**, if you were to use $C[i+1]$ you could use the whole remaining $A[i+1..n]$ whose elements sums to $\frac{2}{3}S$, and you would get only two partitions.

Indeed if one of this suffix sums to $S/3$, say $A[j..n]$, then we are sure that $A[i+1, j-1]$ sums to $S/3$

**Said Easy:**
- say that you have $C$ and $P$, the prefix sum array of $A$
- iterate $P$ using the index $i$
	- is $P[i] == \frac{S}{3}$?
		- then we have found one of the three parts that summed give $S$
		- **the number of suffixes of the suffix** $A[i+2,n]$ are the third part
			- if the whole $A[i+2,n]$ sum up to $S/3$ than we have found the three parts: $A[1,i], A[i+1,i+1], A[i+1, n]$ 
			- counting the other suffixes makes it work: we are counting the indexes $(i,j)$ such that we can decompose $A$ in three parts, counting the $j$s is as counting the $i$s
## 14.3 - Prefix Sum Solution, O(n)
The solution is based on the following **mathematical property:**
**Observation:** any two prefix sums that are not next to each other with the same mod $k$, or a prefix sum with mod $k = 0$ that is not the first number will yield a valid subarray.

**The property said better:**
![[Pasted image 20240229124500.png | center ]]
The algorithm behaves as follows: 
- build `prefixSum`
- create a map *modulo -> index of prefixSum that gives that modulo*
- for every element `(i, pS)` in `prefixSum`
	- compute `modulo = pS % k`
	- if `modulo == 0 && i != 0` we return true
	- if `modulo` is not contained in the map we insert it `modulo -> i`
	- if `modulo` is present and the value associated with it is less than `i-1` we return true

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
	for (i, prefixSum) in prefixSumArray.enumerate() 
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
# 15 - Update the Array
You have an array $A$ containing $n$ elements initially all $0$. 
You need to do a number of update operations on it. 
In each update you specify `l`, `r` and `val` which are the starting index, ending index and value to be added. 
After each update, you add the `val` to all elements from index `l` to `r`. 
After `u` updates are over, there will be `q` queries each containing an index for which you have to print the element at that index.

**Basically we have to support two operations:**
1) `access(i)`, which returns $A[i]$
2) `range_update(l,r,v)`, which update the entries in $A[l..r]$ by adding $v$
 
To efficiently solve this problem we introduce a new data structure, the **Fenwick Tree**
## 15.1 - Fenwick Tree
**The Fenwick Tree, also known as the Binary Indexed Tree (BIT), is a data structure that maintains the prefix sums of a dynamic array.** 
**With this data structure we can update values in the original array and still answer prefix sum queries, both in logarithmic time**.

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
currently we are not supporting queries for positions within the ranges between consecutive powers of 2. 
Look at the image above: positions that falls in the range (subarray) `[5, 7]`, which falls between the indices $4$ ($2^2$) and $8$ ($2^3$), are not supported. 
In fact we can't make the query `sum(5)`.
**Enabling queries for this subarray is a smaller instance of our original problem.**

We apply the **same strategy by adding a new level** to our tree. 
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

**We can make some observations:** 
1) While we have represented our solution as a tree, it can also be represented as an array of size n+1, as shown in the figure above
2) We no longer require the original array `A` because any of its entries `A[i]` can be simply obtained by doing `sum(i) - sum(i-1)`. This is why the Fenwick tree is an **implicit data structure**
3) Let be $h = \lfloor\log(n)+1\rfloor$, which is the length of the binary representation of any position in the range $[1,n]$. Since any position can be expressed as the sum of at most $h$ powers of $2$, the tree has no more than $h$ levels. In fact, the number of levels is either $h$ or $h-1$, depending on the value of $n$ 

Now, let’s delve into the details of how to solve our `sum` and `add` queries on a Fenwick tree.
### 15.1.1 - Answering a `sum` query
This query involves beginning at a node `i` and traversing up the tree to reach the node `0`. Thus `sum(i)` takes time proportional to the height of the tree, resulting in a time complexity of $\Theta(\log n)$. 

Let's consider the case `sum(7)` more carefully. 
We start at node with index 7 and move to its parent (node with index 6), its grandparent (node with index 4), and stop at its great-grandparent (the dummy root 0), summing their values along the way. 
This works because the ranges of these nodes ($[1,4], [5,6], [7,7]$) collectively cover the queried range $[1,7]$. 

Answering a `sum` query is straightforward **if we are allowed to store the tree's structure.**
However a significant part of the Fenwick tree's elegance lies in the fact that storing the tree is not actually necessary. 
This is because **we can efficiently navigate from a node to its parent using a few bit-tricks, which is the reason why the Fenwick trees are also called Binary Indexed Trees.**
#### 15.1.1.1 - Compute the Parent of a Node
We want to compute the parent of a node, and we want to do it quickly and without representing the structure of the tree.

Let's consider the binary representation of the indexes involved in the query `sum(7)`

![[Pasted image 20240110150658.png | center ]]

**Theorem:** the binary representation of a node's parent can be obtained by removing the trailing one (i.e., the rightmost bit set to 1) from the binary representation of the node itself.
### 15.1.2 - Performing an `add`
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
### 15.1.2.1 - Computing the Siblings
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
## 15.1.2 - Fenwick Tree in Rust
The following is a minimal implementation
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
## 15.1.3 - Applications of Fenwick Tree
### 15.1.3.1 - Counting Inversions in an Array
We are given an array $A[1..n]$ of $n$ positive integers. If $1 \le i < j \le n$ and $A[i]>A[j]$, then the pair $(i,j)$ is called an **inversion** of $A$. 
**The goal is to count the number of inversions** of $A$.

We assume that the largest integer $M$ in the array is in $O(n)$. 
This assumption is important because we're using a Fenwick Tree of size $M$ and building such a tree takes $\Theta(M)$ time and space. 
If, on the other hand, $M$ is too large, we need to sort $A$ and replace each element with its **rank in the sorted array** $\clubsuit$. 

---
$\clubsuit$ $$\text{rank($x$)} = |\text{elements less than $x$ in the array}|$$
**Said easy:** the rank of an element is its position in the sorted array. 
If elements are equal than they have the same rank. 
rank goes from $1$ to $n$

---

**The algorithm behaves as follow:****
1. **Base Case Handling**: If the input array `a` is empty, there are no inversions, so the function returns `0`.
2. **Initialization**: The function initialize (all zeroes) s a Fenwick Tree `ft` with length of `max + 1`, where `max` is the maximum value in the array `a`. 
3. **Counting Inversions using Fenwick Tree**:
    - For each element `e` in the input array `a`
        - compute the count of elements greater than `e` **that appear later in the array** $\diamondsuit$. This is done by querying the Fenwick Tree for the range sum between `(e + 1)` and `max`. Since the Fenwick Tree stores cumulative sums, this effectively gives the count of elements greater than `e` encountered so far.
	        - $\diamondsuit$ say that `e` appears at position `i`: the elements that appears at position `j` with `j > i` and such that `a[j] > e` are inversions
		        - `i < j` and `a[j] > a[i]`
        - Increment by one the count of occurrences of the current element `e` in the Fenwick Tree using the `add` method. This step is crucial for accurately counting inversions since we need to keep track of how many times each element appears in the array.
	        - this is because if an element appears two times in the array the inversions related to those "different" elements have an impact
    - The total count of inversions is accumulated in the variable `count`.
4. **Returning the Result**: Finally, the function returns the total count of inversions.

In this particular use case, the Fenwick Tree is used to efficiently calculate the cumulative sum of elements greater than the current element encountered so far. 
This is crucial for determining the count of inversions. 
The `range_sum` method computes the cumulative sum efficiently by exploiting the tree structure, resulting in an overall time complexity of $O(n \log n)$ for the function, where `n` is the size of the input array `a`.

**Consider the following pseudo-implementation:**
```java
countInversions(a[])
	n = a.length()

	max = a.max()
	// use the implementation presented above!
	ft = FenwickTree.withLen(max + 1)

	counter = 0 
	for i = 0 to n
		counter += ft.rangeSum((a[i]+1), max)
		ft.add(a[i], 1)

	return counter
```
## 15.2 - Fenwick Tree Solution
We are given an array `A[1,n]` initially set to 0. 
We want to support two operations: 
- `access(i)` returns the element $A[i]$
- `range_update(l, r, v)`, updates the entries in `A[l,r]` adding to them `v`

The following Fenwick solution solve the problem
1) from `A` we build the Fenwick Tree of length `n` (mind that `A` is initialized with all zeros)
2) the operation `access(i)` is a wrapper of the operation `sum(i)` we have seen before $\ \clubsuit$ 
3) the operation `range_update(l,r,v)` exploit the operation `add(i, v)` of the implementation of the Fenwick Tree: 
	1) first we check that `l` is `<=` than `r`, aka that the interval of entries to update is well formed
	2) then we check that `r <= n`, aka that the interval of entries to update is actually in the array
	3) we perform `add(l,v)`: this trigger the addition of the value `v` to each node whose range include the position `l` in the Fenwick Tree
	4) we perform `add(r, -v)`: this trigger the subtraction of the value `v` to each node whose range include the position `r` in the Fenwick Tree
	5) we have added and subtracted the same same quantity `v` in the Fenwick tree, this means that prefix sum are coherent and the elements in `[l,r]` are increased by `v` 

$\clubsuit$ The operation `sum(i)` we have written in the implementation of the FT results in the access operation in this problem  (`sum(i) == access(i)`) by the construction of range update. 
**Example:** consider the empty FT (as an array): $[0,0,0,0]$
- perform `range_update(1,2,2)`, we obtain:
	- the fenwick tree representation:  `[0,2,0,-2]`
	- the array it represents: `[0,2,2,0]`
- perform `access(2)`, we want the third element of the array
	- `sum(2)` is the prefix sum of the elements $[0,1,2]$ of the FT
		- it gives $2$, as $0 + 2 + 0 = 2$ 
		- the element at position $2$ of the array that the FT represents is $2$

**The solution of the problem is given by the following data structure:**
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
### 15.2.1 - Dynamic Prefix-Sums with Range Update
The range update of the previous problem is paired with the `access(i)` operation.
Here we want to support:
- `range_update(l, r, v)`, same as before
- `sum(i)`, returns $\Sigma_{k=1}^i\ A[k]$, we want to support **the real prefix operation *on* the fenwick tree with range updates**

As before, we notice that 
1) the operation `add(i,v)` provided by the basic implementation of our FT is a special case of `range_update`, specifically `add(i,v) == range_update(i, n, v)` where `n` is the size of the tree
2) `access(i)` is still being supported using `sum(i) - sum(i-1)`

**The difference between this and Update the Array** is that here we have to support `sum`, while we where only supporting `access(i)` in Update the Array. 
Let's say that we use here the same approach we used in Update the Array.
For `range_update(l,r,v)` we modify the Fenwick Tree by adding $v$ ad position $l$ and by adding $-v$ at position $r+1$.
- consider the starting array $[0,0,0,0,0,0]$ on which we build the FT, we use $1$-indexing
- perform `range_update(2,4,2)`
	- the "ideal" array would be `[0,2,2,2,0,0]`
	- in the FT array, using our `range_update`, we have $[0,2,0,0,-2,0]$
- perform `sum(3)`, e.g., $i = 3$
	- the prefix sum result should be $4$
	- in our implementation `ft.sum(3)` is equal to the `access(3)` on the array, and in fact `ft.sum(3)` gives $2$ (the ideal array in position $3$ is $2$)
	- to give the actual prefix sum we try to respond $i\cdot v$, but this gives $6$, which is wrong

**More formally:** consider `range_update(l,r,v)` on a brand new FT (**all zeroes**). 
The **correct result** of a `sum(i)` after the `range_update` is: 
- if $1 \le i < l$, `sum(i)` is $0$
- if $l \le i < r$, `sum(i)` is $v\cdot (i-l + 1)$ 
- if $r \le i$, `sum(i)` is $v\cdot (r-l+1)$ 

Instead the results returned by our method are the following: 
- if $1 \le i < l$, `sum(i)` is $0$
- if $l \le i < r$, `sum(i)` is $v\cdot i = v(l -1) + v(i - l + 1)$ 
- if $r < i$, `sum(i)` is $(v-v)\cdot i = 0$ 

**To fix those problems** we employ another Fenwick Tree, FT2,  which will keep track of these discrepancies. 
When we perform a `range_update(l,r,v)` we:
1) add $v$ at position $l$ in the first fenwick tree: `ft1.add(l,v)`
2) add $-v$ at position $r+1$ in the first fenwick tree: `ft1.add(r+1,-v)`
	- 1) and 2) are as in "Update the Array"
1) add $-v(l-1)$ at position $l$ in the second fenwick tree: `ft2.add(l, -v*(l-1))`
2) and $v\cdot r$ to the position $r+1$ in the second fenwick tree: `ft2.add(r+1, v*r)` 

This revised approach ensures that the result of `sum(i)` can be expressed as $a\cdot i + b$, where:
- $a$ is the sum up to $i$, `sum(i)` in the first fenwick tree 
- $b$ is the sum up to $i$ in FT2

And we get 
```java
sum(i) 
	// a         i     b 
	ft1.sum(i) * i + ft2.sum(i)
```
The value of $b$ from FT2 corrects the errors of the flawed solution, specifically: 
- for $1 \le i < l$, $b = 0$
- for $l \le i \le r,\ b = -v(l-1)$ 
- for $r < i,\ b = v\cdot r - v(l-1) = v(r-l+1)$ 
# 16 - Nested Segments
We are given $n$ segments: $[l_1, r_1],\dots, [l_n, r_n]$ on a line. 
There are no coinciding endpoints among the segments. 
The task is to determine and report the number of other segments each segment contains.
**Alternatively said:** for the segment $i$ we want to count the number of segments $j$ such that the following condition hold: $l_i < l_j \land r_j < r_i$. 

We provide two solutions to this problem: 
- with Fenwick Tree
- with Segment Tree
## 16.1 - Fenwick Tree Solution
**We use a sweep line & Fenwick tree approach.** 

We build an array `events` where every entry is `[l_i, r_i, i]`, and then we sort `events` by start of the respective range, `l_i`.

Then we build the Fenwick tree with size $2n+1$, we scan each event $[l_i, r_i, i]$ and add $1$ in each position $r_i$ in the fenwick tree: `ft.add(r_i, 1)`

Now we scan the events again. 
When we process the event $[l_i, r_i, i]$ we observe that **the segments already processed are only the ones that starts before the current one, as they are sorted by their starting points.**

To find the solution of this problem for the current segment (aka the number of segments contained in the current one) **we need to know the number of the segments that starts after the current one that also end before the current one**, before $r_i$. 
This is computed with a query `sum(r_i - 1)` on the Fenwick Tree. 
Why `sum(r_i - 1)` is the number of segments contained in $[l_i, r_i]$?
Because all the segments that starts before $l_i$ have already been processed and their right endpoint have been removed from the Fenwick Tree $\spadesuit$ (aka we subtracted one to the position $r_i$)
Therefore `sum(r_i - 1)` is the number of segments that **starts after** $l_i$ and **end before** $r_i$

$\spadesuit$ After computing the solution for the current segment we subtract $1$ to position $r_i$, to **remove the contribution of the right endpoint of the current segment in the next queries. This is why the segments that starts before the current one but overlaps with it are not counted**.

The following snippet implement the solution above, using the Fenwick tree previously defined. 
```rust
nestedSegmentFT(segments[])
	n = segments.length()
	List result
	List events
	for (i, segment) in segments.enumerate()
		events.push((segment.start, segment.end, i))
	// sort by the first component of each element, s_i
	events.sort()

	tree = FenwickTree(2*n + 1)
	for event in events
		tree.add(event.end, 1)

	for event in events
		result.push((tree.sum(event.end - 1), event.index))
		tree.add(event.end, -1)

	// sort by the second element, the index, to  
	// restore the original ordering of segments
	result.sort()

	return result
```
## 16.2 - Segment Tree
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
### 16.2.1 - Structure of the Segment Tree
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
#### 16.2.1.1 - Construction
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

**Said easy:**
**The construction is recursive and is like a post-order visit, where the "visit" is the assignment of the value of the current node**
- compute the left child value by recurring on the left subtree
- compute the right child value by recurring on the right subtree
- merge the segments and result in the current node

The **time complexity of the construction** is $O(n)$, assuming that the merge operation is $O(1)$, as the merge operation gets called n times, which is equal to the number of internal nodes in the segment tree.
#### 16.2.1.2 - Sum Queries
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

**Let's now reason about the complexity of the the sum queries**
We have to show that we can compute the sum queries in $O(\log(n))$.

**Theorem:** For each level we only visit no more than four vertices.
And since the height of the tree is $O(\log(n))$, we receive the desired running time. 
**Proof:** we prove that at most four vertices each level by **induction.** 
- **base case:** at the first level we only visit one vertex, the root vertex, so here we visit less than four vertices. 
- **inductive case:** let's look at an arbitrary level. By induction hypothesis, we visit at most four vertices. **If** we only visit at most two vertices, the next level has at most four vertices. That is trivial, because each vertex can only cause at most two recursive calls. **So let's assume** that we visit three or four vertices in the current level. From those vertices, we will analyze the vertices in the middle more carefully. Since the sum query asks for the sum of a continuous subarray, we know that segments corresponding to the visited vertices in the middle will be completely covered by the segment of the sum query. Therefore these vertices will not make any recursive calls. So only the most left, and the most right vertex will have the potential to make recursive calls. And those will only create at most four recursive calls, so also the next level will satisfy the assertion. We can say that one branch approaches the left boundary of the query, and the second branch approaches the right one.

**Inductive case, said easy:**
![[Pasted image 20240229164807.png | center | 600]]

The query works by dividing the input segment into several sub-segments for which all the sums are already precomputed and stored in the tree. And if we stop partitioning whenever the query segment coincides with the vertex segment, then we only need   $O(\log n)$  such segments, which gives the effectiveness of the Segment Tree.
#### 16.2.1.3  - Update Queries
Now we want to modify a specific element in the array, let's say we want to do the assignment $a[i] = x$. And we have to rebuild the Segment Tree, such that it corresponds to the new, modified array. 

This query is easier than the sum query. Each level of a segment tree forms a partition of the array. Therefore an element $a[i]$ only contributes to one segment from each level. 
Thus only $O(\log(n))$ vertices need to be updated. 

It is easy to see that the update request can be implemented using a recursive function. The function gets passed the current tree vertex, and it recursively calls itself with one of the two child vertices (the one that contains $a[i]$) and after that recomputes its sum value, similar how it is done in the build method (that is as the sum of its two children). 

**Example:** given the same array as before, we want to perform the update $a[2] = 3$ 
![[Pasted image 20240112112424.png | center | 450]]
### 16.2.2 Range Update and Lazy Propagation
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
If our update query is for range $3$ to $5$, then we need to update this node and all descendants of this node. 
With Lazy propagation, we update only node with value $27$ and postpone updates to its children by storing this update information in separate nodes called lazy nodes or values. 
We create an array `lazy[]` which represents lazy node. 
The size of `lazy[]` is same as array that represents segment tree.
The idea is to initialize all elements of `lazy[]` as 0. 
A value 0 in `lazy[i]` indicates that there are no pending updates on node `i` in segment tree. 
A non-zero value of `lazy[i]` means that this amount needs to be added to node `i` in segment tree before making any query to the node.
## 16.2.3 Persistency
A **persistent data structure** **is a data structure that remembers its previous state for each modification.** 
**This allows to access any version of this data structure that interest us and execute a query on it.** 

Segment Tree is a data structure that can be turned into a persistent data structure efficiently
- avoid to store the complete ST for every modification
- avoid to loose the $O(\log n)$ time behavior for answering range queries. 

In fact any change request in the Segment Tree leads to a change in the data of only $O(\log n)$ vertices along the path starting from the root. 
So if we store the Segment Tree using pointers, then when performing the modification query we simply need to create new vertices instead of changing the available vertices. 
Vertices that are not affected by the modification query can still be used by pointing to the old vertices. 
Thus for a modification query $O(\log n)$ new vertices will be created, including a new root vertex, and the entire previous version of the tree rooted at the old vertex will remain unchanged. 

**For each modification of the Segment Tree we will receive a new root vertex.** 
To quickly jump between two different versions of the Segment Tree, we need to store this roots in an array. 
To use a specific version of the Segment Tree we simply call the query using the appropriate root vertex.

**The following image gives an idea of how we can make persistent ST:**
![[Pasted image 20240229171701.png | center | 500]]
With the approach described above almost any Segment Tree can be turned into a persistent data structure.
## 16.3 - Segment Trees Solution
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
		- the number of nested segments in the the current segment is the range sum on the range `[segments[axis[i].index, axis[i].endpoint]` 
			- we store this range sum on a variable `res`
		- we push the tuple `(axis[i].index, res)` in the array of results
		- we increase by one the start of the segment indexed with `axis[i].index` in `segments`
	- sort `results` by the indexes and return it

**Why it works?**
![[Pasted image 20240212162551.png | center | 600]]
![[IMG_0416.png | center | 600]]
**In words:**
Consider the segments $[(s_0, e_0), \dots, (s_{n-1}, e_{n-1})]$, sorted by $s_i$.
When we find the end of a segment $i$, namely $e_i$ we do the range sum of $(s_i, e_i)$ to get the number of segments contained in the segment $(s_i, e_i)$. 
Then we increase by $1$ the segment $(s_i, s_i)$ in the segment tree (aka we `add(s_i, 1))

This works because we increase by one the start $s_i$ when we find the end of its segment, the endpoint $e_i$. 
**The range sum** on $(s_i, e_i)$ **will count only segments that starts after the current segment** $s_i$ **and have already been closed** (otherwise they would be $0$ in the tree).
## 16.4 - Summary
![[Pasted image 20240301112709.png | center | 600]]
# 17 - Powerful Array
An array of positive integers $a_1,\dots,a_n$ is given. 
Let us consider its arbitrary subarray $a_l, a_{l+1},\dots, a_r$, where $1 \le l \le r \le n$.
For every positive integer $s$ we denote with $K_s$ the number of occurrences of $s$ into the subarray.
We call the **power** of the subarray the sum of products $K_s \cdot K_s \cdot s$ for every positive integer $s$
The sum contains only finite number of non-zero summands as the number of different values in the array is indeed finite. 

You should calculate the power of $t$ given subarrays.

**Besides the trivial solutions, we introduce a new algorithmic technique.**
## 17.1 - Mo's Algorithm 
The Mo’s Algorithm is a powerful and efficient technique for **solving a wide variety of range query problems.** 
It becomes particularly **useful for kind of queries where the use of a Segment Tree or similar data structures are not feasible.** 
**This typically occurs when the query is non-associative, meaning that the result of a query on a range cannot be derived by combining the answers of the subranges that cover the original range.**

Mo’s algorithm typically achieves a time complexity of $O((n+q)\sqrt n)$, where $n$ represents the size of the dataset, and $q$ is the number of queries.

**Let's first consider an easier problem than Powerful Array**
## 17.1.1 - Three or More
We are given an array $A[0,n-1]$ consisting of colors, where each color is represented by an integer within $[0,n-1]$. 
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
Specifically, for left endpoints, we must remove all the colors in $A[l',l-1]$ if $l' < l$, or we need to add all the colors in $A[l,l']$ if $l < l'$. 
The same applies to right endpoints $r$ and $r'$. 

The rust implementation below uses two closures, `add` and `remove` to keep `answer` and `counters` updated as we adjust the endpoints
```rust
pub fn three_or_more(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    let max = a.iter().max().unwrap()
    let mut counters: Vec<usize> = vec![0; max];
    let mut answers = Vec::with_capacity(queries.len());

    let mut old_l = 0;
    let mut old_r = 0; // here right endpoint is excluded
    let mut answer = 0;

    for &(l, r) in queries {
        let mut add = |i| {
            counters[a[i]] += 1;
            if counters[a[i]] == 3 {
                answer += 1
            }
        };
	    let mut remove = |i| {
            counters[a[i]] -= 1;
            if counters[a[i]] == 2 {
                answer -= 1
            }
        };

        while old_l > l {
            old_l -= 1;
            add(old_l);
        }
        
        while old_l < l {
            remove(old_l);
            cur_l += 1;
        }
        // same reasoning as before
        while old_r <= r {
            add(old_r);
            old_r += 1;
        }
        while old_r > r + 1 {
            old_r -= 1;
            remove(old_r);
        }
        answers.push(answer);
    }
    answers
}
```

The time complexity of the algorithm remains $\Theta(q\cdot n)$. 
**However we observe that a query now executes more quickly if its range significantly overlaps with the range of the previous query.** 

This implementation is **highly sensitive to the ordering of the queries.**
**The previous figure becomes now a best-case for the new implementation** as it takes $\Theta(n)$time. Indeed, after spending linear time on the first query, any subsequent query is answered in constant time.
**Mind that it is enough to modify the ordering of the above queries to revert to quadratic time (alternate between very short and very long queries).**

The above considerations lead to a question: **if we have a sufficient number of queries, can we rearrange them in a way that exploits the overlap between successive queries to gain an asymptotic advantage in the overall running time?**
Mo's Algorithm answers positively to this question by providing a reordering of the queries such that the time complexity is reduces to $\Theta((q+n)\sqrt n)$  

The idea is to conceptually partition the array $A$ into $\sqrt n$ buckets, each of size $\sqrt n$, named $B_1,B_2,\dots,B_{\sqrt n}$. 
A query **belongs** to a bucket $B_k$ if and only if its left endpoint $l$ falls into the $k-\text{th}$ bucket. 
In other terms: 
$$
	\text{the query }(l,r) \in B_k \iff \lfloor l/\sqrt{n}\rfloor = k
$$

Initially we group the queries based on their corresponding buckets, and within each bucket the queries are solved in ascending order of their right endpoints (aka the queries are sorted by their $r$)

The figure shows this bucketing approach and the queries of one bucket sorted by their right endpoint.
![[Pasted image 20240116094255.png | center | 600]]

**Let's analyze the complexity of the solution using this ordering**
It is sufficient to count the number of times we move the indexes `old_l` and `old_r`. This is because both `add` and `remove` take constant time, and thus **the time complexity is proportional to the overall number of moves of these two indexes.** 

**Let's concentrate on a specific bucket.** 
As we process the queries in ascending order of their right endpoints, the index `old_r` moves a total of at most $n$ times ($n$ times if all the queries are in this bucket)
On the other hand, the index `old_l` can both increase and decrease but it is limited within the bucket, and thus it cannot move more than $\sqrt n$ times per query. 
Thus, for a bucket with $b$ queries, the overall time to process its queries is $\Theta(b\sqrt n + n)$. 

Summing up over all buckets the time complexity is $\Theta(q\sqrt n + n\sqrt n)$, aka $\Theta((n+q)\sqrt n))$

**Here's a Rust implementation of the reordering process.** 
We sort the queries by buckets, using their left endpoints, and within the same bucket we sort them in ascending order by their right endpoints. 
We also have to compute a `permutation` to keep track of how the queries have been reordered. This permutation is essential for returning the answers to their original ordering. 
```rust
pub fn mos(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    // Sort the queries by bucket, get the permutation induced by sorting.
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
## 17.1.2 Final Considerations on Mo's Algorithm
**Mo’s algorithm is an offline approach, which means we cannot use it when we are constrained to a specific order of queries or when update operations are involved.**

When implementing Mo’s algorithm, the most challenging aspect is implementing the functions `add` and `remove`. 
There are query types for which these operations are not as straightforward as in previous problems and require the use of more advanced data structures than just an array of counters
## 17.1.3 - An application of Mo's Algorithm
### 17.1.3.1 - Mo's for Queries on a Tree
Consider a tree with $n$ vertices. 
Each vertex has some color. Assume that the vertices are numbered from $1$ to $n$. 
We represent the color of a vertex $v$ with $c_v$. 
The tree root is the vertex with number $1$.

We need to answer $m$ queries. 
Each query is described by two integers $v_j, k_j$. 
The answer to the query is the number of colors $c$ that occurs at least $k_j$ in the subtree of vertex $v_j$. 

This problem can be solved in $\Theta((m+n)\sqrt{n})$ time using the Mo's algorithm, how?

![[IMG_0424.png | center]]
## 17.2 - Solution
We can just use Mo's Algorithm and a little bit of attention in updating the answer after a `add` or a `remove`.

The solution is identical to the one seen in the previous problem, with one difference. 
We are not interested anymore in the number of occurrences of $i$, denoted $K_i$, in a given subarray, but we want to compute $$\Sigma_i\ K_i^2\cdot i,\ i\in [l,r]$$**When we increase the number of an occurrence we have to first remove the number obtained when we thought that there was one less occurrence.** 
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
# 18 - Longest Common Subsequence
Given two strings, `S1` and `S2`, the task is to find the length of the longest common subsequence, i.e. longest subsequence present in both strings. 
**Observation:** subsequence != substring. A subsequence do not have to be contiguous. 

**There are many ways to attack this problem, we use it to talk about Dynamic Programming.**
## 18.1 - Dynamic Programming
Dynamic Programming, like divide-and-conquer, solves problems by combining solutions of subproblems. 
Divide-and-Conquer algorithms partitions the problem into disjoint subproblems, solve the subproblems and then combine their solutions to solve the original problem. 
In contrast, **dynamic programming applies when subproblems overlap, that is, when sub-problems share sub-sub-problems.**
In this context a divide-and-conquer algorithm does more work than necessary, repeatedly solving the common sub-sub-problems. 
**A dynamic programming algorithm solves each sub-sub-problem just once and then saves its answer in a table, avoiding the work of recomputing the answer every time it solves each sub-sub-problem.** 
### 18.1.2 - A first easy problem: Fibonacci
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

	return F[n-1]```
### 18.1.3 Memorization vs Tabulation
Tabulation and Memorization are two common techniques used in dynamic programming to optimize the solution of problems by avoiding redundant computations and storing intermediate results.
1. **Tabulation - Bottom-Up:**
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
2. **Memorization - Top - Down:**
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
### 18.1.4 - Rod Cutting
Serling Enterprises buys long steel rods and cuts them into shorter rods, which then sells. Each cut is free. 
The management of Serling Enterprises wants to know the best way to cut up the rods. 

Consider a rod of length $n$, we know that for any $i \in [1,n]$ the price of a rod of length $i$ is $p_i$.
The goal is that of determining the maximum revenue $r_n$ obtainable by cutting up the rod and selling the pieces. 

**Solution:** **Bottom-Up Dynamic Programming**
We fill an array $r$ of size $n+1$, initialized with all zeroes, where the entry $r[i]$ stores the maximum revenue obtainable by a rod of size $i$ (which consider the best number of cuts).
Assuming we have already solved all the subproblems of size $j < i$, what's the value of $r[i]$?

**Let's list all the possibilities:**
- we do not cut, the revenue is $p_i$ (aka $p_i + r[0]$). 
- we make a cut of length $1$ and we optimally cut the remaining rod of size $i-1$ 
	- the revenue in this case is $p_i+ r[i-1]$ 
- we make a cut of length $2$ and we optimally cut the remaining rod of size $i-2$
	- the revenue in this case is $p_2 + r[i-2]$
- ...

The value of $r[i]$ is the maximum among all these revenues. 

The following pseudo-implementation clarifies the reasoning: 
```java
rodCutting(rods[])
	n = rods.length()
	r[n+1] // intialized with all zeroes

	// lenght of the cut
	for i = 1 to n 
		q = 0 // max profit for the current rod cuttings
		
		// which is the way to cut the sub-rod of length i
		// to maximize its profit?
		for j = 1 to i
			q = Math.max(q, rods[i].p + r[i-j])
		r[j] = q

	return r[n]
```

`q = Math.max(q, rods[i].p + r[i-j]`
- `q` is the current best profit for a potential cut of length $i$
- we take the maximum between `q` and the sum of
	- `rods[i].p`, we make a cut of the length `i`: we have a rod of length `i` and consider its price, plus
	- `r[i-j]`, the best possible revenue obtainable with the other piece of the rod, which is long `i-j`

The algorithm obviously computes in $O(n^2)$ time. 
### 18.1.5 - Minimum Cost Path
We are given a matrix $M$ of $n\times m$ integers. 
The goal is to find the minimum cost path to move from the top-left corner to the bottom-right corner (you **can not** move diagonally). 

**Solution: Bottom-Up Dynamic Programming**
The idea is to fill the matrix $W$, $n \times m$, as it follows ($1-$indexing)
$$W[i][j] = M[i][j] + 
\begin{cases}
	\begin{align}
		&0\ &\text{if}\ i = j = 1  \\
		&W[i][j-1]\ &\text{if}\ i = 1 \land j > 1 \\
		&W[i-1][j]\ &\text{if}\ j = 1 \land i > 1 \\
		&\min(W[i-1][j], W[i][j-1])\ &\text{otherwise}
	\end{align}
\end{cases}$$
**Said easy:**
You can think of $M[i][j]$ as the "cost of being in" $M[i][j]$, as if the value of a cell of $M$ is its tax to be on it. 
You **always pay** the cost of being in a cell when computing $W$.

Basically the first row and column of $W$ are the prefix sum of the first row and column of $M$, as you can reach them only by going always right or always down.
**Intuition:** to reach $M[1][4]$ (aka the first row, fourth column) the only path is through the first row. 

Then, to reach an "internal" cell $M[i][j]$ where $i,j > 1$ we have to pay:
- the price to enter $M[i][j]$
- the smallest price between arriving from the cell above or from the cell to its right

**Basically you only have two moves:**
- $\rightarrow$
- $\downarrow$
## 18.2 - Solution
Now that we have refreshed dynamic programming we can go back to our problem. 

The subproblems here ask to compute the longest common subsequence (LCS) of prefixes of the two sequences `S1` and `S2`: given two prefixes `S1[1,i]` and `S2[1,j]` our goal is to compute `LCS(S1[1,i], S2[1,j])`

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
		\max
			\begin{cases}
				\text{LCS(S1[1, i], S2[1, j-1])} \\
				\text{LCS(S1[1, i-1], S2[1, j])}
			\end{cases}
		\ &\text{otherwise}
	\end{cases}
	\end{align}
$$
**The pseudo-code of the problem is the following** 
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

	return dp[n-1][m-1]
```
**In practice** we create a matrix `dp[n+1][m+1]` where `n` is the length of `S1` and `m` is the length of `S2`, we set the first row and column to `0` (as they represent the LCS with prefix length `0`) and then iterate, using two nested loop, to check if `s[i] = s[j]`.
In such case increase by one the previous LCS (`dp[i-1][j-1]`), otherwise we pick the max between the LCS of `S1` without the current character and `S2` without the current character.
Then we return `dp[n][m]`, which is the LCS of $S1[1..n]$ and `S2[1..m]` aka `S1` and `S2`
# 19 - Minimum Number of Jumps
Consider an array of `N` integers `arr[]`. 
Each element represents the maximum length of the jump that can be made forward from that element. 
This means if `arr[i] = x`, then we can jump any distance `y` such that `y <= x`.  
Find the minimum number of jumps to reach the end of the array starting from the first element. 
If an element is 0, then you cannot move through that element.  
**Note:** Return -1 if you can't reach the end of the array.
## 19.1 - Solution
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
			if j + array[j] >= i && jumps[j] != MAX
				jumps[i] = Math.min(jumps[i], jumps[j]+1)
				break

	return jumps[n-1]
}
```

**The key is the if guard inside the two loops:** 
- `j + arr[j] >= i`: we want to reach `i` and we are in `j`. If `j + array[j] >= i` it means that `i` is reachable from `j` doing the available number of jumps in `j`, which is `array[j]`
- `jumps[j] != MAX`: we can actually reach `j` from the start

Then the minimum number of jumps required to reach `i` is the minimum between the current number of jumps to reach `i` and the number of jumps required to reach `j` plus one more jump (to reach `i`)
![[Pasted image 20240116171229.png |center| 600]]
# 20 - Partial Equal Subset Sum
Given an array `array[]` of size `N`, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.

This problem is a well-known NP-Hard problem, which admits a pseudo-polynomial time algorithm. 
The problem has a solution which is almost the same as **0/1 Knapsack Problem.**
## 20.1 - 0/1 Knapsack Problem 
We are given $n$ items. Each item $i$ has a value $v_i$ and a weight $w_i$. 
We need to put a subset of these items in a knapsack of capacity $C$ to get the maximum total value in the knapsack. 
**It is called 0/1 because each item is either selected or not selected.** 

We can use the following solutions: 
1) if $C$ is small we can use **Weight Dynamic Programming.** The time complexity is $\Theta(Cn)$
2) If $V = \Sigma_i\ v_i$ is small we can use **Profit Dynamic Programming.** The time complexity is $\Theta(Vn)$ 
3) if both $V$ and $C$ are large we can use *branch and bound*, not covered here. 

**Weight Dynamic Programming**
The idea is to fill a $(n+1)\times (C+1)$ matrix $K$. 
Let $K[i][A]$ be the max profit for weight $\le A$ using items from 1 up to $i$.
![[Pasted image 20240116180507.png | center | 600]]
**Profit Dynamic Programming**
The idea is similar. 
We can use a $(n+1)\times (V+1)$ matrix $K$. 
Let $K[V][i]$ be the minimum weight for profit at least $V$ using items from $1$ up to $i$. 
Thus we have:
$$K[V][i] = \min(K[V][i-1], K[V-v[i][i-1])$$
The solution is $\max(a:K[V,n]\le C)$ 
## 20.2 - Solution
As in the 0/1 knapsack problem we construct a matrix $W$ with $n+1$ rows and $v+1$ columns. 
Here the matrix contains booleans. 

The entry $W[i][j]$ is `true` if and only if there exists a subset of the first $i$ items with sum $j$, false otherwise. 

The entries of the first row $W[0][]$ are set to false, as with $0$ elements you can not make any sum. 
The entries of the first column $W[][0]$ are set to true, as with the first $j$ elements you can always make a subset that has sum $0$, the empty subset. 

Entry $W[i][j]$ is true either if $W[i-1][j]$ is true or $W[i-1][j - S[i]]$ is true. 
- $W[i-1][j] = \text{T}$, we simply do not take the $i$-th element, and with the elements in $[1, i-1]$ we already can make a subset which sum is $j$ 
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

	// always exists a subset of the first i elements that 
	// sum up to 0, the empty subset
	for i = 0 to n+1
		dp[i][0] = true
	// never exists the empty subset that sum up to j >= 1
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
# 21 - Longest Increasing Subsequence
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
## 21.1 - Solution
Due to optimal substructure and overlapping subproblem property, we can also utilize Dynamic programming to solve the problem. Instead of memoization, we can use the nested loop to implement the recursive relation.

The outer loop will run from `i = 1 to N` and the inner loop will run from `j = 0 to i` and use the recurrence relation to solve the problem.

**The reasoning is the following:**
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
**Said easy:**
- outer loop (`for i in 1..n`) iterates over each element of the array starting from the second element.
	- The inner loop (`for j in 0..i`) iterates over elements before the current element `arr[i]`.
		- **if** 
			1) the current element `array[i]` is greater than the element `array[j]` (hence, the element at position `i` could be the next element of the sequence that ends at position j)
			2) the longest increasing subsequence that ends in `array[i]` is shorter than the sequence that ends in `j` (`lis[j]`) plus 1 (the maybe added element `array[i])
		- **then:** we say that `lis[i] = lis[j] + 1`
## 21.2 - Smarter Solution: Speeding up LIS
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

**Said easy:**
- the length of `ans` is the length of the current LIS. 
- when we substitute, we insert `nums[i]` in position `low` of `ans`, using binary search to find `low`
	- `low` is the index of the smallest element grater than `nums[i]` in `ans`, we can substitute `ans[low]` with a smaller element without affecting the longest increasing subsequence, that remains valid. 
- substituting `ans[low]` with `nums[i]` let us "consider" a new LIS, starting from `nums[i]`, this is because when we substitute we insert in a place that is still part of the current LIS, but if a new LIS would start entirely from the current element every element would be substituted **starting from that element**
## 21.3 - DP as Longest/Shortest Path of a DAG
It is often useful to reduce a problem to a (single source) longest path computation on a suitable **DAG** (directed acyclic graph).

Let's consider again the LIS problem. 
Our DAG $G = (V,E)$ has a vertex for each element of the sequence plus a dummy vertex that marks the end of the sequence. 
We use $v_{n+1}$ to denote the dummy vertex, and we $v_i$ to denote the vertex corresponding to element $S[i]$. 

Every vertex as an edge to $v_{n+1}$, and 
$$(v_j, v_i)\in E \iff j<i \land S[j] < S[i]$$
**Said easy:** we have and edge $(v_j,v_i)$ if and only if $S[i]$ can follow $S[j]$ in an increasing subsequence. 

By construction exists a one-to-one correspondence between increasing subsequences of $S$ and paths from $v_1$ to $v_{n+1}$ in $G$. 
Thus any longest path on $G$ corresponds to a LIS on $S$. 

A longest path on a DAG $G$, even weighted ones, can be computed in time proportional to the number of edges in $G$. 
The DAG has at most $n^2$ edges and, thus, the reduction gives an algorithm with the same time complexity of the previous solution. 
# 22 - Coin Change
We have $n$ types of coins available in infinite quantities where the value of each coin is given in the array $C = [c_1, c_2,\dots,c_n]$. 
The goal is to find out how many ways we can make the change of the amount $K$ using the coins given. 

**Example:** $C = [1,2,3,8]$, there are $3$ ways to make $K = 3$, $i.e., \{1,1,1\},\{1,2\},\{3\}$ 

The solution is similar to the one for $0/1$ Knapsack and Subset Sum. 
We build a matrix $W$ of size $(n+1)\times (K+1)$, i.e., we compute the number of ways to change any amount smaller that or equal to $K$ by using coins in any prefix of $C$. 

**The following pseudo-implementation is clarifies the solution**, which is **very** similar to Partial Equal Subset Sum
```java 
coinChange(coins[], k) {
	n = coins.length()
    // store the number of ways to make change for each amount
    // dp[i][j] = number of ways we can change the values j having the
    // coins in coins[0..i]
	// rows = coins "cut"
	// column = amount, up to k
	dp[n+1][k+1]

	//one way to make change for amount 0, i.e., using no coins
	for i = 0 to n+1
		dp[i][0] = 1

	for i = 1 to n+1
		for j = 1 to k+1
			// the current coin is itself greater than k
			// we can not use it to change k
			if coins[i-1] > j
				dp[i][j] = dp[i-1][j]
			else
				// the number of ways to change j is the sum between 
				// 1) the number of ways without using the coin i
				// 2) the number of ways using the coin i plus * (see later)
				dp[i][j] = 
					dp[i-1][j] + 
					dp[i][j-coins[i]]
			
        return dp[n][amount]
```

`*` If we include the current coin, we subtract its value from the current amount `j`, and then consider the number of ways to make change for the reduced amount, including the current coin.
**This allows for the same coin to be used multiple times, as we explore all possible combinations of coins to make change for a given amount.** 
Each cell in the dynamic programming table represents the number of ways to make change using a specific combination of coins.
# 23 - Longest Bitonic Sequence
Given an array `arr[0 … n-1]` containing $n$ positive integers, a subsequence of `arr[]` is called **bitonic** if it is first increasing, then decreasing. 
Write a function that takes an array as argument and returns the length of the longest bitonic subsequence. 
A sequence, sorted in increasing order is considered Bitonic with the decreasing part as empty. Similarly, decreasing order sequence is considered Bitonic with the increasing part as empty. 

**Solution**
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
# 24 - Largest Independent Set on Trees
Given a tree $T$ with $n$ nodes, find one of its **largest independent sets**. 
An independent set is a set of nodes $I$ such that there are no edges connecting any pair of nodes in $I$. 

**Example:** The nodes in red form the largest independent set of the following tree

![[Pasted image 20240222092807.png | center | 400]]
Mind that generally the largest independent set is not unique. 

**Solution**
Consider a bottom-up traversal of the tree. 
For every node $u$ we have two possibilities: 
1) add $u$ to the independent set. 
2) don't add $u$ to the independent set. 

In the case 1) , $u$'s children cannot be part of the independent set but its grandchild could. 
In the case 2) $u$'s children could be part of the independent set. 

Let $LIST(u)$ be the size of the independent set of the subtree rooted at $u$, and let $C_u$ and $G_u$ be the set of children and grandchildren of $u$. 
We then have the following recurrence: 
$$
LIST(u) = \begin{cases}
\begin{align}
&1\ &\text{if}\ u\ \text{is a leaf}\\
&\max
	\begin{cases}
		1 + \Sigma_{v\in G_u} LIST(v) \\
		\Sigma_{v\in C_u} LIST(v)
	\end{cases}
	 &\text{otherwise}
\end{align}
\end{cases}
$$
Thus the problem is solved with a post-order visit of $T$ in linear time. 
**Observation:** the same problem on general graphs is NP-Hard. 
# 25 - Meetings in One Room
There is **one** meeting room in a firm. 
There are `N` meetings in the form of `(start[i], end[i])` where `start[i]` is start time of meeting `i` and `end[i]` is finish time of meeting `i`.  
Find the maximum number of meetings that can be accommodated in the meeting room, knowing that only one meeting can be held in the meeting room at a particular time.

**Note:** Start time of one chosen meeting can't be equal to the end time of the other chosen meeting.

There are various ways to solve the problem, we use it to present **greedy algorithms.**
## 25.1 - Greedy Algorithms
A **greedy algorithm** is any algorithm that follows the problem-solving heuristic of making the locally optimal choice at each stage. 
In many problems a greedy strategy **does not produce an optimal solution**, but a greedy heuristic **can yield locally optimal solutions** that approximate a globally optimal solution in a reasonable amount of time. 

**Example:** a greedy strategy for the Travelling Salesman Problem (TSP) is the heuristic "at each step of the journey visit the nearest city". 
This heuristic does not intend to find the best solution, but it terminates in a reasonable number of steps; finding an optimal solution to such a complex problem typically requires unreasonably many steps.
**Note:** this greedy approach is not very good on the TSP, as the current optimal choice is based on previous choices, and greedy algorithms never reconsider the choices they have made so far.

**Most problems for which greedy algorithms yields good solutions (aka good approximation of the globally optimal solution) have two property:** 
1) **greedy choice property:** we can make whatever choice seems best at the moment and then solve the subproblems that arise later. The choice made by a greedy algorithm may depend on choices made so far, but not on future choices or all solutions to the subproblem. It iteratively makes one greedy choice after another, reducing each given problem into a smaller one. In other words **a greedy algorithm never reconsider its choices**. This is **the main difference from dynamic programming**, which is exhaustive and is guaranteed to find the solution. After every stage, dynamic programming makes decisions based on all the decisions made at the previous stage and may reconsider the previous stage's algorithmic path to the solution
2) **optimal substructure:** a problem exhibits optimal substructure if an optimal solution to the problem contains optimal solutions to the sub-problems. 

**In summary:**
**Greedy is an algorithmic paradigm that builds up a solution piece by piece, always choosing the next piece that offers the most obvious and immediate benefit.** 

Greedy algorithms are used for optimization problems. 
An optimization problem can be (optimally) solved using Greedy if the problem has the following property: 
- at every step, we can make a choice that looks best at the moment, and we get the optimal solution to the complete problem.
### 25.1.1 - Activity Selection
You are given $n$ activities with their start and finish times. Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time. 

**Solution:**
The greedy choice is to always pick the next activity whose finish time is the least among the remaining activities and the start time is more or equal to the finish time of the previously selected activity. 

The algorithm behaves as follows: 
- sort the activities according to their finish time
- select the first activity from the sorted activities
- iterate through the remaining activities
	- if the start time of this activity is greater than or equal to the finish time of the previously selected activity then select this activity 
	- update the "finish" of the activity for future selection

The following pseudo-implementation clarifies the reasoning: 
```java
activitySelection(activities[])
	n = activities.length()

	// sort by activities[i].finish component
	activities.sort()
	res = 1; 
	lastFinishTime = activities[0].finish

	for i = 1 to n
		// if the current activity starts after the end of the 
		// last selected activity then we can do it
		if activities[i].start >= lastFinishTime
			res++
			lastFinishTime = activites[i].finish

	return res
```
### 25.1.2 - Job Sequencing 
You are given an array of jobs where every job have a deadline and associated profit that can be made if the job is completed before its deadline. 
It is also given that every job takes a single unit of time,  so the minimum possible deadline for any job is 1. 
Maximize the total profit if only one job can be scheduled at a time. 

**Solution:**
We use a greedy approach: we select first the most paying jobs and we schedule them for as late as possible (while still respecting their deadline). 
Choosing the jobs with maximum profit first, by sorting the jobs in decreasing order of their profit, help to maximize the total profit as choosing the job with maximum profit for every time slot eventually maximize the total profit. 

**The algorithm behaves as follows:**
- sort the jobs by their profit in decreasing order: from the highest paying to the lowest paying
- create an array of strings (the ids of the jobs) `calendar` with size equal to the greatest of all the deadlines among the jobs
	- the array is initialized to `""`, the empty string
- iterate through jobs using `j`
	- find the biggest free slot (index `i)` in `calendar` that is smaller than `j.deadline`. this is where we place `j.id`.  
		- to find it you can either go backwards from `calendar` or use a semi-bs approach
		- set `calendar[i] = j.id`
		- `res += j.profit`
	- if such an `i` cannot be found simply skip the job
- return res
### 25.1.3 - Fractional Knapsack
Given $N$ items, each of them with its weight $w_i$ and value $v_i$ of $N$ items.
We want to put these items in a knapsack of capacity $W$ to get the maximal profit in the knapsack. 
In the fractional knapsack **we can break items** for maximizing the total value. 

**The greedy approach is to calculate the ratio profit/weight for each item and sort them on the basis of this ratio.** 
Consider the ratio $\frac{\text{profit}}{\text{weight}}$ and remember that it is
- $< 1$ if an item weights less than its value, aka is more valuable than its weight, is very much convenient 
- $= 1$ if an item has equal weight and value
- $> 1$ if an item weights more than its value

The ratio is the **value per unit of weight,** we want to select the item that have highest ratio, so we sort non increasingly the items by the ratio.

Then take the item with the highest ratio and add them as much as we can (can be whole or a fraction of it). 
**This will always give the maximum profit because in each step it adds an element such that this is the maximum possible profit for that much weight.** 
### 25.1.4 - Boxes
There are $n$ boxes. 
For each box $i$ we have its 
- weight: $w_i$
- strength/durability: $s_i$

You can choose some subset of boxes and rearrange them in any order you like.
Find the maximum possible number of boxes that can form a tower (the subset $T$), where for each box $j$ used in the tower the following constraint holds: 
$$s_j \ge \Sigma_{k \in T}\ w_k$$
**Said easy:** the constraint is "structural", a box has to be strong/durable enough to hold the weight of all the boxes over it.

**Solution:**
![[IMG_0425.png | center | 600]]

**Said better:**
![[Pasted image 20240301092147.png | center | 600]]

**We always choose the box that maximize the residual strength of the tower**, which maximize the number of boxes that we can still put on top of the last box, hence the height of the tower is maximized. 
### 25.1.5 - Hero
Bitor has $H$ health points and must defeat $n$ monsters. 
The $i$-th monster deals $d_i$ damage but after death it drops a potion that restore $a_i$ health points, and $a_i$ can be greater than the initial health of Bitor. 
Decide if Bitor can find an order of fighting that makes him defeat all monsters. 

**Observation:** if Bitor reaches $0$ health points then it is dead, if it reaches $0$ but the monster dies it wont count as a win, a dead Bitor cannot drink the potion.

**Solution:**
We have to consider at the same time the damage that a monster deals and the health it provides once killed. 
**This leads to considering a ratio (as in the fractional knapsack).** 

We consider the ratio
$$\frac{\text{damage that the monster $i$ deals}}{\text{health recovered once $i$ is killed}}$$
This ration is useful because it is
- $< 1$ if we gain health defeating the monster
- = 1 if we do not gain health defeating the monster (the monster deals as damage as it heal)
- $> 1$ if we loose health defeating the monster

**We then simply choose to kill monsters that make our health grows first.**

**The algorithm behaves as follows:** 
- compute for each monster the ratio above
- sort the monsters using that ratio in increasing order
- try to kill all the monsters, if $H$ ever reaches $0$ than Bitor cannot kill all the monsters.
## 25.2 - Solution
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
# 26 - Wilbur and Array
Wilbur the pig is tinkering with arrays again. 
He has the array $a_1,\dots,a_n$ initially consisting on $n$ zeroes.
At one step he can choose any index $i$ and either add $1$ to all elements $a_i,\dots, a_n$ or subtract $1$from all elements $a_i,\dots,a_n$. 
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
## 26.1 - Solution
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
# 27 - Woodcutters
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
## 27.1 - Solution
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
# 28 - Bipartite Graph
You are given an adjacency list of a graph **adj**  of V of vertices having 0 based index. Check whether the graph is bipartite or not.

**The Problem in detail:**
A **Bipartite Graph** **is a graph whose vertices can be divided into two independent sets**, $U$ and $V$, such that every edge $(u, v)$ either connects a vertex from $U$ to $V$ or a vertex from $V$ to $U$. 
In other words, for every edge $(u, v)$, either $u$ belongs to $U$ and $v$ to $V$, or $u$ belongs to $V$ and $v$ to $U$. 
We can also say that there is no edge that connects vertices of same set.

**A bipartite graph is possible if the graph coloring is possible using two colors such that vertices in a set are colored with the same color.**
## 28.1 - Graphs 101
![[Pasted image 20240222110350.png | center | 700]]
- **a)** an undirected graph $G= (V,E)$
- **b)** an adjacency-list representation of $G$
- **c)** the adjacency-matrix representation o f $G$
### 28.1.2 - Breadth-First Search
Given a graph $G = (V,E)$ and a distinguished **source** vertex $s$, BFS systematically explores the edges of $G$ to "discover" every vertex that is reachable from $s$.
It computes the distance from $s$ to each reachable vertex, where the distance to a vertex $v$ equals the smallest number of edges needed to go from $s$ to $v$. 

BFS also produces a "breadth-first tree" with root $s$ that contains all reachable vertex. 
For any vertex $v$ reachable from $s$, the simple path in the breadth-first tree from $s$ to $v$ corresponds to a shortest path from $s$ to $v$ in $G$ (a path containing the smallest number of edges.)

Breadth-First search is so named because it expands the frontier between discovered and undiscovered vertices uniformly across the breadth of the frontier. 
**You can think of it as discovering vertices in waves emanating from the source vertex.**

**Remember:** a queue is a FIFO data structure
- insert the new element as the end/tail, it "appends" elements
- remove elements at the start/head, it pop the first 

```java
bfs(graph, s)
	for v in graph.nodes()
		if v != s
			v.color = "white" // node not seen yet
			v.distance = Math.MAX // maximum distance, unreachable for now
			v.predecessor = null // no predecessor in the BF-Tree

	s.color = "gray" // a vertex seen for the first time becomes gray
	s.distance = 0 // the distance to the source is 0, we start there
	s.predecessor = null // the source has no predecessor

	Queue frontier
	// add s as the last element of the queue
	frontier.add(s)

	while !frontier.isEmpty
		// pop the element at the head of the frontier
		u = frontier.pop()
		for v in u.neighbors()
			if v.color == "white"
				v.color = "gray"
				v.distance = u.distance + 1
				v.predecessor = u
				frontier.push(v)
		u.color = "black" // explored all neighbors: the node becomes black
```

**Observation:**
- an unreachable node remains "white"
- the frontier contains only "gray" nodes
- once a node becomes black it is never "explored" again (never re-enters the frontier)
### 28.1.3 - Depth-First Search
As its name implies, DFS searches "deeper" in the graph whenever is possible. 
DFS explores edges of the most recently discovered vertex $v$ that still has unexplored edges leaving it.
This process continues until all vertices are reachable from the original vertex have been discovered. 

Unlike BFS, whose predecessor subgraph forms a tree, depth-first search produce a predecessor graph that might contains several trees, because the search may repeat form multiple sources.
Therefore we define the **predecessor subgraph** of a depth-first search slightly differently from that of a BFS: it always includes all vertices, and it accounts for multiple sources. 

Specifically for a depth-first search the predecessor subgraph is $G_\pi = (V, E_\pi)$, where 
$$E_\pi = \{(v.\pi, v): u\in V \land v.\pi \ne\ \text{NIL}\}$$
The predecessor subgraph of a DFS forms a depth-first **forest** comprising several depth-first trees. 
The edges in $E_\pi$ are tree edges. 

Besides creating a depth first forest, DFS also **timestamps** each vertex. 
Each vertex $v$ has two timestamps: 
- $v.d$ records when $v$ is first discovered (and greyed)
- $v.f$ records when the search finishes examining $v$'s neighbors (and blackens $v$)
```java 
dfs(graph)
	for u in graph.vertices()
		u.color = "white"
		u.predecessor = null

	global time = 0 // global variable
	for u in graph.vertices()
		if u.color == "white"
			dfsVisit(G, u)


dfsVisit(graph, u)
	time++ // time at which u has been discovered
	u.d = time
	for v in u.neighbors()
		if v.color == "white"
			v.predecessor = u
			dfsVisit(G, v)
	time++ 
	u.f = time
	u.color = BLACK
```

**Observation:** DFS uses a **stack** instead of a queue to explore the nodes. In the above implementation the stack is the activation records stack since the implementation is recursive.
#### 28.1.2.1 - Strongly Connected components
An application of DFS is the decomposition of a directed graph into its strongly connected components. 

A **strongly connected component** of a graph $G = (V,E)$ is a maximal set of vertices $C \subseteq V$ such that for every pair of vertices $u$ and $v$ in $C$ we have both $u \rightarrow v$ and $v \rightarrow u$; that is, vertices $u$ and $v$ are reachable from each other.  

The algorithm for finding strongly connected components of a graph $G = (V,E)$ use the **transpose** of $G$, which is defined as $G^T = (V,E^T)$ where $E^T = \{(u,v) : (v,e) \in E\}$. 
Given an adjacency-list representation of $G$ the time to create $G^T$ is $O(V+E)$. 

**Observation:** $G$ and $G^T$ have the same strongly connected components. 

The following linear time (i.e., $\Theta(V+E)$ time) algorithm computes the strongly connected components of a directed graph $G = (V,E)$ using two depth-first searches, one on $G$ and one on $G^T$. 
```java
stronglyConnectedComponents(graph)
	// this is to compute u.f of each node
	dfs(graph)

	graphT = graph.transpose() 

	// same as DFS but in the main loop consider the 
	// the vertices in order of decreasing u.f
	customDFS(graphT)

	// print the vertices of each tree in the depth-first forest obtained
	// by customDFS as a separate strongly connected component. 
	printRes()
```

**Said Easy:**
1. Initialize an empty stack to keep track of the finishing times of nodes during the first DFS traversal.
2. Perform a DFS traversal on the original graph, and upon completing the traversal of a node, push it onto the stack.
3. Build the transposed graph (a graph with all edges reversed).
4. Pop nodes from the stack one by one. Each popped node represents the start of a DFS traversal in the transposed graph.
6. Perform DFS on the transposed graph starting from the popped node. Collect all nodes reachable from this starting node. These nodes form one SCC.
7. Repeat step 4 and step 5 until all nodes are visited.
### 28.1.3 - Single Source Shortest Path
Given a direct or undirect graph $G = (V,E)$ the **objective is to find the shortest path from a single source vertex to all the other vertices in the graph.** 

**Said differently:** given a graph with weighted edges (where each edge has a non-negative weight) the goal is to find the shortest path from a specified vertex (the source) to all other vertices in the graph. 

**From the CCLR:**
Given a graph $G = (V,E)$ find a shortest path from a given source vertex $s\in V$ to every vertex $v\in V$. 
We then introduce: 
- **the weight function:** $w: E \rightarrow \mathbb{R}$ that maps edges to real values, the weights
	- the weight of a path $p = (v_0,\dots,v_k)$ is the sum of the weights of its constituent edges: $w(p) = \Sigma_{i=1}^k w(v_{i-1}, v_i)$ 
- the **shortest path weight:** $\delta(u,v)$ from $u$ to $v$, is equal to
	- $\min\{w(p) : u \xrightarrow{p} v\}$ is there is a path from $u$ to $v$
	- $-\infty$ otherwise

This problem has an **optimal substructure**, which we have seen when we talked about greedy algorithms and dynamic programming. 
**Lemma:** given a weighted directed graph $G = (V,E)$ let $p = (v_0, \dots, v_k)$ be the shortest path from $v_0$ and $v_k$. 
Consider the sub-path $p_{ij}$ from the vertex $v_i$ to $v_j$, where $0 \le i \le j \le k$. 
The sub-path $p_{ij}$ is the shortest path from $i$ to $j$.

There are two things we have to consider when solving this problem: 
1) **negative weighted edges**
	1) if the graph has no negative cycles that are reachable from the source that we have no problem, even if there are negative edges/cycles
	2) if the graph has a negative cycle that is reachable from the source then the shortest-path are not well defined
		1) if there is a negative weight cycle on some path from $s$ to $v$ then we define $\delta(s,v) = -\infty$ 
2) **cycles**
	1) as we have seen we cannot have negative cycles
	2) it also cannot have a positive cycles since removing a cycle from a path $s$ to $v$ produces a shortest path
		1) $0$-weighted cycles can be removed
	3) **without loss of generality we can assume that the shortest path have no cycles, that is they are simple paths**
		1) since any acyclic path in the graph contains at most $|V|$ distinct vertices it also contains at most $|V|-1$ edges.

**Any shortest path contains at most n-1 edges, where n is the number of nodes in the graph**

There are several algorithms to solve the problem: 
1) **Dijkstra's Algorithm:** it starts from the source vertex and iteratively explore the vertices with the smallest tentative distance until all vertices have been visited
2) **Bellman-Ford Algorithm:** iterates through all edges multiple times, updating the shortest distance until convergence is reached
#### 28.1.3.1 - Dijkstra's Algorithm
Dijkstra's algorithm is a method for finding the shortest paths from a single source vertex to all other vertices in a weighted graph. 
It works for both directed and undirected graphs with non-negative edge weights. 
**The algorithm maintains a set of vertices whose shortest distance from the source vertex is known, and continually expands this set by considering vertices with the minimum distance from the source.**

**The algorithm behaves as follows:** 
1. **Initialization**: Assign a distance value to every vertex in the graph. Set the distance of the source vertex to 0 and initialize the distances of all other vertices to infinity. Also, maintain a priority queue (often implemented using a min-heap) to store vertices based on their current estimated distances from the source.
2. **Relaxation**: Repeat the following steps until all vertices have been processed:
    - Extract the vertex with the minimum distance from the priority queue. This vertex is considered as visited.
    - For each neighboring vertex that has not been visited yet, calculate its tentative distance from the source by adding the weight of the edge connecting it to the current vertex to the distance of the current vertex. If this new distance is smaller than the previously known distance of the neighboring vertex, update its distance value in the priority queue.
3. **Termination**: Once all vertices have been visited, the algorithm terminates, and the distances calculated represent the shortest paths from the source vertex to all other vertices in the graph.

**Consider the following pseudo-implementation:**
```java
dijikstra(graph, source)
	source.distance = 0
	PriorityQueue queue // a min-priority queue

	for v in graph.nodes()
		if v != source
			v.distance = MAX
		queue.add(v)

	while !q.isEmpty
		// u is a vertex in Q such that distances[u] is minimum
		// quueue is a priority queue 
		u = queue.pop()
		for v in graph.getNeighborsOf(u)
			weight = graph.getWeightOfEdge(u,v)
			// this update is reflected on the priority queue
			// that might have to "resort" elements
			v.distance = min(v.distance, u.distance+weight)

	distances[n]

	for i = 0 to n
		distances[i] = graph.getNode(i).distance
	return distances
```

**Time complexity of Dijkstra:**
The time complexity of Dijkstra's algorithm depends on the data structures used to implement it. 
Using a priority queue based on a **binary heap**, the time complexity is typically $O((V+E)\log(V))$ where $V$ is the number of vertices and $E$ is the number of edges in the graph.
**Remember the complexity of a binary heap**, seen them in *8.3*
**Breakdown of the complexity**
1. Initialization:    
    - Inserting vertices into the priority queue: O(V * log(V))
    - Each insertion takes O(log(V)) time, and there are V vertices.
2. Main loop:
    - In each iteration of the loop, we extract the minimum vertex from the priority queue and relax its outgoing edges.
    - The loop runs V times because each vertex is extracted at most once.
    - For each vertex, we relax its outgoing edges, which takes constant time for each edge relaxation.
3. Updating priorities:
    - Decreasing the priority of a vertex: O(log(V))
    - Since each edge relaxation may require updating the priority of a vertex in the priority queue, the total time spent on updating priorities is O(E * log(V)).
#### 28.1.3.2 - Bellman-Ford Algorithm
The Bellman-Ford algorithm is used to find the shortest paths from a single source vertex to all other vertices in a weighted graph, even in the presence of negative weight edges, as long as there are no negative weight cycles reachable from the source vertex.

**The algorithm behaves as follows:**
1. **Initialization**: Set the distance to the source vertex as 0 and all other distances to infinity.
2. **Relaxation**: 
	- **Iterate Through All Edges**:
	    - For each vertex `u` in the graph:
	    - For each edge `(u, v)` connected to `u`, where `v` is the destination vertex, and `weight` is the weight of the edge
	- **Check for Improvement**:
	    - Check if the distance to vertex `v` can be improved by considering the edge `(u, v)`.
	    - The distance to `v` can be improved if the distance to `u` plus the weight of edge `(u, v)` is less than the current known distance to `v`.
	- **Update Distance**:
	    - If the above condition is met, update the distance to vertex `v` to be the sum of the distance to `u` and the weight of edge `(u, v)`.
	    - This update represents the shortest path discovered so far from the source vertex to vertex `v`.
	- **Repeat**:
	    - Repeat this process for all vertices and edges in the graph for a total of `|V| - 1` iterations, where `|V|` is the number of vertices in the graph.
	    - Each iteration potentially refines the distance estimates until the shortest paths are found.
1. **Check for Negative Cycles**: After $|V| - 1$ iterations, if any distances can still be improved, then there exists a negative weight cycle in the graph reachable from the source vertex. This step is optional if you're just interested in finding the shortest paths and not detecting negative cycles.

**Consider the following pseudo-implementation:** 
```java
bellmanFord(graph, source)
	// step 1) initialization
	n = graph.getNumberOfNodes()
	distances[n]
	for i = 0 to n
		distances[i] = Math.MAX
	distances[source] = 0

	// step 2) relaxation
	for i = 0 to n-1 // |V|-1 times
		for u in graph.getNodes()
			for v in graph.getNeighborsOf(u)
				weight = graph.getWeightOfEdge(u,v)
				distances[v] = min(distances[v], distances[u] + weight)
		
	// step 3) check negative cycles
	for u in graph.nodes()
		for v in graph.getNodes()
			weight = graph.getWeightOfEdge(u,v)
			if distances[u] + distance  < distances[v]
				throw NegativeCycleException

	return distances
```

**Time Complexity of Bellman-Ford**
1. **Iteration over Vertices (|V|)**:
    - In each iteration of the outer loop (`for _ in range(len(graph) - 1)`), we iterate through all vertices in the graph to relax their outgoing edges. This contributes O(V) to the time complexity.
2. **Iteration over Edges (|E|)**:
    - For each vertex `u` in the graph, and for each edge `(u, v)` connected to `u`, where `v` is the destination vertex, the inner loop iterates through the edges. This inner loop contributes O(E) to the time complexity.

Combining the complexities of iterating over vertices $(O(V))$ and edges $(O(E))$, and considering that we iterate over all vertices for a total of $|V| - 1$ times, we get a time complexity of $O((|V| - 1) * |E|)$, which is simplified to $O(V * E)$
**Observation:** we assume that each node as a constant number of neighbors ($O(1)$), for a complete graph the complexity is $O(n^3)$.
### 28.1.4 - Minimum Spanning Tree
A **spanning tree** is a subset of a graph that includes all the graph's vertices and some of the edges of the original graph, intending to have no cycles. 
A spanning tree is not necessarily unique.

 A **minimum spanning tree (MST)** is a subset of the edges of a connected, edge-weighted graph that connects all the vertices together without any cycles and **with the minimum possible total edge weight.** 
 **It is a way of finding the most economical way to connect a set of vertices.** 
 A minimum spanning tree is not necessarily unique. 
 **All the weights of the edges in the MST must be distinct. If all the weights of the edges in the graph are the same, then any spanning tree of the graph is an MST.** 
 
 The edges of the minimum spanning tree can be found using the greedy algorithm or the more sophisticated Kruskal or Prim's algorithm.
 
A minimum spanning tree has precisely $n-1$ edges, where $n$ is the number of vertices in the graph.

**Kruskal's Algorithm**
In Kruskal’s algorithm, sort all edges of the given graph in increasing order. 
Then it keeps on adding new edges and nodes in the MST if the newly added edge does not form a cycle. 
It picks the minimum weighted edge at first and the maximum weighted edge at last. Thus we can say that it makes a locally optimal choice in each step in order to find the optimal solution, aka **it is a greedy algorithm.**

**The algorithm behaves as follows:**
1) sort all the edges of the graph in non-decreasing order of their weight
2) pick the smallest edge: if this edge do not form a cycle with the spanning tree formed so far then include the edge in the spanning tree, otherwise discard it
3) repeat the step 2) until there are $V-1$ edges in the spanning tree.
### 28.1.5 - Disjoint Sets Data Structure
In a room there are $N$ persons, and we will define two persons as friends if they are directly or indirectly friends (a friend of a friend is considered a friend, and so on)
A **group of friends** is a group of persons where any two pair of persons are friends. 

Given the list of persons that are directly friends find the number of group of friends and the number of persons in each group.
**Let's see an example:** consider $N = 5$ and the list of friends is $[1,2], [5,4], [5,1]$
We can see that using the "transitive property of friendship" at the end we will have two groups of friends: $[1,2,4,5], [3]$ 

![[Pasted image 20240228121022.png | center | 150]]

**Solution:**
This problem can be solved using a BFS but we use it to introduce data structures for disjoint sets. 
**A disjoint-set data structure is a structure that maintains a collection S1, S2, S3.. Sn of dynamic disjoint sets.**
**Two sets are disjoint if their intersection is null.** 
In a data structure of disjoint sets every set contains a **representative**, which is member of the set. 

Consider the example above.
At the beginning there are $5$ groups: $[1], [2], [3], [4], [5]$. Nobody is anybody's friends and everyone is the representative of their own group.

The next step is that $1$ and $2$ becomes friends, hence the groups $[1]$ and $[2]$ become one group. Now we have $4$ groups: $[1,2],[3],[4],[5]$.
Next $5$ and $4$ become friends. The groups will be $[1,2], [3], [4,5]$. 
And in the last step $5$ and $1$ become friends and the groups will be $[1,2,4,5], [3]$. 
The representative of the first group will be $5$ and the representative of the second group will be $3$ (we will see later why we need representatives.)

**How can we check if two persons are in the same group?**
We use the representative elements comes in.
Let's say we want to check if $3$ and $2$ are in the same group. 
If the representative of th set that contains $3$ is the same as the representative of the set that contains $2$ then they are in the same group (and therefore are friends). 
We see that this is not the case, as the representative of $2$ is $5$ while the representative of $3$ is $3$.

**Some operations:**
Let's define the following operations: 
- `create-set(x)` creates a new set with one element `x`
- `merge-sets(x,y)` merges into one set the set that contains `x` and the set that contains `y` (`x` and `y` are in different sets), the original sets will be destroyed
- `find-set(x)` returns the representative or a pointer to the representative of the set that contains `x`

Then we can now solve the problem using those operations: 
```java
n = read(stdin) // number of friends
for x from 1 to n
	create-set(x)

for (x,y) friends 
	if find-set(x) != find-set(y)
		merge-sets(x,y)
```

Now, if we want to see if two persons `(x,y)` are in the same group we check if `find-set(x) == find-set(y)`

We will analyze the running time of the disjoint-set data structure in terms of $N$ and M, where $N$ is the number of times that `create-set(x)` is called, and $M$ is the total number of times that `create-set(x)`, `merge-sets(x, y)` and `find-set(x)` are called. 

Since the sets are disjoint, each time `merge-sets(x, y)` is called one set will be created and two will be destroyed, giving us one less set. 
If there are $n$ sets then after $n-1$ calls of `merge-sets(x,y)` there will remain only one set. 
That’s why the number of `merge-sets(x,y)` calls is less than or equal to the number of `create-set(x)` operations.
#### 28.1.5.1 - Implementation with Linked List
One way to implement disjoint set data structures is to represent each set by a linked list. 
Each element (object) will be in a linked list and will contain a pointer to the next element in the set and another pointer to the representative of the set. 

Here is a figure of how the example of the problem will look like after all operations are made. T**he blue arrows are the pointers to the representatives and the black arrows are the pointers to the next element in the sets.**

![[Pasted image 20240228163239.png | center | 180]]

Representing sets with linked lists we will obtain a complexity of $O(1)$ for` create-set(x)` and `find-set(x)`. 
`create-set(x)` will just create a new linked list whose only element (object) is `x`, the operation `find-set(x)` just returns the pointer to the representative of the set that contains element (object) `x`.

Now let’s see how to implement the `merge-sets(x, y)` operation. 
The easy way is to append `x`’s list onto the end of `y`’s list. 
The representative of the new set is the representative of the original set that contained y. 
We must update the pointer to the representative for each element (object) originally on `x`’s list, which takes linear time in terms of the length of `x`’s list. 

It’s easy to prove that, in the worst case, the complexity of the algorithm will be $O(M^2)$ where $M$ is the number of operations `merge-sets(x, y)`. 
- if we have $M$ operations `merge-sets` where in each of them we update the pointers of the representative, the worst possible case is that we update, for each call, exactly $M$ pointers, as updating $M$ pointers for each calls maximize the number of updated pointers.

With this implementation the complexity will average $O(N)$ per operation where $N$ represents the number of elements in all sets.

**TODO**
## 28.2 - Solution
The solution is basically an adapted version of BFS, the implementation is crystal clear. 
```java
isBipartite(graph[][], src) 
	// number of nodes
	n = graph[0].length()

	// assign the "non-color" to all the nodes
	colors[n]
	for i = 0 to n
		colors[i] = -1

	// assign a color to the src
	colors[src] = 1
	
	// fifo queue (like BFS)
	Queue frontier 
	frontier.add(src)

	while !frontier.isEmpty()
		// extract a node from the queue
		i = frontier.poll()

		// there is a self loop: a node is connected to itself
		// hence a node is the neighbor of himself, hence not bipartite
		if graph[i][i] == 1
			return false

		for j = 0 to n
			// the node i and j are neighbors
			if graph[i][j] == 1
				// two neighbors have the same color
				if colors[i] == colors[j]
					return false
					
				// the neighbor j has no color assigned
				// it is a node that we have not seen before
				if color[j] == -1
					color[j] = 1 - color[u]
					frontier.add(j)
	return true
```
