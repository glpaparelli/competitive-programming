# 1) Contiguous Subarray with Max Sum
Given an array with elements $x \in \mathbb{Z}$, find the subarray with maximum sum. 
### Solutions
#### Brute Force, $O(n^2)$
Compute the sum of every possible subarray and store the maximum.
The complexity is given by the two nested for loops
#### Optimal Solution: Kadane's Algorithm, $O(n)$
Kadane's Algorithm is based on two observations: 
1) the sum of values in any prefix of the optimal subarray is positive
2) the element before the first element of the optimal subarray is negative

```java
kadane_algorithm(a[])
	n = a.length()
	max = Integer.MIN_VALUE
	sum = 0
	
	for i = 0 to n
		if sum > 0 
			sum += a[i]
		else 
			sum = a[i]

		max = Math.max(sum, max)
		
	return max
```

**Example:**
![[IMG_0394.png | center | 500]]
# 2) Trapping Rain Water
Given an array $h$ with `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining. 
![[Pasted image 20240506090641.png | center | 450]]
### Solutions
#### Brute Force, $O(n^2)$
For each element find the highest bar to its left and right. Take the smaller of the two bars and call it $t$. The difference between the height of the current bar and the $t$ is the amount of water that can be stored over the current bar.
#### Smart Solution: Precomputation, $O(n)$ time, $O(n)$ space
**The point is to think locally, not globally.**
We compute two support arrays, the array of left and right leaders.

An element of an array is a **right leader** if it is greater than or equal to all the elements to its right side. The rightmost element is always a leader. Left leaders are analogous.
Computing leaders cost $O(n)$: to compute the right leaders traverse the array backwards.

We then have that: 
- $LL[i]$ contains the left leader for the position $i$
- $RL[i]$ contains the right leader for the position $i$

The water that can be stored over the element $i$ is given by
$$w(i) = \min(LL[i], RL[i]) - h[i]$$
we can then compute the sum with just one pass over $h$.
#### Optimal Solution: Two Pointers Trick, $O(n)$
Idea based on the previous solution plus the observation that **we do not need all the leaders**, but just the two currently meaningful leaders. 

```java
max_water(heights[]) 
	n = heights.length()
	left = 0, right = n-1
	result = 0

	left_max = 0 // max height seen by "left" so far, curr LL
	right_max = 0 // max height seen by "right" so far, curr RL

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

![[Pasted image 20240105122707.png | center | 500]]
# 3) Binary Search
Divide-and-conquer algorithm that searches an element within a sorted array: $O(\log(n))$
- **divide:** array is divided into two equal halves, centering around the middle element 
- **solve:** compare the middle element of the array with the searched key. 
	- if the middle element is a match, the search stop successfully.
	- If not, we recursively search for the key only in one of the two halves
- **combine:** there is nothing to combine in this case

```java
binary_search(a[], target) 
	n = a.length()
	low = 0
	high = n - 1

	while low < high
		mid = low + (high - low) / 2

		if a[mid] == target
			return mid
		
		if a[mid] < target
			low = mid + 1
		else 
			high = mid

	return -1
```

We can use BS to return the first (or last) occurrence of the element by modifying the first if: 
``` java
if a[mid] == target
	answer = mid
	high = mid
```
## Applications of Binary Search
We can **binary search the answer** of a problem if: 
- the possible answers are restricted to a range of values `[low, high]`
- we have a **monotone boolean predicate** `pred` that is true if the answer is good for our aims

**Monotone Boolean Predicate:** the predicate $P(x)$ transitions from `true` to `false` at most once as $x$ increases. 
This monotonicity makes it possible to use binary search to efficiently find the boundary where the predicate changes its value from `true` to `false`.

If no assumption is done on `pred` we have no other way than evaluating `pred` on every element inside `[low, high]`, hence complexity of $O(\text{high} - \text{low}) = O(n)$. 
But if the predicate is **monotone** we can binary search the answers in `[low, high]`. 

Say that we want the **biggest** possible answer, then: 
```java
rangeBS(low, high, pred) 
	answer = -1
	while low < high
		mid = low + (high - low) / 2

		if pred(mid) == true
			answer = mid
			low = mid+1 // we want the biggest
		else 
			high = mid

	return answer 
```
### 4) Square Root
We have a non-negative integer $v$ and we want to compute the square root of $v$ down to its nearest integer. 
We can use the previous approach as: 
- the answers fall in the interval $[0, v]$
- the predicate $p(x) \doteq x^2 <= v$ is monotone

Hence we can simply call `rangeBS(0, v, p)` and return the smallest element that verifies the predicate.
### 5) Social Distancing
Consider a sequence of $n$ mutually-disjoint intervals. 
The extremes of each interval are non-negative integers. 
We aim to find $c$ integer points within the intervals such that the smallest distance $d$ between consecutive selected point is **maximized**. 

If a certain distance is feasible (i.e., exists a selection of points at that distance), then any smaller distance is also feasible. **The feasibility is a monotone boolean predicate.**
As the candidate answers range from $1$ to $l$, where $l$ is the overall length of the intervals, the solution takes $\Theta(\log(l))$ evaluations of the predicate.  

**What's the cost of evaluating the predicate?** 
We sort the intervals by their left endpoints and then evaluate any candidate distance $d'$
by scanning the sorted intervals from left to right. 
First we select the left extreme of the first interval as the first point. 
Then we move over the intervals and we choose greedily points at a distance at least $d'$ from the previous one. 
Thus an evaluation of the predicate takes $\Theta(n)$ time.

The total running time is $\Theta(n\log(n))$. 

```java
pred(intervals, distance, c) {
	// the first point is the start of the first interval
	lastSelected = intervals[0].start 
	counter = 1
	// for every interval in intervals
	for i in intervals
		// we select as many points as we can in every interval.
		// we get the max using i.start because if a point falls
		// in the empty space between two intervals we place it 
		// at the start of the current interval
		while Math.max(lastSelected + distance, i.start) <= i.end
			lastSelected = Math.max(lastSelected + distance, i.start)
			counter++
	
	// returns true if we placed at least c points
	return counter >= c
}
socialDistancing(intervals, c) {
	// l is the maximum length of an interval
	if l < c
		return -1 // there is no solution

	// sort the intervals by the start
	intervals.sort()
	
	// do a tailored rangeBS on all the candidate answers
	// for each of them compute the predicate
	rangeBS(1, l+1, pred)
}
```
# 6) Find Peak Element
A peak element is an element strictly greater than its two neighbors. 
Given an integer array `nums` of $n$ elements find a peak element and return its index. 
Mind that $\text{nums}[-1] = \text{nums}[n] = -\infty$. 
Write an algorithm that solves the problem in $O(\log(n))$ time
### Solution, $O(\log(n)$
We use a binary search approach to solve the problem. 
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

This solution works because a peak is guaranteed to be found (as a last resource, in the first or last element of the array).

![[Pasted image 20240211104316.png | center | 500]]
# 7) Tree Traversals
**Tree traversals are a form of graph traversal and refers to the process of visiting each node in a tree data structure exactly once.**
**Such traversals are classified by the order in which the nodes are visited.**

Traversing a tree involves iterating over all the nodes in some manner.
Since from a given node there is more than one possible next node some nodes must be **deferred**, aka stored, in some way for later visiting. This is often done via a stack (LIFO) or queue (FIFO). 
**As a tree is a recursively defined data structure, traversal can be defined by recursion.**
In these cases the deferred nodes can be implicitly stored in the call stack.
### Depth-First Search
In DFS **the search tree is deepened as much as possible before going to the next sibling.**
To traverse a binary tree with DFS we perform the following operations on each node:
- if the current node is empty then return
- else execute the following three operations in a **certain** order: 
	- **N:** visit the current node
	- **L:** recursively traverse the current node's left subtree
	- **R:** recursively traverse the current node's right subtree

![[Screenshot from 2024-01-08 09-19-39.png | center | 350]]

**The choice of the order of N, L, R determines the color and the type of visit:**
1) **N-L-R, pre-order visit:** The preorder traversal is a topologically sorted one, because the parent node is processed before any of its child
	1) "visit" the current node
	2) recursively traverse the current node's left subtree
	3) recursively traverse the current node's left subtree
2) **L-N-R, in-order visit:** In a BST this visit retrieves the keys in ascending order
	1) recursively traverse the current node's left subtree
	2. visit the current node
	3. recursively traverse the current node's right subtree
3. **L-R-N, post-order visit:** 
	1) recursively traverse the current node's left subtree
	2) recursively traverse the current node's right subtree
	3) visit the current node

**Complexity Analysis:**
- **Time Complexity:** $O(n)$, where $n$ is the number nodes
- **Auxiliary Space:**
	- $O(1)$ if no recursion stack space is considered
	- otherwise $O(h)$, where $h$ is the height of the tree 
		- worst-case scenario: $h = n$, the tree is skewed
		- best-case scenario: $h = \log(n)$, the tree is a complete tree
# 8) Size of Every Subtree
Given a binary tree $T$ we want to compute the subtree size of node `u`. 
### Solution
```java
// returns the size of the subtree rooted in u
subtree(u) 
	if u == null
		return 0
	l = subtree(u.left)
	r = subtree(u.rigth)
	return l + r + 1
```

**The idea is that the information is flowing bottom-up,** as the size of an empty node is known (the size of tree rooted in a leaf) and with that info from the bottom you build up the solution for the upper nodes.
![[Pasted image 20240506143517.png | center | 350]]
# 9) Depth of a Node
Given a binary tree $T$ we want to compute depth for every node in the subtree rooted in `u`, where the depth of a node is defined as the distance from the root. 
### Solution
```java
depth(u, d)
	if u == null
		return 
	depth(u.left, d+1)
	print(u, d)
	depth(u.right, d+1)
```
**The information is flowing top-down.**
To print the depth of every node you simply call `depth(r, 0)`, where `r` is the root.
# 10) Path Sum equals to Subtree Sum
Given a tree $T$ count the nodes `u` such that the sum of keys on the path root-to-`u` equals the sum of the keys in `u`'s subtree.
![[Pasted image 20240506152610.png | center | 300]]
### Solution 
```java
// u is the target node, psum is the path sum so far from r to u
// this function returns a pair:
//	- counter (number of nodes that satisfy pathsum == treesum) 
//  - subtree sum: the sum of the subtree rooted in u

// the initial call of this function will be pathSubtreeSum(r, 0)
pathSubtreeSum(u, psum)
	if u == null
		return 0,0
		
	// cl / cr = counter left / counter right
	// sl / sr = left subtree sum / right subtree sum
	cl, sl = pathSubtreeSum(u.left, psum+u.k)
	cr, sr = pathSubtreeSum(u.right, psum+u.k)

	// sum subtree rooted in u
	su = sl + sr + u.k
	// counter subtree rooted in u
	cu = sl + sr
	// is u a node such that the sum of its subtree 
	// is equal to the sum of the keys on its path to the root?
	if su == psum
		cu = cu + 1 // also count u as a node that respect the property
	
	return cu, su
```
# 11) Max Sum Leaf-to-Root Path
Given a binary tree, find the maximum sum path from a leaf to the root. 
**Example:** consider the following binary tree:
```plaintext 
					              10  
				                 /  \  
				               -2    7  
				              /  \       
				             8   -4
```
Since we have three leaves ($8, -4, 7$) we have three leaf-to-root path: 
- $8 \rightarrow -2 \rightarrow 10$
- $-4 \rightarrow -2 \rightarrow 10$
- $7 \rightarrow 10$

The sum of these paths are $16, 4$ and $17$ respectively. 
Therefore the maximum sum leaf-to-root path is the last one: $7\rightarrow 10$.
### Solution
The solution is made of the following two steps: 
- find the leaf node that is on the maximum sum leaf-to-root path
- once we have the target leaf node, we can print the maximum sum path by traversing the tree

**The following snippet clarifies the process:**
```java
maxPathSum = MIN
targetLeaf = null

maxLeafRootPath(root)
	targetLeaf = findTargetLeaf(root, 0)
	print maxPathSum 

// set targetLeaf to the leaf with maximum sum leaf-to-root path
findTargetLeaf(node, currentPathSum)
	if node == null 
		return

	// compute the current path sum from root to this node
	currentPathSum += node.value

	// the current node is a leaf
	if node.isLeaf == true
		// this leaf has the maximum sum leaf-to-root 
		// path so far: store the leaf and save the sum
		if currentPathSum > maxPathSum
			maxPathSum = currentPathSum
			targetLeaf = node

	// this node is not a leaf: recur on children
	findTargetLeaf(node.left, currentPathSum)
	findTargetLeaf(node.right, currentPathSum)
```

**Time Complexity**
The time complexity of the above solution is $O(n)$ as it involves two tree traversals
# 12) Maximum Path Sum
Given a binary tree find the maximum path sum from a leaf to another. 
**Example:**
![[Pasted image 20240108100401.png | center | 450]]
### Solutions
#### Trivial Solution, $O(n^2)$
The obvious approach is to reuse the solution of the problem seen before. 
For every node `x` we compute: 
- max sum `x.left`-to-leaf path 
- max sum `x.right`-to-leaf path
- take the maximum between the two
- add the value stored in `x`
- if this sum is greater than the current maximum then we update it

For every node `x` we call twice `maxLeafRootPath`. Since the function cost $O(n)$ we have that the overall time complexity is $O(n^2)$. 
#### Optimal Solution, O(n)
**We can find the maximum sum using a single traversal of binary tree**.
The idea is to **maintain two values** in every recursive call: 
1) maximum sum leaf-to-root path for the subtrees rooted under the current node
2) maximum path sum between leaves, the desired output

To solve the problem we use a custom **post-order visit** `maxPathSum` that exploits a global variable `maxSum` to store the result. 
Our visit `maxSumPath` receive in input the root of the tree and then
- if the passed root is `null` we return $0$
- `left = maxSumPath(root.left)`, we recur on the left subtree
	- the function **returns the sum of the path from the root to a leaf**, hence `left` is the **path sum from the current node to a leaf** 
- `right = maxSumPath(root.right)`, we recur on the right subtree
- `maxSum = Math.max(left+right+root.value, maxValue)`, update the `maxSum` if we found a bigger path from leaf to leaf passing for the current node

The following snippets further clarifies the algorithm: 
```java
maxValue = 0

solution(root)
	maxPathSum(root)
	print(maxValue)

// calculates two things:
// 1) maximum path sum between two leaves, which is stored in maxValue.
// 2) max current-node-to-leaf path sum, which is returned.
maxPathSum(node)
	if node == null
		return 0

	left = maxPathSum(node.left)
	right = maxPathSum(node.right)
	
	maxValue = Math.max(maxValue, left+right+node.value)

	return Math.max(left, right) + node.value
```

**In "one pass" (aka one traversal) we do what we did in the trivial solution.**
The reasoning is the following: 
- Take a node `x`
- Compute the max path sum from `x.left` to a leaf `i` and the max sum path from `x.right` to a leaf `j`. 
- The sum path from `i` to `j`, that passes through `x`, is greater than the one previously stored? Then we update the max

The result is `maxValue` when the program terminates.
# 13) Two Sum in a Sorted Array
Given a sorted array and an integer `x`, find if there exists any pair of elements `a[i], a[j]` such that `x = a[i] + a[j]`
### Solutions
#### Brute Force, $O(n^2)$
Scan the array twice and we return `true` if we find two numbers that adds up to `x`
```java
pairSum(a[], x)
	n = a.length()
	for i = 0 to n
		for j = i+1 to n
			if a[i] + a[j] == x
				return true
			if a[i] + a[j] > x
				break // a is sorted, need a new "i"
	return false
```
#### Optimal Solution: Two Pointers, $O(n)$
We use two pointers, `left` and `right`, to find the solution. 
The pointer `left` will point to the first element of the array, and `right` to the last. 
If the sum of the elements pointed by `left` and `right` is greater than `x` we will shift backwards by one `right`, in the other case we shift `left` one position to the right. 
The solution takes advantage of the sortedness of the array.

```java
pairSum(a[], x)
	n = a.length()
	left = 0, right = n-1

	while left < right
		if a[left] + a[right] == x
			return true
		if a[left] + a[right] < x
			left++
		else 
			right++

	return false
```
# 14) Binary Search Trees
**A Binary Search Tree (BST) is a rooted binary tree data structure where each node's key is bigger than all the keys stored in the node's left subtree, and smaller than all the keys stored in the node's right subtree.** 

Since the nodes in a BST are laid out so that each comparison skips about half of the remaining tree, the lookup takes $O(h) = O(\log(n))$ time.
The same goes for the other two operations: insertion and removal.

In the worst case (keys are descending or ascending values) a BST degrade to that of a singly linked list: $O(n)$. 
This actually never happen in reality as it is always used a Balanced BST (BBST): if you have a sorted array that would make a skewed BST simply select the median element as the root of the tree.
In this notes when we say BST we actually mean a BBST.
### Lookup
**Searching for a specific key can be done recursively or iteratively.**
![[Screenshot from 2024-01-19 15-34-31.png | center | 500]]
### Predecessor and Successor
For certain operations, given a node $x$, finding the successor or predecessor of $x$ is crucial. 
Assuming that all the keys of the BST are distinct, **the successor of a node** $x$ in BST **is the node with the smallest key greater** than $x$'s key. 
On the other hand, **the predecessor of a node** $x$ in BST **is the node with the largest key smaller** than $x$'s key. 

**Said Easy:**
**Predecessor** and **Successor** takes $O(\log(n))$ and we have that: 
- the **successor of a node** is the node with the smallest key that is bigger than the current node
	- once to the right, then left as long as possible
- the **predecessor of a node** is the node with the greatest key that is smaller than the current node
	- once to the left, then right as long as possible
### Removing an Element 
**Removing an element** from a BST costs $O(h)$
- if it is a leaf just remove it
- if it is a node with a single child: copy the value of the child into the current node and delete the child
- if it is a node with two children: find the **in-order successor** of the node, copy the contents of the in-order successor in the current node and then remove the in-order successor
# 15) Is T a BST?
Check if the binary tree $T$ is a binary search tree.
### Naive Solution 
We remember the property that a binary search tree has to respect: 
$$\forall u . \begin{cases} \max\{v.k\ |\ \forall v\in u.left \} \le u.k \\ \land \\ \min\{v.k\ | \ \forall v\in u.right \} > u.k \end{cases}$$
Then we try the following approach: 
```java
// returns false if any node in u's subtrees don't respect the property
checkBST(u)
	if u == null 
		return true

	bL = checkBST(u.left) // is the left child a BST?
	bR = checkBST(u.right) // is the right child a BST?

	maxL = maxTree(u.left) // max element in the left subtree
	minR = minTree(u.right) // min element in the right subtree

	bU = maxL <= u.k && minR > u.key // u respect the property?

	return bL && bR && bU 

maxTree(u)
	if u = null
		return MIN // return -infinite
	mL = maxTree(u.left)
	mR = maxTree(u.right)
	return max(mL, mR, u.k)
```

This solution works but it is too slow as for every node  `i` of  the subtree rooted in `u` we check if the subtree rooted  in `i` respect the property.
We do a lot of repeated work.
In the worst case, (a skewed BST (therefore a non balanced BST)) the cost to pay is $O(n^2)$, where $n$ is the number of nodes.
### Optimal Solution
We build upon the previous solution
```java
// return false if any node do not respect the property
checkBST(u)
	if u == null
		// (is a BST, max and min in the subtree)
		return true, MAX, MIN

	bL, maxL, minL = checkBST(u.left) 
	bR, maxR, minR = checkBST(u.rigth)

	bU = maxL <= u.k && minR > u.key // u respect the property?

	return (bL && bR && bU), max(u.k, maxL, maxR), min(u,k, minL, minR)
```
# Predecessor Problem
Given a set $S$ of keys we would like to support the following operations
- `insert(x)`
- `delete(x)`
- `lookup(x)`, is `x` in $S$?
- `min/max`
- `predecessor(x)`: $\max\{z\ |\ z\in S \land z\le x\}$, `x` may not be in $S$
- `successor(x)`: $\min\{z\ |\ z \in S \land z\ge x\}$, `x` may not be in $S$

With BBST all the operations are solved in $O(\log(n))$ times. 
Insertions, search and retrieving the min/max are quite obvious (with the catch of balancing when we insert in the tree). 
The other operations are a bit trickier.
**TODO**
# 16) Frogs and Mosquitoes
There are $n$ frogs sitting on the coordinate axis $Ox$. 
For each frog we have two values:
- $x_i$, the position of the $i$-th frog on the coordinate axis (all positions $x_i$ are different)
- $t_i$, the initial length of the tongue of the $i$-th frog

We then have $m$ mosquitoes, one by one are landing to the coordinate axis. 
For each mosquito we have two values: 
- $p_j$, the coordinate of the position where the $j$-th mosquito lands
- $b_j$, the size of the $j$-th mosquito. 

**Frogs and mosquitoes are represented as points on the coordinate axis.**

**The frog can eat a mosquito if the mosquito is in the same position with the frog or to its right, and the distance between them is not greater than the length of the tongue of the frog.**

If at some moment several frogs can eat a mosquito the leftmost frog will eat it (with minimal $x_i$). 
**After eating a mosquito the length of the tongue of a frog increases with the value of the size of the eaten mosquito.** 
**It's possible that after it the frog will be able to eat some other mosquitoes: the frog should eat them in this case.**

A mosquito is landing to the coordinate axis only after the frogs eat all possible mosquitoes landed before. 
**Mosquitoes are given in order of their landing to the coordinate axis.**

For each frog print two values: the number of eaten mosquitoes and the final length of the tongue.

The target complexity is $O((n+m)\cdot \log(n+m))$
### Solution
**We store the position of the frogs in a BST.**
When a mosquito lands on position `b`, to know which frog eats it we simply do a `predecessor(b)` query on the frog tree.

**This base solution needs to be adjusted to account for three main issues:**
- **Overlapping segments:** Each frog can cover the segment `(p, p+tongue)`. To enforce that the leftmost frog has the priority on the landing mosquito we preprocess the input segments to force no overlap. If two frogs share a common segment the rightmost frog gets moved and we assign to it assigned the segment `(r+1, r+1+tongue)` where `r` is the maximum distance reached by the tongue of the left frog. If a frog segment is contained by another frog then that frog is removed entirely.
- **Dynamic segments:** When a frog eats its tongue grows, which means that it now can cover a longer segment. This new segment might fully contain other segments, which can be found with a successor query (on the position of the frog that has eaten) on the tree of frogs. In this case we simply delete the contained segments from the tree, as this frog(s) will never eat. If the new segment partially overlaps with another we reuse the solution used for the first issue (iteratively until we find no overlaps)
	- **mixed case:** the segment fully contain another frog and partially overlaps the next one, remove one and modify the other. 
	- mind that no other mixed case is possible: to partially overlaps a segment either you just do that or fully contain other segments first
- **Uneaten mosquitoes:** A mosquito may be left uneaten when it first lands if there are no frogs that can reach it. We store uneaten mosquitoes in another BST with their landing position as key. When a frog eats we check on this tree if the frog can eat other mosquitoes. This can be found with a successor query of the position of the frog that has eaten on the mosquitoes tree.

**Time Complexity:**
We have $n$ frogs and $m$ mosquitoes.

**Forcing no overlap in the starting segments** takes $O(n)$.
**Predecessor queries:** $O(n\cdot\log(n))$
**Changing dynamic segments:** $O((n+m)\cdot \log(n))$
- removing frogs costs $O(n\log(n))$ as in the worst case we might remove every frog
- modifying frogs costs $O(m\log(n))$ as in the worst case we modify one frog after every eaten mosquito, and there are $m$ mosquitoes
**Searching for uneaten mosquitoes** takes $O(m\cdot\log(m))$ time overall because we can search for a maximum of $m$ mosquitoes.

Then the **overall time complexity** is dominated by $O((n+m)\cdot \log(n))$ and $O(m\cdot \log(m))$. 
And we see that 
$$
\begin{align}
(n+m)\log(n) + m\log(m) \\ 
&\le (n+m)\log(n) + (n+m)\log(m) \\ 
&\le (n+m)\log(n+m) + (n+m)\log(n+m) \\
&= (n+m)(\log(n+m) + \log(n+m)) \\
&= (n+m)(2\log(n+m)) \\
&= O((n+m)\log(n+m))\ \ \square
\end{align}
$$
The above operations can be made as $n$ and $m$ are positive, and $\log$ is a positive 

**Space Complexity:** $O(n)$ to store the two trees.
# 17) Sliding Window Maximum
Given an array `a[0, n-1]` and an integer `k` find the maximum for each subarray, aka window, of `a` of size `k`.
## Solutions
### Brute Force, $O(n\cdot k)$
The trivial approach is to consider each of the `n-k+1` windows independently. 
For each window we compute its maximum by scanning it completely, which takes $O(k)$
We do that for each element of the array, hence $O(n\cdot k)$ 

**The problem in this approach is that we do not use the result of previous windows to compute the maximum of the current one.**
### Observations
To solve efficiently this problem we need a data structure that handle the next window while using the progress made in processing the previous one. 

**Notice that:** 
1) We can represent a window as a multiset $\mathcal{M}$ of size `k`. In this setting the result for the current window is simply the biggest element in $\mathcal{M}$
2) When we move to the next window **only two elements change:** the first element of the window exist and the a new one enters as the last. Said easy: **the window slides one position through the right.**

We then require a data structure capable of performing three operations on a (multi)set:
- insert a new element
- delete an arbitrary element
- retrieve the maximum element
### BST-Based Solution, $\Theta(n\cdot \log(k))$
A BBST supports the three operations in $\Theta(\log(|\mathcal{M}|))$, and it is optimal in the comparison model (lowest possible complexity in terms of comparison). 
We can then solve the problem in $\Theta(n\cdot \log(k))$ time:
- insert the first `k` elements of the array in the tree and store the maximum
- iterate until we have covered all the windows:
	- shift the window
		- remove the leftmost element of the window
		- insert the new rightmost element in the window
	- store the maximum
- return the maximums
### Heap-Based Solution, $\Theta(n\cdot \log(n))$ 
A heap is a tree-based data structure that satisfy the **heap property:**
- **max-heap:** for any given node `C`, if `P` is a parent of `C` than the key of `P` is $\geq$ than the key of `C`
- **min-heap:** the key of `P` is $\le$ than the key of `C`

The main operations of a heap are: 
- **insert**, $O(\log(n))$
	- insert the new element at the bottom of the heap (next available position)
	- **heapify-up**: swap the element with its parent until the heap property is satisfied again
- **remove min/max**, $O(\log(n))$
	- remove the root element, the mix or max of the tree
	- replace the root with the last element in the heap
	- **heapify-down**: swap the element with one of its children until the heap property is satisfied again
- **peek**, $O(1)$
	- return the maximum (or the minimum) without removing it 

The solution can be then found using a max-heap and scanning through the array left to right: 
- populate the max-heap with the first `k` elements, along with their respective positions
- as we move on to process the remaining elements one-by-one, we insert each current element with its position. We then request the heap to provide the current maximum: if it is paired with an index that is outside the window we discard it and ask again until we get it right

We insert elements with their respective index so that we can now if it is part of the window or not. 
If the index of an element in the tree is smaller than `i - (k-1)`, where `i` is the current iteration index, then the element is outside the window.

Using the above approach we have `n` insertions and at most `n` extractions of the maximum (window of size $1$). 
Since the maximum number present in the heap at any given time is up to `n`, each of these operations takes $O(\log(n))$ time. 
The overall complexity is $O(n\cdot \log(n))$.
### Optimal Solution: Sliding Window, $O(n)$
The BST solution can do more than what is requested, e.g. we can have the second largest element in the window.
**The fact that we can do much more than what is requested, it’s an important signal to think that a faster solution could exist.**

The best solution uses a **deque**, a double-ended queue, which supports constant time insertion, removal and access at the front and back of the queue.

The algorithm behaves as follows: 
1. Iterate through the array, one element at a time
2. While processing each element we maintain a deque `Q` that store only elements that are relevant for the current window. Specifically: `Q` stores only elements that are maximum candidates for the current and for future windows. 
3. At each step we do: 
	1. Remove elements from the front of `Q` that are outside the current window. This is done by checking if the indices of those elements are beyond the current window boundary
	2. Remove elements from the back of `Q` that are $\le$ to the current element. Those elements cannot be the maximum within the current or any future windows and we only keep elements that might become maximum elements for a window
	3. Append the index of the current element to the back of `Q`, as it could be a maximum element in the current or future windows
4. Once we have processed all the elements and maintained the deque as described, the maximum element for each window are stored in the front of the deque, collect these values and return them as result

```java
// q = [a, ..., z]
//      front    back
//      head     tail
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
#### Correctness
The correctness of the algorithm is based on two theorems: 
1) the deque is sorted in decreasing order
2) at any iteration the deque contains all and only the right leaders of the current window

**1) Theorem:** The elements in `Q` are sorted in decreasing order. 
**Proof:** we prove the theorem by induction on the number of iterations.
- **base case:** `Q` is empty, true
- **inductive case:** consider `Q` after `i` iterations. By inductive hypothesis `Q` is sorted. The current iteration `i+1` will remove elements only from the front and the back of the deque, hence no change in the order of elements. Then we insert the current element `a[i+1]` as the tail of `Q`, just after an element that is strictly greater than it (if any). Thus the queue remains sorted.

We now introduce the definition of right leaders of the window to show that the largest element within the window is at the top (head) of the queue. 
Given a window an element is called **right leader** if and only if it is larger than any other element of the window at its right.
As an example: the red elements are right leaders: 

![[Pasted image 20240219095000.png | center | 700]]

**2) Theorem:** At every iteration `Q` contains all and only the right leaders of the current window. 
**Proof:**
1) Any right leader cannot be removed from `Q` as all the subsequent elements are smaller than it. 
2) Any non-right leader will be removed as soon as the next right leader enters `Q`. 
3) Any element outside the window cannot be in `Q`. 
	1) By contradiction, let us assume that `Q` contains one such element, say 𝑎. Let 𝑟 be the largest right leader. On one hand, 𝑎 cannot be `<=` 𝑟, otherwise 𝑎 would be removed when inserting 𝑟 in `Q`. On the other hand, 𝑎 cannot be larger than 𝑟, otherwise, it would be in front of `Q` and removed by the first inner loop.

We derive correctness of the algorithm by combining the sortedness of `Q` with the fact that the largest leader is the element to report.
#### Time Complexity
We have a loop that is repeated `n` times. 
The cost of an iteration is dominated by the cost (and thus the number) of pops. 
However in certain situations we may pop all the elements in the deque. 
As far as we know there may be up to `n` elements in the deque and thus an iteration may cost $O(n)$, therefore the overall cost is $O(n^2)$.

The above analysis is **wrong**.
It is true that there may exist very costly iterations but they are amortized by many cheap ones.
The overall number of pop iterations cannot be larger than `n` as an element is not considered anymore by the algorithm as soon as it is removed from `Q`.
Each of them costs constant time, thus the complexity is $O(n)$.
# 18) Sweep Line
The Sweep Line Algorithm is an algorithmic paradigm used to solve a lot of problems in the computational geometry efficiently. 
**Sweep Line is used to solve problems on a line or on a plane.**

The idea is to use an imaginary line sweeping over the x-axis. 
As it progresses we maintain a running solution to the problem at hand. 
The solution is updated as the vertical line reaches key points where some event happen. 
The type of event tells us how to update the solution. 
# 19) Maximum Number of Overlapping Intervals
Consider a set of intervals $[s_i, e_i]$ on a line. 
We say that two intervals $[s_i, e_i]$ and $[s_j, e_j]$ overlaps if and only if their intersection is not empty. 
**The objective is to compute the maximum number of overlapping intervals.**** 

**Example:**
![[Pasted image 20240109094209.png | center | 550]]
We have a set of $10$ intervals and the maximum number of overlapping intervals is $5$, at positions $3$ and $4$ on the axis.
### Solution: Sweep Line, $O(n)$
**We apply the sweep line approach to the problem.**
The idea is to let the line sweep left to right and stop at the beginning or at the end of every interval: the endpoints of the intervals are the meaningful events.
We maintain a counter which keeps track of the number of intervals that are currently intersecting the sweep line, along with the maximum value of the counter. 
**For each point** we first add to the counter the number of intervals that begin at that point, then we subtract the number of intervals that ends at that point. 

**Observation:** the sweep line only touches points on the x-axis where an event occurs. 
This is important as the number of considered points, and thus the complexity, is proportional to the number of intervals, and not to the size of the x-axis.

**The algorithm behaves as follows:**
- create an array `axis` of pairs `(p, isStart)`
	- `p` is either $s_i$ or $e_i$ of an interval
	- `isStart` is true if `p` is the start of an interval (aka, a $s_i$), false otherwise
- sort `axis` by the first element of each pair
- set `counter = 0` and `max = 0`
- iterate over `axis` using the iteration variable `i`
	- if `axis[i]` is the start of an interval (`axis[i].isStart == true)
		- `counter++`
		- `max = Math.max(counter, max)`
	- otherwise `counter--`
- return `max`
# 20) Closest Set of Points
Consider a set of `n` points on a plane. 
The goal is to **find the closest pair of points** in the set. 
The distance between two points $(x_1, y_1)$ and $(x_2, y_2)$ is the Euclidian distance:
$$d((x_1, y_1), (x_2, y_2)) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$
A **brute force** approach would simply compute the distance between all possible pairs, resulting in $O(n^2)$. 
**We tackle the problem using Sweep Line.**
### Solution: Sweep Line, $O(n\cdot\log(n))$
We start by **sorting the points in increasing order of their x-coordinate.** 
We use $\delta$ to keep track of the current shortest distance seen so far. 
Initially $\delta$ is set to $+\infty$. 
Then we use a vertical sweep line to iterate through the points, attempting to improve $\delta$

Consider the point $p = (x,y)$ just reached by the sweep line. 
We can improve $\delta$ if the closest point **to the left** of $p$ is closer than $\delta$. 
If such point exists it must have a x-coordinate in the interval $[x - \delta, x]$ as it has to be to the left of $p$, and a y-coordinate in the interval $[y - \delta, y + \delta]$. 
**The following figure shows the rectangle within the point must lie:** 

![[Pasted image 20240219103626.png | center | 250]]

There can be **at most 6 points** within this rectangle. 
The 6 circles within the perimeter of this rectangle represents points that are at distance exactly $\delta$ from each other. 
**Four our purpose a slightly weaker result is enough, which states that the rectangle contains at most 8 points.**
To understand why lets consider the 8 squares in the figure above. 
Each of these squares, including its perimeter, can contain **at most one point**. 
Let's assume that a square contains two points: $q$ and $q'$.
The distance between $q$ and $q'$ is surely smaller than $\delta$. If the point $q'$ exists it would already been processed by the sweep line since it has a smaller x-coordinate than $p$. 
This is not possible: the value of $\delta$ would be smaller than its current value. 

**Let's use the above intuition to solve the problem.**
We build step-by-step a BBST with points sorted by their y-coordinates. 
When processing the point $p = (x, y)$ we iterate over points with y-coordinates in the interval $[y - \delta, y + \delta]$. 
**If** the current point $u$ retrieved from the BST has a x-coordinate smaller (which means farthest) than $x-\delta$  then we remove $u$ from the BST, and it will never be considered again. 
This is because the points are sorted by x on the axis: if the point $u$ is already at a greater distance than $\delta$ to the current point $p$ then the next point on the axis will be to an even greater distance.
**Else** we compute the distance of $u$ and $p$, and update $\delta$ if needed.
Every time we update $\delta$ we store $u$ and $p$
Before moving the sweep line to the next point we insert $p$ in the BST as it is may relevant for the next point in the axis (this $p$ could be the $u$ for the next $p$)
#### Complexity
What is the complexity of the algorithm? 
Let's consider the following costs: 
- Identifying the range of points with the required y-coordinates takes $\Theta(\log(n))$ $\ \spadesuit$
- Removing a point takes $\Theta(\log(n))$ 
- Iterating over the points in this range takes constant time per point
- How many points we do iterate over? There can be at most 8 points that have an x-coordinate greater or equal to $x - \delta$  and therefore survive. 
- There can be many points with smaller x-coordinates. However since each point is inserted and subsequently removed from the set (BST) of points at most once during the execution, the cost of dealing with all these points is at most $\Theta(n\cdot\log(n))$

$\spadesuit:$ 
1. **Start at the root node**: Begin traversal from the root of the BBST. 
2. **Recursively traverse the tree**:
    - If the current node's key falls within the given range, add it to the result
    - If the current node's key is smaller than the range, recursively search the right subtree
    - If the current node's key is larger than the range, recursively search the left subtree
3. **Continue until a base case is reached**:
    - If the current node is NULL, stop the traversal
    - If the current node's key is outside the given range, prune the subtree and stop traversal in that direction
4. **Return the result**:
    - Accumulate all nodes within the range while traversing the tree
    - Return the collected nodes as the result

The **overall complexity** is therefore $O(n\cdot \log(n))$. 
# 21) Check if all Integers are Covered
Consider an array `ranges` of ranges $[s_i, e_i]$ and two integers, `left` and `right`. 
Return true if each integer $x \in$ `[left, right]` is covered **by at least one** interval in  `ranges`. 
An integer $x$ is covered by an interval `ranges[i] =` $[s_i, e_i]$ if $s_i \le x \le e_i$. 
### Solutions
#### Brute Force, $O(n \cdot (\text{right} - \text{left}))$
Check that every integer `x` $\in$ `[left, right]` is covered by at least one interval in `ranges`
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
#### Sweep Line, $O(\max(\text{right - left}, n))$ 
We use sweep line and a map to solve the problem. 
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
# 22) Longest K-Good Segment
Consider an array of integers. 
Let's call a sequence of one or more consecutive elements in the array a segment (aka subarray)
We say that a segment is $k$-good if it contains no more than $k$ different values. 
Find any longest $k$-good segment.

**Observation:** We can use different data structures to solve this problem. 
The time complexities are: 
- $O(n\cdot\log(k))$ if we use a BST
- $O(n\cdot k)$ if we use an array of pairs
- $O(n)$ w.h.p with Hashing
### Solution, Hashing
The problem has some similarities with Sliding Window Maximum: there you wanted the max element for any window of size $k$, while here you want the longest window with at most $k$ distinct elements.
**We use two pointers to simulate a dynamic sliding window.** 
We use a hash-set to store elements 

```java
findMaxSegment(a[], k) 
	n = a.length
	left = 0, right = 0 // pointers current window
	maxLeft = 0, maxRight = 0 // pointers current longest k-good seg
	Set uniques

	while left < right
		if uniques.size() <= k 
			uniques.add(a[right])
			if uniques.size() <= k && right - left > maxRight - maxLeft
				maxRight = right
				maxLeft = left

			right++
		else
			uniques.remove(a[left])
			left++
		
	return (maxLeft, maxRight)
```

**Why it works?**
The key is in the first two ifs. 
If `uniques.size() <= k` we insert the current element `a[right]`. At this point we have two cases: 
1) `uniques.size()` has grown
	1) this means that `a[right]` is a never-seen-before element
	2) if `uniques.size() <= k` and `right - left >= maxRight - maxLeft` than this is a new longest k-good segment, and therefore we update `maxLeft` and `maxRight`
	3) if `uniques.size() > k` this is not a k-good segment and in the next iteration(s) we will remove the tail element `a[left]` until we return in k-good segment scenario
2) `uniques.size()` is the same as before
	1) this means that `a[right]` was seen before
	2) surely `uniques.size() <= k`
	3) if the segment is the longest so far we do as in the previous case, we update `maxLeft` and `maxRight`
# 23) Prefix Sums
The prefix sum, also known as cumulative sum, is used to solve a wide range of problems that involve querying cumulative information about a sequence of values or elements.

The essence of prefix sums lies in **transforming a given array of values into another array, where each element at a given index represents the cumulative sum of all preceding elements in the original array.**
### Applications of Prefix Sums
#### 24) Ilya and Queries
Consider a string $s = s_1s_2\dots s_n$ consisting only of characters $a$ and $b$. 
We need to answer $m$ queries. 
Each query $q(l,r)$ where $1 \le l < r \le n$ asks for the number of positions such that $s_i = s_{i+1}$. 

Consider the following example ($1$-indexing):
- $s = a\ a\ b\ b\ b\ a\ a\ b\ a$
- $q = (3,6) = 2$
	- the substring is $b\ b\ b\ a$
	- the positions 3, 4 and 6 make true $s_i = s_{i+1}$, but $s_6 = s_{6+1}$ do not count since $6+1 = 7 \not\in [3,6]$ 

**To solve the problem we can use the prefix sums.** 
We can compute the binary array $B[1,n]$ such that
$$B[i] = \begin{cases}1\ \text{if}\ s_i = s_{i+1} \\ 0\ \text{otherwise}\end{cases}$$
At this point the answer to the query $q(l,r)$ is $\Sigma_{i=l}^r\ B[i]$. 
Now every query can be solved in constant time **by computing the prefix sum array of the vector $B$.** 
Let's call $F$ the prefix sum array of $B$: we have that $q(l,r) = F[r] - F[l]$.
#### 25) Little Girl and Maximum
We are given an array $A[1, n]$ and a set $Q$ of queries. Each query is a range sum, the query $[i,j]$ returns the sum of the elements in $A[i\dots j]$. 
The goal is to permute the elements in $A$ in order to maximize **the sum of the results of the queries** in $Q$. 

The key is to observe that if we want to maximize the sum we have to assign the largest values to the most frequently accessed entries. 
Hence the solution consists in sorting $A$ in descending order, then sort in descending order the frequency of access of every element and pair the two.
Once we have computed the frequencies the solution takes $\Theta(n\cdot\log(n))$.

**We are left with the problem of computing the frequencies.**
We want to compute an array $F[1,n]$ where $F[i]$  is the number of times the index $i$
belongs to a query of $Q$.
Computing this array by updating every entry of $F$ for every query takes $O(n\cdot |Q|)$, and thus is unfeasible. 
**We can solve this problem using the prefix sum.** 

Let's consider an array $U[1,n]$ such that its prefix sums are equal to our target array $F$. 
We need to only modify two entries of $U$ to account for a query of $Q$. 
This is done as follows: 
- set all the entries of $U$ to $0$
- for every query $q = [l,r] \in Q$ we do
	- $U[l]$++
	- $U[r+1]$-- 
- the prefix sum of $U$ is 
	- unchanged for the indexes less than $l$
	- increased by one for the indexes in $[l,r]$
	- unchanged for the indexes greater than $r$

Therefore **the prefix sum of $U$ up to $i$ is equal to $F[i]$.** 
This algorithm takes $O(|Q|+n)$ time: we have $|Q|$ queries to go through, plus $n$, which is the time needed to build $U$ and its prefix sum array. 

**The code clarifies the algorithm:**
```java
littleGirlMax(a[], q[]) 
	n = a.length()
	
	u[n] // compute U
	for (l, r) in q
		u[l]++
		if r + 1 < n 
			u[r+1]--

	f[n] // compute the prefixSum of U, aka F
	pS = 0
	for i = 0 to n
		pS += u[i]
		f[i] = pS

	// sort the array and the frequencies in decreasing order
	a.sortDecreasing()
	f.sortDecreasing()

	// return the maximized sum of the queries
	res = 0
	for i = 0 to n
		res += a[i] * f[i]

	return res
```
#### 26) Number of Ways 
Given an array $A[1,n]$ count the number of times the array can be divided into three subarrays that have the same sum. 
Formally we need to find the number of pairs of indices $i, j$ (with $2 \le i < j \le n-1$) such that: $$\Sigma_{k=1}^{i-1} A[k] = \Sigma_{k=i}^{j} A[k] = \Sigma_{k = j+1}^{n} A[k]$$
Let's say that the sum of the whole array is $S$. 
If $S$ cannot be divided by $3$ we immediately return $0$. 

Then we compute an array $C$ that stores, at position $i$, the number of suffixes of the suffix $A[i\dots n]$ that sums up to $S/3$. 
$C$ starts with every entry set to $0$, then: 
```java
if A[n] == S/3
	C[n] = 1

// iterate through A[] backwards to fill up C
suffixSum = A[n]
for i = n-1 to 1
	suffixSum += A[i]
	if suffixSum == S/3 
		C[i] = C[i+1] + 1
	else 
		C[i] = C[i+1]
```

At this point we scan $A$ left to right to compute its prefix sum. 
Every time the prefix sum at position $i$ is $S/3$ we add $C[i+2]$ to the result. 
This is because the part $A[1\dots i]$ sums up $S/3$ and can be combined with any pair of parts of $A[i+1\dots n]$ that also sums up to $S/3$. 
Since the values in $A[i+1\dots n]$ sums to $2\frac{S}{3}$, the number of such pairs is the number of suffixes that sums to $S/3$ in $A[i+2\dots n]$. 

**Clarification:** $C[i+2]$ because we want **three partitions**, if you were to use $C[i+1]$  you could use the whole remaining $A[i+1\dots n]$ whose elements sums to $2\frac{S}{3}$ , and you would get only two partitions.

**Said Easy:**
- say that you have $C$
- the prefix sum array of $A$ is represented with $P$
- iterate $P$ using the index $i$
	- is $P[i] == \frac{S}{3}$?
		- then we have found one of the three parts that summed give $S$
		- **the number of suffixes of the suffix** $A[i+2,n]$ are the third part
			- if the whole $A[i+2,n]$ sum up to $S/3$ than we have found the three parts: $A[1,i], A[i+1,i+1], A[i+2, n]$ 
			- counting the other suffixes makes it work: we are counting the indexes $(i,j)$ such that we can decompose $A$ in three parts, counting the $j$s is as counting the $i$s
# 27) Contiguous Subarray Sum
Given an integer array `nums` and an integer `k`, return true if `nums` has a **good subarray** or false otherwise. 
A **good subarray** is a subarray where: 
- its length is at least 2
- the sum of the elements of the subarray is a multiple of `k`
### Solutions
#### Brute Force, $O(n^2)$
From each element `i` in the array we compute every possible subarray starting in `i` and check that the sum of the elements of the subarray is a multiple of `k`, storing the maximum length so far.
#### Optimal Solution: Prefix Sum, $O(n)$
The solution is based on a math property: 
1) any two prefix sums that are not next to each other and have the the same value mod $k$ implies that there is a good subarray (the subarray between the last element of the first subarray and the first element of the second one)
2) a prefix sum with mod $k = 0$ that is not the first number will, yield a valid subarray. 

**The algorithm then is as follows:** 
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

		// if modulo is 0 and not the first ok
		if modulo == 0 && i != 0 
			return true

		// first time we see this modulo
		if modsToIndices.contains(modulo) == false
			modsToIndices.insert(modulo, i)
		else 
			// this modulo has been seen
			previousIndex = modsToIndices.get(modulo);
			// not the previous
			if previousIndex < i - 1
				return true

	return false
```
# 28) Fenwick Trees
The Fenwick Tree, aka Binary Indexed Tree, is a data structure that maintains the prefix sum of a **dynamic array.**
**With this data structure we can update values in the original array and still answer prefix sum queries, both in logarithmic time.** 

Consider the following problem: we have an array $A[1\dots n]$ of integers and we would like to support the following operations: 
- `sum(i)` returns the sum of elements in $A[1\dots i]$
- `add(i,v)` adds the value `v` to the entry $A[i]$

To solve the above problem we can use two trivial solutions: 
- Store $A$ as it is. This way `sum(i)` is solved by scanning the array in  $\Theta(n)$ time, and `add(i,v)` costs $O(1)$
- Store the prefix sum of $A$. this way `sum(i)` is solved in $O(1)$, and `add(i,v)` is solved by modifying all the entries from $i$ to $n$ in $\Theta(n)$ time. 

Both solutions are unsatisfactory in the tradeoff between the two operations.
**The Fenwick Tree provide a better tradeoff for the problem** as it efficiently handles these queries in $\Theta(\log(n))$ while using linear space. 
In fact the Fenwick Tree is an **implicit data structure**, which means it requires only $O(1)$ additional space to the space needed to store the input data. 

Let's introduce the Fenwick Tree with a running example. 
Consider the following 1-indexed array: 

![[Pasted image 20240110113032.png | center | 350]]

and consider a simpler version of the problem: let's solve `sum` queries only for positions that are powers of $2$, hence $1,2,4,8$. 
The solution of this simplified problem will be **the first level of the fenwick tree:**

![[Pasted image 20240110114748.png | center | 300]]
We notice that: 
1) there is a dummy root node $0$ 
2) **over every node** there is the corresponding index in $A$
3) **the value of every node** is the prefix sum up to that index
4) **under every node** we have a range of integers: those are the positions covered by that node

With this solution we have that: 
- `sum(i)`, simply access the node `i`, provided that `i` is a power of $2$ 
- `add(i,v)`, add `v` to all the nodes that covers ranges that include the position `i` 
	- **example:** `add(3,10)`, we need to add $10$ too the nodes $4$ and $8$ since they cover the ranges $[1,4]$ and $[1,8]$, and both of those ranges include $3$
	- **generally:** `add(i,v)`,  find the smallest power of $2$ that is bigger than `i`, this is the end of the range covered by the first node you have to update. Say that the corresponding node is indexed with `j`. We have to add `v` to the nodes $j, j^2, j^4, \dots$ 

We observe that `sum` takes constant time and `add` takes $\Theta(\log(n))$ time. 
This is promising: we have to extend the solution to positions that are not power of $2$. 
Formally we are not supporting queries positions that falls within the ranges of two consecutive powers of $2$, e.g., `sum(6)` is not supported. 
Fortunately **enabling queries for such positions is a smaller instance of our problem,** we can apply the same strategy by adding a new level to our tree. 
The children of a node stores partial sums **starting from the next element:**

![[Pasted image 20240110121552.png | center | 300]]

The emphasis is on **starting.** The node indexed with $6$ covers the range $[5,6]$ and its value is the prefix sum of the subarray $A[5,6]$, which is $7 - 3 = 4$. 

**Lemma:** our two-level tree can handle `sum(i)` queries for positions that are either a power of $2$ or a sum of two powers of $2$.
**Proof:** 
- consider a position $i$ expressed as $2^{k'} + 2^k$, where $k' > k$
- we can decompose the range $[1, i]$ into two subranges: 
	- $[1, 2^{k'}]$
	- $[2^{k'}+1, 2^{k'} + 2^k]$
- both of these two ranges are covered by the nodes of our tree
	- the range $[1, 2^{k'}]$ is covered by a node in the first level
	- the range $[2^{k'}+1, 2^{k'} + 2^k]$ is covered by a node by the node `i` in the second level

**Example:** how to compute `sum(5)`
- $i = 5$
	- $k' = 2, k = 0$
	- $2^{k'} + 2^k = 4 + 1 = 5$
- the range of the query, `[1,5]` is then divided in two subranges
	- $[1,4]$ covered by the node $4$
	- $[5,5]$ covered by the node $5$
- the result $16$ is obtained by summing the values of the nodes

**Which positions are still not supported?**
The positions that are neither a power of $2$ or the sum of two powers of $2$. 
In our setting, we can't compute `sum(7)`. 
As before we just add a new level to our tree, so we can support positions that are the sum of three powers of $2$. 

![[Pasted image 20240110141449.png | center | 300]]

**We are done**, the above is the complete Fenwick Tree of the array $A$. 
**We can make some observations:** 
- While we have represented our solution as a tree, it can also be represented as an array of size $n+1$, as shown in the figure above. 
- We no longer require the original array $A$ because any of its entries $A[i]$ can be obtained by doing `sum(i) - sum(i-1)`. This is why the Fenwick Tree is **an implicit data structure**
- Let be $h = \lceil \log(n)+1\rceil$, which is the length of the binary representation of any position in the range $[1,n]$. Since any position can be expressed as the sum of at most $h$ powers of $2$, the tree as no more than $h$ levels. 
### Answering a `sum(i)` Query
This query involves beginning at node $i$ and traversing up the tree to reach the node $0$.
Thus `sum(i)` takes time proportional to the height of the tree, resulting in a time complexity of $\Theta(\log(n))$. 

Let's consider `sum(7)`. 
We start at node with index $7$ and move to its parent (the node with index $6$) then to its grandparent (the node with index $4$), and finally stop at its great-grandparent (the dummy root $0$), summing their values along the way. 
This works because the ranges of these nodes ($[1,4],[5,6],[7,7]$) collectively cover the queried range $[1,7]$. 

Answering a `sum` query is straightforward **if we are allowed to store the tree's structure.**
However a significant part of the Fenwick tree's elegance lies in the fact that storing the tree is not actually necessary.
The point is that we can **efficiently navigate from a node to its parent using a bit trick**, which is why Fenwick Trees are also called **Binary Indexed Trees.**

**Bit Trick: How to Compute the Parent of a Node**
We want to compute the parent of a node and we want to do it quickly, without representing the structure of the tree. 
Let's consider the binary representation of the indexes involved in the query `sum(7)`:
![[Pasted image 20240110150658.png | center ]]
**Theorem:** the binary representation of a node's parent can be obtained by removing the trailing one (i.e., the rightmost bit set to $1$) from the binary representation of the node itself. 
### Performing an `add` 
Now let's consider the `add(i,v)` operation. 

We need to add the value `v` to each node whose range includes the position `i`. 
Additionally the right siblings of the node `i` also include the position `i` in their ranges: siblings share the same starting positions and right siblings have increased sizes.
The right siblings of the parent of `i`, the right siblings of its grandparent and so on can also contain the position `i`. 
**It might seem like we have to modify a large number of nodes** but in reality the number of nodes to be modified is at most $\log(n)$. 
This is because **each time we move from a node to its right sibling or to the right sibling of its parent, the size of the covered range at least doubles.**
A range cannot double more than $\log(n)$ times. 

**Example:** if we want to perform `add(5,x)` we just need to modify the nodes in red

![[Pasted image 20240110152607.png | center | 300]]

**Bit Trick to Compute the Sibling of a Node**
Continuing with the example above, starting with `i = 5`, the next node to modify is its right sibling, the node $6$. 
Their binary representation is 
![[Pasted image 20240110152844.png| center]]

We see that if we isolate the trailing one in the binary representation of $5$ (isolate the trailing one = `0001`) and add it to the representation of $5$ itself we obtain the binary representation of $6$. 

**The binary representation of a node and its sibling matches, expect for the position of the trailing one.**
When we move from a node to its right sibling this trailing one shift one position to the left. 
Adding this trailing one to a node accomplishes the required shift.

The **time complexity** is once again $O(\log(n))$ as we cannot have more than $\log(n)$ shifts.
## Applications of Fenwick Trees
### 29) Range Sums
We can use Fenwick Trees to perform range sums, that is the cumulative sums of any range $[l, r]$ in the array. This is done by simply returning `sum(r) - sum(l)`.
### 30) Counting Inversions in an Array
We are given an array $A[1\dots n]$ of $n$ positive integers. 
If $1 \le i < j \le n$ and $A[i] > A[j]$ then we say that the pair $(i,j)$ is an **inversion** of $A$. 
The goal is to count the number of inversions of $A$. 

We assume that the largest integer $M$ in the array is in the order of $O(n)$. 
This assumption is important because we are using a Fenwick Tree of size $M$, and building such a tree takes $\Theta(M)$ space and time. 
If this assumption does not hold then we sort $A$ and replace each element with its **rank** in the sorted array, where the rank of an element is its position in the sorted array.  
If elements are equal then they have the same rank.  
**Main point:** rank goes from $1$ to $n$.

To solve the problem we use a Fenwick Tree `FT` on an array $B$ with $M$ elements, initially all set to $0$. 
`FT` is used to **keep track of elements that we have already processed.**
We initialize a counter variable `res` and scan the array $A$ from left to right
When we process $A[i]$ we do:
- `counter += FT.range_sum(A[i]+1, n)`
	- this counts the number of elements in the array that are greater than $A[i]$
- `FT.add(A[i], 1)`

Since both `range_sum` and `add` cost $\Theta(\log(n))$ the **overall running time** is $\Theta(n\cdot \log(n))$

**Said Easy:**
Let's see it in action in the first two iterations. 
We scan the array using $i$. 
When $i = 0$ we have a clean `FT`, the array $B$ is still initialized to all zeroes.
We process $A[i]$:
- `counter += FT.range_sum(A[i]+1, n)`, which is zero as we have just started
- `FT.add(A[i], 1)`, which "means" $B[A[i]] = 1$ in this context

Now we go to the next element, $A[i+1]$. Let's say that $j = i+1$ for clarity.
We process $A[j]$: 
- `counter += FT.range_sum(A[j]+1, n)`
	- `FT.range_sum(A[j]+1)` counts the number of elements bigger than $A[j]$ that we have already processed
	- If $A[i]$ $>$ $A[j]$ then the $1$ we added when we processed $A[i]$ will have an impact on the range sum ($A[j]+1$, $n$), aka the sum of $B[A[j]+1\dots n]$
		- **fenwick tree are used to efficiently compute prefix sums!**
	- and in fact if $A[i]$ is bigger than $A[j]$ than $(i,j)$ is an inversion
# 31) Update the Array
Consider an array $A$ containing $n$ elements that are currently all zeroes. 
Consider two operations to support: 
- `access(i)`: returns $A[i]$
- `range_update(l,r,v)`: update the entries in $A[l\dots r]$ by adding `v`. 

You have $u$ updates to perform. Afterwards there will be $q$ queries, each containing an index corresponding the element of the array you have to print. 
### Solution: Fenwick Trees
We use a Fenwick Tree to efficiently solve the problem: 
- from $A$ we build a Fenwick Tree `FT` of length $n$ (remember, $A$ is initially all zeroes)
- the operation `access(i)` is a wrapper of the query `sum` we have seen so far: $\clubsuit$
- the operation `range_update(l,r,v)` exploit the operation `add(i,v)` 
	- check limit cases: $l < r$, $r < n$, ...
	- `add(l,v)`: add `v` to each node whose range include the position `l` in `FT`
	- `add(r+1, -v)`: subtract `v` to each node whose range includes `r` in `FT` 
	- we have added and subtracted the same quantity `v` in `FT`, this means that the prefix sums are coherent and the elements in `[l,r]` are increased by `v`

**Clarification:** $\clubsuit$
Better said: `sum(i)` in the current setting results in the `access` operation: `sum(i) == access(i)`
**Example:** Consider the empty FT represented as an array $[0,0,0,0]$
- perform `range_update(1,2,2)` 
	- Fenwick Tree representation: `[0,2,0,-2]`
	- array "represented" by the FT: `[0,2,2,0]`
- perform `access(2)`; we want the third element of the array
	- `sum(2)` is the prefix sum of the elements `[0,1,2]` of the FT
		- `sum(2)` = $0 + 2 + 0 = 2$ 
	- the element at position $2$ of the array that the FT represents is $2$
### 32) Dynamic Prefix-Sums with Range Update
**Note:** we use 1-indexing.
The range update of the previous problem is paired with the `access` operation. 
Now we want to support the following operations: 
- `range_update(l,r,v)`, same as before
- `sum(i)`, returns $\Sigma_{k=i}^n A[k]$: **we want to support the real prefix sum** operation on the fenwick tree with range updates

We notice that :
1) `add(i,v)` is a special case of range update:
	1) we can solve `range_update(l,r,v)` with `r-l+1` call to `add`, but this would cost $O((r-l+1)\cdot \log(n))$ while we want to achieve $\Theta(\log(n))$
2) `access(i)` will be supported by performing `sum(i) - sum(i-1)`

In our initial approach we follow a similar strategy to the one used in the "Update the Array" problem. 
For a `range_update(l,r,v)` we modify the fenwick tree by adding `v` in position `l` and subtracting `v` in position `r+1`. 
When querying `sum(i)` we multiply the result by `i`. 
**This approach is flawed:**
- consider the starting array $[0,0,0,0,0,0]$ on which we build the Fenwick Tree `FT`
- perform `range_update(2,4,2)`
	- the "ideal" array would be `[0,2,2,2,0,0]`
	- `FT` is `[0,2,0,0,-2,0]` as a result of the range update query
- perform now `sum(3)`
	- the prefix sum should be $4 = 0 + 2 + 2$
	- as before in this setting we have that `sum(i) == access(i)`, and in fact `sum(3)` on `FT` gives us `2`, hence `sum(3) = 2 * 3 = 6`, **which is wrong**

**Formally:** consider `range_update(l,r,v)` on a brand new FT (**all zeroes**). 
**The results returned by our method are the following:** 
- if $1 \le i < l$, `sum(i)` is $0$
- if $l \le i < r$, `sum(i)` is $v\cdot i = v(l -1) + v(i - l + 1)$ 
- if $r < i$, `sum(i)` is $(v-v)\cdot i = 0$ 

The **correct result** of a `sum(i)` **after** the `range_update` is: 
- if $1 \le i < l$, `sum(i)` is $0$
- if $l \le i < r$, `sum(i)` is $v\cdot (i-l + 1)$
	- `v` times the number of entries modified by the range update
- if $r \le i$, `sum(i)` is $v\cdot (r-l+1)$ 
	- `v` times the length of the range $[l,r]$

**In words:** our implementation reports the correct result for $1 \le i \le l$ but introduces errors in the other cases. 
Specifically, for $l \le i \le r$ it introduces an additional term $v(l-1)$, while it reports $0$ instead of the correct value $v(r-l+1)$ in the latter case. 

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
# 33) Segment Trees
**A Segment Tree is a data structure that stores information about array intervals as a tree.**
This allows answering **range queries** over an array efficiently, while still being flexible enough to **allow quick modification of the array.**
**Range queries are more general than range sums:** we can find the sum of consecutive array elements $A[l\dots r]$ or find the minimum element in a segment in $O(\log(n))$ time. 
Between answering such queries we can modify the elements by replacing one element of the array, or even change the elements of a whole subsegment, e.g., assign all elements in $A[l,r]$ to any value.

We consider the simplest form of Segment Trees and we want to answer sum queries efficiently. 
Formally: given an array $A[0\dots n-1]$ our segment tree must be able to perform the following operations in $O(\log(n))$ time: 
- find the sum of the elements in the range $[l,r]$: $\Sigma_{i = l}^rA[i]$
- change values of elements in the array: $A[i] = x$
### Structure of the Segment Tree
We can take a divide-and-conquer approach when it comes to array segments. 
We compute and store the sum of elements of the whole array, i.e., the sum of the segment $A[0\dots n-1]$. 
Then we split the array into two halves $A[0\dots n/2 -1]$ and $A[n/2 \dots n-1]$, compute the sum of each and store it. Then we split again, and again, until we reach segment size $1$. 
We can view this segment as forming a binary tree: the root is the whole array segment, and each vertex (except leaves) have exactly two children. 
**Example:** consider the array $A = [1,3,-2,8,-7]$
![[Pasted image 20240112095556.png | center | 450]]
**We can see that the segment tree only requires a linear number of vertices.** 
The first level contains a single node (the root), the second level will contains two nodes, the second will have four nodes and so on, until we reach $n$ (we will have $n$ leaves as we have $n$ elements in the array)
Thus, the number of vertices in the worst case can be estimated by the sum 
$$1 + 2 + 4 + \dots + 2^{\lceil\log(n)\rceil+1} < 4n$$
**The height of a segment tree** is $O(\log(n))$ since when we are going down from the root the size of the segments decrease (approximately) by half. 
### Construction of the Segment Tree
Before constructing the segment tree we need to decide
1) **the value that get stored in each node:** in a sum segment tree a node will store the sum of the elements in that segment
2) **the merge operation that combine two siblings:** in a sum segment tree the two nodes corresponding to the ranges $A[l_1\dots r_1]$ and $A[l_2\dots r_2]$ will be merged into a node that covers the range $A[l_1\dots r_2]$

Note that a vertex is a leaf if its corresponding segment covers only one element of the original array. 

**To build a segment tree** we start at the bottom level, the leaves, and assign them their respective values. On the basis of these values we can compute the values of the previous levels, using the merge function, until we reach the root. 
It convenient to describe this operation recursively in the other directions: **from the root to the leaves.**
In other terms: **the construction is a post-visit, where the "visit" phase is the assignment of the value of the current node:**
- compute the left child by recurring on the left subtree
- compute the right child by recurring on the right subtree
- merge the segments and result in the current node

The **time complexity** of the construction is $O(n)$, assuming that the merge operation costs $O(1)$ as in the case of the sum segment trees.
### Range Sum Queries
Say that we receive two integers, $l$ and $r$, and we want to compute the sum of the elements in the segment $A[l\dots r]$ in $O(\log(n))$ time.
We can achieve this by traversing the tree and use the precomputed sums of the segments. 

Assume that we are currently in a node that covers the segment $A[tl\dots tr]$. 
There are three possible cases: 
1) The segment $A[l\dots r]$ is equal to the corresponding segment, aka $l = tl \land r = tr$. We simply return the value stored in the current node. 
2) The queried segment is fully contained in the segment of either the left or right child of the current node. We simply recur over that child.
3) The queried segment intersect both the left and right children of the current node. Then we recur on both children:
	1) go to the left child and compute a partial answer for this vertex: the sum of values of the intersection 
	2) go to the right child and do the same
	3) combine the results 

In other terms: **processing a sum query is done by a recursive function that calls itself once on either the left or the right child without changing the query boundaries, or calls itself twice, once for child, splitting the query in two subqueries.**
The recursion ends whenever the boundaries of the current query segment coincides with the boundaries of the segment of the current vertex.  
In that case the answer will be the precomputed value of the sum of this segment, which is stored in the tree.

**Example:** consider $A = [1,3,-2,8,-7]$ and the query sum $(2,4)$. 
![[Pasted image 20240112103329.png | center | 400]]
**We now have to show that we can compute sum queries in O(log(n)).** 
**Theorem:** for each level we only visit no more than four nodes, and since the height of the three is $O(\log(n))$ we get that a sum query costs at most $O(4\cdot \log(n)) = O(\log(n))$. 
**Proof:** we prove the theorem by induction: 
- **base case:** at the first level we visit only one vertex, the root.
- **inductive case:** 
	- let's say that we are at an arbitrary level of the tree
	- by induction hypothesis we visit at most $4$ nodes of the current level
	- if in the current level we visit only two nodes then we will visit at most four nodes in the next level: each vertex can trigger **at most** two recursive calls
	- let's then assume that we visit $3$ or $4$ nodes in the current level: we analyze carefully the vertices in the "middle"
	- since the sum query asks for the sum of a **contiguous** subarray we know that the segments corresponding to the nodes "in the middle" will be completely covered by the segment of our query
	- **the middle nodes will not make any recursive calls**
	- only the leftmost and rightmost nodes might make recursive calls on their
	- hence in the next level we will visit at most $4$ nodes $\square$ 

**A graphical intuition of the above proof:**
![[Pasted image 20240229164807.png | center | 500]]
### Update Queries
Now we want to modify a specific element in the array, as in $A[i] = x$. 
This query is easier than the sum. Each level of a segment tree forms a partition of the array, therefore an element $A[i]$ contributes only to one segment for each level. 
Thus only $O(\log(n))$ nodes need to be updated. 

**We can perform this query by employing a recursive function.** 
The function receive the current tree vertex and recursively calls on the child that contains $A[i]$, and then recomputes its sum value, similar to how it is done in the build method. 

**Example:** consider $A = [1,3,-2,8,-7]$ and $A[2] = 3$:
![[Pasted image 20240112112424.png | center | 400]]
### 34) Range Update and Lazy Propagation
Segments Trees also allows applying modification queries to an entire segment of contiguous elements with the same time cost: $O(\log(n))$. 
When we need to update an interval we will update a node and mark the child that needs to be updated, but actually update it later, only when needed.
To do so we can add to every node a field that marks if the current node has a pending update or not.
Then, when we perform another range update or a sum query, if nodes with a pending update are involved, we first perform the updates and then solve the query. 

**In other words:** when there are many updates and updates are done on a range, we can postpone some updates (avoid recursive calls in update) and do those updates only when required. 
Remember that a node in segment tree stores or represents result of a query for a range of indexes: if this node’s range lies within the update operation range, **then all descendants of the node must also be updated.**

**Example:**
![[Pasted image 20240212165129.png | center | 500]]
Consider the node with value $27$ in the above tree. That node stores the sum $A[3\dots 5]$. 
If our update query is for the range $(3,5)$ then we need to update that node and all of its descendants. 
With lazy propagation **we only update that node and postpone the updates on its descendants** by storing this update information: 
- Store it in a apposite filed of the node, as said before
- Store it in separate nodes called lazy nodes. 
	- create an array `lazy[]`, which represents lazy nodes, all initialized to zeroes. 
	- if `lazy[i] = 0` indicates that there are no pending updates on the node `i`. 
	- if `lazy[i] != 0` than this amount needs to be added to the node `i` in the segment tree before making any query that involve the node.
### 35) Persistency
A **persistent data structure is a data structure that remembers its previous state for each modification. 
This allows to access any version of this data structure that interests us and execute a query on it.**

We can efficiently create **persistent segment trees.** 
In fact any change request in the segment tree leads to a change of only $O(\log(n))$ vertices along the path form the root. 
If we store the segment tree using pointers then to perform the modification query we simply need to create new vertices instead of actually changing the available ones. 
Vertices that are not affected by the modification can be reused, just use pointers to refer to them.
**For each modification of the segment tree we will receive a new root vertex.**
To quickly jump between versions we need to store the roots in an array, then to use a specific version of the segment tree we simply call the query on the right root vertex. 

**The following image gives an idea of how we can make a persistent segment tree:**
![[Pasted image 20240229171701.png | center | 450]]
# 36) Nested Segments
We are given $n$ segments $[l_1, r_1]\dots [l_n, r_n]$ on a line. 
There are no ends of some segments that coincide.
For each segment find the number of segments it contains. 

**In other words:** for each segment $i$ we want to count the number of segments $j$ such that the following condition holds:
$$l_i < l_j \land r_j < r_i$$
### Solutions
#### Fenwick Tree Solution
**This solution involves a fenwick tree and the sweep line approach.** 

Let's build an array `events` where every entry is of the form $[l_i, r_i]$. 
Then sort `events` by the first component of every entry, $l_i$. 
At this point we build a fenwick tree with size $(2n+1)$ and iterate through `events`. 
For each event $[l_i, r_i]$ we add $1$ in position $r_i$ of the fenwick tree: `ft.add(r_i, 1)`. 

Now we scan the events again: when we process the event $[l_i, r_i]$ we observe that **the segments already processed are only the ones that starts before the current one, as they are sorted by their starting points.**
To find the solution for the current segment (aka the number of segments contained in the current segment) **we need to know the number of the segments that starts after the current one that also ends before the current one,** before $r_i$. 
This is computed with a query `sum(r_i - 1)` on the fenwick tree. 
Then we perform `ft.add(r_i, -1)` to subtract $1$ from $r_i$ in the ft.

Why `sum(r_i - 1)` is the number of segments contained by $[l_i, r_i]$? 
Because all the segments that starts before $l_i$ have already been processed and their right endpoints have been removed from the fenwick tree: we subtracted $1$ to the position $r_i$. ($\clubsuit$)
In other words: the only segments that have $1$ in position $r$ are the ones that starts after
the current segment.
Thus `sum(r_i - 1)` is the number of segments that **starts after** $l_i$ and **end before** $r_i$. 

$\clubsuit$: After computing the solution for the current segment we subtract $1$ to the position $r_i$ to remove the contribution of the right endpoint of the current segment in the next queries. 
This is why segments that starts before the current one but overlaps with it are not counted. 

**The following snippets clarifies the algorithm:**
```java
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
#### Segment Tree Solution
**Let's now solve the problem using a segment tree and sweep line.**

Given an input array `segments` of ranges $[s_i,e_i]$ we compute the maximum endpoint of the segments and we call it $n$. 
We then create a segment tree based on an array of size $n$ that is initialized with all zeroes. 
At this point we create an array `axis` that stores triples, where:
- the first element store an endpoint of a segment, either a $s_i$ or a $e_i$
- the second element is the index of the segment of the endpoint, $i$
- the third is a boolean value `isStart` which is `true` if the endpoint of this triple is an $s_i$, false otherwise

Now we sort axis by the first component, the segment endpoints.
Finally we sweep over the axis:
- for every element $i$ in the axis
	- **if** this element is the end of a segment (`isStart == false`)
		- then the number of segments contained in the segment $[s_i, e_i]$ is given by the range sum on $[s_i,e_i]$ on the segment tree
		- increase by one the start of the segment in the ST: `st.add(s_i, 1)`
	- **else** skip

**Why it works?**
When we find the end of the segment $i$, namely $e_i$, we do the range sum of $[s_i,e_i]$ in the segment tree to get the number of segments contained by the segment $i$. 
Then we increase by $1$ the segment $[s_i,s_i]$ (aka `st.add(s_i,1)`) in the segment tree.
This works because we increase by one the start $s_i$ only when we find the end $e_i$. 
**We add 1 only when we find the end of a segment.** 
The range sum on $[s_i,e_i]$ **will count only segments that starts after the current segment, as segments are sorted by their endpoints, and that have already been closed: if they weren't closed they would not be counted as they would be zero, we wouldn't have done the add by one yet.**
# 37) (Dynamic) Range Minimum Queries (RMQ)
Given $A[1\dots n]$ we want to provide the following two operations: 
- `add(i,v)`: $A[i]=A[i]+v$
- `rmq(l,r)`: $\text{argmin}(A[l\dots r])$, in other terms the **position of the minimum element** of $A$ in the range $[l,r]$
### Solution
We use a segment tree to solve the problem. 
The range $[l,r]$ will be covered by some segments of the segment tree. If we use a segment tree that stores the minimum element (and its position) of each segment and combine the answers we can derive the answer of the query.

The approach is identical to the one seen for computing the range sums. 
We simply modify the construction of the segment tree so that every node stores the minimum element and its position.
Then we perform a query similar to the range sum that visit nodes and use the information of the visited nodes to build the answer
# 38) RMQ with Occurrences
Given $A[1\dots n]$ we want to provide the following two operations: 
- `add(i,v)`: $A[i]=A[i]+v$
- `rmq(l,r)`: $\min(A[l\dots r])$ with the number of its occurrences
### Solution
We use a segment tree to solve the problem. 
As in the previous problem (and as always with STs) we **store the information on the nodes.**
You build again the tree by saving the minimum and the number of occurrences from the ground up. 
The minimum of a leaf (a segment of length $1$) will be the element itself and the number of its occurrences will be $1$. 
Going up we merge the info of the children. If an element is smaller than the other than the info stored in that node will equal to the info that stores the smaller element. 
If the minimums stored in the two children are equal then we take the same element as the minimum and sum the number of occurrences.

As before we simply modify the construction method of the tree, and the query behaves as the range sum query.
# 39) Shortest Prefix Sum
Consider an array $A[1\dots n]$ with non-negative values. 
We want to provide the following three operations: 
- `add(i,v)`, same as before
- `range_sum(l,r)`, same as before
- `search(s)`: report the position $i$ of the shortest prefix of $A$ such that $\Sigma_{k=1}^i\ A[k] \ge s$
### Trivial Solution, $O(\log^2(n))$ 
Prefix Sums are monotone (we have non-negative values), non-decreasing. In other words the prefix sums are sorted in increasing order. 
We can **binary search the answer.** 
We build a segment tree to efficiently answer range sum queries (which include prefix sums, as a prefix sum is a range sum with left endpoint equal to $0$).
The range of possible answers is $[0, n-1]$, where $n$ is the length of the array $A$. 
At this point we binary search the answer: we get to the middle element `mid` of the range of possible answers and perform a query `range_sum(0, mid)`.
If the answer to that query is a value that is `>=` $s$ than the answer is feasible, and we binary search the left half of the range of possible answers, searching for a smaller `mid`. Otherwise we recur on the right half of the array. 
At the end we will return the smallest index `mid` such that $\Sigma_{k=1}^{\text{mid}}\ A[k] \ge s$.

For each iteration of binary search we perform a `range_sum` to verify that the predicate $\Sigma_{k=1}^i\ A[k] \ge s$ holds. 
Binary search recur $O(\log(n))$ times and for each one of them we pay $O(\log(n))$ for the predicate. 
Therefore a query `search(s)` costs $O(\log(n^2))$.
### Optimal Solution
In the trivial solution we do not fully exploit the segment tree. 
In fact we do not need to binary search the answer. 
We want the smallest index `i` such that the sum of the elements in $A[0, i]$ is `>=` $s$. 

**The algorithm is presented with the following example:**

![[IMG_0481.png | center | 600]]

**In other words:** we try to go left, if the range sum stored in left node is `>=` $s$ then we continue going left, as we are trying to find the smallest index, and eventually return the right endpoint of the segment.
If in the left child the sum is smaller than $s$ then surely the prefix $A[0\dots i]$, where $i$ is the right endpoint of the segment covered by the left child, is not enough and we have to go right. 
Going right we just need to find the range sum $[i+1, j]$ such that `range_sum(0, i) + range_sum(i+1, j) >= s`. 
Therefore we subtract to $s$ the range sum of $[0, i]$ and continue the search to the right.

**Time Complexity:**
It is clear that we resolve a `search(s)` query with just one inspection of the tree, and we discard one of the two subtrees in every recursion. Therefore the complexity is $O(\log(n))$. 
# 40) Select: Position of the i-th Occurrence of 1
Given a binary vector $B[0\dots n-1]$. 
We want to provide the following operations: 
- `add(i,b)`, $B[i] = (B[i]+b)\mod 2$ 
- `range_sum(l,r)` 
- `select(i)`, report the position of the $i$-th occurrence of $1$ in $B$
### Solution
The `select(i)` operation is equal to the operation `search(s)` we have seen in the previous problem, therefore the approach to solve the problem is identical. 
The key insight is to notice that if the $i$-th occurrence of $1$ is in position $4$, then the prefix sum of $B$, starting from $0$ to $4$, will be exactly $i$.
# 41) Hard Select
This problem is an harder variation of the previous one. 
Given a binary vector $B[0\dots n-1]$. 
We want to provide the following operations: 
- `add(i,b)`, $B[i] = (B[i]+b)\mod 2$ 
- `range_sum(l,r)`, also called `rank_1(l,r)` for binary vectors
- `hard_select(l,r,k)`, report the $k$-th occurrence of $1$ in the range `[l,r]`
### Solution 
We build upon the "Select" we have seen before, and we notice that actually `hard_select(l,r,k) == select(k + range_sum(0, l-1))`.
# 42) Number of Occurrences
Consider an array $A[0, n-1]$ of integers. 
We want to provide 
- `add(i,v)`
- `occs(l,r,x)`, reports the number of occurrences of `x` in the subarray $A[l,r]$
### Solution
As we have seen before we have to store information on nodes, and build upon that info to retrieve the info of the parent.
In this case we store, for every node, an hash-map that maps an element of the array with the number of occurrences of that element in the segment covered by the node.

When we build we will start, as always, from the leaves node which will have only the mapping `element -> 1`. 
Going up we will "merge" the hash-map, updating the occurrences of elements that appear in both the map of the left and right child, and inserting the elements that appear only in one of the two child.

Once the tree is built (the building process will be slower than the one seen for range sums, as there is the merging of maps) a query `occs(l,r,x)` simply cost $O(\log(n))$ (w.h.p, but depends on the implementation of the map), and the approach is the same to all the queries we have seen before (e.g., `range_sum`).

**What is the space requirement?**
**todo**
Every element can be at most in $\log(n)$ nodes' hash-maps by "construction" of the segment tree.
Therefore the space requirement is $O(n\cdot \log(n))$. 
# 43) Successor Queries in a Subarray
Consider an array $A[0\dots n-1]$. 
We want to provide the following operations: 
- `add(i,v)`
- `successor(l,r,x)`, reports the successor of `x` in $A[l\dots r]$ 
### Solution
We again use a segment tree. We have to store meaningful information in every node of the tree so that we can use it to build the info of a parent node. 
In this case for every node we need a fast way to retrieve the successor of an element `x`

We store a BST in every node of the segment tree. 
The successor query on a BST requires $O(\log(n))$ time. 
When we traverse the ST we combine the answers by taking the minimum. 
The minimum of all the BST successor queries gives us the smallest element that is bigger than `x`, aka its successor. 

The overall time for a `successor(l,r,x)` query is $O(\log^2(n))$. 
We see at most $O(\log(n))$ nodes of the ST to cover the range $[l,r]$. For every BST belonging to these nodes we perform a successor query, which costs $O(\log(n))$. 

**What is the space usage?**
**todo**
Every BST takes linear space in the number of elements in the subtree. Every single element is at most in $O(\log(n))$ BST (aka nodes), therefore overall we use $O(n\cdot \log(n))$ space.
# 44) Triplets 
Given an array of integers $A[0,n-1]$. 
We want to count the number of triplets $i,j,k$ with $0 \le i < j < k <n$ such that 
$$A[i] < A[j] < A[k]$$
**Example:** Given an array $A = [1,2,3,4,1]$ the number of triplets is $4$, as $(1,2,3), (1,2,4), (1,3,4), (2,3,4)$ are triplets that satisfy the constraint above.
### Easy Solution: $O(n^2)$
We consider any pair $(i,j)$ with $i < j$ in the array. 
Once we fixed $i$ and $j$ we check if $A[i] < A[j]$. If that is not the case then we just move to the next pair.
If $A[i]$ is smaller than $A[j]$ we then add to the result the number of indices $k$ such that $j < k \land A[j] < A[k]$. 

We need a way to efficiently compute the number of indices $k$. 
The key lies in the fact that we can choose the pairs in the order we want.
We process pairs so that when a $j$ is fixed then we try all the possible indices $i$, so scanning $A[j+1,n-1]$ is amortized over all possible $i$. 
**In other words:** when we choose a $j$ we then try all possible $i$s, and for every possible $i$s we count the $k$s. This is way better than fixing $i$ and $j$ and then count the $k$s, because for a successive pair $(i', j)$ you would have to scan again and count the $k$s, which you could have already done.
The overall cost $O(n^2)$. 
### Solution: Segment Tree, $O(n\cdot\log(n))$
**Can we do better?** An alternative to the previous solution is:
- fix a $j$
- in $A[0,j-1]$ count how many $i$s such that $A[i] < A[j]$ and call this quantity $I$
- in $A[j+1,n-1]$ count how many $k$s such that $A[j]< A[k]$ and call this quantity $K$
- add to the result $I \cdot K$

The above approach is still quadratic as we are scanning, for every fixed $j$, the whole array (two nested loops).
However this approach is insightful. For every $j$ we would like to spend $\log(n)$ time compute $I$ and $K$.

**Summary of the problem:**
The index $j$ moves left to right. For every $j$ we want to do
$$\text{result = result +}\ |\{i\ |\ i < j \land A[i] < A[j]\}|\cdot |\{k\ |\ j < k \land A[j] < A[k]\}|$$
**Notation:** 
- $|\{i\ |\ i < j \land A[i] < A[j]\}|$ is represented with $I$
- $|\{k\ |\ j < k \land A[j] < A[k]\}|$ is represented with $K$

We then need an efficient data structure to compute $I$ and $K$. 
Basically $I$ and $K$ are result of a count operation in a prefix and a suffix of $A$. When $j$ moves an element "enters" the prefix (potentially changing $I$), and an element exists the suffix (potentially changing $K$).
The said data structure must be able to tell me, for every $j$, the number of elements smaller than $A[j]$ in that prefix and the number of elements bigger than $A[j]$ for that suffix.

In order to do this we remap the values in $A$ with values in $[0,n-1]$. 
This is done by sorting the array and replace every element with its rank in the sorted array. It is obvious that if a triple were to be counted before it will also be counted now. 
The complexity is given by the sorting, thus $O(n\cdot\log(n))$ 

At this point all the elements are in $[0,n-1]$: we can use them to index **a segment tree.**
We can use a segment tree $ST_p$ to efficiently query the prefixes. 
**TODO**
$ST_p$ covers all the array and in position $l$ we store the number of values that are **equal** to $l$ in $A[0,j-1]$. 
At this point to compute $I$ for a given $j$ we can simply perform the query `sum(A[j]-1)` to $ST_p$, which is the number of elements in the prefix $A[0,j-1]$ that are smaller than $A[j]$. 
Similarly we use a segment tree $ST_s$ such that in position $l$ it stores the number of values equal to $l$ in $A[j+1, n-1]$. 
To compute $K$ we then perform the query `range_sum(A[j]+1, n-1)`, which is equal to the number of elements in $A[j+1, n-1]$ that are greater than $A[j]$. 

For every $j$ we perform two queries that cost $O(\log(n))$ and therefore the complexity is $O(n\cdot\log(n))$. 
Then we need to add the cost of updating the prefix and suffix every time we move $j$. Since this is an update the cost is also $O(\log(n))$, and therefore the overall cost is $O(n\cdot\log(n))$. 
# 45) Mo's Algorithm
The Mo's algorithm is a technique for solving a wide variety of range query problems. 
It becomes particularly **useful for kind of queries where the use of segment trees or similar data structures are not feasible.** 
**This happen when queries are non-associative**: the result of query on a range cannot be derived by combining the answer of subqueries.

Mo's algorithm typically achieves a time complexity of $O(\sqrt{n}(n+q))$, where $n$ is the size of the dataset and $q$ is the number of queries. 

**Let's show the Mo's Algorithm in action through the problem "Three of More"**
We are given an array $A[0\dots n-1]$ consisting of colors, where each color is represented by an integer within $[0, n-1]$. 
Additionally we are given a set of $q$ queries called `three_or_more`: the query `three_or_more(l,r)` aims to count the number of colors that occur at least three times in the subarray $A[l\dots r]$. 

The trivial solution of the problem is to use an additional array as a counter to keep track of occurrences of each color within the range and scan the queried subarray. 
Whenever a color reaches $3$ the answer is incremented by $1$. 
The catch here is that **we have to reset the array of counters after each query.**
It is clear that this solution has a time complexity of $\Theta(q\cdot n)$. 
The figure below illustrate an input that showcases the worst-case running time. 
Say that we have $n$ queries, the first query range has a length of $n$ and spans the entire array. The subsequent queries are one unit shorter until the last one that has a range of one. 
The total length of these ranges is $\Theta(n^2)$, which is also the complexity of the solution: 
![[Pasted image 20240116092201.png | center | 600]]
**Let's improve the solution using the Mo's Algorithm.**
Suppose we have just answered the query for the range $[l', r']$ and we are now addressing the query that ranges over $[l,r]$. 
Instead of starting from scratch we can build upon the previous query by adding or removing the contributions of colors that are new in the query range but not in the previous one, or vice versa.
**Specifically:** 
- for left endpoints
	- if $l' < l$ we must remove all the colors in $A[l', l-1]$ 
	- if $l < l'$ we must add all the colors in $A[l, l']$
- for right endpoints is analogous 

**Mind that the time complexity of the algorithm is unchanged**, it is still $\Theta(q\cdot n)$. 
The key insight here is that **a query executes faster it its range significantly overlaps with the range of the previous query.**
The previous figure becomes now a best case scenario: the first query takes $\Theta(n)$ and all the subsequent queries are resolved in constant time.

We see that **the order of the queries have a great impact on the time complexity.**
Even now we can reorganize the queries to go back to quadratic time. 
However this consideration leads to a question: **if we have a sufficient number of queries, can we rearrange them in a way that exploits the overlap between successive queries in order to gain an asymptotic advantage in the overall running time?**
**Mo's Algorithm** answers positively to this question by providing a reordering of the queries such that the time complexity is reduced to $\Theta(\sqrt{n}\cdot(n+q))$. 

**How to reorganize the queries.**
We conceptually divides the array in $\sqrt{n}$ buckets, each of size $\sqrt{n}$, and we name the buckets $B_1\dots B_\sqrt{n}$
A query **belongs** to a bucket $B_k$ $\iff$ its left endpoint $l$ falls into the $k$-th bucket. 
**In other words:** 
$$q_i = (l,r) \in B_k \iff \lfloor\frac{l}{\sqrt{n}}\rfloor = k$$

Initially we group the queries based on their corresponding buckets, and within each bucket we sort the queries in **ascending order of their right endpoints.** 
The figure shows the bucketing approach and how the queries of one bucket are sorted
![[Pasted image 20240116094255.png | center | 600]]

**Let's analyze the complexity of the solution using this ordering.**
We count the number of times we move the indexes $l'$ and $r'$, the endpoints of the previous query. 
This is because adding and removing the contribution of a color takes constant time, and thus the time complexity is proportional to the overall number of moves of these two indices.
**Let's consider a specific bucket.** 
As we process the queries in ascending order of their right endpoints, the index $r'$ can only move backwards and moves at most $n$ times (limit case if all the queries are in this bucket). 
On the other hand the index $l'$ can both increase and decrease but it is limited within this bucket and thus cannot move more than $\sqrt{n}$ times per query. 
Hence, for a bucket with $b$ queries, the overall time to process its queries is $\Theta(b \sqrt{n}+ n)$. 

Summing up over all buckets the time complexity is $$\Theta(q\sqrt{n} + n\sqrt{n}) = \Theta(\sqrt{n}\cdot (n+q))\ \ \ \square$$
When $m = \Omega(n)$ the time per query is $\Theta(\sqrt{n}))$ amortized. 

**The following code is the implementation of three or more and its application using the Mo's Algorithm:**
```rust
pub fn three_or_more(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    let max = a.iter().max().unwrap()
    let mut counters: Vec<usize> = vec![0; max];
    let mut answers = Vec::with_capacity(queries.len());

    let mut old_l = 0; // l'
    let mut old_r = 0; // r'
    let mut answer = 0;

    for &(l, r) in queries {
	    // macros that add or remove the contributions  
	    // of colors of the current query from the previous one
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

pub fn mos(a: &[usize], queries: &[(usize, usize)]) -> Vec<usize> {
    let mut sorted_queries: Vec<_> = queries.iter().cloned().collect();
	let mut permutation: Vec<usize> = (0..queries.len()).collect();

    let sqrt_n = (a.len() as f64) as usize + 1;
    // sort queries by bucket and get the permutation induced by sorting.
	// the latter needed to permute the answers to the original ordering
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

**Final Considerations on Mo's Algorithm:**
**Mo’s algorithm is an offline approach, which means we cannot use it when we are constrained to a specific order of queries or when update operations are involved.**
When implementing Mo’s algorithm, the most challenging aspect is implementing the functions that add and remove contributions from the previous query result.  
There are query types for which these operations are not as straightforward as in previous problem and require the use of more advanced data structures than just an array of counters. 
# 46) Powerful Array
Consider an array $A[1\dots n]$ of positive integers. 
Let us consider its arbitrary subarray $a_l\dots a_r$, where $1 \le l < r \le n$. 
For every positive integer $s$ we denote with $K_s$ th number of occurrences of $s$ into the subarray. 
We call the **power of the subarray** the sum of products $K_s \cdot K_s \cdot s$ for every positive integer $s$ in the subarray.
The sum contains only finite number of non-zero summands as the number of different values in the array is indeed finite.
You should calculate the power of $t$ given subarrays.
### Solution: Mo's Algorithm
We can use Mo's Algorithm and a little bit of attention in updating the answer after an `add` or a `remove`. 
The solution is identical to the one seen in the problem "Three or More" with one difference: we are not interested anymore in the number of occurrences of $i$, denoted with $K_i$, in the subarray, but we instead want to compute $$\Sigma_i\ K_i^2\cdot i,\ i\in [l,r]$$When we increase the number of occurrences we have to first remove the old "power" that we obtained when we thought that there was one less occurrence. 
**In other words:**
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
# 47) (Static) Range Minimum Queries (RMQ)
**TODO**
Given an array $A[1\dots n]$ we want to be able to answer the query `rmq(l,r)` that returns the position of the minimum element in $A[l\dots r]$.
In other words:
$$\text{rmq(l,r)}= \text{argmin}(A[l\dots r])$$
In contrast to the solution that uses segment trees, which requires $O(\log(n))$ time, we want to answer the query in $O(1)$ time and $O(n)$ space.
**Static** means that $A$ will not be changed or updated, this is why we can do better than the dynamic version of the problem.
### Trivial Solution: $O(1)$ time, $O(n^2)$ space
A simple solution is to create a 2D array `lookup[][]` where an entry `lookup[i][j]` stores the minimum value in $A[i\dots j]$. 
**Note:** `lookup[i][j]` store the **index** of the minimum element in the range `[i,j]`.
Once the table is built we can answer a query in $O(1)$ time.

**To process of building the table is the following:** 
```java
preprocess(a[])
	n = a.length()

	lookup[n][n]
	for i = 0 to n 
		// the minimum for the ranges [i,i] (of length 1) is 1
		lookup[i][i] = 1

	for i = 0 to n 
		for j = i+1 to n
			if a[lookup[i][j-1]] < a[j]
				lookup[i][j] = lookup[i][j-1]
			else
				lookup[i][j] = j
```

The time to build the table is $\Theta(n^2)$ and the space requirement to store it is also $\Theta(n^2)$. 

At this point any query `rmq(l,r)` is directly answered with `a[lookup[l][r]]` in constant time.
### Solution: $O(1)$ time, $O(n\cdot \log(n))$ space
The idea is to precompute a minimum of all subarrays of size $2^j$ where $j$ varies from $0$ to $\log(n)$. 
As before we build a lookup table. Here `lookup[i][j]` contains (the index) of the minimum of the range starting from $i$ and of size $2^j$. 
For example, `lookup[0][3]` contains the minimum of the range `[0,7]` (starting from $0$ and of size $2^3$). 

**How we fill up this table:**
We fill it in a bottom-up manner using previously computed values.
For example to find a minimum of range `[0,7]` we can use a minimum of the following two: 
- minimum of range `[0,3]`
- minimum of range `[4,7]`

Based on the above example: 
```java
// eg: if a[lookup[0][2]] <= a[lookup[4][2]]
//          lookup[0][3] = lookup[0][2]
if a[lookup[i][j-1]] <= a[lookup[i+2^(j-1)][j-1]]
	lookup[i][j] = lookup[i][j-1]

// eg: if a[lookup[0][2]] > a[lookup[4][2]
//          lookup[0][3] = lookup[4][2]
else 
	lookup[i][j] = lookup[i+2^(j-1)][j-1]
```

**How we compute the Query:**
For any arbitrary query `[l,r]` we need two ranges that are in powers of $2$. 
The idea is to use the closest power of $2$. We always need to do at most one comparison (compare the minimum of two ranges which are powers of $2$). 
One range starts with `l` and ends with `l + closest-power-of-2`. 
The other range starts with `r - closest-power-of-2 + 1` and ends with `r`. 
For example, if the given range is `[2,10]` we compare the minimum of the two ranges `[2,9]` and `[3,10]`. 

**Formally:** 
```java
// for (2,10), j = floor(Log2(10-2+1)) = 3
j = floor(Log(r-l+1))  
  
// if a[lookup[0][3]] <=  a[lookup[3][3]] then rmq(2,10) = lookup[0][3]  
if a[lookup[L][j]] <= a[lookup[r- 2^j +1][j]]  
   rmq(l, r) = lookup[l][j]  
  
// if a[lookup[0][3]] >  a[lookup[3][3]], then rmq(2,10) = lookup[3][3]  
else 
   rmq(l, r) = lookup[r- 2^j +1][j]
```
# 48) Dynamic Programming
Dynamic Programming, like divide-and-conquer, solves problems combining solutions of subproblems.
Divide-and-conquer algorithms partition the problems into disjoint subproblems, solve them, and then combine their solution to solve the original problem.
In contract, **dynamic programming applies when subproblems overlap, that is when sub-problems share sub-sub-problems.** 
In this context a divide-and-conquer problem does more work than necessary, repeatedly solving the shared sub-sub-problems.
**A DP algorithm solve each sub-sub-problem just once and saves its solution in a table, avoiding the work of recomputing the answer every time it solves each sub-sub-problem.**

**Let's consider a first easy example: Fibonacci's Numbers**
Fibonacci's numbers are defined as 
$$\begin{align}
&F_0 = 0 \\
&F_1 = 1 \\ 
&F_n = F_{n-1} + F_{n-2}
\end{align}$$
Our goal is to compute the $n$-th Fibonacci's number. 
Let's consider the following trivial recursive algorithm: 
```java
fibonacci(n) 
	if n == 0
		return 0
	if n == 1 
		return 1

	return fibonacci(n-1) + fibonacci(n-2)
```
In computing `fibonacci(n-1)` we will compute `fibonacci(n-2)` and `fibonacci(n-3)`, and in computing `fibonacci(n-2)` we will compute `fibonacci(n-3)` and `fibonacci(n-4)`, and so on.
**There are lots of the same Fibonacci numbers that are computed every time from scratch.**

To avoid the above waste of computations we use the **top-down** approach called **Memoization**.
Whenever we compute a Fibonacci number we store it in an array `M`. 
Every time we need a Fibonacci number we check if it is already in the array. If the number is present we just reuse it, otherwise we compute it and store it. 
This algorithm requires linear time and space.
```java
memFibonacciDP(n)
	if n == 0
		return 0
	if n == 1
		return 1

	if M[n] == -1
		M[n] = fibonacci(n-1) + fibonacci(n-2)

	return M[n]
```

**In DP we also have a bottom-up approach, called Tabulation**, which uses linear time and **constant** space.
This approach typically depends on some natural notion of "size" of a sub-problem, such that solving any particular sub-problem depends only on solving "smaller" sub-sub-problems.
Basically we solve the subproblems in ascending order by their size. 
When solving a particular subproblem we are sure that we have already solved all the smaller sub-sub-problems its solution depends upon.
```java
tabFibonacci(n)
	F[n]
	F[0] = 0
	F[1] = 1

	for i = 2 to n 
		F[i] = F[i-1] + F[i-2]

	return F[n-1]
```

**Memoization vs Tabulation**
Tabulation and Memoization are two common techniques used in DP to optimize the solution of problems by **avoiding redundant computations and storing intermediate results.**

**To summarize:** 
- **Tabulation, Bottom-Up approach:** solve a problem by building a table (usually an array or a matrix) and filling it in a bottom-up manner. The table is filled iteratively, starting from the base cases and moving towards the final solution.
	- **pros:** 
		- often more intuitive and easier to implement 
		- usually lower memory requirements since it only needs to store the necessary values in the table
	- **cons:**
		- it may compute values for all subproblems, even those that are not actually needed for the final solution, leading to a potentially higher time complexity
- **Memoization, Top-Down approach:** solve a problem by storing the results of expensive function calls and returning the cached result when the same inputs occur again.
	- **pros:**
		- space-efficient as it only stores results for the subproblems encountered during the actual computation
	- **cons:**
		- may introduce overhead due to function calls and the need for a cache, potentially resulting in a slightly slower runtime compared to tabulation.
# 49) Fastest Algorithm for Fibonacci's Numbers
The fastest way is to exploit matrix multiplication:
![[Pasted image 20240508140841.png | center | 650]]
# 50) Matrix Exponentiation
How can we compute efficiently $A^n$, with $A$ being a matrix?
### Solution
Let's consider $n$: 
- if $n$ is even then $A^n = A^\frac{n}{2} A^\frac{n}{2}$
- if $n$ is odd then $A^n =A^\frac{n}{2} A^\frac{n}{2} A$

Then you keep on decomposing by half until you reach an immediate result, which you then propagate backwards. 
Since we can decompose by half $\log(n)$ times the cost of this algorithm is $O(\log(n))$
# 51) Snakes and Ladders
You are given an $n\times n$ integer matrix `board` where the cells are labelled from $1$ to $n^2$ in a Boustrophedon style, eg: 
![[Pasted image 20240508142026.png | center | 450]]
starting from the bottom left of the board (i.e. `board[n-1][0]`) and alternating direction each row, e.g:

![[Pasted image 20240519103213.png | center | 400]]

The objective is to start from `1` and reach `100` using a dice.
	**You move following the order of numbers:** starting from $1$ you roll a dice, if it gives $5$ then you move from $1$ to $5$. 
The problem is that in the board here are both **snakes** and **ladders**. 

![[Pasted image 20240519103757.png | center | 550]]

Snakes and Ladders impose a behavior for the player: 
- if you reach the head of a snake then you move back to the tail of the snake
- if you reach the start of a ladder you climb it 

In other words: ladders are good, snakes are bad. 

**Given a board, what is the minimum number of moves to reach 100 from 1?**
### Solution
We are kinda asked the shortest "sequence" of cells we traverse to reach the destination. 
Turns out we can actually transform the game into a graph and then apply the shortest path in a DAG: 
- every cell is a node
- every node has an edge to the 6 following cells (nodes) as they can be reached from it
- if a following cell is the start of a ladder then the node has an edge straight to the ladder destination, as ladders (and snakes) are mandatory
- same for snakes

![[Pasted image 20240519110856.png | center | 500]]
In the above scheme we see in blue the edges that represent ladders, in green the edges that represent snakes. 
In red we see the case where from $21$ we can directly go to the ladder destination. 
The same would be for snakes. 
Once you have inserted "the red edges" you can actually forget about snakes and ladders.
Since the source is $1$ and the destination is $100$ we can now apply the algorithm Shortest Path of a DAG, where the cost of each edge is $1$.
Mind that the computation of the shortest path on a unitary costs DAG translates to a BFS: as soon the destination enters the frontier you have found the shortest path. 

Since we use a BFS we have that the cost is $O(|V| + |E|)$. 
Since $|V| = n^2$ and the number of edges is constant ($6n + k$, where $k$ is the number of edges plus snakes) we have that the cost is $O(|V|)$, linear in the number of nodes.
# 52) No Consecutive Zeroes
How many ways are there to make a $N$ bit binary string in which there are no consecutive zeroes?
**Example:** $N = 3$, there are $5$ binary strings with no consecutive zeroes: $010,101,111,110,011$. 
### Solution: Tabulation
The subproblems in this case are clear: just consider shorter strings. 
Consider an array `M[n]` where $M[i]$ is the number of ways we can create a string of $i$ bits with no consecutive zeroes. 
It is obvious that: 
- `M[0] = 1`, as with $0$ bits we can only create one string, the empty one
- `M[1] = 2`, as with $1$ bit we can create two strings with no consecutive zeroes: $0$ and $1$ 
- `M[2] = 3` ($11, 01, 10$)
- `M[3] = 5`

**Formally:** $M(n) = M(n-1) \clubsuit + M(n-2)\spadesuit$, where 
- $\clubsuit:$ append $1$
- $\spadesuit:$ append $10$ 

**Its the Fibonacci's sequence:** we can compute it in $\Theta(n)$.
# 53) Board Tessellation
Consider a $2\times n$ board and domino tiles (of dimensions $1\times 2$). 
How many tessellation are there for the board?
![[Pasted image 20240508145940.png | center | 500]]
### Solution: Tabulation
It is Fibonacci again, only "shifted by one"
![[IMG_0484.png | center | 600]]
# 54) Rod Cutting
Serling Enterprises buys long steel rods and cuts them into shorter rods, which then sells. Each cut is free.
What is the best way to cut up rods?

Consider a rod of length $n$, we know that for any $i \in [1,n]$ the price of a rod of length $i$ is $p_i$. 
The goal is to determine the maximum revenue $r_n$ obtainable by cutting up the rod and selling its pieces. 
### Solution: Tabulation DP, $O(n^2)$ 
We fill an array $R[n+1]$ initialized with all zeroes. 
Every entry $R[i]$ stores the maximum revenue obtainable by a rod of size $i$, which consider the best number of cuts. 
Assuming that we have already solved all the subproblems of size $j < i$, what it the value of $r[i]$?
**Let's list all the possibilities:**
- we do not cut, the revenue is $p_i$, aka $p_i + R[0]$
- we make a cut of length $1$ and we optimally cut the remaining rod of size $i-1$
	- the revenue in this case is $p_1 + R[i-1]$
- we make a cut of length $2$ and we optimally cut the remaining rod of size $i-2$
	- the revenue in this case is $p_2 + R[i-2]$
- ...

The value of $R[i]$ is the **maximum among all these revenues.** 

In a tabulation fashion **we compute the max profit for all rods of length up to $n$**.
The following snippet clarifies the algorithm: 
```java
rodCutting(rod[])
	n = rod.length()
	R[n+1] // initialized with all zeroes

	for i = 1 to n 	// length of the rod
		q = 0 // max profit for the "current" rod

		// best way to cut?
		for j = 0 to i
			// q is the max between q and price of the rod of length i
			// plus the maximum profit obtainable by the the rod of 
			// length i - j
			q = Math.max(q, rod[i].p + R[i-j])
		R[j] = q

	return R[n]
```
# 55) Minimum Cost Path
We are given a matrix $M$ of $n\times m$ integers. 
The goal is to find the minimum cost path to move from the top-left corner to the bottom-right corner. 
Mind that you **only have two possible moves:**
- $\rightarrow$
- $\downarrow$
### Solution: Tabulation DP
The idea is to fill a $n\times m$ matrix $W$, using $1$-indexing, as it follows: 
$$W[i][j] = M[i][j] + 
\begin{cases}
	\begin{align}
		&0\ &\text{if}\ i = j = 1  \\
		&W[i][j-1]\ &\text{if}\ i = 1 \land j > 1 \\
		&W[i-1][j]\ &\text{if}\ j = 1 \land i > 1 \\
		&\min(W[i-1][j], W[i][j-1])\ &\text{otherwise}
	\end{align}
\end{cases}$$
**Said Easy:**
We can think of $M[i][j]$ as the "cost of being in the cell $M[i][j]$", as if the value of a cell of $M$ is a tax to be paid for being on it. 
You always pay the cost of being in the cell you are when computing $W$. 

Basically the first row and column of $W$ are the prefix sum of the first row and column of $M$, as you can reach them only by going always right or always down.
**Example:** to reach $M[1][4]$ the only path is through the first row.

Then, to reach an "internal" cell of $M[i][j]$ where $i,j > 1$, we have to pay:
- the price to enter $M[i][j]$
- the smallest price between arriving from the cell above ($W[i-1][j]$) or from the cell to the left ($W[i][j-1]$) 

The time complexity is $O(n\cdot m)$. 
# 56) Longest Common Subsequence
Given two strings, `S1` and `S2`, of length $n$ and $m$ respectively, the task is to find the length of the longest common subsequence. 
**Observation:** subsequence $\ne$ substring. A subsequence do not have to be contiguous.
### Solution: Tabulation DP
The subproblems here ask to compute the longest common subsequence (LCS) of prefixes of two sequences, `S1` and `S2`. 
**Subproblem:** Given two prefixes `S1[1..i]` and `S2[1..j]` our goal is to compute the LCS of those two prefixes. 

Assume that we already know the solutions to the following smaller problems:
- `LCS(S1[1..i-1], S2[1..j-1])`
- `LCS(S1[1..i], S2[1..j-1])` $\spadesuit$
- `LCS(S1[1..i-1], S2[1..j])` $\clubsuit$

Then we have that: 
- if `S1[i] == S2[j]` we can extend the LCS of `S1[1..i-1]` and `S2[1..j-1]` by adding one character, `c = S1[i] = S2[j]`
- otherwise we can only consider the maximum between $\spadesuit$ and $\clubsuit$ 

**Summarizing:**
$$\text{LCS(S1[1, i], S2[1, j])} = 
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
	\end{align}$$

Therefore we use a $n\times m$ matrix `dp` to store the intermediate results. 
Formally `dp[i][j] = LCS(S1[1,i], S2[1,j])`. 
We set the first row and column to `0` (as they represent the LCS with prefix length `0`) and iterate, using two nested loop, to check if `S1[i] == S2[j]`.  
In such cases we increase by one the previous LCS (`dp[i-1][j-1]`), otherwise we pick the max between the LCS of `S1` without the current character and `S2` without the current character.  
We return `dp[n][m]`, which is the LCS of `S1[1..n]` and `S2[1..m]`, aka `S1` and `S2`.

```java
LCS(S1, S2)
	n = S1.length() + 1
	m = S2.length() + 1

	// first row and colum are initialized to zeroes
	// as the LCS of two strings using 0 chars of one is 0
	dp[n][m]
	for i = 0 to n
		dp[i][0] = 0
	for j = 0 to m
		dp[0][j] = 0

	for i = 1 to n 
		for j = 1 to m
			if S1[i] == S2[j]
				dp[i][j] = dp[i-1][j-1] + 1
			else 
				dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j])

	return dp[n][m]
```
# 57) Minimum Number of Jumps
Consider an array $A$ of $n$ integers. 
Each element $A[i]$ represents the maximum length of a jump that can be made to the right from the position $i$. 
This means that if $A[i] = x$ then we can jump any distance $y$ such that $y \le x$. 
Find the minimum number of jumps to reach the end of the array starting from the beginning of the array. Return $-1$ if it is not possible to reach the end. 
### Solution: Tabulation DP
We use an array `jumps` to store intermediate results. 
Formally `jumps[i]` is the minimum number of jumps needed to reach the position $i$ starting from the start of the array. 

The solution is in the following snippet: 
```java
minJumps(A[])
	n = A.length
	jumps[n]
	// before starting the only reachable position
	// is the starting position, which requires 0 jumps. 
	jumps[0] = 0 
	for i = 1 to n
		jumps[i] = MAX

	for i = 1 to n // subproblems: jumps up to n, the end of the array
		for j = 0 to i // compute the solution for each subproblem
			if j + A[j] >= i && jumps[j] != MAX
				jumps[i] = Math.min(jumps[i], jumps[j] + 1)
				break

	return jumps[n-1]
```

The key is the **if** guard inside the two loops: 
- `j + A[j] >= i`: we want to reach the position `i` and we are currently in the position `j`. If `j + A[j] >= i` it means that from `j`, plus the length of the jump we can make from `j` (`A[j]`), we can actually reach `i`
- `jumps[j] != MAX`: we can actually reach the position `j` from the start of the array

**Then** the minimum number of jumps required to reach `i` is the minimum between the current minimum number of jumps to reach `i` (`jumps[i]`) and the jumps required to reach `j` plus the one jump to reach `i` (`jumps[j] + 1`). 
# 58) 0/1 Knapsack Problem
We are given $n$ items. Each item $i$ has a value $v_i$ and a weight $w_i$. 
We want to put a subset of these items in a knapsack of capacity $C$ and we want to maximize the total value in the knapsack. 
**It is called 0/1 because an item is either selected or not.**

**We have three solutions to this problem:**
1) If $C$ is small then we can use **weight dynamic programming:** $\Theta(C\cdot n)$ time
2) if $V = \Sigma_i\ v_i$ is small then we can use **profit dynamic programming:** $\Theta(V\cdot n)$ time
3) if both $V$ and $C$ are large then we can use **branch and bound:** not covered here. 

**1) Solution with Weight Dynamic Programming**
We fill a $(n+1)\times (C+1)$ matrix $K$.
We then have: 
![[Pasted image 20240429111518.png | center | 650]]

The **last entry of the matrix** $K[n][C]$ contains the solution.

**2) Solution with Profit Dynamic Programming**
The idea is similar to the previous one. 
We use a matrix a $(n+1)\times (V+1)$ matrix $K$. 
Formally $K[v][i]$ is the minimum weight for a profit at least $v$ using items from $1$ up to $i$. 
We then have that 
$$K[v][i] = \min(K[v][i-1], K[v - v_i][i-1] + w_i)$$
The solution is: $\max{a : K[v][n] \le C}$. 
# 59) Max Value with K Coins
Consider $n$ pairs of coins: 
$$\begin{align}
&c_{1,1},\ c_{1,2},\dots,\ c_{1,n} \\
&c_{2,1},\ c_{2,2},\dots,\ c_{2,n}
\end{align}$$
The pair are "vertical", as in $(c_{1,1}, c_{2,1})$. 
The value of the coin $(i,j)$ is $c_{i,j}$. 
The objective is to select $k$ coins to maximize the total value but you can select $c_{2,i}$ only if $c_{1,i}$ is selected. 
### Solution: Longest Path on a DAG, $O(n^2)$
Every coin is a node in a graph. 
![[IMG_0486.png | center]]
**Longest Path on a DAG:** the complexity is proportional to the number of edges and we have $O(n^2)$ edges.
### Solution: Tabulation, $O(n\cdot k)$ 
This problem can be reduced to a specific instance of the Knapsack Problem.
In this case we have that the capacity is $k$, every object weight $1$ and every object value is the associated coin value.
The difference from the classical knapsack is that we also have to consider the constraint on the coin pairs. 

**The idea is the following:**
![[IMG_0487.png | center | 600]]

We have to fil a matrix that is $2n$ rows and $k$ columns, therefore the running time is $\Theta(n\cdot k)$, which is better than the previous solution if $k$ is smaller than $n$.
# 60) Partition Equal Subset Sum
Given a set $S$ of $n$ non-negative integers, check if it can be partitioned into two parts such that the sum of elements in both parts is the same. 
The problem is a well-known NP-Hard problem, which admits a pseudo-polynomial time algorithm. 
**pippo**
The solution is almost identical to the one of the 0/1 Knapsack Problem. 
### Solution: Tabulation DP, $\Theta(v\cdot n)$
We construct a boolean $(n+1)\times (v+1)$ matrix $W$, where $v = (\Sigma_{i=1}^n S[i])/2$.
Formally $W[i][j]$ is `true` if and only if there exists a subset of the first $i$ items with sum equal $j$, false otherwise. 

The entries of the first row, $W[0][j]$, are set to `false` as with $0$ elements you cannot make any sum.
Similarly, the entries of the first column, $W[i][0]$, are set to `true` as with the first $i$ elements you can always create a subset of elements that sum up to $0$: the empty subset. 
Then we have that:
$$W[i][j] = \text{true} \iff W[i-1][j]\lor W[i-1][j-S[i]]$$
**As in the 0/1 Knapsack:**
- **don't take the current element $i$
	- $W[i-1][j]$: do not use the $i$-th element, using the elements $[1\dots i-1]$ we can already make a subset whose elements sum is $j$
- **take the current element $i$**
	- $W[i-1][j - S[i]]$: with one less element we can make a subset of elements whose sum is $j-S[i]$, therefore using the element $i$ we can exactly make a subset that sums up to $j$

**To sum up:**
$$W[i][j] = 
\begin{cases}
\text{T}\ \text{if $i=0 \land j >= 0$} \\ 
\text{F}\ \text{if $i \ge 1 \land j = 0$} \\
W[i-1][j] \lor W[i-1][j-S[i]]\ \text{otherwise}
\end{cases}$$
# 61) Subset Sum Problem 
Given a set $S$ of $n$ positive integers and a target value $V$ return true if there is any subset of $S$ with sum $V$. 
### Solution: Tabulation, $O(n\times V)$
The solution is almost identical to problems we have already seen. 
The idea is to fill a table `W` of `n+1` rows and `V+1` columns, where `W[i][j]` is `true` if using a subset of `i` elements we can make a sum equal to `j`. 
**We then have:** 
$$W[i][j] = 
\begin{cases}
\text{T if $j = 0$} \\ 
\text{F}\ \text{if $i = 0 \land j \ne 0$} \\
W[i-1][j] \lor W[i-1][j-S[i]]\ \text{otherwise}
\end{cases}$$
# 62) Longest Increasing Subsequence
Given an array of integers $S$ of length $n$, find the length of the strictly **longest** increasing subsequence (LIS) in the array.
Mind that in general the LIS is not unique.
**Observation:** as before, subsequence $\ne$ subarray, it is not necessarily contiguous.
### Solution: Tabulation DP, $O(n^2)$ 
Given $S$ let `LIS(i)` be the longest increasing subsequence of the prefix $S[1\dots i]$. 
Then we have that: 
$$LIS(i) = \begin{cases}1 + \max(LIS(j)\ |\ 1\le j < i \land S[j] < S[i])\\ 
1 \text\ \ \ \ \text{if such $j$ does not exists}
\end{cases}$$
We use an array `lis` of size `n+1`, where `lis[i]` is the LIS of the prefix $S[1\dots i]$.  
**The solution is directly given as code:** 
```java
LIS(S) 
	n = S.length()
	n++
	lis[n]
	// the LIS of the null prefix is 0
	lis[0] = 0
	for i = 1 to n
		lis[i] = 1

	// i is the current element
	for i = 1 to n 
		// j is used to to compute the LIS of the prfix up to i
		for j = 0 to i 
			if S[i] > S[j] && lis[i] < lis[j] + 1
				lis[i] = lis[j] + 1

	return lis.max()
```

- **if**
    1. the current element `S[i]` is greater than the element `S[j]` (meaning that the element at position `i` could be the next element of the sequence that ends at position `j`)
    2. the longest increasing subsequence that ends with `S[i]` is shorter than the sequence that ends in `j` (`lis[j]`) plus `1`, the "maybe added" element `S[i]
- **then:** 
	- `lis[i] = lis[j] + 1`
### Optimal Solution: Speeding up LIS
**todo**
The main idea of this approach is to simulate the process of finding a subsequence by maintaining a list of "buckets", where each bucket represents a valid subsequence. 

We start with an empty list `ans` and iterate through the input array `S`.
For each `e` in `S`: 
- **If** `e` is greater than the last element of the last bucket, aka the largest element in the current subsequence, we append `e` to the end of `ans`.  
- **Else** we perform a binary search on `ans` to find the position of smallest element that is `> e` in `ans`. This step help us to maintain the property of increasing elements in the buckets (`ans`). Once we have found such position we replace the element in that position with `e`: this keeps `ans` sorted and ensures that we have the potential for a longer subsequence in the future. 

**Consider the following implementation:**
```java
LIS(S)
	n = S.length()
	List ans
	ans.add(S[0])

	for i = 1 to n
		if S[i] > ans.last()
			ans.add(S[i])
		else 
			// smallestBS return the index of the   
			// smallest element that is >= S[i] in ans
			low = smallestBS(ans, S[i])
			ans[low] = S[i]

	return ans.length()
```

**In other words:**
- the length of `ans` is the length of the current LIS
- when we substitute we insert `S[i]` in `ans[low]`
	- `low` is the index of the smallest element that is `>= e` in `ans`: we can always substitute it with a smaller element `e` without affecting the LIS
	- this substitution let us consider a new LIS starting from `e`: when we substitute we insert in a place that is still part of the current LIS, but if a new LIS would start entirely from `e` every element would be also substituted!
### Reduction to Longest/Shortest Path on a DAG
It is often useful to reduce a problem to a (single source) path computation on a suitable DAG (directed acyclic graph). 

**Let's consider again the Longest Increasing Subsequence problem:** we build a suitable DAG $G$ so that finding the longest path on $G$ is as finding the LIS on the sequence $S$. 

Let's build $G = (V, E)$. 
$G$ has a vertex for every element of the sequence $S$, plus a dummy node that will be used to mark the end of the sequence. 
We use $v_{n+1}$ to denote this dummy vertex while we use $v_i$ to the represent the vertex that corresponds to $S[i]$. 
Every vertex has an edge to $v_{n+1}$ and we have that 
$$(v_j, v_i) \in E \iff j < i \land S[j] < S[i]$$
**In other words:** we have an edge $(v_j, v_i)$ if and only if $S[i]$ can follow $S[j]$ in an increasing subsequence.

**By construction exists a one-to-one correspondence between increasing subsequences in $S$ and paths from $v_1$ to $v_{n+1}$ in $G$.**
**Any longest path on $G$ corresponds to a LIS in $S$.** 

Our $G$ has at most $n^2$ edges and thus this reduction gives us an algorithm with the same complexity of the (naive) solution of LIS: $O(n^2)$. 
# 63) Erdős–Szekeres theorem
For any $r$ and $s$, every sequence $S$ of $n= (r-1)(s-1)+1$ distinct numbers contains either: 
- an increasing subsequence of length $r$
- a deceasing subsequence of length $s$

In other words: any sequence of length $n$ contains either a decreasing or increasing subsequence of length $\sqrt{n}$.
### Proof
Let's consider the sequence $S[1\dots n]$.
Let be `LIS[]` and `LDS[]` the dynamic programming table for the longest increasing and decreasing subsequence of $S$.
Formally, `LIS[i]` and `LDS[i]` are the longest increasing and longest decreasing subsequences that end in `S[i]`, respectively.

Consider for each position `i` the pair `<LIS[i], LDS[i]>`. 
![[Pasted image 20240519133253.png | center | 450]]
**Claim:** **there are no equal pairs.**
**Proof:** 
Consider two positions `i` and `j`, with `i < j`. 
Since the elements in `S` are distinct we have only two cases: 
1) `S[i] < S[j]`, and therefore `LIS[i] + 1 <= LIS[j]`
2) `S[i] > S[j]`, and therefore `LDS[i] >= LDS[j]+1`

Let's assume for absurd that the LIS is of length $< r$ and the LDS is of length $<s$. 
In other words we assume that LIS $\le r-1$ and LDS $\le s-1$.
In this scenario how many pairs we can make? Since LIS is at most $r-1$ and LDS is at most $s-1$ we can make $(r-1)(s-1)$ pairs.
Now, since $(r-1)(s-1$) is less than $n$ we have reached an absurd: `LIS` and `LDS` have length $n$, I should be able to make $n$ pairs!
For the pigeon hole principle this means that some pair must be repeated, but we know that there cannot be repeated pairs. 
Therefore the length of the LIS and/or LDS have to greater, at least $r$ or $s$ respectively, qed.
# 64) Coin Change
We have $n$ types of coins, available in infinite quantities. 
Each coin is given in the array $C = [c_1,\dots,c_n]$. 
The goal is to find out in how many ways we can change the amount $K$ using the given coins. 
**Example:** $C = [1,2,3,8]$ and $K = 3$. The answer is $3$ as we can change $3$ with $\{1,1,1\}, \{1,2\}, \{3\}$. 
### Solution: Tabulation DP
The solution is inspired to the one of "0/1 Knapsack" and even more so to "Partial Equal Subset Sum". 
We build a $(n+1)\times (K+1)$ matrix $W$ where $W[i][j]$ is the number of ways we can change the amount $j$ using a subset of coins up to $i$-th coin "cut".

**The code clarifies the solution:**
```java
coinChange(coins[], k)
	n = coins.length()
	W[n+1][k+1]

	// the number of ways to change 0 with no coin
	W[0][0] = 0
	// with any subset of coin you can change 0, just dont use coins
	for i = 1 to n+1
		W[i][0] = 1

	for i = 1 to n+1
		for j = 1 to k+1
			if coins[i-1] > j
				// the current coin is alone bigger than k, 
				// we cannot use it to change k
				W[i][j] = W[i-1][j]
			else
				// the number of ways to change j is the sum between 
				// 1) #ways without using the coin i 
				// 2) #ways using the coin i to make (j - coins[i])
				W[i][j] = 
					W[i-1][j] + 
					W[i][j-coins[i]]

	return W[n][k]
```
# 65) Longest Bitonic Subsequence
Given a sequence $S[1\dots n]$ find the length of its longest bitonic subsequence. 
A bitonic subsequence is a subsequence that first increases and then decreases. 
### Solution, Dynamic Programming
This problem is a variation of the Longest Increasing Subsequence. 
We construct two arrays `lis[]` and `lds[]` using the solution of LIS:
- `lis[i]` stores the length of the longest increasing subsequence ending with `S[i]`
- `lds[i]` stores the length of the longest decreasing subsequence starting from `S[i]`

The solution is given by finding the maximum value `lis[i] + lds[i] - 1`, where `i` is from `0` to `n-1`. 
# 66) Largest Independent Set of a Tree
Given a tree $T$ with `n` nodes find one of its largest independent sets. 
An independent set is a set of nodes $I$ such that there are no edges connecting any pair of nodes in $I$. 
**Example:** the nodes in red form a largest independent set in their tree:

![[Pasted image 20240222092807.png | center | 300]]

### Solution: Bottom-Up DP
For every node $u$ we have two possibilities: 
1) add $u$ to the the independent set: $u$'s children will not be part of the independent set, but $u$'s grandchildren might be
2) don't add $u$ to the independent set: $u$'s children might be part of the independent set

Let `LIST(u)` be the size of the independent set of the subtree rooted in $u$. 
We then define: 
- $C_u$ as the set of children of $u$
- $G_u$ as the set of grandchildren of $u$

We can now define the following recurrence: 
$$LIST(u) = \begin{cases}
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

Then to algorithm is given by the following snippet
```java
solution(tree)
    n = tree.size() // num nodes
    lis[n] // initialized with all -1
	res = list(root, lis)

list(u, lis)
	if u == null
		return 0

	if lis[u] != -1
		return lis[u]

	// we use u and maybe his grandchildren
	uUsed = 1
	if u.left != null
		uUsed += Max(list(u.left.left, lis), list(u.left.right, lis))
	if u.right != null
		uUsed += Max(list(u.rigth.left, lis), list(u.right,right, lis))

	// we dont use u, maybe his children
	uNotUsed = 0
	uNotUsed += list(u.left, lis) + list(u.right, lis)

	// is better to use u and his grandchildren or only his children
	res = Max(uUsed, uNotUsed)
	lis[u] = res
	return res
```
# 67) Greedy Algorithms
A **greedy algorithm** is any algorithm that follows the problem-solving heuristic of **making the locally optimal choice** at each stage. 
In many problems a greedy strategy **does not produce an optimal solution**, but a greedy heuristic **can yield locally optimal solutions** that approximate a globally optimal solution in a reasonable amount of time.

Most problems for which greedy algorithms yields good solutions (aka good approximation of the globally optimal solution) have two properties: 
- **greedy choice property:** We can make whatever choice seems best at the moment and then solve the subproblems later. **A greedy algorithm never reconsider its choices**. 
- **optimal substructure:** a problem exhibits this property if an optimal solution to the problem contains optimal solutions to the sub-problems.

**Greedy is an algorithmic paradigm that builds up a solution piece by piece, always choosing the next piece that offers the most obvious and immediate benefit.**
# 68) Alice and Bob's Coins
Consider a sequence of $n$ coins, where $n$ is an even number. 
Every coin has an associated value.
We have two players, Alice (the 1st player) and Bob (the 2nd player).
The game start with Alice that can take a coin, either at start or at the end of the sequence. Then is the turn of Bob, and he can take either the first of last coin of the "updated" sequence. 
The game goes on until all there are coins.
The game is won by the player that collected the coins with largest value. In other words the richest player wins.

**Is there a strategy for Alice that guarantees that the value she collect is at least the same as the amount of Bob?**
In other words: is there a strategy that guarantees `Money Alice >= Money Bob`?
### Solution: Greedy
We can label a coin with `E` if the coin is in an even position in the sequence, and we can label coins in odd positions with `O`. 
Alice starts first: she can force Bob to take only odd or even coins!
Then the solution is to:
- compute the sum $S_E$ of coins in even positions
- compute the sum $S_O$ of coins in odd positions
- if $S_E > S_O$ then Alice will take a coin in an even position, either from start or end of the sequence, forcing Bob to take coins that sum up to a lower (or equal) value
- if $S_E < S_O$ then Alice will take a coin in an odd position, forcing Bob to take coins in even positions, that sum up to a lower (or equal) value

**This greedy strategy is not the most obvious strategy but it is correct.**
When attacking problems with a greedy strategy you have to be extra careful, as intuitive strategies might be negated by tricky counterexamples. 
# 69) Activity Selection
You are given $n$ activities with their starting and finishing times. 
Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at the time, aka no overlaps are allowed.
### Solution: Greedy
The greedy solution is achieved by sorting the activities in ascending order **by their finish. This takes care of all the tricky overlaps. 
```java
activitySel(activities[])
	n = activities.length()
	// sort by ascending order of activities[i].end
	activities.sort()

	res = 1
	last = activities[0]
	for i = 1 to n
		// if the current activity starts at least when the
		// the last activity ends than it can be done. 
		// we are sure that the current activity will end after the
		// end of the last activity as they are sorted by ends
		if activities[i].start >= last.end
			res++
			last = actitvities[i]

	return res
```

**Why it Works?**
**Lemma:** Exists at least an optimal solution which includes the activity with the smallest ending time, which is the first activity selected by our strategy.
**Proof:** 
![[Pasted image 20240430164009.png | center | 600]]
# 70) Job Sequencing
You are given an array of jobs where every job has an id, a deadline and a profit that can be achieved if the job is completed before its deadline. 
It is also given that every job takes a single unit of time, so the minimum possible deadline for any job is $1$. 
We start at time $0$ and every time we perform a job we move "the clock" by one, summing to the total profit the profit associated with the job we have just completed.
Maximize the total profit, considering that only one job can be done at a time. 
### Solution, Greedy
If we relax the problem so that a job must be executed exactly as its deadline is very easy to find an optimal solution. For jobs with same deadline just pick the most profitable one. Start selecting "backwards": pick the most profitable job with the latest deadline and schedule it for that time, then move back to the previous time slot and do the same. 

**What about the general problem?** We can schedule a job to be executed any time before its deadline. 
The approach has some similarity with the previous one. We still proceed backwards and put the most profitable job with latest deadline to be executed exactly on the time slot of its deadline. 
But now we must also consider the case where there is another job with the same deadline. Before this job would be discarded but now it might survive, and more than that it: might be crucial to achieve the optimal solution.
We put that job to the previous time slot, with jobs that have as deadline that time slot, and select the job with highest profit. 
The discarded jobs go backwards again, and so on.

**Said Easy:**
Put the most profitable job with latest deadline to be executed exactly on the time slot of its deadline. 
For each time slot we might have many jobs with that deadline: we select the most profitable one. 
The jobs that are not selected move backward, in the group of jobs that might be scheduled for the "previous" time slot, and repeat.

**How it is Implemented?**
We process time slots: for each time slot we have the jobs that have that time slot as deadline.
Then we use a **Max-Heap** to store the jobs that might be scheduled. 
The max-heap is used to efficiently insert and retrieve jobs with maximum profit. 
Going backwards from the last time slot: we insert all the jobs that might be scheduled there in the max-heap and then ask retrieve the maximum. That is the scheduled job for that time slot. 
We then move backward to the previous time slot. Insert the jobs that might be done there in the max-heap and retrieve once again the max. 
And so on. 

The overall total complexity is $O(n\cdot \log(n))$.
# 71) Fractional Knapsack
Consider $n$ items. For each item $i$ we have its value $v_i$ and its weight $w_i$. 
We want to put these items in a knapsack of capacity $C$, and we want to maximize the value of the knapsack. 
In this version of the problem **we can break the items** for maximizing the total value.
### Solution: Greedy
The greedy approach is to calculate the ratio profit/weight for each item and then sort them on the basis of this ratio.
Consider that: 
$$\frac{v_i}{w_i} = \begin{cases}< 1\ \text{if $i$ is more heavy than valuable: disconvinent}\\ = 1\ \text{if $i$ is as heavy as valuable: neutral} \\ > 1\ \text{if $i$ is more valuable than heavy: convinent}\end{cases}$$
In other words: **the above ratio is the value per unit of weight:** we want to select the item that have the highest ratio first. 
**This will always give the maximum profit because in each step it adds an element such that this is the maximum possible profit for that much weight.**
# 72) Boxes
Consider $n$ boxes. 
For each box $i$ we have
- $w_i$, its weight
- $d_i$, its durability

You can choose **1) any subset of boxes** and **2) arrange them in any order** you like. 
Find the maximum number of boxes that can be used to form a tower, where for each box $j$ you use in the tower the following constraint must hold: 
$$d_j \ge \Sigma_{k\in T}\ w_k$$
**In other words:** the constraint is "structural": a box has to be strong enough to hold the weight of all the boxes over it. 
### Solution: Greedy
To solve the problem we divide it in two tasks: 
1) assume we already know the (optimal) ordering of the boxes and just select a subset of boxes that makes the heighest tower
2) find the best ordering

Then in the final solution you use the answer of **2)** and then **1).**

**Step 1)**
Assume we know an optimal ordering of boxes and let's find the best selection. 
We use a $(n+1) \times (n+1)$ matrix $W$ where $W[i][k]$ is the minimum weight of a tower with height $k$ using the first $i$ boxes. 

In the first row of the matrix, $W[0][k]$, we have all zeroes, as the minimum weight of a tower built with zero boxes is zero. 
Similarly, the first column of the matrix $W[i][0]$ is also set to zeroes, as the the weight of a tower of height zero is zero (no box are used). 
When a cell $W[i][j]$ is equal to zero than it means that it is not possible to build a tower of height $j$ using a subset of the first $i$ boxes.
Then we have 
$$W[i][k] = \min \begin{cases}W[i-1][k]\ \clubsuit\\ W[i-1][k-1] + w_i\ \spadesuit\ \text{if}\ d_i \ge W[i-1][k-1]\end{cases}$$
where: 
- $\clubsuit:$ do not use the current box $i$ in the tower
- $\spadesuit:$ use the $i$-th box as base of the tower, and in fact we add its weight $w_i$ (only if the box is durable enough to support the tower)

The solution is then stored in the last row $i = n$. 
We traverse the last row right to left (as we want the heighest possible tower) and stop at the first entry that is not zero. 
The corresponding column index is the answer to the problem.

Filling this matrix requires $O(n^2)$. 

**Step 2)**
The above method works because we assumed the optimal ordering. 
The optimal ordering guarantee that the box that we consider now, if usable, can be used as a base of the tower in an optimal solution. 
**When we add a box we always add it as a base.**

Consider two boxes $A$ and $B$. 
In the **ordering**, $A$ is before **(in the ordering)** $B$ if 
$$\clubsuit\ d_A - w_B \le d_B- w_A$$
or, in words, if the durability of $A$ minus the weight of $B$ is less or equal to the durability of $B$ less than the weight of $A$. 
This is correct because $B$ will be lower than $A$ **in the tower:** therefore, when going through boxes, fist I see $A$ and use it as base of the tower. Then I see $B$ and I can use it as base of the tower. 
**Lower in the tower means later in the ordering.**
Remember: the optimal ordering guarantee that the currently considered box can be used as a base of the tower in an optimal solution (if it can be used). 

Then we can rewrite $\clubsuit$ as 
$$d_A + w_A \le d_B + w_B$$
To find the best ordering we compute $d_i + w_i$ for every box $i$ and sort in ascending order: $\Theta(n\cdot \log(n)).$
# 73) Hero
Bitor has $H$ health points and must defeat $n$ monsters.
Each monster $i$ deals $d_i$ damage but gives $a_i$ health after being defeated. 
Mind that Bitor can have more exceed $H$ health. 
Find an ordering of monsters such that Bitor can defeat them all. Return $-1$ if not possible.

**Observation:** if Bitor reaches $0$ health then he dies, even if the monster dies at the same time. The rewarded health $a_i$ requires an alive hero to be applied. 
### Solution: Greedy
The greedy choice is based on the following classification of monsters: 
- a monster $i$ is **good** if $a_i > d_i$
- a monster $i$ is **bad** if $d_i > a_i$

We sort the **good monsters by increasing damage** and the **bad monsters by decreasing health bonus.** 

**In words:**
As Bitor face good monsters he will increase his health, as a good monsters gives more than he take.
This means that when Bitor will have defeated all the good monsters he will have maximum health.
Now Bitor will fight bad monsters, starting with monsters that restore the most possible health. After each fight he will have the most health possible in that situation, which will be used to fight the next monster.
If he can defeat all the bad monsters then Bitor is victorious. 

**The idea is given by the graph:**
![[IMG_0477.png | center | 550]]
# 74) Meetings in One Room
There is **one** meeting room in a firm.
There are $N$ meetings in the form of $(s_i, e_i)$, where $s_i$ is the starting time of the meeting $i$ and $e_i$ is its end. 
Find the maximum number of meetings that can be accommodated in the meeting room, knowing that only one meeting can be held in the meeting room at a particular time.
### Solution: Greedy
The solution is very similar to the one seen in "Activity Selection": 
- sort the meetings in increasing order of their end
- do the first meeting in the sorted list and store its ending time
- iterate from the second meeting to the last one 
	- if the starting time of the current meeting is `>=` the ending time of the last meeting then we can have the meeting
		- increase the counter of meetings
		- update the last ending time with the current meeting's ending time
- return the counter
# 75) Wilbur and Array
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
### Solution: Greedy
The solution is based upon the observation that the minimum number of operations is equal to the sum of differences (in absolute value) between consecutive elements of the target array. 
Given the array $b[1\dots n]$ we have that
$$\text{result} = b[1] + \Sigma_{i=2}^n\ |b_i-b_{i-1}|$$
Therefore the solution is given by:
```java
wilburAndArray(a[], b[])
	n = a.length()
	result = Math.abs(b[0])

	for i = 1 to n
		result += Math.abs(b[i] - b[i-1])

	return result
```
# 76) Woodcutters
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
### Solution: Greedy
We use a greedy approach to solve the problem. 
1) we always cut the first tree, making it fall to the left
2) we always cut the last tree, making it fall to the right
3) we **prioritize** **left falls**, meaning that if we can make the current tree falling to the left we always will: consider the current tree $i$
	1) if the previous tree is at a point $x_{i-1}$ that is smaller (aka farther) that where the current tree would fall, then we cut the tree and make it fall to the left
	2) else, if the current tree position $x_i$ plus its height $h_i$ do not fall over next tree $i+1$ we cut and make it fall to the right
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
			// update the position of the current tree
			trees[i].x = trees[i].x + trees[i].h
			continue

	return result
}
```
# 77) Magic Numbers
A number is said to be magic if it is formed by a concatenation of $1$, $14$, $144$. 
For example:
- $14$ is a magic number
- $114$ is a magic number
- $141441$ is a magic number
- $1444$ is **not** a magic number

**Determine if a given number is magic.**
### Solution: Greedy
The approach is called **greedy parsing**.  The idea is to **find the longest match.**
This is based on the **Lempel-Ziv compression**. 
![[IMG_0488.png | center | 700]]
**Always selecting the longest match** is optimal, in Lempel-Ziv it can be proved that the greedy approach has the smallest number of phrases. 

Therefore the **optimal way to check if a number is magic is to parse it greedily**, that is to "decompose" it by exploiting the longest possible matches:
- go through the number to see if there are numbers that are not $1$ or $4$, if so return false
- now start matching using longest matches
	- if you can arrive at the end of the number then return true
	- otherwise (that is at the first "mismatch") return false

**Example:** 
To check that $141441$ is a magic number we match using the phrases $1, 14, 144$. 
The longest matches going through the number are: $14$, $144$, $1$
For the case of $1444$. We first match $144$, but then cannot match the last $4$ and therefore we return false.
**todo**
Why the Lempel-Ziv (greedy parsing) is optimal (lowest number of matches)?
# 78) Lexicographically Maximum Subsequence
**todo**
Given a string `S` of length $n$, find the lexicographically maximum (longest) subsequence.
**Example:**
- `S = a b a b b a`
- `LMS = b b b a`

Why? If you select any other subsequence `s` than `s` is lexicographically smaller than `LMS`.
**The lexicographically largest character will be the first character in our answer.**
### Solution: Greedy
We scan the sequence `S` right to left, we keep track of the largest symbol we have seen so far and we select any letter which is at least as large. Any time we select we update the largest symbol seen so far if needed. 
# 79) Queue
You have a queue of people waiting for something. 
The person $i$ in the queue has
- $h_i$, the height of the person
- $a_i$, number of other people who are taller than $i$ are before $i$ in the queue

The queue is destroyed (the persons in the queue are randomly shuffled).
The objective is to rebuild the queue. 
### Solution: Greedy
Sort the people in a non-decreasing order using $a_i$. 
If every person in a position $i$ has as $a_i$ a value that is smaller or equal to $i$ then a solution is possible, otherwise a solution cannot be found.

![[Pasted image 20240509143145.png | center | 700]]
# Graphs
![[Pasted image 20240222110350.png | center | 700]]
- **a)** an undirected graph $G = (V, E)$
- **b)** adjacency list representation of $G$
- **c)** adjacency matrix representation of $G$
### 80) Breadth-First Search
Given a graph $G = (V,E)$ and a source vertex $s$ a BFS explores the edges of $G$ to "discover" every vertex that is reachable from $s$. 
BFS computes the distance from $s$ to every reachable nodes, where the distance is defined as the smallest number of edges needed to go from $s$ to a node $v$. 

Another effect of the BFS is to produce a breadth-first tree rooted in $s$. 
The tree contains all the nodes reachable from $s$. 
For any vertex $v$ reachable from $s$ the path in the tree from $s$ to $v$ corresponds to a shortest path from $s$ to $v$ in $G$. 

Breadth-First search expands the frontier between discovered and undiscovered vertices uniformly across the breadth of the frontier.  
**You can think of it as discovering vertices in waves emanating from the source vertex.**

**Remember:** a queue is a FIFO data structure: 
- insert the new element as its end/tail, it "appends" elements
- remove elements from the start/head, pop from the start

The idea is the following: 
```java
bfs(G, s)
	for v in G.nodes()
		if v != s
			v.color = "white" // node never "seen" (reached) before
			v.distance = Math.MAX // maximum distance, unreachable
			v.predecessor = null // no predecessor in the BF-tree

	s.color = "gray" // a vertex seen for the first time becomes gray
	s.distance = 0 // the distance from s to s is zero
	s.predecessor = null // s has no predecessor

	Queue frontier
	// add s as the last element of the frontier
	frontier.add(s)

	while !frontier.isEmpty()
		// pop element from the head of the queue
		u = frontier.pop()
		for v in u.neighbors() // consider all the neighbors of u
			if v.color == "white" // if v was unreachable now is reachable
				v.color = "gray"
				v.distance = u.distance + 1
				v.predecessor = u
				frontier.push(v) // add v to the frontier
		u.color = "black" // explored all neighbors: the node is black
```

**Observation:**
- an unreachable node remains "white"
- the frontier contains only "gray" nodes
- once a node becomes black it is never "explored" again (never re-enters the frontier)

**What if there is a cycle?**
- **a)** use a priority queue (Dijkstra, see later on)
- **b)** use a DFS (stack instead of a queue)

**Time Complexity:** $O(V+E)$
This cost arises from the need to visit each vertex once and traverse each edge once during the BFS traversal.
### 81) Depth-First Search
As its name implies, DFS searches "deeper" in the graph whenever is possible.
DFS also timestamps each vertex:
- `v.d` records when `v` is first discovered (and "greyed")
- `v.f` records when the search finishes examining `v`'s neighbors ("blackened")

```java
time // time is a global variable
dfs(G)
	for u in G.nodes()
		u.color = "white"
		u.predecessor = null

	time = 0 // time is initialized
	for u in G.nodes()
		if u.color = "white"
			dfsVisit(G, u)

dfsVisit(G, u)
	time++ 
	u.d = time // time at which u has been discovered (u starting time)
	u.color = "gray"
	for v in u.neighbors()
		if v.color = "white"
			v.predecessor = u
			dfsVisit(G, v)
	time++ 
	u.f = time // u's finishing time, u.f is in [1, 2*|V|]
	u.color = "black"
```

**Observation:** DFS uses a **stack** instead of a queue to explore the nodes.
In the above implementation the stack is the activation records stack since the implementation is recursive.

Unlike BFS, whose predecessor subgraph forms a tree, depth-first search produce a predecessor graph that **might contains several trees:**
1. **Backtracking**: In DFS, when we reach a dead end (a vertex with no unvisited neighbors), we backtrack to the nearest vertex with unvisited neighbors. This process can create multiple branches of exploration, leading to the formation of separate trees.
2. **Visited Nodes**: As DFS progresses, it marks vertices as visited to avoid revisiting them. This ensures that each component is explored exactly once, and hence each component forms a separate tree in the forest.

Therefore we define the **predecessor subgraph** of a depth-first search slightly differently from that of a BFS: it always includes all vertices, and it accounts for multiple sources. 

Specifically for a depth-first search the predecessor subgraph is $G_\pi = (V, E_\pi)$, where 
$$E_\pi = \{(v.\pi, v): u\in V \land v.\pi \ne\ \text{NIL}\}$$
The predecessor subgraph of a DFS forms a depth-first **forest** comprising several depth-first trees. 
The edges in $E_\pi$ are tree edges. 

**Note:** DFS  builds a DFS-tree when applied to a single connected component of an undirected graph or to the entire graph if it's connected. 
However, in the case of a graph with multiple **connected components**, DFS produces a forest of DFS trees, often referred to as a DFS forest.
A **strongly connected component** of a graph $G = (V,E)$ is a maximal set of vertices $C\subseteq V$ such that for every pair of vertices $u$ and $v$ we both have $u\longrightarrow v$ and $v\longrightarrow u$. 
In other words: for every pair of vertices $u,v \in C$ there exists a path from $u$ to $v$ and vice versa.

**During the execution of the DFS, different type of edges can be encountered.** 
Let's break down each type of edge:
- **tree edges:** Tree edges are the edges that are discovered during the traversal and form the structure of a DFS tree. When DFS visit a new vertex $v$ from a vertex $u$, the edge $(u,v)$ becomes a tree edge. 
- **forward edges:** Forward edges are edges that connects a vertex to one of its descendants in the DFS tree. In other words, a forward edge $(u,v)$ exists if the vertex $v$ is a descendant of $u$, but not a direct child of $u$
- **back edges:** Back edges are edges that connect a vertex to one of its ancestors in the DFS tree. A back edge $(u,v)$ exists if the vertex $v$ is an ancestor of the vertex $u$ in the DFS tree
- **cross edges:** Cross edges are edges that connect two vertices neither of which is an ancestor of the other in the DFS tree. In other words a cross edge $(u,v)$ connects two vertices that are not in a direct ancestor-descendant relationship

**Parenthesis Theorem:**
Descendants in a depth-first-search tree have an interesting property. 
If $v$ is a descendant of $u$, then the discovery time of $v$ is later than the discovery time of $u$. 
In any DFS traversal of a graph $G = (V,E)$, for any two vertices $u$ and $v$, exactly one of the following statements is true: 
- the intervals `[u.d, u.f]` and `[v.d, v.f]` are entirely disjoint and neither `u` nor `v` is a descendant of the other in the depth-first forest
- the interval `[u.d, u.f]` is contained within the interval `[v.d, v.f]`, and `u` is a descendant of `v` in a depth-first tree
- the interval `[u.v, v.f]` is contained within the interval `[u.d, u.f]`, and `v` is a descendant of `u` in a depth-first tree

**White-Path Theorem**
A vertex `v` is a descendant of `u` if and only if there is a path from `v` to `u` with only white vertices at time `u.d`

**Time Complexity:**
Typically $O(V + E)$. 
This is because, in each recursive call, the algorithm explores all the edges incident to the current vertex, leading to a total of $V + E$ operations.
# 82) Is G a DAG?
Consider a graph $G = (V,E)$. 
How can we check weather $G$ is a DAG?
### Solution, DFS
We can simply run a DFS: if there exists a back edge than $G$ is not a DAG. 
Practically, when we are visiting the neighbors of a node $u$ and we find a grey vertex $v$ (a vertex that was already visited) **and** $v$ is not the parent of $u$ (i.e., $v$ is not the vertex from which $u$ was discovered), then there exists a back edge $(u,v)$, and therefore a cycle. 
# 83) Strongly Connected Components
**An application of DFS is the decomposition of a directed graph into its strongly connected components.**
### Solution: Kosaraju's Algorithm, $O(V + E)$
The solution use the **transpose** of $G$, which is defined as $G^T = (V, E^T)$ where 
$$E^T = \{(u,v) : (v,u)\in E\}$$
Given an adjacency-list representation of $G$ the time needed to create $G^T$ is $O(V+E)$. 
**Observation:** by construction $G$ and $G^T$ have the same strongly connected components. 

**The algorithm is the following:**
1. Initialize an empty stack to keep track of the finishing times of nodes during the first DFS traversal.
2. Perform a DFS traversal on the original graph, and upon completing the traversal of a node, push it onto the stack.
3. Build the transposed graph (a graph with all edges reversed).
4. Pop nodes from the stack one by one. Each popped node represents the start of a DFS traversal in the transposed graph.
5. Perform DFS on the transposed graph starting from the popped node. Collect all nodes reachable from this starting node. These nodes form one SCC.
6. Repeat step 4 and step 5 until all nodes are visited.
# 84) Single Source Shortest Path
Given a direct or undirect graph $G = (V,E)$ the objective is to **find the shortest path from a single source vertex to all the other vertices** in the graph. 

**Said differently:** given a graph with weighted edges (where each edge has a non-negative weight) the goal is to find the shortest path from a specified vertex (the source) to all other vertices in the graph.

**From the CCLR:**
Given a graph $G = (V,E)$ find the shortest path from a single source $s\in V$ to every other vertex $v\in V$. 
Then we introduce: 
- **the weight function:** $w: E\rightarrow \mathbb{R}$, maps edges to real values
	- the weight of a path $p = (v_0\dots v_k)$ is the sum of the weights of its constituent edges
- the **shortest path weight** from $u$ to $v$ is represented with $\delta(u,v)$ and it is 
	- $\min\{w(p): u\xrightarrow{p}v\}$ if there is a path $p$ from $u$ to $v$
	- $-\infty$ otherwise 

This problem has an **optimal substructure.**
**Lemma:** given a weighted directed graph $G = (V,E)$, let $p = (v_0\dots v_k)$ be the shortest path from $v_0$ to $v_k$. Consider the sub-path $p_{ij}$ from the node $v_i$ to $v_j$, where $0\le i \le j \le k$. Then the sub-path $p_{ij}$ is the shortest path from $v_i$ to $v_j$. 

**There are two things to consider when solving this problem:**
1) **negative weighted edges:** If the graph has a negative cycle that is also reachable from the source then the shortest path are not well defined. 
	- **solution:** if there is a negative weight cycle on some path from $s$ to $v$ then we set $\delta(s,v) = -\infty$ 
2) **cycles:**
	- as we have seen we cannot have negative cycles and we have a way to "remove" them
	- we also don't want positive cycles: removing a cycle from a path $s$ to $v$ surely produces a shortest path 
	- **without loss of generality we can assume that the shortest path have no cycles, or in other words, it is a simple path:** since any acyclic path in the graph contains at most $V$ distinct vertices we see that any shortest path contains $O(|V|-1)$ edges
###  85) Solution: Dijkstra's Algorithm
Dijkstra's Algorithm is a method for finding the shortest path from a single source vertex to all other vertices 
It works for both directed and undirected graphs with non-negative edge weights. 
The algorithm **maintains a set of vertices whose shortest distance from the source is known, and continually expands this set by considering vertices with the minimum distance from the source.**

**The algorithm behaves as follows:**
1) **initialization:** assign a distance value to every vertex in the graph
	1) set the distance of the source vertex to $0$
	2) initialize the distances of all the other vertices to infinity. 
	3) maintain a priority queue (often a min-heap) to store vertices based on their current estimated distances from the source
2) **relaxation:** repeat the following steps until all vertices have been processed
	1) extract the vertex $v$ with minimum distance from the queue
		1) this vertex is considered visited
	2) for each neighbor $u$ of $v$ that is not yet been visited calculate its tentative distance from the source by adding the weight of the edge $(v,u)$: if this new distance is smaller than the previously known distance of $u$ update its distance value in the queue
3) **termination:** once all vertices have been visited the algorithm terminates
	1) the distances calculated represent the shortest paths from the source vertex to all other vertices in the graph

The algorithm is implemented in the following snippet: 
```java
dijkstra(G, s)
	PriorityQueue q // a min priority queue
	
	s.d = 0 
	for v in G.nodes()
		if v != s
			v.d = Math.MAX
		q.add(v)

	while !q.isEmpty()
		// u is the vertex in Q with minimum distance from s
		u = q.pop()
		for v in u.neighbors()
			weight = G.getWeight(u,v)
			v.d = min(v.d, u,d + weight)

	distances[n]
	for i = 0 to n
		distances[i] = graph.node(i).d

	return distances
```

**Time Complexity:**
The time complexity of Dijkstra's algorithm depends on the data structures used to implement it. 
Using a priority queue based on a **binary heap**, the time complexity is typically $O(\log(V)\cdot(V+E))$ where $V$ is the number of vertices and $E$ is the number of edges in the graph.
**Remember the complexity of a binary heap**
**Breakdown of the complexity**
1. **Initialization**: Inserting vertices into the priority queue, $O(V \cdot \log(V))$. Each insertion takes $O(\log(V))$ time, and there are $V$ vertices
2. **Main loop**: In each iteration of the loop, we extract the minimum vertex from the priority queue and relax its outgoing edges
    - the loop runs $V$ times because each vertex is extracted at most once
    - for each vertex, we relax its outgoing edges, which takes constant time for each edge relaxation
3. **Updating priorities**: Decreasing the priority of a vertex: $O(\log(V))$
    - Since each edge relaxation may require updating the priority of a vertex in the priority queue, the total time spent on updating priorities is $O(E \cdot \log(V))$
### 86) Solution: Bellman-Ford's Algorithm
The Bellman-Ford algorithm is used to find the shortest paths from a single source vertex to all other vertices in a weighted graph, even in the presence of negative weight edges, as long as there are no negative weight cycles reachable from the source vertex.

**Consider the following implementation:** it takes in a graph, represented as lists of vertices (represented as integers $[0..n-1])$ and edges, and fills two arrays (distance and predecessor) holding the shortest path from the source to each vertex.
```java
BF(G, s)
	distances[n]
	predecessors[n]

	// step 1) initialization
	for v in G.vertices 
		distances[v] = inf
		predecessors[v] = null
	distances[s] = 0

	// step 2) relaxation
	 for v in G.nodes()
		for (u,v) in G.edges()
			w = G.getEdgeWeight((u,v))
			if distances[u] + w < distances[v] 
				distances[v] = distances[u] + w
				predecessors[v] = u

	// step 3) check for negative cycles
		for (u,v) in G.edges()
			w = G.getEdgeWeight((u,v))
			if distances[v] > distances[u] + w
				return NegativeCycle

	return distances, predecessors
```

The Bellman-Ford algorithm has a time complexity of $O(V\cdot E)$, where V is the number of vertices and E is the number of edges in the graph. 
In the worst-case scenario, the algorithm needs to iterate through all edges for each vertex, resulting in this time complexity. 
# 87) Disjoint Sets Data Structure
**Consider the following toy Problem:**
In a room, there are $N$ persons and we define two persons as friends if they are directly or indirectly friends. 
A group of friends is a group of friends where any two pairs of persons are friends. 
Given a list of persons that are directly friends find the number of groups of friends and the number of persons for each group.
**Let's see an example:** consider $N = 5$ and the list of friends is $\{[1,2], [5,4], [5,1]\}$. 
Using the "transitive property of friendship" we will have only two group of friends: $[1,2,4,5], [3]$

![[Pasted image 20240228121022.png | center | 150]]

This problem can be solved using a BFS (in $O(V+E)$ time) but we use it to introduce data structures for disjoint sets.

A **Disjoint-Sets** data structure is a structure that efficiently maintains a collection $S_1\dots S_n$ of dynamic disjoint sets. 
Two sets are disjoint if their intersection is null. 
In such a data structure every set contains a **representative**, which is a particular member of the set.
**Consider the example above:**
- at the beginning there are $5$ groups, $[1], [2], [3], [4], [5]$: nobody is anybody's friends and everyone is the representative of its own group
- the next step is that $1$ and $2$ become friends, hence the groups $[1]$ and $2$ merge: we now have $4$ groups, $[1,2],[3],[4],[5]$
- next $5$ and $4$ become friends and we have 3 groups: $[1,2], [3], [4,5]$
- in last step $5$ and $1$ become friends leaving us with $2$ groups: $[1,2,4,5], [3]$
	- the **representative** of the first group is $5$ and the representative for the second group is $3$

**How can we check if two persons are in the same group?**
**This is where the representatives come in.** 
If a person $i$ belongs to a set with representative $a$, and a person $j$ belongs to a set with the same representative $a$, then $i$ and $j$ belongs to the same set and therefore they are friends  

Lets define some operations: 
- `create-set(x)`: create a new set with only the element `x`
- `merge-sets(x,y)`: merges the set that contains `x` with the set that contains `y`
- `find-set`(x): returns the representative (or a pointer to it) of the set that contains `x` 
Then, the above problem can be solved with the following algorithm: 
```java 
read(N) // N is the number of persons
for x = 1 to N
	create-set(x)

for (x,y) in friends
	if find-set(x) != find-set(y)
		merge-sets(x,y)
```

Since sets are disjoint every time `merge-sets(x,y)` is called it will destroy two sets and create a new one. 
If there are $n$ sets then after $n-1$ calls of `merge-sets(x,y)` there will remain only one set. Thus the number of `merge-sets(x,y)` calls is $\le$ to the number of `create-sets(x)`

We will analyze the running time of the disjoint-set data structure in terms of $N$ and M, where $N$ is the number of times that `create-set(x)` is called, and $M$ is the total number of times that `create-set(x)`, `merge-sets(x, y)` and `find-set(x)` are called. 
### Implementation with Linked List
One way to implement disjoint set data structures is to represent each set by a linked list. Each element will be in a linked list and will contain a pointer to the next element in the set, paired with another pointer to the representative of the set. 

**Example:** the blue arrows are the pointers to the representative and the black ones are the pointers to the next element in the set

![[Pasted image 20240228163239.png | center | 180]]

Using this implementation results in a complexity of $O(1)$ for `create-set(x)` and `find-set(x)`.
The first will simply create a new linked list whose only element  `x` and the second just returns the pointer to the representative of the set that contains `x`. 

Let's now see how to implement the `merge-sets(x,y)`. 
The easy way is to append `x`'s list onto the end of `y`'s list. 
The representative of the newly created set is the representative of the original set that contained `y`. After merging we must update the pointer of the representative of each element that was in `x`'s list, which takes linear time in terms of `x`'s list length. 
It is easy to prove that **in the worst case the complexity of the algorithm** will be $O(M^2)$, where $M$ is the number of operations `merge-sets(x,y)`. 
With this implementation the complexity will average $O(N)$ per operation where $N$ is the number of elements in all sets.

**The Weighted Union Heuristic**
Let's see how a heuristic will make the algorithm more efficient. 
Let's say that the representative of a set contains information about how many objects (elements) are in that set as well. The optimization is to always append the smaller list into the longer one and, in case of ties, append arbitrarily. 
Basically the `merge-sets` will modify the representative of small set.
Say that we have a small set `X`. The set `X` could be involved in $N$ merge operations but its representative will change $O(\log(N))$ times, as you can't merge it more than that. 
Since changing the representative costs $O(1)$ for each element the cost of $N$ union operations in $O(N\log(N))$. 
**This will bring the complexity of the algorithm** to $O(M+N\log(N))$ where $M$ is the number of operations. 

So far we reached an algorithm that solve the problem in $O(M+N\cdot\log(N))$ where $N$ is the number of persons, $M$ is the number of friendships 
Space wise the cost is $O(N)$. 
The BFS solves the problem in $O(M+N)$ time and $O(M+N)$ space. 
We have optimized space, but not time. 
### Implementation with Root Trees
The next step is to see what we can do for a faster implementation of disjoint set data structures. 
Let's represent sets by rooted trees, with each node containing one element and each tree representing one set. Each element will point only to its parent and the root of each tree is the representative of that set (and its own parent).

**Let's see how the trees will look in a step-by-step execution of the previous example:**

**Step 1:** Nobody is anybody friend
![[Pasted image 20240505195324.png | center | 250]]
We have $5$ trees and each tree has a single element, which is the root and the representative of that tree.

**Step 2:** $1$ and $2$ are friends, `merge-sets(1,2)`
![[Pasted image 20240505195510.png | center | 250]]
We now have $4$ trees, one tree contains $2$ elements and have $1$ as its root. 

**Step 3:** $5$ and $4$ are friends, `merge-sets(5,4)`
![[Pasted image 20240505195641.png | center | 250]]
**Step 4:** $5$ and $1$ are friends, `merge-sets(5,1)`
![[Pasted image 20240505195741.png | center | 180]]
For now this implementation is not faster than the algorithm that uses linked lists. 

**Two Heuristics:**
We now see how, using two heuristics, we will achieve the asymptotically fastest disjoint set data structure known so far, which is almost linear in terms of the number of operations made.
These two heuristics are called "**union by rank**" and "**path compression**".
The idea of **union by rank** is to make the root of the tree with fewer nodes point to the root of the tree with more nodes. For each node we maintain a **rank** that approximates the logarithm of the size of the subtree rooted in that node, and it is also an upper-bound of the height of the node. 
When `merge-sets(x,y)` is called the root with smaller rank is made to point to the root with larger rank. 
The idea in **"path compression"**, used for `find-set(x)` operations, is to make each node on the find path point directly to the root. This will not change any ranks. 

To **implement a disjoint set forest** with these heuristic we must keep track of ranks.
With each node `x` we keep the integer value `rank[x]`, which is `>=` the number of edges in the longest path between `x` and a sub-leaf. **todo: why?**
When `create-set(x)` is called the initial rank is zero (`rank[x] = 0`). 
When a `merge-sets(x,y)` is called the root with higher rank will become the parent of the root of lower rank (or in case of a tie we randomly chose one and increment its rank). 

We then have that the operations are defined as follows:
```java
P[N] // parents array, N number of elements
// P[x] is the parent of x 
create-set(x)
	P[x] = x
	rank[x] = x

merge-sets(x,y)
	PX = find-set(x)
	PY = find-set(y)
	if rank[PX] > rank[PY]
		P[PY] = PX
	else 
		P[PX] = PY
	if rank[PX] == rank[PY]
		rank[PY]++

find-set(x)
	if x != P[x] 
		P[x] = find-set(P[x])
	return P[x]
```

**Let's see why the above implementation is faster than the previous ones.**
If we only use the "union by rank" heuristic then we will get the same running time we achieved with the "weighted union" heuristic when we used lists. 
When we use both "union by rank" and "path compression", the worst running time is $O(M\cdot \alpha(M,N))$, where $\alpha(M,N)$ is the very slowly growing inverse of Ackerman's function.
In application $\alpha(M,N) \le 4$, which is asymptotically a constant, and that is why we can say that the running time is linear in terms of $M$. 

**Back to the problem:**
The problem then requires $O(N + M\alpha(M,N)) = O(N+M)$.
The problem of friendship is solvable in $O(N+M)$ time and $O(N)$ space using disjoint set data structure. The difference for time execution is not big if the problem is solved with BFS, but we do not need to keep in memory the vertices of the graph. 

Consider a slight variation of the problem: in a room there are $N$ persons and you have $Q$ queries. 
A query can be of two forms:
- `x y 1`, meaning that `x` is friend with `y`
- `x y 2`, meaning that we ask if `x` and `y` are in the same group of friends at that moment in time

In this case the solution with disjoint-set data structure is the fastest, giving a complexity of $O(N+Q)$. 
# 88) Minimum Spanning Tree
A **minimum spanning tree (MST)** is a subset of the edges of a connected, edge-weighted graph that connects all the vertices together without any cycles and **with the minimum possible total edge weight.**  
**It is a way of finding the most economical way to connect a set of vertices.**
**A MST is not necessarily unique.**
If all the weights of the edges in the graph are the same, then any spanning tree of the graph is a MST.

A MST of a graph $G= (V,E)$ has precisely $|V| -1$ edges. 
### 89) Solution: Kruskal's Algorithm, $O(V+E\cdot\log(E))$
Kruskal's Algorithm is a greedy algorithm and it behaves as follows: 
1) sort all the edges of the graph in ascending order of their weight
2) pick the lightest edge (the first in the ordering): if this edge do not form a cycle in the current MST then include it in the MST, otherwise discard it
3) repeat the step previous step until there are $|V| -1$ edges in the spanning tree

To solve the problem we use a **disjoint set data structure** 
```java
kruskal(G):
    Set mst // empty set, store edges of G
	G.sortEdges() // sort edges in ascending order of weight

	for v in G.nodes()
		makeSet(v) // create a disjoint set for each node

	// edges are given sorted
	for (u,v) in G.edges()
		if find(u) != find(v) // check if (u,v) make a cycle in MST
			mst.add((u,v)) // include the edge in the mst
			union(u,v) // merge the set of u and v

	return mst
```

Where: 
- `makeSet(v)` initializes a disjoint set for the vertex `v`
- `find(v)` returns the representative (root) of the set that contains `v`
- `union(v,u)` merges the set containing the nodes `u` and `v`

**Time Complexity:**
The complexity depends on the sorting and on the disjoint-set operations. 
1) **sorting** the edges takes $O(E\cdot \log(E))$ 
2) **disjoint-set operations** 
	1) `makeSet(v)` takes $O(1)$ per vertex, thus $O(V)$
	2) `find(v)` and `union(u,v)` are nearly constant time on average thanks to the union-by-rank and path-compression heuristics, therefore $O(E)$

Overall the time complexity of Kruskal's algorithm in dominated by the sorting. 
However, since $E$ is $O(V^2)$ we have that $\log(E) = O(\log(V^2)) = O(2\log(V)) = O(\log(V))$, and thus the complexity is $O(E\cdot \log(V))$.
### 90) Solution: Prim's Algorithm, $O(E + V\cdot \log(V))$
Prim's algorithm is a greedy algorithm used to find the minimum spanning tree (MST) of a weighted undirected graph.

The algorithm starts with an empty spanning tree. 
The idea is to maintain two sets of vertices. 
The first set contains the vertices already included in the MST, and the other set contains the vertices not yet included. At every step, it considers all the edges that connect the two sets and picks the minimum weight edge from these edges. 
After picking the edge, it moves the other endpoint of the edge to the set containing MST.

The algorithm behaves as follows: 
1) determine an arbitrary vertex as the starting vertex of the MST
2) follow steps 3) to 5) until there are vertices that are not included in the MST, aka fringe vertices
3) find edges connecting any tree vertices (the one in MST) with fringe vertices
4) find the minimum among these edges
5) add the vertex connected to this edge in the MST if it do not form any cycle
6) return the MST

![[Pasted image 20240523095654.png | center | 650]]
# 91) Bipartite Graph
Consider an adjacency list of a graph `adj` of $V$ vertices ($0$-indexed). 
Check whether the graph is bipartite or not. 

A **bipartite graph** is a graph whose vertices can be divided into two independent sets $U$ and $V$ such that every edge $(u,v)$ either connect a vertex from $U$ to a vertex in $V$ or vice versa. 
In other words: for every edge $(u,v)$ either $u\in U \land v\in V$ or $u\in V \land u \in V$. 
We can also say that there is no edge that connects vertices of same set.

**A graph is bipartite if the graph coloring is possible using only two colors and vertices in the same set have the same color.**
### Solution
**The solution is basically an adapted version of BFS:**
```java
bipartite(List<List<Integer>> G)
	n = G.length() // nuber of nodes
	colors[n]
	for i = 0 to n
		colors[i] = -1 // every node has no color in the beginning

	// frontier as in BFS
	Queue q // q stores pairs: (node,color)

	// needed in case G is not a connected graph
	for i = 0 to n
		// the current node is not colored
		if colors[i] == -1
			// we color the node and add it to the frontier
			colors[i] = 0
			q.add((i, 0))

			while !q.isEmpty()
				p = q.pop()
				v = p.frist // current vertex
				c = p.second // color of the current vertex

				// iterate through the neighbors of v
				for j in G.get(v)
					if colors[j] == c // neighbor with same color!
						return false 

					if colors[j] == -1 // uncolored neighbor
						colors[j] = (c == 1) ? 0 : 1
						q.add((j, colors[j]))

	return true
```

**Observation:** if $G$ is connected we can randomly pick a node and use it as a source and avoid the outer for.

**The time complexity** is $O(V+E)$ (it is nothing more than a BFS).
# 92) DAG Topological Sorting 
Topological sorting is a way to order the nodes in a Directed Acyclic Graph (DAG) such that for every directed edge `u -> v`, node `u` comes before node `v` in the ordering. In simpler terms, it's a linear ordering of vertices in such a way that for every directed edge `u -> v`, vertex `u` comes before `v` in the ordering.

The algorithm to compute the topological ordering of a DAG $G$ is the following: 
- run DFS to compute `v.f` for each vertex `v`
- as each vertex is finished (`v.f` gets valorized, the vertex become black) we insert it into a stack
- the final stack is a topological sorting