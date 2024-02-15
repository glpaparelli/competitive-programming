**Extra Condensed Summaries for Oral Preparation**
# Contiguous Sub-Array Max Sum
- **dumb solution:** 
	- two nested loops, the outer one from `i = 0 to n` and the inner one from `j = i to n`.
	- at every iteration of the outer loop we set `sum = 0`. 
	- then for every iteration of `j` we add to sum `a[j]`. If `sum` is greater than `max` we update `max`
- **kadane's algorithm:**
	- based on two properties: 
		- the sum of any prefix of the optimal subarray is positive 
		- the value that precedes the optimal subarray is negative
	- iterate from start to finish
		- if `sum < 0` then `sum = nums[i]`, this let us "restart" the subarray in consideration
		- otherwise `sum += nums[i]` and if `sum > max` we update max
# Trapping Rain Waters
- **dumb solution:** for each element `i` find the max height from `[0, i-1]` and `[i+1, n-1]`, take the minimum of the two heights and subtract `h[i]`, that is the amount of water that can be stored over the element `i`, add it to the result
- **precomputation:** 
	- compute two arrays that stores the left and right leaders, `ll`, `rr`. this can be done efficiently: to compute the left leaders we traverse the array right to left, to compute the right leaders we traverse the array left to right. Every time we find a new maximum we have found a leader. 
	- the amount of water over `i` is `min(ll[i], lr[i]) - h(i)`, sum it to the result 
- **two pointers trick:**
	- `left = 0`, `right = n-1`
	- `left_max = 0`, `right_max = 0`, stores the max height seen so far from left and right respectfully 
	- iterate while `left < right`
		- if `h[left] < h[right]`
			- if `h[left] > h[left_max]`
				- `left_max = left` and no water can be stored here
			- else 
				- the amount of water stored in this element is `left_max - h[left]`
			- shift left by one to the right, `left++`
		- otherwise same reasoning
# Find Peak Element
A peak is an element that is greater than its neighbors: `a[i] > a[i-1] && a[i] > a[i+1]`. 
Note that we assume that `a[-1] = a[n] = -infinte`. 
The problem can be solved in $O(n)$ simply iterating through the array left to right. 
It can also be solved with a **binary search** approach in $O(\log(n))$. 

**Binary Search:** divide-et-conquer approach: 
- **divide:** partition the problem into smaller, more manageable, instances, in this case we divide the array in roughly two halves
- **solve:** the subproblems are solved recursively. when they reach a small enough size to be immediate they are solved. in this case the search is continued into one of the two halves of the array
- **combine:** combine the sub-solution to get the solution of the original problem. in this case there is no combination. 

**To solve the problem** we use a modified version of binary search: 
- `left = 0, right = n-1`
- `while left < right`
	- compute the middle element
	- `if a[mid] < a[mid+1]`
		- `low = mid + 1`, mid+1 could be a peak
	- otherwise `high = mid`, mid-1 could be a peak
- return low
# Maximum Path Sum
Given a tree where the nodes have an integer key, find the maximum path sum from **one leaf to another.**

First we go over **tree traversals** real quick. 
Tree traversals are a form of graph traversals and refers to the process of visiting each node in the tree exactly once. 

Such traversals are **classified by the order in which the node are visited**
- **pre-order:** visit the node, then visit the left and right subtrees
- **in-order:** visit the left subtree, visit the node, visit the right subtree
- **post-order**: visit the left subtree, visit the right subtree and then visit the current node

We also have **BFS** that visit every node in a level before going in the next level (also known as level traversal) and **DFS** that deepen the tree as much as it can before exploring other nodes

**To solve the problem** we use a custom **post-order visit** that exploits a global variable `maxSum` to store the result. 
The visit `maxSumPath` receive in input the root of the tree and then
- if the passed root is null we return $0$
- `left = maxSumPath(root.leftST)`, we recur on the left subtree
- `right = maxSumPath(root.rightST)`, we recur on the right subtree
	- the function returns the sum of the path from the root to a leaf, hence `left` is the sum from the current node to a leaf in the left subtree, analogous for `right`
- `maxSum = max(left+right+root.value, maxValue)`, update the `maxSum` if we found a bigger path from leaf to leaf passing for the current node

When the visit terminates the result is stored in the global variable `maxSum`
# Frogs and Mosquitoes
To solve the problem we first need to review **Binary Search Trees**. 
A **BST** is a tree where the key of each internal node is bigger than the keys stored in its left subtree and smaller than the keys stored in its right subtree. 

**Insertion** and **Search** are obvious and takes $O(\log(n))$. 
**Predecessor** and **Successor** also takes $O(\log(n))$ and we have that: 
- the **successor of a node** is the node with the smallest key that is bigger than the current node
	- once to the right, then left as long as possible
- the **predecessor of a node** is the node with the greatest key that is smaller than the current node
	- once to the left, then right as long as possible

**Removing an element** from a BST is harder and costs $O(h)$
- if it is a leaf just remove it
- if it is a node with a single child: copy the node of the child into the current node and delete the child
- if it is a node with two children: find the **in-order successor** of the node, copy the contents of the in-order successor in the current node and then remove the in-order successor 

**To solve the problem** we use a BST for frogs and one for mosquitoes. 
We call them `frogBST` and `mosquitoBST`: 
- in `frogBST` we store the frogs by their position
- in `mosquitoBST` we store the mosquitoes that could not be eaten when they arrived, by their landing position

We use `axis` to store mosquitoes, sorted by their position and with no overlaps
- partial overlaps are solved by moving the frogs
- total overlaps are solved by removing frogs that would never eat

We iterate through the mosquitoes by their arrival: 
- we find the predecessor of the mosquito's landing position in `frogBST`. which is the frog that may eat it
- if it can eat then it eats and its tongue grows
	- find successors of the frog in `mosquitoBST` to check if now it is possible to eat an uneaten mosquito, if so eat it and repeat
	- solve overlaps as before: either move frogs or remove them entirely

Removing elements costs $O(h)$ and we need to remove at most $n$ nodes. so the cost of the algorithm is dominated by the worst-case deletion of nodes (it happens if there is always the first frog that eats, and every time it eats it covers the next frog). 
# Maximum Number of Overlapping Intervals
We have a list of intervals $[s_i, e_i]$. 
We say that two intervals overlaps if their intersection is not empty, we need to find the maximum number of overlapping intervals. 

We use a **sweep-line approach:** we use an imaginary vertical line that sweeps over the x-axis. 
As it progresses we maintain a working solution for the problem at hand.
The working solution is updated when the vertical line reaches certain key points where some event happen.

**To solve the problem** we create an array `axis` with both starts and end of intervals, then we sort it. 
Then we sweep the axis: when we find the start of an interval we increment a counter by one, and store it if it is the maximum so far. 
When we find the end of an interval we decrease by one the interval. 
Finally  we return the maximum. 
# Check if all Integers are Covered
You have a list or ranges $[s_i, e_i]$ and an interval `[left, right]`.
We have to check that every $x \in [\text{left, right}]$ is contained by at least one of the ranges in the least. 

**dumb solution:** two nested loops, the outer one iterates over `[left, right]` and the inner one over ranges. as soon as we find a range that covers the current element in `[left, right]` we go to the next iteration. If the element is not covered by any interval we return false

**smart solution:** we use the sweep line technique to solve this in less than quadratic time. 
- create a map that associate points to the number of "active" ranges in that point
	- iterate through ranges
		- for every $s_i$ increment by one the value associated with $s_i$ in the map
		- for every $e_i$ decrease by one the value associated with $e_i + 1$ in the map (ranges are inclusive)
- `openRangesNow = 0`
- iterate with `i` from `0` to `left-1` and add to `openRangesNow` the values in the map associated with `i`
- iterate with `point` from `left` to `right`
	- `openRangesNow += map.getOrDefault(point, 0)`
	- if `openRangesNow` is `0` then the point is not covered, and we return false
# Longest K-Good Segment 
We have an array `a` and we call a sequence of **consecutive** elements of `a` a **segment.**
We call a segment **k-good** if it contains no more than `k` distinct elements

**Note:** to elements are **consecutive** if their difference in absolute value is `1`. 

**To solve the problem** we use a set and a sliding-window approach. 
Classic sliding window solution, intuitive. 
# Contiguous Subarray Sum
Given an integer array `nums` and an integer `k`, return `true` if `nums` has a **good subarray** or false otherwise. 
A **good subarray** is a subarray where: 
- its length is **at least** 2 
- the sum of the elements of the subarray is a multiple of `k`
	- an integer `x` is a multiple of `k` if there exists an integer `n` such that `x = n * k`. 
	- `0` is **always** a multiple of `k`.

**dumb solution:** brute force, from each element we compute every possible subarray starting there and check if it is a multiple of `k`

**To solve the problem** we use a support array `prefixSum` to compute the prefix sum of `nums`, where `prefixSum[i]` is the sum of the elements up to `i` in `nums`. 
Then we exploit the property "any two prefix sums that are not next to each other with the same modulo k, or a prefix sum that modulo k is zero and it is not the prefix sum of the first element will yield a valid subarray". 
We then employ a map that associate a modulo value with the index of `prefixSum` that gives that value mod `k`.

The algorithm is the following: 
- build `prefixSum`
- create a map *modulo -> index of prefixSum that gives that modulo*
- for every element `(i, pS)` in `prefixSum`
	- compute `modulo = pS % k`
	- if `modulo == 0 && i != 0` we return true
	- if `modulo` is not contained in the map we insert it `modulo -> i`
	- if `modulo` is present and the value associated with it is less than `i-1` we return true
# Fenwick Tree
We now introduce **Fenwick Trees**, also known as **Binary Indexed Trees (BIT).**

**The Fenwick Tree is a data structure that is used to maintain the prefix sum of dynamic arrays.**

Say that we have an array `a[1..n]` and we want to support two operations: 
- `sum(i)` returns the sum of all the elements of `a` up to `i`
- `add(i,v)` perform the operation `a[i] += v`

We can use **trivial** approaches that have **unsatisfactory trade-offs between the two operations**
- use only the array `a`
	- the operation `sum(i)` requires performing the prefix sum of `i` element, we pay $O(n)$ each time
	- the operation `add(i,v)` is done in constant time
- use a prefix sum support array
	- the operation `sum(i)` is done in constant time
	- the operation `add(i,v)` requires to update the prefix sum array, we pay $O(n)$ each time

A Fenwick Tree can perform both operations in **logarithmic time** while using linear space.
The Fenwick Tree is an **implicit data structure**, which means it only requires $O(1)$ additional space to the space of the input array. 

**In every node we store:** 
- the index of that number in the array
- the prefix up to that element in the array
- the range of the array that the current node it covers

**How the tree is built**
- in the first level we have nodes that covers ranges that ends with an index that is a power of $2$
- in the second level we have nodes that covers ranges that ends with an index that is a sum of two powers of $2$
	- the second level nodes covers the partial sum *starting* from the corresponding index in the array
- in the third level we have nodes that covers ranges that ends with an index that is a sum of three powers of $2$
- ...
![[Pasted image 20240110141449.png | center | 300]]

**Observations:**
- besides the tree representation we can represent the data structure as an array, as shown in the image above
- we **no longer require the original array**, any of its entry `i` can be obtained simply by doing `sum(i) - sum(i-1)`. this is why Fenwick trees are an **implicit data structure**
- consider $h = \lfloor\log(n) + 1\rfloor$, which is the length of the binary representation of any of the positions in the range $[1,n]$ (the positions of the array). Since any position can be expressed as the sum of at most $h$ power of $2$ the tree has at most $h$ levels
## Computing the sum(i) query
This query involves beginning at a node `i` and traversing up the tree to reach the node `0`
Thus `sum(i)` takes time proportional to the height of the tree, resulting in a time complexity of $\Theta(\log n)$. 

Let's consider the case `sum(7)`. 
Start at node with index $7$ and move to its parent, the node with index $6$.
Then we move to its grandparent, the node with index $4$, and stop at its great-grandparent (the dummy root 0), summing their values along the way. 
This works because the ranges of these nodes ($[1,4], [5,6], [7,7]$) collectively cover the queried range $[1,7]$. 

We can also reason the other way around: $7$ is the sum of $2^{k_2} + 2^{k_1} + 2^{k_0}$ where 
- $k_2 = 2$
- $k_1 = 1$
- $k_0 = 0$

And we can decompose the range $[1,7]$ as: 
- $[1, 2^{k_2}] = [1,4]$, that we find in the first level (the right boundary is a power of $2$)
- $[2^{k_2} + 1, 2^{k_1} + 2^{k_2}] = [5, 6]$, that we find in the second level (the right boundary is a sum of two powers of $2$)
- $[2^{k_1} + 2^{k_2} + 1, 2^{k_1} + 2^{k_2} + 2^{k_0}] = [7,7]$, that we find in the third level (the right boundary is a sum of three powers of $2$)

Answering a `sum` query is **straightforward if we are allowed to store the tree's structure.**
But it turns out that we do not strictly require this: **we can efficiently navigate from a node to its parent using bit tricks.**
This is the reason why Fenwick trees are also called **binary indexed trees**. 

**Theorem:** the binary representation of a node's parent can be obtained by removing the trailing one (i.e., the rightmost bit set to 1) from the binary representation of the node itself.
This theorem works wonders with the array representation of the fenwick tree.
# Computing the add(i,v) query
We want to perform an `add(i, v)`. 

We need to add the value `v` to each node whose range include the position `i`.

Surely, the node with index `i` is one of these nodes as its range ends in `i`. 
Additionally, the right siblings of node with index `i` also encompasses the position `i` in their ranges. 
This is because **siblings share the same starting positions, and right siblings have increasing sizes.**
The right siblings of the parent node of `i`, the right siblings of the grandparent, and so on can also contain position `i`. 

It might seem like we have to modify a large number of nodes, however we can show that **the number of nodes to be modified is at most log(n).** 
This is because **each time we move from a node to its right sibling or to the right sibling of its parent, the size of the covered range at least doubles. 
And a range cannot double more than \log(n) times.**

**Finding the Sibling:**
As before we can exploit a binary trick to efficiently find the sibling of a node. 
**The binary representation of a node and its sibling matches, except for the position of the trailing one**. When we move from a node to its right sibling, this trailing one shifts one position to the left. 

**The time complexity** of `add` is $\Theta(\log(n))$, as we observe that each time we move to the right sibling of the current node or the right sibling of its parent, the trailing one in its binary rep. shifts at lest one position to the left, and this can occur at most $\lfloor\log(n)\rfloor+1$ times.
# Update the Array
We are given an array `A[1,n]` initially set to 0. 
We want to support two operations: 
- `access(i)` returns the sum `A[1..i]`
- `range_update(l, r, v)`, updates the entries in `A[l,r]` adding to them `v`

**To solve the problem we exploit a Fenwick Tree:**
- build the Fenwick Tree from `A`
- `access(i)` is a wrapper of `sum(i)` of the Fenwick Tree operation
- `range_update(l,r,v)`: we use the operation `add(i, v)` we have seen before
	- `add(l, v)`: this trigger the addition of the value `v` to each node whose range include the position `l` in the Fenwick Tree
	- `add(r, -v)`:  this trigger the subtraction of the value `v` to each node whose range include the position `r` in the Fenwick Tree
	- we have added and subtracted the same same quantity `v` in the Fenwick tree, this means that prefix sum are coherent and the elements in `[l,r]` are increased by `v` 
# Segment Tree
A Segment Tree is a data structure that stores information about array intervals as a tree. 
This allows answering **range queries** over an array efficiently, while still being flexible enough to **allow quick modification of the array**.
The key point here is **range queries**, not only range sums!

Between answering such queries **we can modifying the elements by replacing one element of the array, or even changing the elements of a whole subsegment** (e.g., assigning all elements `a[l..r]` to any value, or adding a value to all element in the subsegment)

We consider a simple version of segment trees and the formal definition of our task is the following: given an array $a[0,\dots,n-1]$, the Segment Tree must be able to perform the following operations in $O(\log(n))$ time
1) find the sum of elements between the indices $l$ and $r$: $\Sigma_{i=l}^r\ a[i]$ 
2) change values of elements in the array: $a[i] = x$

Mind that doing that with the trivial data structures (the array itself, a prefix sum array) leads to trade-offs that are not acceptable, as we have already said with Fenwick Trees.

The structure of the segment tree is very intuitive: the root is the whole array, the segment $[0, n-1]$, then the children are the left and right halves of the parent segment. 
This is repeated until we reach leaves, that are the segments of length $1$ (the entry of the array itself)

**Example:** consider the array $a = [1,3,-2,8,-7]$
![[Pasted image 20240112095556.png | center | 450]]
The number of vertices in the worst case can be estimated by the sum 
$$1+2+4+\dots+ 2^{\lceil\log(n)\rceil+1\ \equiv\ h}<4n$$
Mind that whenever $n$ is not a power of 2, not all levels of the Segment Tree will be completely filled, as shown in the image. 
**The height of a Segment Tree** is $O(\log(n))$, because when going down the root to the leaves the size of the segments decrease approximately by half.
## Construction
Before construction: 
- decide which information we store in the node
- decide how to merge information to form the parent node

The construction if recursive and is like a post-order visit, where the "visit" is the assignment of the value of the current node
- compute the left child value by recurring on the left subtree
- compute the right child value by recurring on the right subtree
- merge the segments and result in the current node

We start the construction at the root vertex, hence we are able to compute the entire segment tree. 
The **time complexity of the construction** is $O(n)$, assuming that the merge operation is $O(1)$, as the merge operation gets called n times, which is equal to the number of internal nodes in the segment tree.
## Sum Queries
We receive two integers $l$ and $r$, and we have to compute the sum of the segment $a[l,\dots,r]$ in $O(\log(n))$ time. 
To do this we will traverse the tree and use the precomputed sums of the segments. 

Let's assume that we are currently at the vertex that covers the segment $a[tl,\dots,tr]$. 
There are three possible cases: 
1) the segment $a[l,\dots,r]$ is equal to the corresponding segment of the current index, then we are finished and we return the sum that is stored in the vertex
2) the segment of the query fall completely into the domain of either the left or the right child. In this case we can simply go to that child vertex, which corresponding segment covers the query segment, and execute the algorithm described here with that vertex
3) the query segment intersects with both children. In this case we have no other option as to make two recursive calls, one for each child. First we go to the left child, compute a partial answer for this vertex (i.e. the sum of values of the intersection), then go the right child, compute the partial answer using that vertex, and then combine the answers.

**Theorem:** for each level we only visit no more than four vertices.
And since the height of the tree is $O(\log(n))$, we receive the desired running time. 
## Update Queries
Now we want to modify a specific element in the array, let's say we want to do the assignment $a[i] = x$. 
We have to rebuild the Segment Tree, such that it corresponds to the new, modified array. 

Each level of a segment tree forms a partition of the array. Therefore an element $a[i]$ only contributes to one segment from each level. 
Thus only $O(\log(n))$ vertices need to be updated. 

It is easy to see that the update request can be implemented using a recursive function. 
The function gets passed the current tree vertex, and it recursively calls itself with one of the two child vertices (the one that contains $a[i]$) and after that recomputes its sum value, similar how it is done in the build method (that is as the sum of its two children). 
# Nested Segments
We are given $n$ segments: $[l_1, r_1],\dots, [l_n, r_n]$ on a line. 
There are no coinciding endpoints among the segments. 
The task is to determine and report the number of other segments each segment contains.
**Alternatively said:** for the segment $i$ we want to count the number of segments $j$ such that the following condition hold: $l_i < l_j \land r_j < r_i$. 
## Fenwick Tree Solution
**We use a sweep line & Fenwick tree approach.** 

We build an array `events` where every entry is `[l_i, r_i, i]`, and then we sort `events` by start of the respective range, `l_i`.

We then we build the Fenwick tree with size $2n+1$ and we scan each segment $[l_i, r_i]$ and add $1$ in each position $r_i$ in the fenwick tree. 

Now we scan the segments again. 
When we process the segment $[l_i, r_i]$ we observe that the segments already processed are only the ones that starts before the current one, as they are sorted by their starting points.
Now to find the solution of this problem for the current segment (aka the number of segments contained in the current one) we need to know the number of these segments (the ones that starts before the current one) that also end before the current one, before $r_i$. 
This is computed with a query `sum(r_i)` on the Fenwick Tree.
After computing the solution for the current segment we subtract $1$ to position $r_i$, to remove the contribution of the right endpoint of the current segment. 

## Segment Tree Solution