**Extra Condensed Summaries for Oral Preparation**
# Contiguous Sub-Array Max Sum
- **dumb solution:** 
	- two nested loops, the outer one from `i = 0 to n` and the inner one from `j = i to n`.
	- at every iteration of the outer loop we set `sum = 0`. 
	- then for every iteration of `j` we do `sum += a[j]`. If `sum` is greater than `max` we update `max`
- **kadane's algorithm:**
	- based on two properties: 
		- the sum of any prefix of the optimal subarray is positive 
		- the value that precedes the optimal subarray is negative
	- set `sum = 0`
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
- the prefix sum up to that element in the array
- the range of the array that the current node covers `[i..j]`

**How the tree is built**
- in the first level we have nodes that covers ranges that ends with an index that is a power of $2$
- in the second level we have nodes that covers ranges that ends with an index that is a sum of two powers of $2$
	- the second level nodes covers the partial sum *starting* from the corresponding index in the array
- in the third level we have nodes that covers ranges that ends with an index that is a sum of three powers of $2$
- ...
![[Pasted image 20240110141449.png | center | 300]]

**Observations:**
- besides the tree representation we can represent the data structure as an array, as shown in the image above
- we **no longer require the original array**, any of its entry `i` can be obtained simply by doing `sum(i) - sum(i-1)`. this is why the fenwick tree is an **implicit data structure**
- consider $h = \lfloor\log(n) + 1\rfloor$, ($+1$ due to the 1-indexing) which is the length of the binary representation of any of the positions in the range $[1,n]$ (the positions of the array). Since any position can be expressed as the sum of at most $h$ power of $2$ the tree has at most $h$ levels
## Computing the sum(i) query
This query involves beginning at a node `i` and traversing up the tree to reach the node `0` (the dummy root).
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
**A Segment Tree is a data structure that stores information about array intervals as a tree.** 
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

Then we build the Fenwick tree with size $2n+1$, we scan each event $[l_i, r_i, i]$ and add $1$ in each position $r_i$ in the fenwick tree. 

Now we scan the events again. 
When we process the event $[l_i, r_i, i]$ we observe that the segments already processed are only the ones that starts before the current one, as they are sorted by their starting points.

To find the solution of this problem for the current segment (aka the number of segments contained in the current one) we need to know the number of the segments that starts after the current one that also end before the current one, before $r_i$. 
This is computed with a query `sum(r_i)` on the Fenwick Tree. 

After computing the solution for the current segment we subtract $1$ to position $r_i$, to remove the contribution of the right endpoint of the current segment in the next queries.
This is why the segments that starts before the current one but overlaps with it are not counted
## Segment Tree Solution
**Let's now solve nested segments with a Segment Tree and Sweep Line**

**In words:**
Consider the list of segments `segments:` $[(s_0, e_0), \dots, (s_{n-1}, e_{n-1})]$.

Create an array `axis` where each entry is `[endpoint_i, i, isStart]`, where: 
- `endpoint_i` is either $s_i$ or $e_i$
- `i` is the index of the segment of the endpoint in `segments`
- `isStart` is true if `endpoint_i` is a $s_i$, false otherwise

Sort `axis` by the first element, the segments endpoints. 

Now we iterate over `axis`: when we find the end of a segment $i$, namely $e_i$ we do the range sum of $(s_i, e_i)$ to get the number of segments contained in the segment $(s_i, e_i)$. 
Then we increase by $1$ the segment $(s_i, s_i)$ in the segment tree. 
This works because we increase by one the start $s_i$ when its segment $i$ has been closed. 
The range sum on $(s_i, e_i)$ will count only segments that starts after $s_i$ and have already been closed (otherwise they would be $0$ in the tree).

**Alternatively said:**
- find the end of the segment $i$, $e_i$. 
- do the range sum $(s_i, e_i)$
	- all the segments $(s_j, e_j)$ that starts after $s_i$ and have already been closed have caused the increment by one of $s_j$
		- starts after $i$ and already been closed, fully contained in $i$
- the segment $i$ has been closed, increase by one $(s_i, s_i)$ in the tree.
# Powerful Array
An array of positive integers $a_1,\dots,a_n$ is given. 
Let us consider its arbitrary subarray $a_l, a_{l+1},\dots, a_r$, where $1 \le l \le r \le n$.
For every positive integer $s$ we denote with $K_s$ the number of occurrences of $s$ into the subarray.
We call the **power** of the subarray the **sum** of products $K_s \cdot K_s \cdot s$ for every positive integer $s$
The sum contains only finite number of nonzero summands as the number of different values in the array is indeed finite. 

You should calculate the power of $t$ given subarrays.
## Mo's Algorithm 
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
Consider the following worst case: we have $n$ queries, and the first query range has a length of $n$ and spans the entire array. 
Then, the subsequent queries are each one unit shorter, until the last one, which has a length of one. 
The total length of these ranges is $\Theta(n^2)$, which is also the time complexity of the solution.

**Let's now see the solution using the Mo's Algorithm.**
Suppose we have just answered the query for the range $[l',r']$ and are now addressing the range $[l,r]$. 
Instead of starting from scratch, we can update the previous answer and counters by adding or removing the contributions of colors that are new in the query range but not in the previous one, or vice versa.
Specifically, for left endpoints, we must remove all the colors in $A[l',l-1]$ if $l' < l$, or we need to add all the colors in $A[l,l']$ if $l < l'$. The same applies to right endpoints $r$ and $r'$. 

The time complexity of the algorithm remains $\Theta(qn)$. 
However we observe that a query now executes more quickly if its range significantly overlaps with the range of the previous query. 

This implementation is **highly sensitive to the ordering of the queries.**

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
Thus, for a bucket with $b$ queries, the overall time to process its queries is $O(b\sqrt n + n)$. 

Summing up over all buckets the time complexity is $\Theta(q\sqrt n + n\sqrt n)$, aka $\Theta((n+q)\sqrt n))$. 

**Final Considerations on Mo's Algorithm**
Mo’s algorithm is an **offline approach**, which means we cannot use it when we are constrained to a specific order of queries or when update operations are involved.

When implementing Mo’s algorithm, the most challenging aspect is implementing the functions `add` and `remove`. 
There are query types for which these operations are not as straightforward as in previous problems and require the use of more advanced data structures than just an array of counters
## Solution
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

# Dynamic Programming
**Dynamic Programming solves problems by combining solutions of subproblems.** 
Divide-and-Conquer algorithms partitions the problem into disjoint subproblems, solve the subproblems and then combine their solutions to solve the original problem. 
In contrast, **dynamic programming applies when subproblems overlap, that is, when sub-problems share sub-sub-problems.**

**A dynamic programming algorithm solves each sub-sub-problem just once and then saves its answer in a table, avoiding the work of recomputing the answer every time it solves each sub-sub-problem.** 

**Classic example on how Fibonacci computes the same stuff over and over** and how this can be fixed with dynamic programming, either using **tabulation** or **memoization**
- **tabulation:** completely fills a "table" (an array or a matrix) and then return the last element as a result. it is intuitive but may compute sub-solution that are not really used for the computation of the target result 
- **memoization:** compute the result of a sub-problem only if it is the first time it encounters it and then save the result in a table, so that every time the same subproblem is found again the answer is already in "cache"
# Longest Common Subsequence
Given two strings, `S1` and `S2`, the task is to find the length of the longest common subsequence, i.e. longest subsequence present in both strings. 
**Observation:** subsequence != substring. A subsequence do not have to be contiguous. 

The subproblems here is to compute the LCS on prefixes of `S1` and `S2`: given two prefixes `S1[1..i]` and `S2[1..j]` we want to compute `LSC(S1[1..i], S2[1..j])`

If we assume that we have already computed
1) `LCS(S1[1,i-1], S2[1,j-1])`
2) `LCS(S1[1,i], S2[1,j-1])`
3) `LCS(S1[1,i-1], S2[1,j])`

Then we have that 
$$\text{LCS(S1[1, i], S2[1, j])} = 
	\begin{align}
	\begin{cases}
		0\ &\text{if i = 0 or j = 0} \\
		\text{LCS(S1[1, i-1], S2[1, j-1]) + 1}\ &\text{if S1[i] == S2[j]} \\
		\max(\text{LCS(S1[1, i], S2[1, j-1]), LCS(S1[1, i-1], S2[1, j])})\ &\text{otherwise}
	\end{cases}
	\end{align}$$

**In practice** we create a matrix `dp[n+1][m+1]` where `n` is the length of `S1` and `m` is the length of `S2`, we set the first row and column to `0` (as they represent the LCS with prefix length `0`) and then iterate, using two nested loop, to check if `s[i] = s[j]` and in such case increase by one the previous LCS (`dp[i-1][j-1]`), otherwise to pick the max between the LCS of `S1` without the current character and `S2` without the current character
Then we return `dp[n][m]`, which is the LCS of $S1[1..n]$ and `S2[1..m]` aka `S1` and `S2`
# Minimum Number of Jumps
Consider an array of `N` integers `arr[]`. 
Each element represents the maximum length of the jump that can be made forward from that element. 
This means if `arr[i] = x`, then we can jump any distance `y` such that `y <= x`.  
Find the minimum number of jumps to reach the end of the array starting from the first element. 
If an element is 0, then you cannot move through that element.  
**Note:** Return -1 if you can't reach the end of the array.

**To solve this problem** we use **DP Tabulation**:
Create an array `jumps[n]`, our dp array, where `jumps[i]` will be the minimum number of jumps needed to reach the position `i`. 
`jumps` is initialized with `Integer.MAX` everywhere besides `jumps[0]`, which is `0` by definition (we start at position `0`, no jumps are needed). 

Then we use two nested loop to solve the problem
- outer loop: `i = 1 to n`
	- inner loop: `j = 0 to i`
		- we want to reach `i`: is `i` less than the number of jumps required to reach `j` (**if j is reachable**) plus the number of jumps we can make from `j`? 
			- then `jumps[i]` = the minimum between itself and `arr[j]` + `jumps[j]`

return `jumps[n-1]`
# Knapsack 0/1
We are given $n$ items. 
Each item $i$ has a value $v_i$ and a weight $w_i$. We need to put a subset of these items in a knapsack of capacity $C$ to get the maximum total value in the knapsack. 
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
# Partial Equal Subset Sum
Given an array `array[]` of size `N`, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.

As in the 0/1 knapsack problem we construct a matrix $W$ with $n+1$ rows and $v+1$ columns. 
Here the matrix contains booleans. 
The entry $W[i][j]$ is `true` if and only if there exists a subset of the first $i$ items with sum $j$, false otherwise. 
The entries of the first row $W[0][]$ are set to false, as with $0$ elements you can not make any sum, and entries of the first column $W[][0]$ are set to true, as with the first $j$ elements you can always make a subset that has sum $0$, the empty subset. 

Entry $W[i][j]$ is true either if $W[i-1][j]$ is true or $W[i-1][j - S[i]]$ is true. 
- $W[i-1][j] = \text{T}$, we simply do not take the $i$-th element, and with the elements in $1, i-1$ we already can make a subset which sum is $j$ 
- $W[i-1][j-S[i]] = \text{T}$, as before: if the subset with one element less than $i$ has sum equal to $j - S[i]$ it means that if we take $i$ we reach exactly a subset with sum $j$

**Said easy:**
- we divide the sum of the array by $2$: if the sum is not divisible by $2$ it means that there cannot be two partitions that summed gives the the sum.
- once divided is the same problem above: 
	- exists a subset of the elements that summed gives the half of the sum?
		- if yes then the answer will be true, false otherwise
# Longest Increasing Subsequence
Given an array of integers, find the **length** of the **longest (strictly) increasing subsequence**
from the given array.
**observation:** subsequence, as before, it is not a contiguous. 

Consider the sequence $S[1,n]$, let $LIS(i)$ be the $LIS$ of the prefix $S[1,i]$ whose last element is $S[i]$. 
$$LIS(i) = \begin{cases}1 + \max(LIS(j)\ |\ 1\le j\le i\ \text{and}\ S[j] < S[i]\\ 
1 \text\ \ \ \ \text{if such $j$ does not exists}
\end{cases}$$

We create a `lis` dp array where `lis[i]` = longest increasing subsequence of `S[1..i]` and we use it to solve the problem.
Obviously `lis` is all initialized to `1` as every element is by itself an increasing subsequence of length 1.

The reasoning is the following:
- outer loop (`for i in 1..n`) iterates over each element of the array starting from the second element.
	- The inner loop (`for j in 0..i`) iterates over elements before the current element `arr[i]`.
		- **if** 
			- the current element `array[i]` is greater than the element `array[j]` (hence, the element at position `i` could be the next element of the sequence that ends at position j)
			- the longest increasing subsequence that ends in `array[i]` is shorter than the sequence that ends in `j` (`lis[j]`) plus 1 (the maybe added element `array[i])
		- **then:** we say that `lis[i] = lis[j] + 1`

We finally return the maximum value of `lis`. 
This solution takes $O(n^2)$
## Speeding up LIS
We can also exploit **binary search** to compute the longest increasing subsequence more efficiently, in $O(n\log(n))$ time. 

- create a list `ans`
- insert the first element of the array in `ans`
- iterate through the element of the array, `i = 1 to n`
	- **if** `array[i]` is greater than the last element in `ans` then we append it at the end
	- **else** we binary search `ans`: we find the index, `low`, of the smallest element that is greater than `array[i]` in `ans` and we replace it: `ans[low] = array[i]`
- we return the length of `ans`

**Said easy:**
- the length of `ans` is the length of the current LIS. 
- when we substitute, we insert `nums[i]` in position `low` of `ans`. 
- the position `low` is the index of the smallest element grater than `nums[i]` in `ans`. 
	- this is obvious: we can substitute `ans[low]` with a smaller element without affecting the longest increasing subsequence, that remains valid. 
- substituting `ans[low]` with `nums[i]` let us "consider" a new LIS starting from `nums[i]`, this is because when we substitute we insert in a place that is still part of the current LIS, but if a new LIS would start entirely from the current element every element would be substituted **starting from that element** thanks to binary search
# Longest Bitonic Subsequence
Given an array `array[]` containing $n$ positive integers, a subsequence is called **bitonic** if it is first increasing, then decreasing. 
Write a function that takes an array as argument and returns the length of the longest bitonic subsequence. 
A sequence, sorted in increasing order is considered Bitonic with the decreasing part as empty. Similarly, decreasing order sequence is considered Bitonic with the increasing part as empty. 

**Reminder:** as always, subsequences are not necessarily contiguous elements.

**To solve the problem** we simply reapply LIS, first left to right and then right to left. 
# Greedy Algorithms
A **greedy algorithm** is any algorithm that follows the problem-solving heuristic of making the locally optimal choice at each stage. 
In many problems a greedy strategy **does not produce an optimal solution**, but a greedy heuristic **can yield locally optimal solutions** that approximate a globally optimal solution in a reasonable amount of time. 

Most problems for which greedy algorithms yields good solutions (good approximation of the globally optimal solution) have two property: 
1) **greedy choice property:** we can make whatever choice seems best at the moment and then solve the subproblems that arise later. A **a greedy algorithm never reconsider its choices**. This is **the main difference from dynamic programming**, which is exhaustive and is guaranteed to find the solution. 
2) **optimal substructure:** a problem exhibits optimal substructure if an optimal solution to the problem contains optimal solutions to the sub-problems. 
# Meetings in One Room
There is **one** meeting room in a firm. 
There are `N` meetings in the form of `(start[i], end[i])` where `start[i]` is start time of meeting `i` and `end[i]` is finish time of meeting `i`.  
Find the maximum number of meetings that can be accommodated in the meeting room, knowing that only one meeting can be held in the meeting room at a particular time.

**Note:** Start time of one chosen meeting can't be equal to the end time of the other chosen meeting.

**To solve this problem** we use a greedy approach: 
- sort the meetings by their starting time
- schedule the first meeting and save its ending in a variable `ending`
- iterate from the second meeting to the last one, `for meeting in meetings`
	- if `meeting` starts after the ending of the last meeting, aka if `meeting.end >= ending` then we can hold this meeting
		- increase the counter of meetings 
		- update `ending` with the end of this scheduled meeting
- return the counter
# Wilbur and Array
You have an array `a[n]` with all zeroes, and a target array `b[n]` that you want to reach. 
The only operation available on `a` is `plusOne(i)` that increase by 1 all the elements in `a[i..n]` and `minusOne(i)` that decrease by 1 all the elements in `a[i..n]`

**Example:** 
- input: 
	- 5, the size of the array $a$ `= [0,0,0,0,0]`
	- `[1,2,3,4,5]`, the target array $b$
- output: 
	- 5, as we need five `+1` operation, one for every element `i`
		- `i = 0: +1` $\rightarrow$ `a = [1,1,1,1,1]`
		- `i = 1: +1` $\rightarrow$ `a = [1,2,2,2,2]`
		- ...

What is the minimum number of operations that can be done to match `a` and `b`?

**Observation:**
The number of operations is the the difference in absolute value between an element and its successor: 
- if `a[i] = x` and `b[i] = y` with `x > y` then we will have to make `add(i)` `y - x` times
- same reasoning with negative values ecc. 

**Said easy:**
Based on the previous observation we can say that
$$\text{result} = v[1] + \Sigma_{i=2}^n\ |v_i-v_{i-1}|$$
# Woodcutters
We use a greedy approach to solve the problem. 
1) we always cut the first tree, making it fall to the left
2) we always cut the last tree, making it fall to the right
3) we **prioritize** **left falls**, meaning that if we can make the current tree falling to the left we always will: consider the current tree $i$
	1) if the previous tree is at a point $x_{i-1}$ that is smaller (aka farther) that where the current tree would fall, then we cut the tree and make it fall to the left
	2) else, if the current tree position $x_i$ plus its height $h_i$ do not fall over next tree $i+1$ we cut to the right
		1) in this case we also update $x_i$ as $x_i + h_i$, the tree has fallen to the right and from the standpoint of the next tree its position is $x_i+h_i$
	3) otherwise we do nothing
# Bipartite Graph
A **Bipartite Graph** **is a graph whose vertices can be divided into two independent sets**, $U$ and $V$, such that every edge $(u, v)$ either connects a vertex from $U$ to $V$ or a vertex from $V$ to $U$. 
In other words, for every edge $(u, v)$, either $u$ belongs to $U$ and $v$ to $V$, or $u$ belongs to $V$ and $v$ to $U$. 
We can also say that there is no edge that connects vertices of same set.

**A bipartite graph is possible if the graph coloring is possible using two colors such that vertices in a set are colored with the same color.**

You are given an adjacency list of a graph **adj**  of V of vertices having 0 based index. Check whether the graph is bipartite or not.

**To solve the problem** we use a 1-level Breadth-First Search (BFS) approach 
- iterates through nodes
	- if the node has not a color assigned then we start our BFS
		- color it "blue"
		- iterate through all of its neighbors
			- if the neighbor is not colored, color it red
			- it the neighbor is colored and it has the same color (blue) then return false, the graph is not bipartite
	- return true
