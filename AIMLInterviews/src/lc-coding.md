# <a name="coding"></a> General Coding Interview (Algorithms and Data Structures) :computer:

ML engineers are expected to have a solid understanding of software engineering fundamentals, particularly algorithms, data structures, complexity analysis, and writing correct, readable code.

Depending on the company and seniority level, an interview loop usually includes one or two general coding rounds. These rounds are similar to software engineering coding interviews. Interviewers typically evaluate whether you can:

- Clarify an ambiguous problem and state assumptions
- Select appropriate data structures and algorithms
- Explain time and space complexity
- Turn an initial solution into a more efficient one
- Write readable, testable code without relying heavily on an IDE
- Communicate your reasoning and test edge cases

## Preparation approach

Use patterns to organize your preparation instead of memorizing isolated solutions:

1. Learn the core data structures and the signals associated with each pattern.
2. Solve a small set of representative problems with full explanations.
3. Revisit missed problems until you can derive the solution independently.
4. Mix patterns so the problem type is not revealed in advance.
5. Finish with timed, interview-style practice and company-targeted sets.

For each practice session:

- **Time-box each problem to about 15-20 minutes.** If you are stuck, study the solution, tag the problem for review, and re-attempt it within 48 hours without looking at the answer.
- **Track your work in a sheet:** problem, pattern, difficulty, status (solved or redo), and time spent. Use the results to identify weak patterns for later review and timed mocks.
- **Prioritize quality over raw count.** Solving 300 problems with strong pattern recall is more useful than completing 400 that are only half-understood. Pair this practice with [Chapter 2: ML Coding](./MLC/ml-coding.md) so both coding-round formats are covered.

## LeetCode problem volume and difficulty

[LeetCode](https://leetcode.com/) remains one of the most useful platforms for coding interview practice. I practiced around 350 problems. A reasonable target is **300-400 well-reviewed problems**, with **about 350** as a strong reference point. Quality of review matters more than maximizing the raw count.

| Target total | Easy (35%) | Medium (55%) | Hard (15%*) |
| ---: | ---: | ---: | ---: |
| 300 | ~105 | ~165 | ~45 |
| **350 (recommended)** | **~120** | **~190** | **~50** |
| 400 | ~140 | ~220 | ~60 |

\* The 35/55/15 split totals approximately 105%, so treat these counts as directional rather than exact quotas. Most preparation should center on Medium problems; use Easy problems to build fluency and Hard problems to deepen patterns that are relevant to your target roles.

You can find examples of questions I practiced in [My Leet Sheet](https://docs.google.com/spreadsheets/d/1Ry_JLeOp1fw9mSbbp4ji5l3CZNiCwf8Atevpp0YML6E/edit?usp=sharing).

## Core coding patterns

| Pattern focus | Common signals and techniques |
| --- | --- |
| Arrays, strings, and hashing | Frequency counting, lookup tables, prefix sums, sorting |
| Two pointers and sliding window | Contiguous ranges, pairs, deduplication, expanding or shrinking windows |
| Stack and monotonic stack | Matching delimiters, next greater element, expression evaluation |
| Binary search | Sorted data, monotonic answer spaces, boundary finding |
| Linked lists | Pointer manipulation, fast and slow pointers, reversal, cycle detection |
| Intervals | Merge, overlap, scheduling, sweep line |
| Trees (BFS, DFS, BST) | Traversal, recursion, path problems, ordering properties |
| Tries, heaps, and priority queues | Prefix lookup, top-k, streaming minima or maxima |
| Backtracking | Combinations, permutations, subsets, constraint search |
| Graphs | Matrix traversal, BFS/DFS, topological sort, union-find |
| Advanced graphs and greedy | Shortest paths, minimum spanning trees, locally optimal choices |
| 1-D dynamic programming | State transitions over sequences, take/skip decisions |
| 2-D dynamic programming | Grids, two-sequence problems, multi-variable state |
| Math, geometry, and bit manipulation | Invariants, coordinates, masks, XOR, bitwise state |

Do not practice these only in separate blocks. Once the foundations are stable, use mixed sets so you must recognize the pattern from the problem statement.

## Company-targeted and timed practice

- Use company-tagged problems for your target company, emphasizing questions reported in the **last 6-12 months**.
- Prioritize recurring patterns and frequently reported questions over one-off reports.
- Practice under the target company's expected format: fixed time, no hints, and executable or whiteboard-style code as appropriate.
- Run mixed timed mocks rather than selecting a known topic in advance.
- After each mock, classify misses by pattern, implementation, complexity, communication, or testing, then redo the weakest areas.
- Avoid memorizing company-tagged answers. You should be able to explain why the approach works and adapt it when constraints change.

## During the interview

1. Restate the problem and clarify inputs, outputs, constraints, and edge cases.
2. Describe a correct baseline before optimizing, unless the optimal approach is immediately clear.
3. Explain the chosen data structure, algorithm, and complexity before coding.
4. Write in small, verifiable steps and keep the interviewer informed.
5. Test normal cases, boundaries, empty inputs, duplicates, and failure cases.

## Educative.io

[Educative.io](https://www.educative.io/) provides interactive, text-based explanations and visualizations that are useful for learning interview patterns. Relevant coding resources include:

- [Grokking the Coding Interview Patterns](https://www.educative.io/courses/grokking-coding-interview)
- [Grokking Dynamic Programming Patterns for Coding Interviews](https://www.educative.io/courses/grokking-dynamic-programming-interview)
- [Grokking the Machine Learning System Design Interview](https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview)

For LLM, RAG, agentic AI, and GenAI system-design courses, see [GenAI Learning Resources](./genai-resources.md).
