# Lab 3: Search and A*

Course: Artificial Intelligence (CS F407)

Every number below comes from `results.txt`, produced by `python3 astar.py` (pure Python 3, no libraries). Anyone can reproduce the experiments by running that one command.

---

## 1. Formulation of the search problem (Task 0)

```
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
```

| Component | Specification |
|---|---|
| State S | The robot's cell `(row, col)`, where the cell is not `#`. The lab map has 64 free cells, so 64 states. |
| Actions A | {Up, Down, Left, Right} |
| Transition T | T((r, c), Up) = (r-1, c), Down = (r+1, c), Left = (r, c-1), Right = (r, c+1), defined only when the target cell is on the map and is not `#`. |
| Initial state s0 | (1, 1), the cell marked S |
| Goal G | {(7, 15)}, the cell marked G |
| Cost c | c(s, a, s') = 1 for every move |

(a) **Information needed to specify a state:** just the robot's row and column. The map is fixed and fully known, so it is part of the problem, not the state.
(b) **What makes an action invalid:** the target cell is an obstacle `#` or lies outside the map.
(c) **Deterministic?** Yes. Each action from a given state has exactly one outcome, and the map is static and fully observable.
(d) **What counts as a solution:** a sequence of valid actions that takes s0 to G. An *optimal* solution is one with the minimum number of moves.

---

## 2. Design of the agent (Task 1)

1. **State in Python:** a tuple `(row, col)`. It is hashable, so it can be a dict key or set member.
2. **Warehouse:** the list of map strings, wrapped in a `Warehouse` class that records `start` and `goal`.
3. **Valid actions:** `Warehouse.successors(cell)` tries all four offsets and yields `(action, next_cell, 1)` only when `is_free(next_cell)` (in bounds and not `#`).
4. **Goal recognition:** `is_goal(cell)` compares with the goal cell. A* applies it when a node is **popped** (expanded), not when it is generated. That is what makes A* optimal.
5. **Frontier contents:** a priority queue of entries `(f, h, tie_counter, state)`. The g values are kept in a dict `g[state]`.
6. **Path reconstruction:** a dict `parent[state] = (previous_state, action)`, followed backwards from the goal and then reversed.

**Reported on termination:** whether a solution was found, the path (as actions and drawn on the map), the path length, the number of states expanded, and whether the path is valid when replayed from S.

---

## 3. The final program

See [`astar.py`](astar.py). Main pieces:

- `Warehouse`: the problem (states, transition function, goal test, rendering).
- `astar(problem, h)`: A* graph search with a pluggable heuristic.
- `bfs(problem)`: breadth-first search on the same problem.
- `true_distances_to_goal(problem)`: h\*(n) for every cell, found by BFS backwards from G. It is used as an oracle to test path optimality and to check admissibility empirically.
- `validate(problem, result)`: replays the returned actions from S and checks every step is legal and that the path ends at G.

---

## 4. Prompts used with the LLM (Claude)

**Task 2 prompt:**

> I am implementing a simple goal-based search agent in Python.
> The environment is a grid represented by an ASCII map. The agent starts at S and must reach G. The symbols # represent obstacles and . represents free cells. The agent can move up, down, left, or right, and every movement has cost 1.
> Implement A* search. Use Manhattan distance as the heuristic: h(n) = |x - xG| + |y - yG|.
> The program should: represent grid positions as states; maintain an appropriate frontier; calculate g(n), h(n) and f(n); avoid repeatedly expanding the same state; reconstruct the path when the goal is reached; report the path and its length; report the number of states expanded.
> Represent a state as a (row, col) tuple, keep the map in a class with a successors() method, and make the heuristic a parameter of the search function so it can be swapped.
> Keep the implementation simple and explain the main components of the code.

**Task 5 prompt:**

> Add a breadth-first search function that solves the same Warehouse problem and reports the same measures (found, path length, states expanded). Do not change the warehouse or the A* function.

**Task 6 prompt:**

> Explain why Manhattan distance is an appropriate heuristic for this warehouse when the robot can only move horizontally and vertically. Then add three alternative heuristics as functions with the same signature: h(n) = 0, Euclidean distance, and 2 times Manhattan distance.

**LLM explanation for Manhattan (summarised and checked):** with only horizontal and vertical unit moves, the robot needs at least |dx| horizontal moves and |dy| vertical moves to reach G, whatever the obstacles. So Manhattan distance never overestimates (it is admissible). It also changes by exactly 1 per move, so it is consistent: h(n) <= 1 + h(n'). Obstacles can only make the true distance longer. I verified the admissibility claim directly: for every free cell on each map, `h(n) - h*(n) <= 0` (the last column of the Task 6 tables).

---

## 5. Test results (Task 3)

| Test | Map | Expected | Result |
|---|---|---|---|
| 1. Original warehouse | lab map | a path, length equal to the true shortest distance (40) | found, length **40**, **63** states expanded, valid on replay. PASS |
| 2. Trivial case | `#SG##` | one move Right | `Right`, length 1, 1 state expanded. PASS |
| 3. No solution | handout map with G walled in | report failure, terminate | "no solution" after 9 expansions (the whole reachable region). PASS |
| 4. Alternative paths | two routes, 8 (top) or 12 (bottom) | the 8-move path | length **8**. PASS |
| 4b. Equal alternatives | two routes of 10 moves each | any path of length 10 | length 10 (top route). PASS |
| extra | S boxed in by walls | failure | PASS |
| extra | map with no border row (`#S.......G`) | stays on the grid, length 8 | PASS |
| extra | lab map | A* length equals BFS length | PASS |

**8/8 pass.**

Path found on the lab map (40 moves):

```
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

**A mistake the tests caught (in my test, not the code):** my first version of Test 4 used a map I believed had routes of lengths 8 and 10. A* returned 10 and the test failed. Counting by hand showed both routes in that map were 10 moves long, so my expectation was wrong. I kept that map as Test 4b ("equally short paths") and built a new Test 4 whose two routes really do differ (8 vs 12). This is a good example of "working output != validated algorithm" cutting both ways: the expected value in a test needs checking as well.

---

## 6. Inspecting the A* code (Task 4)

| Concept | Where in `astar.py` |
|---|---|
| State | `(row, col)` tuples. `Warehouse.start` / `goal`. |
| Action | the keys of `MOVES` (`"Up"`, `"Down"`, `"Left"`, `"Right"`) |
| Transition | `Warehouse.successors(cell)`, which yields `(action, nxt, cost)` for legal moves only |
| Goal test | `problem.is_goal(state)` right after `heapq.heappop` in `astar` |
| g(n) | the dict `g`. `new_g = g[state] + cost` |
| h(n) | `hn = h(nxt, goal)` in the successor loop (and `h0` for the start) |
| f(n) | `new_g + hn`, the first element of each heap entry |
| Frontier | `frontier`, a Python list used as a binary heap through `heapq` |
| Visited states | `closed` (a set of expanded states) together with `g` (best known cost to every generated state) |
| Path reconstruction | `reconstruct(parent, node)`, which walks the `parent` pointers back to S and reverses |

(a) **Frontier data structure:** a binary min-heap (`heapq`) of tuples `(f, h, tie, state)`.
(b) **Choosing the next state:** `heappop` returns the entry with the smallest f. Ties go to the smaller h (the node that looks closer to G), then to the earlier insertion. Entries for states already in `closed` are stale duplicates and are skipped.
(c) **Where h is calculated:** inside the successor loop, `hn = h(nxt, goal)`, once per time a state is (re)generated.
(d) **Is f = g + h explicit?** Yes: `heapq.heappush(frontier, (new_g + hn, hn, tie, nxt))`.
(e) **Preventing repeated exploration:** a state is expanded only if it is not in `closed`. A successor is pushed only if the new g is strictly better than the best g known for it. Lazy deletion (skipping stale heap entries on pop) avoids a costly "decrease-key" operation.

---

## 7. BFS vs A* (Task 5)

**Lab map (64 free cells):**

| Measure | BFS | A* |
|---|---|---|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

(a) Both found a solution. (b) The paths have the same length, 40, which is optimal. (c) Neither expanded fewer states: both expanded 63 of the 64 free cells.

(d) **Why A* did not help here, and when it does:** the lab map is really a maze of one-cell-wide corridors. The path first has to go *right and down*, then *back up and left* (the `Up Up Left x6 Up Up` section), and finally right and down again. The Manhattan heuristic points straight at G, so it pulls A* into the bottom corridor (row 7), a dead end that stops at column 13 right next to G. A* explores that dead end before accepting the detour. Almost every cell ends up being expanded by both algorithms. To see the difference a heuristic can make, I ran the same comparison on two more maps:

| Map | BFS expanded | A* expanded | Path length (both) |
|---|---|---|---|
| two-route map (40 free cells) | 30 | 26 | 16 |
| open 20 x 20 room (400 free cells) | 398 | **38** | 38 |

In the open room, A* expanded only the 38 cells on its path, while BFS expanded almost the whole room. A* expands fewer states when h is informative, meaning h(n) is close to h\*(n). In open space Manhattan distance is *exactly* h\*, so A* goes straight to the goal. In a maze where the true path winds away from G, h badly underestimates and A* behaves much like blind search.

*Counting convention:* BFS applies its goal test when a node is generated. A* applies it when a node is popped, and the goal node is not counted as expanded. The two counts are therefore directly comparable ("states whose successors were generated").

---

## 8. Heuristic investigation (Task 6)

Admissibility was checked empirically. For each heuristic, the "worst h - h\*" column is max over all free cells n of h(n) - h\*(n), with h\* computed exactly by a backward BFS from G. A value of 0 or less means admissible on that map.

**Lab map** (optimal = 40)

| Heuristic | Found | Path length | Expanded | Admissible? | worst h - h\* |
|---|---|---|---|---|---|
| Manhattan | yes | 40 | 63 | yes | 0 |
| h = 0 | yes | 40 | 63 | yes | 0 |
| Euclidean | yes | 40 | 63 | yes | 0 |
| 2 x Manhattan | yes | 40 | 67 | no | +14 |

**Two-route map** (optimal = 16)

| Heuristic | Found | Path length | Expanded | Admissible? | worst h - h\* |
|---|---|---|---|---|---|
| Manhattan | yes | 16 | 26 | yes | 0 |
| h = 0 | yes | 16 | 31 | yes | 0 |
| Euclidean | yes | 16 | 25 | yes | 0 |
| 2 x Manhattan | yes | **24** | 27 | no | +12 |

**Open 20 x 20 room** (optimal = 38)

| Heuristic | Found | Path length | Expanded | Admissible? | worst h - h\* |
|---|---|---|---|---|---|
| Manhattan | yes | 38 | 38 | yes | 0 |
| h = 0 | yes | 38 | 399 | yes | 0 |
| Euclidean | yes | 38 | 362 | yes | 0 |
| 2 x Manhattan | yes | 38 | 38 | no | +38 |

On the two-route map, 2 x Manhattan returned a **24-move path when a 16-move path exists**:

```
optimal (Manhattan, 16)        2 x Manhattan (24)
###########                    ###########
#*********#                    #.........#
#*#######*#                    #.#######.#
#*#.....#*#                    #.#*****#.#
#*#.###.#*#                    #.#*###*#.#
#S#...#.#G#                    #S#***#*#G#
#.###.#.#.#                    #*###*#*#*#
#.....#...#                    #*****#***#
###########                    ###########
```

**Findings:**
1. **h = 0:** A* becomes uniform-cost search (equivalent to BFS for unit costs). It is always optimal, but it has no guidance: in the open room it expanded 399 states against 38 for Manhattan.
2. **Euclidean:** admissible, because a straight line is never longer than a 4-connected path, so it is always optimal. But it is *less informed* than Manhattan (Euclidean <= Manhattan everywhere), so it can expand more states. In the open room it expanded 362 against 38. Off the straight line to G, Euclidean distance underestimates the true (Manhattan) distance, so many cells have f below the optimal cost 38 and all of them must be expanded before G is popped.
3. **2 x Manhattan:** not admissible (it overestimates by up to +38). It makes A* greedier. When the greedy direction is right (the open room) it is as fast as Manhattan. When the greedy direction is misleading (the two-route map), it commits to the route that *looks* closer and returns a **suboptimal** path (24 vs 16). On the lab map it even expanded more states (67) than there are free cells, because the inconsistent heuristic made it re-open cells after finding cheaper routes to them.

**Answer to "what happens when the heuristic is too optimistic or too aggressive?"** An overly optimistic (small) heuristic keeps A* optimal, but it degrades towards blind search and expands more states (h = 0 is the extreme). An overly aggressive (overestimating) heuristic can reduce the expansions but loses the optimality guarantee: A* may return a longer path, as seen on the two-route map. Manhattan sits at the useful point for this problem: it is admissible and consistent, and it is as large as possible without overestimating in open space.

---

## 9. Evaluating the LLM-generated agent (Task 7)

1. **Correct immediately:** the problem representation, the successor function, the heap-based frontier with f = g + h, the goal test on pop, and path reconstruction. A* returned an optimal, valid path on the first run.
2. **Bugs or design problems found:** there were no functional bugs in A*. Design points I changed or added: (i) a tie-breaker `(f, h, counter)` so that the heap never compares two states directly and tie behaviour is deterministic; (ii) lazy deletion with a `closed` set plus re-opening, so the code stays correct when an inconsistent heuristic (2 x Manhattan) is plugged in; (iii) replay validation of every returned path. The only failing test during development was caused by a wrong expected value in my own Test 4 map (Section 5).
3. **How problems were found:** by writing tests whose correct answer I knew independently (trivial map, no-solution map, maps with a known shortest length), and by computing h\* exactly with a backward BFS and comparing.
4. **Unfamiliar terminology or data structures:** "lazy deletion" in a heap, and the difference between *admissible* and *consistent* heuristics. Consistency is what makes the "never re-open a closed node" shortcut safe.
5. **Modified the generated code?** Yes: the tie-breaker, the re-opening logic, the `validate` replay, the h\* oracle, and the extra maps for Tasks 5 and 6.
6. **Most useful tests:** Test 3 (no solution), which checks termination; the h\*-based optimality check; and the two-route map, the only experiment that exposed the practical cost of an inadmissible heuristic.
7. **Could I trust it without testing?** No. On the lab map every heuristic, including the inadmissible one, returned the optimal length 40. Testing on only that map would have wrongly suggested that 2 x Manhattan is "just as good".
8. **What I understand now that I didn't before:** a heuristic only helps when it is informative *for the specific map*. On a corridor maze A* was no better than BFS. And optimality depends on admissibility, which can be checked mechanically when h\* is computable.

**Distinguishing contributions:**

| | |
|---|---|
| Designed myself | the P = (S, A, T, s0, G, c) formulation, the state/frontier/parent design (Sections 1 and 2), all test maps and their expected answers, the h\* admissibility check, and the choice of maps that make the heuristics behave differently |
| LLM suggested | the A* and BFS implementations, the heap usage, the Manhattan explanation |
| Accepted | the overall A* structure, the goal test on pop, the successor generator |
| Changed | tie-breaking, closed-set re-opening, replay validation, experiment harness |
| Tested | Tests 1 to 4 plus 4 extra cases (8/8 pass), BFS vs A* on 3 maps, 4 heuristics on 3 maps with admissibility checked against exact h\* |

---

## 10. Final reflection

1. **Why formulate the problem first?** The formulation decides what a state is, which moves are legal, and what "best" means. If these are unclear, any code, generated or not, is solving an unknown problem, and there is nothing to test it against. Writing down S, A, T, s0, G and c gave me the tests directly. For example, "c = 1 per move" means the correct path length is a number I can compute independently. "Moving into `#` is invalid" gives the replay check.
2. **In what sense is A\* informed?** Besides the cost so far, g(n), it uses domain knowledge about how far each state is from the goal, h(n), to decide which state to expand next. BFS and uniform-cost search order the frontier only by depth or cost from the start. A* orders it by the estimated total cost of a solution through n, so it can ignore regions that point away from the goal. In the open room that cut the expansions from 398 to 38.
3. **Why the heuristic matters:** the heuristic controls both efficiency and correctness. An informative admissible h (Manhattan) gives the optimal path with few expansions. A weak admissible h (0, Euclidean) stays optimal but expands many more states. An inadmissible h (2 x Manhattan) can be faster but returned a path 50% longer than optimal on the two-route map. The quality of h also depends on the map: on the corridor maze even Manhattan gave no benefit.
4. **What the LLM contributed:** a correct first implementation of A* and BFS in seconds, standard idioms (`heapq`, parent-pointer reconstruction), and a clear explanation of why Manhattan distance suits 4-connected grids. It turned a design into working code quickly, leaving my time for designing tests and experiments.
5. **What could go wrong without testing?** Code can produce a plausible-looking path that is not optimal (as 2 x Manhattan did), loop forever on unsolvable maps, step through walls or off the grid, or misreport the path length or expansion counts. Tests on a single map can also hide these problems: every heuristic looked equally good on the lab map. Only a known-answer test suite, an independent oracle (h\*), and varied maps show whether the program implements A* correctly, rather than something that merely resembles it.
