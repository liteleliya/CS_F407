# Lab 2: Agents (Constructing a Goal-Based Agent using an LLM)

Course: Artificial Intelligence (CS F407)

All results below come from `results.txt`, produced by `python3 warehouse_agent.py --tests`.

---

## The problem

```
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

Rows and columns are 0-indexed. S = (1, 1), G = (1, 19).

---

## Task 1: Understanding the problem

1. **Environment:** the warehouse floor, a 7 x 21 grid of free squares (`.`) and shelving (`#`), plus the vehicle. It is fully observable (the whole map is known), deterministic (a move always lands where expected), static (shelves don't move while the agent plans), discrete, and single-agent.
2. **Goal:** reach the dispatch square G = (1, 19) from S = (1, 1) without entering a `#` square. Among such paths we prefer a shortest one.
3. **Actions:** Up, Down, Left, Right. Each moves one square. An action is only legal if the target square is inside the grid and not an obstacle.
4. **Information the agent must keep:** its current position, the goal position, a model of the warehouse (which squares are blocked), and the transition model (what square each action leads to). While searching it also keeps a frontier of squares to explore, the set of squares already reached, and a parent pointer for each square so it can rebuild the path. While executing, it keeps the plan it is following.
5. **Why goal-based and not simple reflex:** a simple reflex agent maps the current percept straight to an action ("if the square to the right is free, go right"). Here that fails: from S, going right leads into a dead end at column 5, because the wall at (1, 6) blocks it. Correct behaviour requires looking ahead and asking "which action sequence will end at G?". The agent picks actions by simulating their consequences against an explicit goal, and that is what makes it goal-based.

**Think about it (twice as large):** BFS is still appropriate, because moves have equal cost, so BFS remains complete and optimal. Its time and memory grow linearly with the number of free squares. I measured this on open n x n warehouses:

| n | free squares | squares expanded | path length | time (ms) |
|---|---|---|---|---|
| 10 | 100 | 98 | 18 | 0.1 |
| 20 | 400 | 398 | 38 | 0.5 |
| 40 | 1,600 | 1,598 | 78 | 2.1 |
| 80 | 6,400 | 6,398 | 158 | 8.6 |
| 160 | 25,600 | 25,598 | 318 | 34.0 |
| 320 | 102,400 | 102,398 | 638 | 153.8 |

Doubling the side length quadruples the area, and the squares expanded and the time roughly quadruple too. BFS explores almost the entire warehouse, because it has no sense of direction. Difficulties that would arise in a larger warehouse:
- memory for the frontier and the visited set grows with the area;
- replanning is expensive if shelves or other vehicles move (the environment is no longer static);
- several vehicles need coordination to avoid collisions with each other;
- if movement costs differ (turns, congestion), BFS is no longer optimal, and we need uniform-cost search or A*.

An informed search such as A* with the Manhattan heuristic would expand far fewer squares, and that is the subject of the Search lab.

---

## Task 2: Designing the agent

| Component | Design |
|---|---|
| Environment | `Environment` class: the grid as a list of rows, the positions of S and G, `is_free(pos)` for bounds and obstacle checks, and `result(pos, action)` as the transition model. |
| Current state | The vehicle's position `(row, col)`. |
| Goal | `goal_test(pos)`: `pos == G`. |
| Actions | `{Up, Down, Left, Right}`, each a `(dr, dc)` offset. |
| Decision-making component | `GoalBasedAgent.search()`: breadth-first search over positions returns an action sequence. `choose_action()` then returns the next action of that plan each step. |

**Block diagram**

```
            +---------------------------------------------+
            |                  AGENT                      |
            |                                             |
 percept    |   +-------------+      +-----------------+  |
 (current --+-->| State:      |----->| "What happens   |  |
 position)  |   | position    |      |  if I do a?"    |  |
            |   +-------------+      | (map + result() |  |
            |                        |  transition)    |  |
            |   +-------------+      +--------+--------+  |
            |   | Goal: at G  |               |           |
            |   +------+------+               v           |
            |          |          +-----------------------+
            |          +--------->| Decision: BFS search  |
            |                     | for action sequence   |
            |                     | reaching the goal     |
            |                     +-----------+-----------+
            |                                 | plan
            |                     +-----------v-----------+
            |                     | next action from plan |
            +---------------------+-----------+-----------+
                                              |
                                              v  action (Up/Down/Left/Right)
                              +-------------------------------+
                              |  ENVIRONMENT: warehouse grid  |
                              +-------------------------------+
```

---

## Task 3: Prompt engineering

**Prompt used (Claude):**

> Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem shown below.
>
> (map pasted here)
>
> S is the start, G is the goal, # is an obstacle, . is free space. The vehicle may move Up, Down, Left or Right, one square per move, and every move costs the same.
> The program should:
> - represent the warehouse as a two-dimensional grid;
> - structure the code as an agent: the agent stores its current position, its goal, and a model of the environment, and has a function that chooses its next action;
> - determine a collision-free path from S to G;
> - avoid all obstacles and never leave the grid;
> - print either the path found (as a list of actions and drawn on the map) or a suitable message if no path exists;
> - explain the search algorithm chosen and why it is appropriate.
>
> Also write tests: the given map, a map where G is walled off, a map where G is next to S, and a map with two routes of different lengths.

**Output on the lab map**

```
Path found: 20 moves, 55 squares expanded
Actions: Right Right Right Down Right Right Right Up Right Right Right Right Right Right Right Right Right Right Right Right
#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

The path is 20 moves long. It is optimal: G is 18 columns to the right of S, so at least 18 horizontal moves are needed, and because (1, 6) is a shelf the vehicle must leave row 1 at least once. That costs at least one Down and one Up, so 18 + 2 = 20 is a lower bound, and BFS achieves it.

**Tests** (`python3 warehouse_agent.py --tests`): 10/10 pass.

| Test | Expected | Result |
|---|---|---|
| Lab map: a path is found | path | PASS |
| Lab map: replaying the plan never hits a wall and ends at G | legal | PASS |
| Lab map: length >= Manhattan distance 18 | yes | PASS |
| Lab map: length = 20 (argument above) | 20 | PASS |
| Lab map: BFS length = independent shortest distance (Bellman-Ford relaxation, no queue) | equal | PASS |
| G completely walled off | "no path" | PASS |
| G adjacent to S | `["Right"]` | PASS |
| Open 3x3 room | 4 moves | PASS |
| Two routes of lengths 4 and 8 | picks 4 | PASS |
| Gap in the outer wall next to S | never steps off grid | PASS |

**Questions**

1. **Did the LLM generate a working program on the first attempt?** Yes. The program ran first time and found a valid path. The one failure during testing was in *my* test, not the generated code. I had guessed by eye that the shortest path was 26 moves and wrote that into a test, which failed because BFS returned 20. Checking by hand (the Manhattan + 2 argument above) and with an independent Bellman-Ford distance computation showed that 20 is correct and my expectation was wrong. This is a useful reminder that tests need a trusted oracle too.
2. **How could the prompt be improved?** State the optimality requirement explicitly ("shortest path"), since "collision-free" alone would accept any path. Specify the coordinate convention and output format. Specify the behaviour at the grid edge (maps with a gap in the border). Ask for the number of squares expanded, so different algorithms can be compared.
3. **Which search algorithm did the LLM choose?** Breadth-first search, with a FIFO queue, a visited set (the `parent` dictionary), and parent pointers for path reconstruction.
4. **Why BFS?** Every move costs 1, so BFS finds a shortest path (it explores in order of distance from S). It is complete on a finite grid, needs no heuristic, and is simple to get right. DFS would find *a* path but not necessarily a short one. A* would also be optimal and faster, but it needs a heuristic, which is not necessary for a grid this small (only 55 squares expanded).

---

## Files

| File | Contents |
|---|---|
| `warehouse_agent.py` | `Environment`, `GoalBasedAgent` (BFS planner + plan execution), test suite, scaling experiment. |
| `results.txt` | Full output of `python3 warehouse_agent.py --tests`. |
| `SUBMISSION.md` | This document. |
