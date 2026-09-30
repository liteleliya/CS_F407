"""
CS F407 Lab: Search and A* - warehouse robot navigation.

Search problem P = (S, A, T, s0, G, c):
  S  = free grid cells (row, col)
  A  = {Up, Down, Left, Right}
  T  = move one cell in that direction if the target cell is free and on the map
  s0 = the cell marked S
  G  = {the cell marked G}
  c  = 1 per move

Implements A* (with pluggable heuristics) and BFS, and runs every
experiment in the handout: Tests 1 to 4, the BFS vs A* comparison, and the
heuristic investigation (Manhattan, zero, Euclidean, 2 x Manhattan).

Usage:
    python3 astar.py
"""

import heapq
import math
from collections import deque

LAB_MAP = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

# Two routes to G. Manhattan distance makes the longer right-hand route look
# attractive, which is useful for the heuristic experiments.
TWO_ROUTE_MAP = """\
###########
#.........#
#.#######.#
#.#.....#.#
#.#.###.#.#
#S#...#.#G#
#.###.#.#.#
#.....#...#
###########"""

# Open 20 x 20 room: no obstacles between S and G.
OPEN_MAP = "\n".join(
    ["#" * 22, "#S" + "." * 19 + "#"]
    + ["#" + "." * 20 + "#" for _ in range(18)]
    + ["#" + "." * 19 + "G#", "#" * 22]
)

MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


# ---------------------------------------------------------------------------
# Problem representation
# ---------------------------------------------------------------------------
class Warehouse:
    def __init__(self, text):
        self.grid = text.strip("\n").splitlines()
        self.start = self.goal = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("map needs both S and G")

    def is_free(self, cell):
        r, c = cell
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]) and self.grid[r][c] != "#"

    def successors(self, cell):
        """Transition function T: yields (action, next_state, step_cost) for valid actions."""
        for action, (dr, dc) in MOVES.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if self.is_free(nxt):
                yield action, nxt, 1

    def is_goal(self, cell):
        return cell == self.goal

    def render(self, path):
        rows = [list(r) for r in self.grid]
        for r, c in path:
            if rows[r][c] == ".":
                rows[r][c] = "*"
        return "\n".join("".join(r) for r in rows)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def manhattan(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def zero(cell, goal):
    return 0


def euclidean(cell, goal):
    return math.hypot(cell[0] - goal[0], cell[1] - goal[1])


def manhattan_x2(cell, goal):
    return 2 * manhattan(cell, goal)


HEURISTICS = {
    "Manhattan": manhattan,
    "h = 0": zero,
    "Euclidean": euclidean,
    "2 x Manhattan": manhattan_x2,
}


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------
class Result:
    def __init__(self, path, actions, expanded):
        self.path = path
        self.actions = actions
        self.expanded = expanded

    @property
    def found(self):
        return self.path is not None

    @property
    def length(self):
        return len(self.actions) if self.found else None


def reconstruct(parent, node):
    path, actions = [node], []
    while parent[node] is not None:
        node, action = parent[node]
        path.append(node)
        actions.append(action)
    return path[::-1], actions[::-1]


def astar(problem, h=manhattan):
    """
    A* graph search.
    Frontier: a binary heap of (f, h, tie, state). Ties on f are broken by the
    smaller h (the node that looks closer to the goal), then by insertion order.
    Stale heap entries (a better g was found later) are skipped on pop.
    """
    start, goal = problem.start, problem.goal
    g = {start: 0}
    parent = {start: None}
    tie = 0
    h0 = h(start, goal)
    frontier = [(0 + h0, h0, tie, start)]
    closed = set()
    expanded = 0
    while frontier:
        f, _, _, state = heapq.heappop(frontier)
        if state in closed:
            continue
        if problem.is_goal(state):
            path, actions = reconstruct(parent, state)
            return Result(path, actions, expanded)
        closed.add(state)
        expanded += 1
        for action, nxt, cost in problem.successors(state):
            new_g = g[state] + cost
            if nxt in closed and new_g >= g[nxt]:
                continue
            if new_g < g.get(nxt, math.inf):
                g[nxt] = new_g
                parent[nxt] = (state, action)
                hn = h(nxt, goal)
                tie += 1
                heapq.heappush(frontier, (new_g + hn, hn, tie, nxt))
                closed.discard(nxt)   # re-open if reached more cheaply (only matters for inconsistent h)
    return Result(None, None, expanded)


def bfs(problem):
    """Breadth-first graph search with a FIFO queue. Goal test on generation."""
    start = problem.start
    if problem.is_goal(start):
        return Result([start], [], 0)
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        for action, nxt, _ in problem.successors(state):
            if nxt in parent:
                continue
            parent[nxt] = (state, action)
            if problem.is_goal(nxt):
                path, actions = reconstruct(parent, nxt)
                return Result(path, actions, expanded)
            frontier.append(nxt)
    return Result(None, None, expanded)


def true_distances_to_goal(problem):
    """h*(n) for every reachable cell, computed by BFS backwards from the goal."""
    dist = {problem.goal: 0}
    q = deque([problem.goal])
    while q:
        cell = q.popleft()
        for _, nxt, _ in problem.successors(cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                q.append(nxt)
    return dist


def validate(problem, result):
    """Replay the actions from S: every step legal and the last state is G."""
    cell = problem.start
    for action in result.actions:
        dr, dc = MOVES[action]
        cell = (cell[0] + dr, cell[1] + dc)
        if not problem.is_free(cell):
            return False
    return problem.is_goal(cell)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def report(name, problem, result, show_map=True):
    print(f"\n{name}")
    if not result.found:
        print(f"  no solution; states expanded = {result.expanded}")
        return
    print(f"  solution found, path length = {result.length}, states expanded = {result.expanded}")
    print(f"  valid when replayed: {validate(problem, result)}")
    print("  actions:", " ".join(result.actions))
    if show_map:
        print("  " + problem.render(result.path).replace("\n", "\n  "))


TEST_MAPS = {
    "Test 1: original warehouse": LAB_MAP,
    "Test 2: trivial case": "#####\n#SG##\n#####",
    "Test 3: no solution": "#######\n#S....#\n###.###\n#...#G#\n#######",
    # Two routes: along the top (8 moves) or down, along the bottom and up (12 moves).
    "Test 4: alternative paths": """\
###########
#S.......G#
#.#######.#
#.........#
###########""",
    # Two routes of equal length (10 each): any shortest path is acceptable.
    "Test 4b: two equally short paths": """\
#########
#.......#
#.#####.#
#S#...#G#
#.#.#.#.#
#.......#
#########""",
}


def run_tests():
    banner("Task 3: testing A* (Manhattan heuristic)")
    passed = 0
    expectations = {
        "Test 1: original warehouse": lambda r, p: r.found and validate(p, r) and r.length == true_distances_to_goal(p)[p.start],
        "Test 2: trivial case": lambda r, p: r.found and r.actions == ["Right"],
        "Test 3: no solution": lambda r, p: not r.found,
        "Test 4: alternative paths": lambda r, p: r.found and validate(p, r) and r.length == 8,
        "Test 4b: two equally short paths": lambda r, p: r.found and validate(p, r) and r.length == 10,
    }
    for name, text in TEST_MAPS.items():
        problem = Warehouse(text)
        print("\n" + text)
        result = astar(problem, manhattan)
        report(name, problem, result, show_map=result.found)
        ok = expectations[name](result, problem)
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}]")
    extra = []
    # Extra checks beyond the handout.
    p = Warehouse("###\n#S#\n###\n#G#\n###")
    extra.append(("S boxed in, G elsewhere: no solution", not astar(p).found))
    p = Warehouse("#S.......G")  # open top edge row with no border above
    r = astar(p)
    extra.append(("map without full border: stays on the grid", r.found and r.length == 8))
    p = Warehouse(LAB_MAP)
    extra.append(("A* on lab map equals BFS path length", astar(p).length == bfs(p).length))
    for name, ok in extra:
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    total = len(expectations) + len(extra)
    print(f"\n{passed}/{total} tests passed")
    return passed == total


def compare_bfs_astar():
    banner("Task 5: BFS vs A* (Manhattan)")
    for name, text in [("lab map", LAB_MAP), ("two-route map", TWO_ROUTE_MAP), ("open 20 x 20 room", OPEN_MAP)]:
        problem = Warehouse(text)
        b = bfs(problem)
        a = astar(problem, manhattan)
        free = sum(problem.is_free((r, c)) for r in range(len(problem.grid)) for c in range(len(problem.grid[r])))
        print(f"\n{name} ({free} free cells)")
        print("| Measure | BFS | A* (Manhattan) |")
        print("|---|---|---|")
        print(f"| Solution found | {b.found} | {a.found} |")
        print(f"| Path length | {b.length} | {a.length} |")
        print(f"| States expanded | {b.expanded} | {a.expanded} |")


def heuristic_investigation():
    banner("Task 6: heuristic investigation")
    maps = {"lab map": LAB_MAP, "two-route map": TWO_ROUTE_MAP, "open 20 x 20 room": OPEN_MAP}
    routes = {}
    for map_name, text in maps.items():
        problem = Warehouse(text)
        hstar = true_distances_to_goal(problem)
        print(f"\n{map_name}:")
        if map_name != "open 20 x 20 room":
            print(text)
        print(f"optimal path length (from BFS) = {hstar[problem.start]}")
        print("| Heuristic | Solution found | Path length | States expanded | Admissible on this map? | worst h(n) - h*(n) |")
        print("|---|---|---|---|---|---|")
        for name, h in HEURISTICS.items():
            r = astar(problem, h)
            over = max(h(n, problem.goal) - hstar[n] for n in hstar)
            admissible = over <= 1e-9
            print(f"| {name} | {r.found} | {r.length} | {r.expanded} | {'yes' if admissible else 'no'} | {over:+.2f} |")
            if map_name == "two-route map" and name in ("Manhattan", "2 x Manhattan"):
                routes[name] = (problem, r)
        b = bfs(problem)
        print(f"| (BFS, for reference) | {b.found} | {b.length} | {b.expanded} | n/a | n/a |")
    for name, (problem, r) in routes.items():
        print(f"\ntwo-route map, path returned with {name} (length {r.length}):")
        print(problem.render(r.path))


if __name__ == "__main__":
    banner("Task 2: A* on the lab warehouse map")
    problem = Warehouse(LAB_MAP)
    report("A* with Manhattan heuristic", problem, astar(problem, manhattan))
    ok = run_tests()
    compare_bfs_astar()
    heuristic_investigation()
    raise SystemExit(0 if ok else 1)
