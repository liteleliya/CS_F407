"""
CS F407 Lab: Agents - a goal-based agent for warehouse navigation.

The agent keeps an internal model of the warehouse (the grid), knows its
current position and its goal, and chooses actions by searching for a
sequence of moves that reaches the goal. Search is breadth-first search:
every move costs 1, so the first time BFS reaches G it has found a shortest
collision-free path.

Usage:
    python3 warehouse_agent.py            # solve the lab map
    python3 warehouse_agent.py --tests    # also run the test suite
"""

import sys
import time
from collections import deque

WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Action name -> (row change, column change)
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class Environment:
    """The warehouse: a 2D grid of characters, plus the positions of S and G."""

    def __init__(self, text):
        self.grid = [list(row) for row in text.strip("\n").splitlines()]
        self.rows = len(self.grid)
        self.cols = max(len(r) for r in self.grid)
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == symbol:
                    return (r, c)
        raise ValueError(f"map has no '{symbol}'")

    def is_free(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < len(self.grid[r]) and self.grid[r][c] != "#"

    def result(self, pos, action):
        """Transition model: the square reached by taking `action` from `pos`."""
        dr, dc = ACTIONS[action]
        return (pos[0] + dr, pos[1] + dc)

    def render(self, path=()):
        out = [row[:] for row in self.grid]
        for r, c in path:
            if out[r][c] == ".":
                out[r][c] = "*"
        return "\n".join("".join(row) for row in out)


class GoalBasedAgent:
    """
    Goal-based agent.

    State kept by the agent: its model of the environment (the map), its
    current position, its goal, and the plan it is executing.
    Decision making: when it has no plan, it searches (BFS) for a sequence of
    actions whose predicted result is the goal, then follows that plan.
    """

    def __init__(self, env):
        self.env = env
        self.position = env.start
        self.goal = env.goal
        self.plan = None
        self.nodes_expanded = 0

    def goal_test(self, pos):
        return pos == self.goal

    def search(self):
        """Breadth-first search from the current position. Returns a list of actions or None."""
        start = self.position
        if self.goal_test(start):
            return []
        frontier = deque([start])
        parent = {start: None}          # also serves as the visited set
        self.nodes_expanded = 0
        while frontier:
            pos = frontier.popleft()
            self.nodes_expanded += 1
            for action in ACTIONS:
                nxt = self.env.result(pos, action)
                if not self.env.is_free(nxt) or nxt in parent:
                    continue
                parent[nxt] = (pos, action)
                if self.goal_test(nxt):
                    return self._reconstruct(parent, nxt)
                frontier.append(nxt)
        return None

    @staticmethod
    def _reconstruct(parent, node):
        actions = []
        while parent[node] is not None:
            node, action = parent[node]
            actions.append(action)
        return actions[::-1]

    def choose_action(self):
        """Agent function: percept (current position) -> action."""
        if self.plan is None:
            self.plan = self.search()
        if not self.plan:
            return None
        return self.plan.pop(0)

    def run(self):
        """Execute the plan step by step in the environment. Returns the visited squares."""
        trajectory = [self.position]
        while not self.goal_test(self.position):
            action = self.choose_action()
            if action is None:
                return None
            nxt = self.env.result(self.position, action)
            assert self.env.is_free(nxt), f"collision at {nxt}"
            self.position = nxt
            trajectory.append(nxt)
        return trajectory


def solve(text, verbose=True):
    env = Environment(text)
    agent = GoalBasedAgent(env)
    plan = agent.search()
    expanded = agent.nodes_expanded
    if plan is None:
        if verbose:
            print("No collision-free path exists from S to G.")
        return None, expanded
    agent.plan = list(plan)
    trajectory = agent.run()
    if verbose:
        print(f"Path found: {len(plan)} moves, {expanded} squares expanded")
        print("Actions:", " ".join(plan))
        print("Squares:", " -> ".join(f"({r},{c})" for r, c in trajectory))
        print(env.render(trajectory))
    return plan, expanded


def validate_path(text, plan):
    """Independent check: replay the plan and confirm it is legal and ends at G."""
    env = Environment(text)
    pos = env.start
    for action in plan:
        pos = env.result(pos, action)
        if not env.is_free(pos):
            return False
    return pos == env.goal


def shortest_distance(text):
    """
    Independent oracle (no queue, no BFS): repeatedly relax
    dist[n] = min(dist[n], dist[m] + 1) over all free neighbours until nothing
    changes (Bellman-Ford on the grid). Returns dist to G, or None.
    """
    env = Environment(text)
    free = [(r, c) for r in range(env.rows) for c in range(len(env.grid[r])) if env.is_free((r, c))]
    inf = float("inf")
    dist = {p: inf for p in free}
    dist[env.start] = 0
    changed = True
    while changed:
        changed = False
        for p in free:
            for a in ACTIONS:
                q = env.result(p, a)
                if env.is_free(q) and dist[q] + 1 < dist[p]:
                    dist[p] = dist[q] + 1
                    changed = True
    d = dist[env.goal]
    return None if d == inf else d


def run_tests():
    print("\n" + "=" * 60 + "\nTests\n" + "=" * 60)
    results = []

    def check(name, cond):
        results.append(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    plan, _ = solve(WAREHOUSE, verbose=False)
    check("lab map: path found", plan is not None)
    check("lab map: path is legal and reaches G", validate_path(WAREHOUSE, plan))
    # S=(1,1), G=(1,19): Manhattan distance 18 is a lower bound on any path.
    check("lab map: length >= Manhattan lower bound 18", len(plan) >= 18)
    # Row 1 is blocked at column 6, so the robot must step down and back up
    # at least once: 18 + 2 = 20 is also an upper bound on the optimum.
    check("lab map: length is 20 (Manhattan 18 + 2 forced vertical moves)", len(plan) == 20)
    check("lab map: BFS length equals independent shortest distance",
          len(plan) == shortest_distance(WAREHOUSE))

    blocked = "#####\n#S#G#\n#####"
    plan, _ = solve(blocked, verbose=False)
    check("walled-off goal: reports no path", plan is None)

    adjacent = "####\n#SG#\n####"
    plan, _ = solve(adjacent, verbose=False)
    check("adjacent goal: one move Right", plan == ["Right"])

    open_room = "#####\n#S..#\n#...#\n#..G#\n#####"
    plan, _ = solve(open_room, verbose=False)
    check("open room: shortest path = Manhattan distance 4", plan is not None and len(plan) == 4)

    # Two routes around a pillar: top is length 6, bottom detour is longer.
    two_routes = """\
#######
#S...G#
#.###.#
#.....#
#######"""
    plan, _ = solve(two_routes, verbose=False)
    check("two routes: picks the shorter one (4 moves)", plan is not None and len(plan) == 4)

    # Robot must not leave the map through a gap in the border.
    edge_gap = "#####\n S.G#\n#####"
    plan, _ = solve(edge_gap, verbose=False)
    check("gap in outer wall: never steps off the grid", plan == ["Right", "Right"])

    print(f"\n{sum(results)}/{len(results)} tests passed")
    return all(results)


def scaling_experiment():
    """'Think about it': how does BFS cost grow as the warehouse grows?"""
    print("\n" + "=" * 60 + "\nScaling: open n x n warehouse, S top-left, G bottom-right\n" + "=" * 60)
    print(f"{'n':>6} {'free squares':>13} {'expanded':>10} {'path length':>12} {'time (ms)':>10}")
    for n in (10, 20, 40, 80, 160, 320):
        rows = ["#" * (n + 2)]
        for r in range(n):
            row = ["."] * n
            if r == 0:
                row[0] = "S"
            if r == n - 1:
                row[-1] = "G"
            rows.append("#" + "".join(row) + "#")
        rows.append("#" * (n + 2))
        t0 = time.perf_counter()
        plan, expanded = solve("\n".join(rows), verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{n:>6} {n * n:>13} {expanded:>10} {len(plan):>12} {ms:>10.1f}")


if __name__ == "__main__":
    print("Warehouse map:")
    print(WAREHOUSE)
    print()
    solve(WAREHOUSE)
    if "--tests" in sys.argv:
        ok = run_tests()
        scaling_experiment()
        sys.exit(0 if ok else 1)
