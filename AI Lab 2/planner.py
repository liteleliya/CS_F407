"""
Simple STRIPS-style planning agent using breadth-first search (BFS).

State representation
---------------------
A state is a frozenset of propositions. Each proposition is a tuple,
e.g. ("At", "Robot", "A") represents At(Robot, A). Using tuples (rather
than strings) avoids any parsing and makes propositions directly
hashable, so states (frozensets of propositions) can be stored in a
Python set for the BFS "visited" check and used as dict/set keys.

Action representation
----------------------
Each Action has:
  - name                : human-readable label, e.g. "Move(A, B)"
  - pos_preconditions    : propositions that must be TRUE in the state
  - neg_preconditions    : propositions that must be FALSE in the state
  - pos_effects          : propositions to ADD to the state
  - neg_effects          : propositions to REMOVE from the state
"""

from collections import deque


class Action:
    def __init__(self, name, pos_preconditions=None, neg_preconditions=None,
                 pos_effects=None, neg_effects=None):
        self.name = name
        self.pos_preconditions = frozenset(pos_preconditions or [])
        self.neg_preconditions = frozenset(neg_preconditions or [])
        self.pos_effects = frozenset(pos_effects or [])
        self.neg_effects = frozenset(neg_effects or [])

    def is_applicable(self, state):
        """S |= Preconditions(a): all positive preconditions must be in
        the state, and none of the negative preconditions may be in it."""
        return self.pos_preconditions.issubset(state) and \
            self.neg_preconditions.isdisjoint(state)

    def apply(self, state):
        """Apply(S, a): remove negative effects, then add positive effects."""
        return (state - self.neg_effects) | self.pos_effects

    def __repr__(self):
        return self.name


def format_state(state):
    facts = []
    for prop in state:
        head, *args = prop
        facts.append(f"{head}({', '.join(args)})")
    return "{" + ", ".join(sorted(facts)) + "}"


def bfs_plan(initial_state, goal, actions):
    """
    Breadth-first search over the space of reachable states.

    initial_state, goal : iterables of propositions (tuples)
    actions              : list of Action objects

    Returns a list of Action objects (the plan) if one is found,
    or None if the goal is unreachable.
    """
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)

    if goal.issubset(initial_state):
        return []  # already satisfied: empty plan

    visited = {initial_state}
    queue = deque([(initial_state, [])])  # (state, action-path-to-reach-it)

    while queue:
        state, path = queue.popleft()
        for action in actions:
            if not action.is_applicable(state):
                continue
            new_state = action.apply(state)
            if new_state in visited:
                continue  # already explored this state via a shorter/equal path
            new_path = path + [action]
            if goal.issubset(new_state):
                return new_path
            visited.add(new_state)
            queue.append((new_state, new_path))

    return None  # goal state is unreachable from initial_state


def run_planner(initial_state, goal, actions, label=""):
    """Runs bfs_plan, then prints the plan and the state after each action."""
    if label:
        print(f"\n=== {label} ===")
    print("Initial state:", format_state(initial_state))
    print("Goal:", format_state(goal))

    plan = bfs_plan(initial_state, goal, actions)

    if plan is None:
        print("No plan found.")
        return None

    if not plan:
        print("Goal already satisfied. Empty plan.")
        return []

    print("\nPlan found:")
    state = frozenset(initial_state)
    for i, action in enumerate(plan, start=1):
        state = action.apply(state)
        print(f"  {i}. {action.name}")
        print(f"     -> state S{i}: {format_state(state)}")
    return plan


# ---------------------------------------------------------------------------
# Warehouse domain (from the lab handout, Section 3)
# ---------------------------------------------------------------------------

def build_warehouse_domain(include_pickup=True, include_irrelevant_move=False):
    """
    Locations A, B, C. Robot and Package both start at A. The robot can
    move between connected locations (A<->B, B<->C), pick up the package
    when co-located with it, and drop it at its current location.
    """
    locations = ["A", "B", "C"]
    connections = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]

    actions = []

    # Move(x, y): robot travels between connected locations.
    for x, y in connections:
        actions.append(Action(
            name=f"Move({x}, {y})",
            pos_preconditions={("At", "Robot", x)},
            neg_preconditions=set(),
            pos_effects={("At", "Robot", y)},
            neg_effects={("At", "Robot", x)},
        ))

    if include_pickup:
        # PickUp(Package, loc): robot and package co-located -> robot holds package.
        for loc in locations:
            actions.append(Action(
                name=f"PickUp(Package, {loc})",
                pos_preconditions={("At", "Robot", loc), ("At", "Package", loc)},
                # Assumption: the robot has one gripper, so it cannot pick up
                # a package it is already holding. This precondition is not
                # spelled out in the handout but is needed to keep the
                # action's effects consistent.
                neg_preconditions={("Holding", "Package")},
                pos_effects={("Holding", "Package")},
                neg_effects={("At", "Package", loc)},
            ))

    # Drop(Package, loc): robot holding package at loc -> package placed at loc.
    for loc in locations:
        actions.append(Action(
            name=f"Drop(Package, {loc})",
            pos_preconditions={("At", "Robot", loc), ("Holding", "Package")},
            neg_preconditions=set(),
            pos_effects={("At", "Package", loc)},
            neg_effects={("Holding", "Package")},
        ))

    if include_irrelevant_move:
        # Test C from the handout: an action that moves the robot but has
        # no effect on the package, to check BFS doesn't mistake "robot
        # reached C" for "package reached C".
        actions.append(Action(
            name="Wave(Robot)",
            pos_preconditions=set(),
            neg_preconditions=set(),
            pos_effects={("Waved", "Robot")},
            neg_effects=set(),
        ))

    return actions


if __name__ == "__main__":
    initial_state = {("At", "Robot", "A"), ("At", "Package", "A")}
    goal = {("At", "Package", "C")}

    # Test A: solvable problem (the handout's original warehouse problem).
    actions = build_warehouse_domain()
    run_planner(initial_state, goal, actions, label="Test A: solvable problem")

    # Test B: impossible problem (PickUp removed -> package can never move).
    actions_no_pickup = build_warehouse_domain(include_pickup=False)
    run_planner(initial_state, goal, actions_no_pickup, label="Test B: impossible problem (no PickUp)")

    # Test C: irrelevant action present (should not affect the plan found).
    actions_with_noise = build_warehouse_domain(include_irrelevant_move=True)
    run_planner(initial_state, goal, actions_with_noise, label="Test C: irrelevant action present")
