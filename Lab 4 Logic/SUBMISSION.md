# AI Laboratory - Logical Reasoning for Planning: Submission

Course: Artificial Intelligence (CS F407) - Undergraduate
Lab: *Laboratory – Logical Reasoning for Planning: Using an LLM to Construct and Test a Simple Planning Agent*

This document follows Section 4 ("Submission") of the lab handout item by item.

---

## 1. Specification of the Planning Problem (Task 0)

**Scenario.** A warehouse robot must deliver a package from location `A` to location `C`. Locations `A`, `B`, `C` are connected as a path: `A–B–C`.

**Initial state**

```
I = { At(Robot, A), At(Package, A) }
```

**Goal**

```
G = { At(Package, C) }
```

**Available actions**

| Action | Positive preconditions | Negative preconditions | Positive effects | Negative effects |
|---|---|---|---|---|
| `Move(X, Y)` for each connected pair (`A,B`),(`B,A`),(`B,C`),(`C,B`) | `At(Robot, X)` | - | `At(Robot, Y)` | `At(Robot, X)` |
| `PickUp(Package, L)` for `L ∈ {A,B,C}` | `At(Robot, L)`, `At(Package, L)` | `Holding(Package)` * | `Holding(Package)` | `At(Package, L)` |
| `Drop(Package, L)` for `L ∈ {A,B,C}` | `At(Robot, L)`, `Holding(Package)` | - | `At(Package, L)` | `Holding(Package)` |

\* The negative precondition on `PickUp` (`¬Holding(Package)`) is **not** stated explicitly in the handout's description of the action. I added it as a modeling assumption (single gripper: the robot cannot pick up a package it is already holding) - see Section 9, "LLM contribution / modifications," for why this was added and flagged rather than silently included.

**Which actions are initially applicable?**

- `PickUp(Package, A)`: preconditions `At(Robot,A)` and `At(Package,A)` both hold in `I`, and `Holding(Package)` is absent from `I` → **applicable**.
- `Drop(Package, C)`: requires `At(Robot,C)` and `Holding(Package)`. Neither holds in `I` (`At(Robot,A)` is true, `At(Robot,C)` is not, and the robot is not holding anything yet) → **not applicable**.

---

## 2. Manually Constructed Plan (Task 1)

Constructed by hand before writing/generating any code, as required:

| State | Facts |
|---|---|
| S0 | At(Robot,A), At(Package,A) |
| S1 | At(Robot,A), Holding(Package) |
| S2 | At(Robot,B), Holding(Package) |
| S3 | At(Robot,C), Holding(Package) |
| S4 | At(Robot,C), At(Package,C) |

Plan: `PickUp(Package,A)`, `Move(A,B)`, `Move(B,C)`, `Drop(Package,C)`.

`S4 ⊨ G` since `At(Package,C) ∈ S4`. This matches the plan later found automatically by BFS in `planner.py` (Section 5, Test A) - confirming the hand-constructed plan and the generated program agree.

---

## 3. Prompt Used with the LLM (Task 2)

The exact prompt given to the LLM (Claude), as specified by the handout's Task 2 box:

> I want to implement a simple planning agent in Python.
> Represent a state as a set of logical propositions.
> Each action should contain:
> - a name;
> - positive preconditions;
> - negative preconditions;
> - positive effects;
> - negative effects.
>
> An action is applicable if all of its preconditions are satisfied by the current state.
> When an action is applied:
> 1. remove its negative effects from the state;
> 2. add its positive effects to the state.
>
> Use breadth-first search to find a sequence of actions that achieves a specified goal.
> The program should also:
> - detect when no plan exists;
> - print the resulting sequence of actions;
> - print the states reached after each action.
>
> Explain the implementation and identify any assumptions you make.

A follow-up prompt asked the LLM to run the generated program on the warehouse problem from the handout, and a further prompt asked it to implement the optional Prolog verifier (Section 7 of the handout, Tasks 6–8).

---

## 4. Generated Python Program

See [`planner.py`](planner.py). Summary of structure:

- `Action` class - stores `name`, `pos_preconditions`, `neg_preconditions`, `pos_effects`, `neg_effects` as frozensets; `is_applicable(state)` and `apply(state)` implement the applicability check and the remove-then-add effect rule exactly as specified.
- `bfs_plan(initial_state, goal, actions)` - breadth-first search over reachable states using a `deque` queue and a `visited` set; returns the action list for the first (shortest) plan found, or `None` if the goal is unreachable.
- `run_planner(...)` - replays a found plan from the initial state, printing each action and the resulting state.
- `build_warehouse_domain(...)` - constructs the specific warehouse domain from the handout, with flags to remove `PickUp` (Test B) or add an irrelevant action (Test C).

---

## 5. Results of Tests (Task 3)

### Test A - Solvable problem (original warehouse problem)

- Initial state: `{At(Robot,A), At(Package,A)}`
- Goal: `{At(Package,C)}`
- Plan found: **yes**
- Plan: `PickUp(Package,A) → Move(A,B) → Move(B,C) → Drop(Package,C)`
- States: S1 `{At(Robot,A), Holding(Package)}` → S2 `{At(Robot,B), Holding(Package)}` → S3 `{At(Robot,C), Holding(Package)}` → S4 `{At(Package,C), At(Robot,C)}`
- Valid: **yes** - every action's preconditions were mechanically checked against the state it was applied to before being added to the plan; matches the hand-constructed plan in Section 2.

### Test B - Impossible problem (`PickUp` action removed)

- Same initial state and goal.
- Plan found: **no** → program prints `No plan found.`
- Valid handling: correct. With no action that can ever produce `Holding(Package)`, the package can never leave `A`; BFS exhausts the (small) reachable state space - the robot can only move between A/B/C - and correctly reports failure rather than inventing an action.

### Test C - Irrelevant action present (`Wave(Robot)`, no preconditions, effect `Waved(Robot)`)

- Same initial state and goal.
- Plan found: **yes**
- Plan: identical to Test A.
- Valid: **yes**, and critically the irrelevant action never appears in the plan - confirms the search is goal-directed (`goal.issubset(new_state)`) and isn't fooled by unrelated reachable facts.

### Additional edge-case tests (constructed independently, beyond the handout's 3)

| # | Scenario | Result | Expected |
|---|---|---|---|
| 1 | Goal already true in the initial state (`{At(Robot,A)}`) | Empty plan `[]` | Empty plan |
| 2 | `PickUp(Package,A)` applicability checked while `Holding(Package)` already true | `is_applicable` → `False` | inapplicable |
| 3 | Goal references a location with no supporting action (`At(Package,Z)`) | `None` (no plan) | no plan |

All 6 checks (3 from the handout + 3 additional) pass. Full transcripts were produced by running `python3 planner.py` and a supplementary edge-case script.

---

## 6. "Think About It" Answers

**(p.5, before Task 1)** *"An action should not be considered applicable merely because it appears in the list of available actions... Are all of its preconditions satisfied in the current state?"*
This is exactly what `Action.is_applicable` enforces: membership in the `actions` list is necessary but not sufficient - `bfs_plan` calls `is_applicable(state)` on every action at every state before considering it, discarding any whose positive preconditions aren't a subset of the state or whose negative preconditions overlap it.

**(p.7, before Task 3)** *Identify where each idea from the specification appears in the program:*

| Spec idea | Where in `planner.py` |
|---|---|
| Preconditions → when is an action applicable? | `Action.is_applicable`: `pos_preconditions.issubset(state) and neg_preconditions.isdisjoint(state)` |
| Effects → how does the state change? | `Action.apply`: `(state - neg_effects) \| pos_effects` |
| Goal → when does planning terminate? | The `goal.issubset(new_state)` check inside `bfs_plan`'s main loop; also the `goal.issubset(initial_state)` check for the trivial empty-plan case |
| BFS → how are alternative plans explored? | The `deque`-based `queue` of `(state, path)` pairs in `bfs_plan`, combined with the `visited` set - states are expanded in order of increasing path length, and each is expanded at most once |

**(p.8–9, Task 4 diagram)** The missing step (`?`) between "Check action preconditions" and "Generate successor state" is: **determine the set of applicable actions** - i.e., filter the full action list down to those whose preconditions the current state satisfies. Completed diagram:

```
Current state
  -> Check action preconditions
  -> Determine applicable actions
  -> Generate successor state (apply each applicable action)
  -> Search over alternatives (BFS queue/visited)
  -> Goal? (does the new state entail G?)
```

*In my own words:* logical reasoning and search are complementary. Logic answers a purely local question - "given this one state, which actions are legal, and what state does each produce?" - and has no notion of a multi-step plan on its own. Search answers a purely global question - "given a graph of states connected by legal transitions, which order of expansion reaches a goal state, and how do I know when to stop?" - but has no way to generate that graph without something else defining which edges exist. Logic generates the edges (successors) one state at a time; search decides which edges to follow and in what order, and recognizes the destination. Neither alone produces a plan: logic without search can check one action but not find a sequence; search without logic would have no valid moves to search over (or would need every transition hand-enumerated, which is what STRIPS-style operators avoid).

**(p.13, Task 7 "Think About It")** *"The Python program generated the candidate action. Prolog is being used independently to check whether the action is consistent... Generate → Independent verification."* This is realized directly: `planner.py`'s BFS **generates** the plan (`Move(a,b)`, `Move(b,c)`); `planner.pl`'s `valid_move/2`, defined and populated independently of the Python code, **verifies** each proposed move against a separately hand-written connectivity fact base - see Section 8 below.

**(p.14, before 7.2 Reflection)** *"Prolog should not be regarded as identical to classical logic... focus on the basic idea: Facts + Rules → Inference → Query Answer."* Acknowledged - in Task 8's `reduce_speed` example (Section 8 below) I describe the result as an SLD-resolution proof, not a claim that Prolog's operational semantics (unification order, backtracking, negation-as-failure) are identical to classical first-order entailment.

---

## 7. Reflection on the Use of the LLM

1. **Why specify preconditions/effects before asking an LLM to write the planner?** It turns an underspecified request ("write a planner") into an exact contract. Without it, the LLM has to invent what "applicable" and "apply" mean, and different invented semantics are not directly comparable or testable - with the spec given up front, the generated `is_applicable`/`apply` can be checked line-by-line against a known definition (as done in Section 6's table) instead of trusted on faith.
2. **Example error from not checking preconditions:** if `is_applicable` were skipped or buggy, the planner could include `Drop(Package,C)` in a plan even though the robot never picked the package up - the plan would "look" complete (the goal proposition appears in the final state on paper) while being physically impossible to execute, since the package was never actually held.
3. **Why a plan that "looks reasonable" isn't necessarily valid:** plausibility is a property of surface form - action names in a sensible-sounding order - not of the underlying state transitions. A plan can name the right actions in the wrong order (e.g. `Drop` before `PickUp`) and still read fine to a human skimming it. Only replaying each action's preconditions against the actual computed state (as `run_planner` does, printing S1…Sn) catches that; this is exactly why Task 3 in the handout insists on constructing adversarial tests rather than eyeballing the output.
4. **What did the LLM contribute?** Translation of the precise specification into working Python (the `Action` class, the BFS loop with a visited set, state formatting/printing) and the warehouse domain encoding. It also surfaced an underspecified modeling choice - `PickUp` needing a negative precondition to prevent double-pickup - and flagged it explicitly as an assumption rather than adding it silently.
5. **What did I have to verify independently?** That `is_applicable`/`apply` actually match the `S ⊨ Preconditions(a)` / add-remove semantics in the spec (checked by direct code inspection, Section 6); that BFS terminates with "no plan found" rather than looping forever on Test B; and - by actually executing the code (not just reading it) - that the plan produced for the warehouse problem is a real, executable sequence matching the hand-constructed plan from Task 1, plus 3 additional edge cases beyond the handout's own tests.
6. **Where is logical reasoning used in this lab?** In two places: (i) within the Python planner, deciding action applicability - whether a state entails an action's precondition literals; (ii) in the optional Prolog extension, using Horn-clause resolution to independently re-derive whether a proposed move is supported by the warehouse's connectivity facts, without reusing any of the Python planner's logic.
7. **How does planning relate to the search algorithms from the previous module?** Planning here *is* search: the state space is a graph whose nodes are logical states and whose edges are the ground actions connecting them, and BFS is the same uninformed-search algorithm covered previously. The only new ingredient is that the graph isn't given explicitly - it's generated on the fly by applying logical preconditions/effects to each state as it's dequeued, rather than being handed a fixed adjacency list.

---

## 8. Optional Extension - Prolog as a Logical Verifier (Tasks 6–8)

See [`planner.pl`](planner.pl) for the full code (facts, rules, and an optional automated `run_tests/0` harness).

### Task 6

```prolog
connected(a,b). connected(b,a). connected(b,c). connected(c,b).
can_move(X,Y) :- connected(X,Y).
```

- **(a) Why does `can_move(a,b)` return `true`?** `connected(a,b)` is asserted directly as a fact; the rule `can_move(X,Y) :- connected(X,Y)` unifies `X=a, Y=b` against that fact and succeeds immediately.
- **(b) Why does it not establish `can_move(a,c)`?** `connected(a,c)` was never asserted, and `can_move/2` only checks *direct* connectivity - there is no rule chaining `connected(a,b)` and `connected(b,c)` into a derived `connected(a,c)` (no transitive closure is defined), so the query fails.
- **(c) Relationship between the Prolog rule and `Connected(X,Y) → CanMove(X,Y)`:** the Prolog clause *is* that implication, written as a Horn clause (head `:-` body). Proving `can_move(a,b)` by resolving against the clause and the fact `connected(a,b)` is operationally the same step as modus ponens on `Connected(a,b) ∧ (Connected(X,Y)→CanMove(X,Y)) ⊢ CanMove(a,b)`.

### Task 7

```prolog
valid_move(X,Y) :- connected(X,Y).
```

`valid_move(a,b)` and `valid_move(b,c)` succeed (direct facts); `valid_move(a,c)` fails, matching the handout's expected outcome.

**Challenge - checking a proposed `Move(a,c)`:** querying `valid_move(a,c)` against this knowledge base fails, since `connected(a,c)` is not a fact and no rule derives it. This independently confirms what the Python BFS planner already respects implicitly (it never generated a direct `A→C` move) - Prolog is being used here purely as a **generate → independent verification** check (see Section 6) on an action the Python planner *could* have proposed, not on the actual plan it did.

### Task 8

```prolog
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.
```

`?- reduce_speed.` succeeds. Reasoning chain:

```
wet_road                      (fact)
wet_road -> slippery          (rule)
slippery -> reduce_speed      (rule)
==================================
reduce_speed                  (conclusion)
```

Prolog proves `reduce_speed` by backward chaining: to prove `reduce_speed` it must prove `slippery`, and to prove `slippery` it must prove `wet_road`, which is a base fact - the proof bottoms out and succeeds, propagating `true` back up through both rules (SLD resolution).

### Section 7.2 Reflection

1. **Fact vs. rule:** a fact is a clause with an empty body - an unconditional assertion, e.g. `wet_road.`. A rule is a clause with a non-empty body (`head :- body`) - true only when the body can itself be proven, e.g. `slippery :- wet_road.`.
2. **How does a query correspond to "follows from a KB"?** `?- G.` asks Prolog to determine whether `G` is a logical consequence of the facts and rules currently loaded, by attempting to construct an SLD-resolution proof of `G` from them; success means `G` is entailed by the knowledge base, failure means it is not (under the closed-world assumption).
3. **Why verify a plan with Prolog?** Prolog reasons over a separately hand-written encoding of the domain's ground truth (the `connected/2` facts), independent of the code that produced the plan. It is a genuinely independent check rather than the planner grading its own homework - a bug shared between the plan-generator and the checker (if they were the same code) couldn't be caught this way, but a bug specific to one of them likely can be.
4. **Advantage of an independent verifier when the plan came from an LLM:** an LLM's own stated justification for its output is not a proof - it can be fluent and wrong (a confabulated explanation that sounds correct without being checked against anything). A separate deterministic logical system either succeeds or fails on a query with no persuasive language involved, so it cannot be talked into a wrong answer the way a free-text explanation can. This is the handout's own stated point in Section 3 ("Think About It," p.10): *"A generated explanation is not the same as independent verification."*

---

## 9. Files in this repository

| File | Contents |
|---|---|
| `planner.py` | Required Python planning agent (Task 2): `Action`, `bfs_plan`, `run_planner`, warehouse domain, and Tests A/B/C from Task 3. |
| `planner.pl` | Optional Prolog verifier (Tasks 6–8): `connected/2`, `can_move/2`, `valid_move/2`, and the `wet_road`/`slippery`/`reduce_speed` chain. |
| `SUBMISSION.md` | This document. |
| `README.md` | Repo overview and how to run everything. |

**LLM-generated vs. hand-modified, explicitly identified (per Section 4's closing instruction):**
- `planner.py` and `planner.pl` were generated in full by an LLM (Claude), from the prompts quoted in Section 3 above.
- One deliberate modification beyond the literal spec was made and flagged during generation: `PickUp`'s negative precondition `¬Holding(Package)`, which the handout's action description does not state - added and called out as a modeling assumption (see Section 1's footnote).
- All test results in Section 5, the code-to-spec mapping in Section 6, and the written reflections in Sections 7–8 were produced by tracing and running the actual code, not merely restated from the LLM's own description of what it wrote - consistent with the handout's "generated explanation ≠ independent verification" principle.
