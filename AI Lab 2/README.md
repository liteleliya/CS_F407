# AI Lab 2 - Logical Reasoning for Planning

CS F407 (Artificial Intelligence) lab: a simple STRIPS-style planning agent
that represents states as sets of logical propositions, checks action
applicability via preconditions, applies effects, and searches for a plan
with breadth-first search.

## Files

- `planner.py` - the required Python planning agent (state/action
  representation, BFS planner, warehouse domain, Tests A/B/C).
- `planner.pl` - optional Prolog extension: an independent logical verifier
  for proposed moves, plus a small chained-inference example.
- `SUBMISSION.md` - full write-up: problem specification, hand-constructed
  plan, LLM prompt used, test results, "Think About It" answers, and
  reflection questions, per the lab handout's Section 4.

## Running it

```bash
python3 planner.py
```

Runs the warehouse robot problem (deliver a package from A to C) and prints
Test A (solvable), Test B (no `PickUp` action - no plan exists), and Test C
(irrelevant action present) with the plan and intermediate states for each.

```bash
swipl -q -g run_tests -t halt planner.pl
```

Runs the Prolog verifier's self-checks (requires SWI-Prolog). Alternatively,
consult `planner.pl` in the SWI-Prolog toplevel and run the `?- ` queries
documented in the file's comments.

See `SUBMISSION.md` for the full explanation and results.
