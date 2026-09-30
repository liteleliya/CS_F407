% ---------------------------------------------------------------------------
% Task 6: Prolog as a Plan Verifier
% ---------------------------------------------------------------------------
% Facts describing the warehouse connectivity graph (Section 3 of the
% handout: locations a, b, c; robot can move between connected locations).

connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% Rule: the robot can move directly between X and Y iff they are connected.
can_move(X,Y) :-
    connected(X,Y).

% Example queries (run these at the swipl toplevel after consulting this file):
%
%   ?- can_move(a,b).
%   true.
%
%   ?- can_move(a,c).
%   false.


% ---------------------------------------------------------------------------
% Task 7: Using Prolog to Check a Proposed Plan
% ---------------------------------------------------------------------------
% The Python planner (planner.py) proposed the sequence:
%   Move(a,b), Move(b,c)
% valid_move/2 checks whether a proposed Move action is supported by the
% warehouse connectivity facts above.

valid_move(X,Y) :-
    connected(X,Y).

% Checking the Python planner's proposed moves:
%
%   ?- valid_move(a,b).
%   true.
%
%   ?- valid_move(b,c).
%   true.
%
%   ?- valid_move(a,c).
%   false.
%
% Challenge: suppose the Python planner had instead proposed Move(a,c)
% directly (skipping b). Prolog determines this is NOT supported by the
% warehouse knowledge, since connected(a,c) was never asserted:
%
%   ?- valid_move(a,c).
%   false.


% ---------------------------------------------------------------------------
% Task 8: Connect Prolog to Logical Reasoning
% ---------------------------------------------------------------------------
% A small, separate example (unrelated to the warehouse) demonstrating
% chained inference: Fact => Rule => Rule => Conclusion.

wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% Query:
%
%   ?- reduce_speed.
%   true.
%
% Reasoning chain:
%   wet_road            (fact)
%   wet_road -> slippery        (rule)
%   slippery -> reduce_speed    (rule)
%   ==================================
%   reduce_speed                (conclusion, by chained modus ponens)


% ---------------------------------------------------------------------------
% Automated self-check (optional convenience - not required by the handout).
% Run with:  swipl -q -g run_tests -t halt planner.pl
% ---------------------------------------------------------------------------

run_tests :-
    format("~n=== Task 6: can_move ===~n"),
    check(can_move(a,b), true),
    check(can_move(a,c), false),

    format("~n=== Task 7: valid_move (checking Python planner's proposed moves) ===~n"),
    check(valid_move(a,b), true),
    check(valid_move(b,c), true),
    check(valid_move(a,c), false),

    format("~n=== Task 8: reduce_speed chain ===~n"),
    check(reduce_speed, true).

check(Goal, Expected) :-
    ( call(Goal) -> Actual = true ; Actual = false ),
    ( Actual == Expected
    -> format("  ~q -> ~q  [OK]~n", [Goal, Actual])
    ;  format("  ~q -> ~q  [MISMATCH, expected ~q]~n", [Goal, Actual, Expected])
    ).
