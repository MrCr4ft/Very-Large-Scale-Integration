                                           
%%% ------------------------- IMPORTS ------------------------- %%%
                                          
include "globals.mzn";

%%% ----------------------------------------------------------- %%% 



%%% ----------------------- PARAMETERS ------------------------ %%%

par int: L;                   % The width of the board
par int: N;                   % The number of the circuits
set of int: CIRCUIT = 1..N;   % The set of circuits' indexes
array[CIRCUIT] of par int: W; % The width of the circuits
array[CIRCUIT] of par int: H; % The height of the circuits

array[CIRCUIT] of CIRCUIT: decreasing_area_order = sort_by(CIRCUIT, [-W[i] * H[i] | i in CIRCUIT]);

par int: H_LB;                % These are computed elsewhere
par int: H_UB;                

%%% ----------------------------------------------------------- %%% 



%%% ----------------------- VARIABLES ------------------------ %%%

var H_LB..H_UB: FINAL_H;           % The height of the board

array[CIRCUIT] of var 0..L: x;     % X coordinates of the circuits
array[CIRCUIT] of var 0..H_UB: y;  % Y coordinates of the circuits


%%% ----------------------------------------------------------- %%% 



%%% ------------------------ PREDICATES -------------------------- %%% 


predicate lex_lesseq_ror(array[int] of var int: A, array[int] of var int: B) =
    let {
      par int: N = length(A),
      array[1..N] of var bool: X;
    } in
    X[1] /\ (X[N] <-> (A[N] <= B[N])) /\ forall(i in 1..N-1) (X[N - i] <-> ((A[N - i] <= B[N - i]) \/ (A[N - i] == B[N - i] /\ X[N - i + 1])));

%%% ----------------------------------------------------------- %%% 



%%% ----------------------- CONSTRAINTS ------------------------ %%%

constraint max(i in CIRCUIT) (x[i] + W[i]) <= L;
constraint max(i in CIRCUIT) (y[i] + H[i]) <= FINAL_H;

constraint diffn(x, y, W, H);
constraint cumulative(y, H, W, L);
constraint cumulative(x, W, H, FINAL_H);

constraint (forall(i in CIRCUIT) (
  forall(j in CIRCUIT where j > i) (
    if W[i] + W[j] > L then (y[i] + H[i] <= y[j] \/ y[i] >= y[j] + H[j]) endif)));

constraint (forall(i in CIRCUIT) (
  forall(j in CIRCUIT where j > i) (
    if H[i] + H[j] > FINAL_H then (x[i] + W[i] <= x[j] \/ x[i] >= x[j] + W[j]) endif)));


% Symmetry breaking constraint that imposes an ordering between the two biggest rectangle.
% The biggest rectangle is placed on the bottom-left area of the board wrt the second biggest one.
% This one is compatible with the lexicographic ordering in case one wants to use both

constraint
  let {
    int: o1 = decreasing_area_order[1],
    int: o2 = decreasing_area_order[2]
  }
  in
  lex_less([y[o1],x[o1]], [y[o2],x[o2]]);


% Symmetry breaking constraint that imposes an ordering between rectangles of the same size.
% This one helps in certain cases
constraint forall(i in CIRCUIT) (
  let {
    array[int] of int: I = [j | j in CIRCUIT where W[j] = W[i] /\ H[j] = H[i] /\ j >= i]
  }
  in
    if length(I) > 1 then
      forall(j in index_set(I) where j > 1) (
        lex_less([y[I[j-1]],x[I[j-1]]], [y[I[j]], x[I[j]]])
      )
    else
      true
    endif
);


% Symmetry breaking constraint. Impose lexicographic ordering to circuit position and reflected one.
constraint lex_lesseq_ror(x, [L - x[i] - W[i] | i in CIRCUIT]);
constraint lex_lesseq_ror(y, [FINAL_H - y[i] - H[i] | i in CIRCUIT]);


%%% ----------------------------------------------------------- %%%


%%% ------------------------- SEARCH -------------------------- %%%


%Can help with gecode
/*
solve :: seq_search([
  int_search(x, dom_w_deg, indomain_split),
  int_search(y, dom_w_deg, indomain_split)]) :: restart_luby(100) minimize FINAL_H;

*/

% can help with chuffed
solve :: restart_luby(100) minimize FINAL_H;


%%% ------------------------ SOLUTION ------------------------- %%%

output ["\(L) \(FINAL_H)\n\(N)\n"] ++ ["\(W[i]) \(H[i]) \(x[i]) \(y[i])\n" | i in CIRCUIT];

%%% ----------------------------------------------------------- %%%
