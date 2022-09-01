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

par int: H_LB;
par int: H_UB;

%%% ----------------------------------------------------------- %%%



%%% ----------------------- VARIABLES ------------------------ %%%

var H_LB..H_UB: FINAL_H; % The height of the board

array[CIRCUIT] of var 0..L: x;                  % X coordinates of the circuits
array[CIRCUIT] of var 0..H_UB: y;               % Y coordinates of the circuits

array[0..H_UB,0..L - 1] of var 1..N + 1: board; % This matrix will represent the board cells, it is needed to break additional symmetries

%%% ----------------------------------------------------------- %%%



%%% ----------------------- CONSTRAINTS ------------------------ %%%


constraint max(i in CIRCUIT) (x[i] + W[i]) <= L;
constraint max(i in CIRCUIT) (y[i] + H[i]) <= FINAL_H;

constraint diffn(x, y, W, H);
constraint cumulative(y, H, W, L);
constraint cumulative(x, W, H, FINAL_H);

% Channeling constraint
constraint forall(i in 0..L - 1, j in 0..H_UB, k in CIRCUIT) (
  board[j,i] == k
  <->
  (
    x[k] <= i /\ i < x[k] + W[k]
    /\
    y[k] <= j /\ j < y[k] + H[k]
  )
);

% This redundant constraint imposes that each rectangle must occur (both in the horizontal and vertical position) either
% at the beginning of a row (column) or immediately next to another rectangle.
constraint (
  forall(i in CIRCUIT) (
    member([0] ++ [x[j] + W[j] | j in CIRCUIT where j != i], x[i])
    /\
    member([0] ++ [y[j] + H[j] | j in CIRCUIT where j != i], y[i]))
);

% Symmetry breaking constraint that imposes an ordering between rectangles of the same size.
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

% Symmetry breaking and domain reduction constraint: places the biggest rectangle in the lower half of the board
constraint (x[decreasing_area_order[1]] <= (L - W[decreasing_area_order[1]]) div 2);
constraint (y[decreasing_area_order[1]] <= (FINAL_H - H[decreasing_area_order[1]]) div 2);

% Symmetry is broken by imposing a lexicographic ordering on the flattened version of the board, with respect to its reflection upon axis and the diagonal
constraint symmetry_breaking_constraint(
  lex_lesseq(array1d(board), [ board[i,j] | i in 0..H_UB, j in reverse(0..L-1)])
  /\
  lex_lesseq(array1d(board),[ board[i,j] | i in reverse(0..H_UB), j in 0..L-1])
  /\
  lex_lesseq(array1d(board), [ board[i,j] | i in reverse(0..H_UB), j in reverse(0..L-1)])
);

%%% ----------------------------------------------------------- %%%


%%% ------------------------- SEARCH -------------------------- %%%

/*
solve ::
	int_search([FINAL_H], input_order, indomain_min, complete)
		:: seq_search([
			int_search([x[i] | i in decreasing_area_order], input_order, indomain_min),
			int_search([y[i] | i in decreasing_area_order], input_order, indomain_min),
		])
	  :: restart_luby(100)
minimize FINAL_H;
*/

solve :: restart_luby(100) minimize FINAL_H;


%%% ------------------------ SOLUTION ------------------------- %%%

output ["\(L) \(FINAL_H)\n\(N)\n"] ++ ["\(W[i]) \(H[i]) \(x[i]) \(y[i])\n" | i in CIRCUIT];

%%% ----------------------------------------------------------- %%%