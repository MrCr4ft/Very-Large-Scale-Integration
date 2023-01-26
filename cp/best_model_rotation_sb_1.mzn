                                           
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

array[CIRCUIT] of var bool: rotated;
array[CIRCUIT] of var int: R_W;
array[CIRCUIT] of var int: R_H;

array[CIRCUIT] of var bool: aux_lex_1;  % These are needed for the ROR decomposition of the lex_lesseq constraint
array[CIRCUIT] of var bool: aux_lex_2;

%%% ----------------------------------------------------------- %%% 




%%% ------------------------ PREDICATES -------------------------- %%% 

predicate lex_lesseq_ror(par int: N, array[int] of var bool: X, array[int] of var int: A, array[int] of var int: B) =
    X[1] /\ (X[N] <-> (A[N] <= B[N])) /\ forall(i in 1..N-1) (X[N - i] <-> ((A[N - i] <= B[N - i]) \/ (A[N - i] == B[N - i] /\ X[N - i + 1])));

%%% ----------------------------------------------------------- %%% 




%%% ----------------------- CONSTRAINTS ------------------------ %%%

constraint (forall(i in CIRCUIT) (R_W[i] = if rotated[i] then H[i] else W[i] endif));
constraint (forall(i in CIRCUIT) (R_H[i] = if rotated[i] then W[i] else H[i] endif));

constraint max(i in CIRCUIT) (x[i] + R_W[i]) <= L;
constraint max(i in CIRCUIT) (y[i] + R_H[i]) <= FINAL_H;

constraint diffn(x, y, R_W, R_H);
constraint cumulative(y, R_H, R_W, L);
constraint cumulative(x, R_W, R_H, FINAL_H);

constraint (forall(i in CIRCUIT) (
  forall(j in CIRCUIT where j > i) (
    if R_W[i] + R_W[j] > L then (y[i] + R_H[i] <= y[j] \/ y[i] >= y[j] + R_H[j]) endif)));

constraint (forall(i in CIRCUIT) (
  forall(j in CIRCUIT where j > i) (
    if R_H[i] + R_H[j] > FINAL_H then (x[i] + R_W[i] <= x[j] \/ x[i] >= x[j] + R_W[j]) endif)));
    
% Symmetry breaking constraint that imposes an ordering between the two biggest rectangle.
% The biggest rectangle is placed on the bottom-left area of the board wrt the second biggest one.
constraint
  let {
    int: o1 = decreasing_area_order[1],
    int: o2 = decreasing_area_order[2]
  }
  in
  lex_less(
    [y[o1],x[o1]],
    [y[o2],x[o2]]
);

% Symmetry breaking constraint: square circuits can be fixed as rotated
constraint forall(i in CIRCUIT where W[i] = H[i]) (rotated[i]);

% Symmetry breaking constraint. Impose lexicographic ordering to circuit position and reflected one.
% Even though this constraint prunes much the solution space, it slows down the search, probably because it is quite heavy.
constraint lex_lesseq_ror(N, aux_lex_1, x, [L - x[i] - W[i] | i in CIRCUIT]);
constraint lex_lesseq_ror(N, aux_lex_2, y, [L - y[i] - H[i] | i in CIRCUIT]);
    
%%% ----------------------------------------------------------- %%% 


%%% ------------------------- SEARCH -------------------------- %%%

/*
solve ::
	int_search([FINAL_H], input_order, indomain_min, complete)
		:: seq_search([
			int_search([x[i] | i in decreasing_area_order], input_order, indomain_min),
			int_search([y[i] | i in decreasing_area_order], input_order, indomain_min),
			bool_search(rotated, input_order, indomain_min, complete)
		])
	  :: restart_luby(100)
minimize FINAL_H;
*/

solve :: restart_luby(100) minimize FINAL_H;



%%% ------------------------ SOLUTION ------------------------- %%%

output ["\(L) \(FINAL_H)\n\(N)\n"] ++ ["\(R_W[i]) \(R_H[i]) \(x[i]) \(y[i])\n" | i in CIRCUIT];

%%% ----------------------------------------------------------- %%% 
