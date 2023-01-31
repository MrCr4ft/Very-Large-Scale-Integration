                                           
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

output ["\(L) \(FINAL_H)\n\(N)\n"] ++ ["\(R_W[i]) \(R_H[i]) \(x[i]) \(y[i])\n" | i in CIRCUIT];

%%% ----------------------------------------------------------- %%% 
