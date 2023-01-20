                                            /* SIMPLE MODEL WITH ADDITIONAL CONSTRAINT */
                                            
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

var H_LB..H_UB: FINAL_H;                   % The height of the board

array[CIRCUIT] of var 0..L: x;             % X coordinates of the circuits
array[CIRCUIT] of var 0..H_UB: y;          % Y coordinates of the circuits

array[CIRCUIT,CIRCUIT] of var bool: lr;    % Whether rectangle i is placed on the left of rectangle j
array[CIRCUIT,CIRCUIT] of var bool: ud;    % Whether rectangle i is placed above rectangle j

%%% ----------------------------------------------------------- %%% 



%%% ----------------------- CONSTRAINTS ------------------------ %%%

constraint max(i in CIRCUIT) (x[i] + W[i]) <= L;
constraint max(i in CIRCUIT) (y[i] + H[i]) <= FINAL_H;

constraint diffn(x, y, W, H);
constraint cumulative(y, H, W, L);
constraint cumulative(x, W, H, FINAL_H);

constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (lr[i,j] -> (x[j] >= x[i] + W[i])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (lr[j,i] -> (x[i] >= x[j] + W[j])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (ud[i,j] -> (y[j] >= y[i] + H[i])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (ud[j,i] -> (y[i] >= y[j] + H[j])));

constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (lr[i,j] \/ lr[j,i] \/ ud[i,j] \/ ud[j,i]));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (not lr[i,j] \/ not(lr[j,i])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (not(ud[i,j]) \/ not(ud[j,i])));


% SYMMETRY BREAKING CONSTRAINTS

%Same sized rectangles
constraint forall(i in CIRCUIT) (forall(j in CIRCUIT where i < j) ((W[i] == W[j] /\ H[i] == H[j]) -> (not(lr[j,i]) /\ (lr[i,j] \/ not(ud[j,i])))));


% Large rectangles constraints
constraint forall(i in CIRCUIT) (forall(j in CIRCUIT where i < j) ((W[i] + W[j] > L) -> (not(lr[i,j]) /\ not(lr[j,i]))));
constraint forall(i in CIRCUIT) (forall(j in CIRCUIT where i < j) ((H[i] + H[j] > FINAL_H) -> (not(ud[i,j]) /\ not(ud[j,i]))));


% Domain Reduction Technique
constraint (x[decreasing_area_order[1]] <= (L - W[decreasing_area_order[1]]) div 2);
constraint forall(i in CIRCUIT where W[i] > (L - W[decreasing_area_order[1]]) div 2) (not(lr[i,decreasing_area_order[1]]));

constraint (y[decreasing_area_order[1]] <= (FINAL_H - H[decreasing_area_order[1]]) div 2);
constraint forall(i in CIRCUIT where H[i] > (FINAL_H - H[decreasing_area_order[1]]) div 2) (not(ud[i,decreasing_area_order[1]]));


%One Pair Of Rectangles Technique (clashes with Domain Reduction Technique)

%constraint(not(lr[decreasing_area_order[2], decreasing_area_order[1]]));
%constraint(not(ud[decreasing_area_order[2], decreasing_area_order[1]]));

%%% ----------------------------------------------------------- %%% 


%%% ------------------------- SEARCH -------------------------- %%%


solve :: restart_luby(100) minimize FINAL_H;


%%% ------------------------ SOLUTION ------------------------- %%%

output ["\(L) \(FINAL_H)\n\(N)\n"] ++ ["\(W[i]) \(H[i]) \(x[i]) \(y[i])\n" | i in CIRCUIT];

%%% ----------------------------------------------------------- %%%