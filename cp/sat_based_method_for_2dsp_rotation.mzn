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

array[CIRCUIT] of var bool: rotated;
array[CIRCUIT] of var int: R_W;
array[CIRCUIT] of var int: R_H;

array[CIRCUIT,CIRCUIT] of var bool: lr;    % Whether rectangle i is placed on the left of rectangle j
array[CIRCUIT,CIRCUIT] of var bool: ud;    % Whether rectangle i is placed above rectangle j

%%% ----------------------------------------------------------- %%% 



%%% ----------------------- CONSTRAINTS ------------------------ %%%

constraint (forall(i in CIRCUIT) (R_W[i] = if rotated[i] then H[i] else W[i] endif));
constraint (forall(i in CIRCUIT) (R_H[i] = if rotated[i] then W[i] else H[i] endif));

constraint max(i in CIRCUIT) (x[i] + R_W[i]) <= L;
constraint max(i in CIRCUIT) (y[i] + R_H[i]) <= FINAL_H;

constraint diffn(x, y, R_W, R_H);
constraint cumulative(y, R_H, R_W, L);
constraint cumulative(x, R_W, R_H, FINAL_H);

constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (lr[i,j] -> (x[j] >= x[i] + R_W[i])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (lr[j,i] -> (x[i] >= x[j] + R_W[j])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (ud[i,j] -> (y[j] >= y[i] + R_H[i])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (ud[j,i] -> (y[i] >= y[j] + R_H[j])));

constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (lr[i,j] \/ lr[j,i] \/ ud[i,j] \/ ud[j,i]));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (not lr[i,j] \/ not(lr[j,i])));
constraint forall(i in CIRCUIT) (forall (j in CIRCUIT where i < j) (not(ud[i,j]) \/ not(ud[j,i])));


% SYMMETRY BREAKING CONSTRAINTS

% Symmetry breaking constraint: square circuits can be fixed as rotated
constraint forall(i in CIRCUIT where W[i] = H[i]) (rotated[i]);


%Same sized rectangles
constraint forall(i in CIRCUIT) (forall(j in CIRCUIT where i < j) ((R_W[i] == R_W[j] /\ R_H[i] == R_H[j]) -> (not(lr[j,i]) /\ (lr[i,j] \/ not(ud[j,i])))));


% Large rectangles constraints
constraint forall(i in CIRCUIT) (forall(j in CIRCUIT where i < j) ((R_W[i] + R_W[j] > L) -> (not(lr[i,j]) /\ not(lr[j,i]))));
constraint forall(i in CIRCUIT) (forall(j in CIRCUIT where i < j) ((R_H[i] + R_H[j] > FINAL_H) -> (not(ud[i,j]) /\ not(ud[j,i]))));


% Domain Reduction Technique
constraint (x[decreasing_area_order[1]] <= (L - R_W[decreasing_area_order[1]]) div 2);
constraint forall(i in CIRCUIT where R_W[i] > (L - R_W[decreasing_area_order[1]]) div 2) (not(lr[i,decreasing_area_order[1]]));

constraint (y[decreasing_area_order[1]] <= (FINAL_H - R_H[decreasing_area_order[1]]) div 2);
constraint forall(i in CIRCUIT where R_H[i] > (FINAL_H - R_H[decreasing_area_order[1]]) div 2) (not(ud[i,decreasing_area_order[1]]));


%One Pair Of Rectangles Technique (clashes with Domain Reduction Technique)

%constraint(not(lr[decreasing_area_order[2], decreasing_area_order[1]]));
%constraint(not(ud[decreasing_area_order[2], decreasing_area_order[1]]));

%%% ----------------------------------------------------------- %%% 


%%% ------------------------- SEARCH -------------------------- %%%


solve :: restart_luby(100) minimize FINAL_H;


%%% ------------------------ SOLUTION ------------------------- %%%

output ["\(L) \(FINAL_H)\n\(N)\n"] ++ ["\(R_W[i]) \(R_H[i]) \(x[i]) \(y[i])\n" | i in CIRCUIT];

%%% ----------------------------------------------------------- %%%