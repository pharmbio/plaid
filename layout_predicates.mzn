include "globals.mzn";


predicate sorted_except_0(array[int] of var int: x, int: num, int: min, int: max) = 
  regular(x, concat(["(0* \(i)){\(min),\(max)} " | i in 1..num]++["0*"]))::domain;
  
  
%% Cost implied constraints, similar to the ones derived for a group automaton constraint  
predicate my_implied_cost(var int:cost, array[int] of var int: x) = 
  let {
    int: L = length(x);
    array[1..L] of var int: c_cost;
  } in
  c_cost[1] = x[1] /\
  c_cost[2] <= 1 /\
  c_cost[L] = cost /\
  forall(i in 1..L-1)(x[i+1] = c_cost[i+1] - c_cost[i] ) /\
  forall(i in 1..L-1)(c_cost[i+1] = x[i+1] + c_cost[i] ) /\
  forall(i in 1..L-1)(c_cost[i] <= c_cost[i+1]) /\
  forall(i in 2..L-1)(c_cost[i+1] <= 1 + c_cost[i-1]) /\
  forall(i in 2..L-1)(c_cost[i+1] - 1 <= c_cost[i-1]);


%% Cost implied constraints, similar to the ones derived for a group automaton constraint
predicate my_implied_cost_ext(var int:cost, array[int] of var int: x) = 
  let {
    int: L = length(x);
    array[1..L] of var int: c_cost;
  } in
  c_cost[1] = x[1] /\
  c_cost[2] <= 1 /\
  c_cost[L] = cost /\
  forall(i in 1..L-1)(x[i+1] = c_cost[i+1] - c_cost[i] ) /\
  forall(i in 1..L-1)(c_cost[i+1] = x[i+1] + c_cost[i] ) /\
  forall(i in 1..L-1)(c_cost[i] <= c_cost[i+1]) /\
  forall(i in 2..L-2)(c_cost[i+2] <= 1 + c_cost[i-1]) /\
  forall(i in 2..L-2)(c_cost[i+2] - 1 <= c_cost[i-1]);
  

function bool: force_spread_controls(int: numcols, int: numrows, int: total_controls, int: numplates) = 
  % 96-well plate (This is a proven upperbound)
  if (numcols = 12 /\ numrows = 8) \/ (numcols = 8 /\ numrows = 12) then (11*numplates >= total_controls) 
  % 96-well plate with empty border (This is a proven upperbound)
  elseif (numcols = 10 /\ numrows = 6) \/ (numcols = 6 /\ numrows = 10) then (7*numplates >= total_controls) 
  % 384-well plate with empty border (I have not proven yet that 33 is unsat)
  % 32 is solvable, but makes the model too slow. 
  % TODO: Think about an intermediate constraint
  elseif (numcols = 22 /\ numrows = 14) \/ (numcols = 14 /\ numrows = 22) then (31*numplates >= total_controls) 
  % 384-well plate (I have not proven yet that 38 is unsat) 
  elseif (numcols = 24 /\ numrows = 16) \/ (numcols = 16 /\ numrows = 24) then (37*numplates >= total_controls) 
  % Some generic attempt that 
  elseif numcols*numrows < 60 then (ceil(numcols*numrows/9)*numplates >= total_controls)
  else false
  endif;



%force_spread_controls = if inner_plate_size <= 60 then (ceil(inner_plate_size/9)*numplates >= total_controls) else (floor(inner_plate_size/9)*numplates >= total_controls) endif;
