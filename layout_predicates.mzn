include "globals.mzn";


predicate sorted_except_0(array[int] of var int: x, int: num, int: min, int: max) = 
  % Possibly only the regular constraint is needed
  forall(i in 1..length(x)-1)(x[i]<=x[i+1] \/ x[i+1]==0) /\
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
  



