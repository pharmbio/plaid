%%%% Find a solution! %%%%

array [Plates,1..experiments] of var {0} union Rows: experiment_plate_row;

constraint forall(l in 1..experiments)((experiment_plate[l] > 0) -> (experiment_plate_row[experiment_plate[l],l] == experiment_row[l]));

constraint forall(i in Plates, l in 1..experiments)((experiment_plate[l] != i) -> (experiment_plate_row[i,l] == 0));



%array [1..numplates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..numplates*2];
 
%array [1..numplates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..numplates*2];

%% Strategy 1: good for example04 and example11 BACKUP, previous version!
%array [1..numplates*3] of ann: search_order_a = [if (i mod 3)==1 then int_search(experiment_plate_row[(i+2) div 3,..], first_fail, indomain_random) elseif (i mod 3) == 2 then int_search(controls_layout[(i+2) div 3,..,..], random, indomain_max) else int_search(plates[i div 3,..,..], first_fail, indomain_random) endif | i in 1..numplates*3]; 

array [1..numplates*3] of ann: search_order_a = [if (i mod 3)==2 then int_search(experiment_plate_row[(i+2) div 3,..], first_fail, indomain_random) elseif (i mod 3) == 1 then int_search(controls_layout[(i+2) div 3,..,..], random, indomain_max) else int_search(plates[i div 3,..,..], first_fail, indomain_random) endif | i in 1..numplates*3]; 


%% The one I normally use?? 
%array [1..numplates*3] of ann: search_order = [if (i mod 3)==2 then int_search(experiment_plate_row[(i+2)div 3,..], first_fail, indomain_random) elseif (i mod 3) == 1 then int_search(controls_layout[(i+2)div 3,..,..], random, indomain_max) else int_search(plates[(i+2) div 3,..,..], first_fail, indomain_random) endif | i in 1..numplates*3]; 

%% Strategy 2:
array [1..numplates*2] of ann: search_order_b = [if (i mod 2) == 1 then int_search(controls_layout[(i+1) div 2,..,..], random, indomain_max) else int_search(plates[(i+1) div 2,..,..], first_fail, indomain_random) endif | i in 1..numplates*2]; 


%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++
solve::seq_search((if sorted_compounds == true then [int_search(experiment_plate, first_fail, indomain_min)] else [int_search(experiment_plate, first_fail, indomain_random)] endif)++
(if (swap_search mod 2 == 0) then search_order_b else search_order_a endif))
%[int_search(controls_layout, random, indomain_max), %int_search(experiment_plate_row, first_fail, indomain_random), 
%int_search(plates, first_fail, indomain_random)]) 
     :: restart_geometric(1.1,5*experiments)
    satisfy ; 

%% Trying to use a more deterministic search strategy for testing
% solve:: if (false) then seq_search([int_search(experiment_plate, first_fail, indomain_min)]++search_order) 
%         else (seq_search([int_search(experiment_plate, first_fail, indomain_random)]++search_order)
%                 :: restart_geometric(1.1,5*experiments)) endif
%      satisfy ; 


%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++
%search_order)
%[int_search(experiment_plate_row, first_fail, indomain_random)]++[int_search(plates[i,..,..], random, indomain_max) | i in Plates]) 
%[int_search(plates[i,..,..], random, indomain_max) | i in Plates ]) 
%      :: restart_geometric(1.1,1000)
 %     :: if testing==true then restart_none else restart_geometric(1.1,5*experiments) endif
     %:: restart_linear(1000)
     %:: restart_constant(10*experiments)
  %   satisfy ; 



%% Chuffed
% solve::seq_search([int_search(experiment_plate, random_order, indomain_min), int_search(experiment_row, random_order, indomain_min), int_search(plates, random_order, indomain_max)]) 
%      :: restart_geometric(1.5,100)
%      satisfy; 


%solve::seq_search([int_search(experiment_plate, random_order, indomain_min), priority_search(experiment_plate,[int_search([experiment_row[i]],input_order,indomain_split_random)| i in 1..experiments],smallest,complete)]) satisfy;


%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, random, indomain_split), int_search(plates, first_fail, indomain_random)]) satisfy; 

%solve satisfy;



%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 


       