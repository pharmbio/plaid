% Copyright 2021 PLAID Authors.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
%
% Description:  A highly effective microplate layout designer
%
% Authors: Maria Andreina FRANCISCO RODRIGUEZ (maria.andreina.francisco@it.uu.se)
% Version: 1.0
% Last Revision: May 2024
%

include "globals.mzn";
include "gecode.mzn";
%include "../layout_predicates.mzn";
 

%%%% Input Data  %%%%

%% Information about constraints %%
opt bool: allow_empty_wells;
bool: replicates_on_different_plates;
bool: replicates_on_same_plate;
bool: concentrations_on_different_rows;
bool: concentrations_on_different_columns;

%% Information about the layout %%
int: horizontal_cell_lines;
int: vertical_cell_lines;
int: size_empty_edge;
bool: inner_empty_edge = true;

%% Compounds %%
int: compounds; %% number of drugs/compounds
array [1..compounds] of int: compound_replicates;
int: replicates = max(compound_replicates++[0]);
array [1..compounds] of int: compound_concentrations;
int: max_compound_concentrations = max(compound_concentrations++[0]);

array[1..compounds] of string: compound_names;
array[1..compounds,1..max_compound_concentrations] of string: compound_concentration_names;

%% Combinations (Deprecated) %%
int: combinations;
int: combination_concentrations;
array[1..combinations] of string: combination_names;
array[1..combination_concentrations] of string: combination_concentration_names;

%% Information about controls %%
int: num_controls;
array [1..num_controls] of int: control_replicates; 
array [1..num_controls] of int: control_concentrations;
int: max_control_concentrations = max(control_concentrations++[0]);
array[1..num_controls,1..max_control_concentrations] of string: control_concentration_names;
array[1..num_controls] of string: control_names;

int: total_controls = sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]);

%% Potentially new parameters %%
bool: interconnected_plates = true; %(numplates==1); %Makes the problem much harder by connecting the plates

%%% Testing %%%
opt bool: testing;% = true;
opt bool: print_all;% = true;
bool: debugging = print_all \/ false;
opt bool: sorted_compounds;

%% Plate size / number of wells
int: num_rows;
int: num_cols;


%% TODO: this could be problematic when there are multiple cell lines
int: numrows = if inner_empty_edge then floor(num_rows/horizontal_cell_lines)-2*size_empty_edge else floor((num_rows-2*size_empty_edge)/horizontal_cell_lines) endif;
int: numcols = if inner_empty_edge then floor(num_cols/vertical_cell_lines)-2*size_empty_edge else floor((num_cols-2*size_empty_edge)/vertical_cell_lines) endif;


%% FIX ME!! (Deprecated?)
array[1..max_compound_concentrations] of string: compound_concentration_indicators;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Datafile validation %%%
constraint assert(compounds >= 0,"Invalid datafile: Number of compounds cannot be less than zero.");
constraint assert(combinations >= 0,"Invalid datafile: Number of combinations cannot be less than zero.");
constraint assert(num_controls >= 0,"Invalid datafile: Number of controls should not be less than zero.");
constraint assert(vertical_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero.");
constraint assert(horizontal_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero.");
constraint assert(numrows > 0,"Invalid datafile: Number of rows should be larger than zero.");
constraint assert(numcols > 0,"Invalid datafile: Number of columns should be larger than zero.");
constraint assert(compounds==0 \/ min(compound_replicates) > 0,"Invalid datafile: Number of replicates should be larger than zero.");
constraint assert(compounds==0 \/ min(compound_concentrations) > 0,"Invalid datafile: Number of concentrations should be larger than zero.");
constraint assert((replicates_on_different_plates /\ replicates_on_same_plate) == false,"Invalid datafile: replicates cannot be both on the same plate and on different plates");
constraint assert(numrows mod 2 == 0, "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of rows.");
constraint assert(numcols mod 2 == 0, "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of columns." );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Number of wells needed. Note that plates might not be full
int: total_wells = sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + total_controls; 

set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: Plates = 1..numplates;

int: inner_plate_size = numcols*numrows;


%%%%% Data validation %%%%%
constraint assert(total_wells > 0, "Invalid data: the plates cannot be completely empty.");
constraint assert(inner_plate_size>0, "Invalid data: There are no wells on the plate.");
constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, "Invalid data: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Number of plates needed
%% max is used to avoid division-by-zero errors
int: numplates = max(ceil(total_wells/inner_plate_size),1);         
              

%% All types of experiments (excluding controls): compounds and combinations
int: experiments = compounds*max_compound_concentrations*replicates;  

%% Used in redundant constraints
int: emptywells = numplates*inner_plate_size - total_wells;   



%%%%%%% Detecting some unsatisfiable cases & Data validation %%%%%%%%%%
constraint assert(ceil(sum(compound_replicates)/numplates)*min(compound_concentrations++[infinity]) + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E01)");  
                                                                
constraint assert((floor(sum(compound_replicates)/numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E02)");  

constraint assert(emptywells >= 0,"Model ERROR! Inner empty wells is negative. This should never happen!");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plates (Main model & Solution) %%%%%%%%%%%
array [Plates,Rows,Columns] of var 0..(experiments+num_controls*max_control_concentrations): plates; 

% 0 = empty wells
% 1... compounds*compound_concentrations*replicates = compounds
% compounds*compound_concentrations*replicates+1 ... experiments = combinations
% experiments+1... = controls -> control1_conc1, control1_conc2, ...control2_conc1, control2_conc2...

%% Alternative (extra) model
array [1..experiments] of var {0} union Plates: experiment_plate;

array [1..experiments] of var {0} union Rows: experiment_row;

array [1..experiments] of var {0} union Columns: experiment_column;



%%%% CONSTRAINTS %%%%
%% Place only the exact amount of compounds, controls, and empty wells.
constraint global_cardinality(plates,[0]++[ i | i in 1..experiments]++[i | i in experiments+1..experiments]++[ experiments+i | i in 1..num_controls*max_control_concentrations],[emptywells]++[(floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1] | i in 1..experiments]++[1|i in experiments+1..experiments]++[control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..num_controls*max_control_concentrations]);

%% Experiments that do not exists are not located in any plate, row, or column.
constraint forall(i in 1..experiments)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((i-1) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) == false <-> (experiment_plate[i] = 0 /\ experiment_row[i] = 0 /\ experiment_column[i] = 0));

constraint forall(i in 1..experiments)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((i-1) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) <-> (experiment_plate[i] != 0 /\ experiment_row[i] != 0 /\ experiment_column[i] != 0));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channelling constraints: an experiment/compound is located at a given plate

constraint forall(l in 1..experiments, i in Plates)(count_eq(plates[i,..,..],l,(experiment_plate[l] == i)));

constraint forall(l in 1..experiments)(count_eq(plates,l,(experiment_plate[l] != 0)));


%% For experiments that do not exist = plate 0 (redundant)
%constraint forall(l in 1..experiments)(
 % count_eq(array1d(1..numplates*inner_plate_size, plates),l,0) <-> (experiment_plate[l] = 0 /\ experiment_row[l] = 0 /\ experiment_column[l] = 0)
%);

%constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*inner_plate_size, plates),l,0) <-> (experiment_row[l] == 0));
%constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*inner_plate_size, plates),l,0) <-> (experiment_column[l] == 0));



%% Channeling constraints between the two models

constraint forall(l in 1..experiments, j in Rows)(count_eq(plates[..,j,..],l,(experiment_row[l] == j)));
constraint forall(l in 1..experiments, k in Columns)(count_eq(plates[..,..,k],l,(experiment_column[l] == k)));

constraint forall(l in 1..experiments)(count_eq(plates,l,(experiment_row[l] != 0)));
constraint forall(l in 1..experiments)(count_eq(plates,l,(experiment_column[l] != 0)));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% All concentrations of a given compound replica must appear on a single plate

constraint forall(l in 1..experiments where ((l mod max_compound_concentrations) == 1))(all_equal([experiment_plate[i] | i in l..(l+compound_concentrations[((((l-1) div max_compound_concentrations)) mod compounds)+1]-1)]));


      
array [1..replicates] of set of int: tt = [r*max_compound_concentrations+1..r*max_compound_concentrations | r in 0..replicates-1]   ;   
      
  
  
constraint forall(i in Plates, j in Rows) (
(among(plates[i,j,..], 1..experiments)::domain) <= numcols-among([plates[i,j,k] | k in Columns], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Spreading concentrations of a compound across different rows and columns 
%%% (consider balancing the number of rows and columns!)
% Consider making it different on different plates



% Might be redundant != implied
constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_plate[plates[i,j,k]]==i);
constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_row[plates[i,j,k]]==j);
constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_column[plates[i,j,k]]==k);


% Spread the compounds across different rows and columns

array [Plates,1..compounds*replicates,Rows] of var 0..1: compounds_in_plate_row;
array [Plates,1..compounds*replicates,Columns] of var 0..1: compounds_in_plate_column;

%% Constraints for compounds_in_plate_row
constraint forall(j in Rows, e in 1..experiments)(
  (experiment_row[e]==j)  -> (compounds_in_plate_row[experiment_plate[e], (((e-1) div max_compound_concentrations)+1),j] == 1)
);
 
constraint forall(i in Plates, j in Rows, e in 1..compounds*replicates)(
  (experiment_plate[(e-1)*max_compound_concentrations+1] != i)  -> (compounds_in_plate_row[i,e,j] == 0)
);
 


%% Implied constraint: how many experiments can there be in a row? Per plate
% I think this still works...
%% TODO: Redundant 
constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])== numcols-among(plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

%% Replaced by constraint below (suggested by Gustav)
%constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])<= numrows-among(plates[i,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

%% Gustav:
constraint forall(i in Plates, k in Columns) (
% sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])
(among(plates[i,..,k], 1..experiments)::domain) <= numrows-among([plates[i,j,k] | j in Rows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

                                      


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Different compound replicas must appear on different plates (if possible)
int: min_plates = min(numplates,replicates);

constraint if replicates_on_different_plates then forall(l in 1..compounds) (nvalue(min(numplates,compound_replicates[l]), [experiment_plate[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])) endif; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Different compound replicas must appear on the same plate
constraint if replicates_on_same_plate then forall(l in 1..compounds) (all_equal([experiment_plate[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])) endif;


int: min_compounds_plate = max(floor((sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]))/inner_plate_size), min(compound_concentrations++[0]) );

constraint global_cardinality_low_up(experiment_plate,[ i | i in Plates],[ min_compounds_plate | i in Plates],[ inner_plate_size | i in Plates]);

%%% NOTE: sorted compounds is removed for now
%% TODO: Think more about this constraint and what happens when there are replicates, etc.
%% This option is possibly going to make some plans unfeasible (when there are different numbers of replicates, etc).
%constraint if sorted_compounds == true /\ max(compound_replicates)<=1 then sorted_except_0(experiment_plate,numplates,at_least_compounds,at_most_compounds) endif;

%% WARNING! This option assumes that compounds can be placed on plates in order! This works, for example, when all compounds have the same 
% number of replicates and concentrations and one would like to generate several layouts with the compounds on the exact same plates.
% Otherwise, experiment_plate needs to be given as input
%constraint if sorted_compounds == true /\ replicates_on_same_plate == true  then sorted_except_0(experiment_plate[1..compounds*max(compound_concentrations)],numplates,at_least_compounds div numplates,at_most_compounds div numplates) endif;



% Balancing compounds between plates (and knowing that there must be some sort of balance!)
%% FIX ME!
%% I want to remove these constraints but I need to make other stronger first
int: at_least_compounds = min(compound_concentrations++[infinity])*(sum(compound_replicates) div numplates);
int: at_most_compounds = max_compound_concentrations*ceil(sum(compound_replicates)/numplates);

%% TODO: False only if there are no compounds
constraint if at_least_compounds <= at_most_compounds then global_cardinality_low_up([experiment_plate[i] | i in 1..experiments] ,[ i | i in Plates],[ at_least_compounds | i in Plates],[ at_most_compounds | i in Plates]) endif;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Balancing controls between plates.
%% Asumes controls can be balanced this way, that is, that the designer expects about the same number of controls on each plate and not a plate full of controls.

array[int] of float: controls_per_plate = [control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])/numplates | i in 1..num_controls*max_control_concentrations];

array[int] of int: min_controls_per_plate = [floor(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

array[int] of int: max_controls_per_plate = [ceil(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

constraint forall(i in Plates)(global_cardinality_low_up([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], min_controls_per_plate, max_controls_per_plate));

array[Plates,1..num_controls*max_control_concentrations] of var int: n_controls_per_plate;
constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], n_controls_per_plate[i,..]));


%% Implied all_different constraint: all experiments/compounds are different
constraint alldifferent_except(plates, {0} union experiments+1..experiments+num_controls*max_control_concentrations):: domain;



%% Variable for the new plate split output 

array[Plates,1..compounds] of var int: replicates_experiment_plate;

constraint forall(i in Plates, c in 1..compounds) (replicates_experiment_plate[i,c] = sum([experiment_plate[(c-1)*max_compound_concentrations+ r*max_compound_concentrations*compounds + 1] == i | r in 0..replicates-1 ]));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                                                                                
%%%% Find a solution! %%%%

array [1..numplates] of ann: search_order_x = [int_search(plates[i,..,..], first_fail, indomain_random) | i in 1..numplates]; 

solve::seq_search((if sorted_compounds == true then [int_search(experiment_plate, first_fail, indomain_min)] else [int_search(experiment_plate, first_fail, indomain_random)] endif)++search_order_x)
     :: restart_geometric(1.1,5*experiments)
    satisfy ; 



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Pretty printing %%%

output [if testing \/ debugging then "\(numplates) plates\n" else "" endif];

output [if debugging /\ (emptywells>0) then "warning: there are \(emptywells) empty wells. Consider using them as controls.\n\n" else "" endif];

output [if debugging then "Plate 1:\n" else "" endif];
         
%%TODO: change output so symbols are not hardcoded
output [if debugging then if fix(plates[i,j,k]) == 0  then "  ." 
        elseif fix(plates[i,j,k]) == (experiments+1)  then "  +" %% Some control
        elseif fix(plates[i,j,k]) == (experiments+2)  then "  -" %% Other control...
        elseif fix(plates[i,j,k]) == (experiments+3)  then "  x" %% ...
        elseif fix(plates[i,j,k]) == (experiments+4)  then "  ~" %% ...
        elseif fix(plates[i,j,k]) == (experiments+5)  then "  *" %% other
        elseif fix(plates[i,j,k]) == (experiments+6)  then "  #" %% other
        elseif fix(plates[i,j,k]) == (experiments+7)  then "  &" %% other
        elseif fix(plates[i,j,k]) == (experiments+8)  then "  @" %% other
        elseif fix(plates[i,j,k]) == (experiments+9)  then "  <" %% other
        elseif fix(plates[i,j,k]) == (experiments+10) then "  >" %% other
        elseif fix(plates[i,j,k]) == (experiments+11) then "  $" %% other
        elseif fix(plates[i,j,k]) ==  (experiments+12) then "  X" %% other      
        elseif fix(plates[i,j,k]) >  (experiments+12) then "  Y" %% other        
        else "   " endif ++
        %% Adding lines and headings %%
        if j== numrows /\ k == numcols /\ i<numplates then "\n\n Plate \(i+1):\n" 
        elseif k == numcols then "\n" else " " endif else "" endif|
        i in Plates, j in Rows, k in Columns];        
         
output [if debugging then "Plate:" else "" endif, if debugging then show(plates) else "" endif];

output [if debugging then "\nCompound's plate:\n" else "" endif, if debugging then show(experiment_plate) else "" endif];

output [if debugging then "\nControls's plate:\n" else "" endif, if debugging then show(n_controls_per_plate) else "" endif];

output [if debugging then "\nTotal Compounds:\n" endif, if debugging then show(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) else "" endif];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CSV Ouput %%%

array[int] of string: letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
         
output [if debugging then "plateID,well,cmpdname,CONCuM,cmpdnum,VOLuL\n" endif];

%%% TODO: FIX ME!!!!!! Problem when there are vertical/horizontal cell lines!
output [if debugging then if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[size_empty_edge+j+(numrows+(1+inner_empty_edge)*size_empty_edge)*(h-1)] ++ if (size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1)) < 10 then "0" else "" endif ++ "\(size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1))," ++ 

%%%% Compounds %%%%

if fix(plates[i,j,k]) <= experiments then compound_names[(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) div max_compound_concentrations)+1] 
++ "," ++ compound_concentration_names[(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) div max_compound_concentrations)+1,(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) mod max_compound_concentrations)+1] ++ "," ++

compound_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1] ++ "_" ++ compound_concentration_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1,((floor((fix(plates[i,j,k])-1))) mod max_compound_concentrations)+1]


%%%% Controls %%%%

else
%control_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1] ++ "," ++  
control_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1] ++ "," ++  
%% control concentration

%control_concentration_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1,floor((fix(plates[i,j,k])-experiments-1)/num_controls)+1] 
control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1] 

++ "," ++ 
%% Latex name of the control. Right now it's the same as the name
control_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1] ++ "_" ++ control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1]  

endif 

++ "\n" else "" endif endif|
        i in Plates, j in Rows, k in Columns, v in 1..vertical_cell_lines, h in 1..horizontal_cell_lines];       
         

%%%%%%%%%

output ["%%% Experiment file generated by PLAID's plate splitting %%%\n\n"++
        "num_rows = \(num_rows); %% height\n"++
        "num_cols = \(num_cols); %% width\n\n"++

        "horizontal_cell_lines = \(horizontal_cell_lines);\n"++
        "vertical_cell_lines = \(vertical_cell_lines);\n\n"++
        
        "allow_empty_wells = \(allow_empty_wells);\n\n"++

        "concentrations_on_different_rows = \(concentrations_on_different_rows);\n"++
        "concentrations_on_different_columns = \(concentrations_on_different_columns);\n\n"++

        "% Restriction: replicates_on_different_plates && replicates_on_same_plate == false\n"++
        "replicates_on_different_plates = \(replicates_on_different_plates);\n"++
        "replicates_on_same_plate = \(replicates_on_same_plate);\n\n"++

        "size_empty_edge = \(size_empty_edge);\n\n"++


        "%%% Compounds %%%\n"++
        "compounds = \(sum([ 1 | c in 1..compounds where fix(replicates_experiment_plate[p,c]) >0 ])); %% number of drugs/compounds\n"++

        "compound_concentrations	 = "++
        "\([ compound_concentrations[c] | c in 1..compounds where fix(replicates_experiment_plate[p,c]) >0 ]);\n"++

        "compound_names = "++
        "\([ compound_names[c] | c in 1..compounds where fix(replicates_experiment_plate[p,c]) >0 ]);\n"++


        "compound_replicates	 = \([ fix(replicates_experiment_plate[p,c]) | c in 1..compounds where fix(replicates_experiment_plate[p,c]) >0 ]);\n"++

        "compound_concentration_names = "++
        "array2d(1..\(sum([ 1 | c in 1..compounds where fix(replicates_experiment_plate[p,c]) >0 ])),1..\(max_compound_concentrations),\([ compound_concentration_names[c,n] | c in 1..compounds, n in 1..max_compound_concentrations where fix(replicates_experiment_plate[p,c]) >0 ]));\n"++

        "compound_concentration_indicators = \(compound_concentration_indicators); \n\n"++


        "%%% Combinations %%%\n"++
        "combinations = 	0;\n"++
        "combination_names = [];\n"++
        "combination_concentration_names = [];\n"++
        "combination_concentrations = 0;\n\n"++


        "%%% Controls %%%\n"++
        "num_controls = \(num_controls);\n"++
        "control_replicates = \(n_controls_per_plate[p,..]);\n"++
        "control_concentrations = \(control_concentrations);\n"++
        "control_names = \(control_names);\n"++

        "control_concentration_names = array2d(1..\(num_controls),1..\(max_control_concentrations),\(control_concentration_names));  \n\n"++
        "%%%%% End of file %%%%%\n\n" | p in Plates];
  
