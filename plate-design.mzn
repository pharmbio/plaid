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
% Authors: Maria Andreina FRANCISCO RODRIGUEZ (maria.andreina.francisco@farmbio.uu.se)
% Version: 1.0
% Last Revision: September 2022
%

include "globals.mzn";
include "gecode.mzn";
include "layout_predicates.mzn";
%include "chuffed.mzn";
 

%%%% Input Data  %%%%

%% Information about constraints %%
bool: allow_empty_wells;
bool: replicates_on_different_plates;
bool: replicates_on_same_plate;
bool: concentrations_on_different_rows;
bool: concentrations_on_different_columns;
bool: force_spread_controls;
bool: force_spread_concentrations;

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
bool: balance_controls_inside_plate = true;
bool: interconnected_plates = true; %(numplates==1); %Makes the problem much harder by connecting the plates
int: control_slack = 0; %Makes the problem harder/easier by constraining how equally should we distribute controls
bool: spread_controls = (ceil(inner_plate_size/2)*numplates >= total_controls); %Makes the problem harder when there are many controls

%% Which individual controls can be spread more than others?
array[1..num_controls*max_control_concentrations] of bool: spread_control = [ ceil(inner_plate_size/4)*numplates >= (control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])) | i in 1..num_controls*max_control_concentrations];


%%% Testing %%%
opt bool: testing;% = true;
opt bool: print_all;% = true;
bool: debugging = print_all \/ false;
%bool: enforce = false;
opt int: swap_search;
opt bool: sorted_compounds;


%% Plate size / number of wells
int: num_rows;
int: num_cols;


%% TODO: this could be problematic when there are multiple cell lines
int: numrows = if inner_empty_edge then floor(num_rows/horizontal_cell_lines)-2*size_empty_edge else floor((num_rows-2*size_empty_edge)/horizontal_cell_lines) endif;
int: numcols = if inner_empty_edge then floor(num_cols/vertical_cell_lines)-2*size_empty_edge else floor((num_cols-2*size_empty_edge)/vertical_cell_lines) endif;
%int: numrows = floor(num_rows/horizontal_cell_lines)-2*size_empty_edge;
%int: numcols = floor(num_cols/vertical_cell_lines)-2*size_empty_edge;

%% TODO: add latex indicators for controls 
%array[1..compound_concentrations] of string: compound_concentration_names;


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
constraint assert(not (num_controls == 1 /\ spread_control[1]==false),"Invalid datafile: There are too many controls of only one kind. This is not allowed at the moment. If you believe this is a mistake, please contact the developers.");
constraint assert(numrows mod 2 == 0, "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of rows.");
constraint assert(numcols mod 2 == 0, "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of columns." );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Number of wells needed. Note that plates might not be full
int: total_wells = sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + total_controls; 

set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: Plates = 1..numplates;

enum Vertical = {upper,lower};
enum Horizontal = {left,right};

array [Plates,Rows] of var 0..numcols: experiments_in_plate_row;
array [Plates,Columns] of var 0..numrows: experiments_in_plate_column;

bool: even_columns = ((numcols mod 2) == 0);
bool: even_rows = ((numrows mod 2) == 0);

int: inner_plate_size = numcols*numrows;


%%%%% Data validation %%%%%
constraint assert(total_wells > 0, "Invalid data: the plates cannot be completely empty.");
constraint assert(inner_plate_size>0, "Invalid data: There are no wells on the plate.");
constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, "Invalid data: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Number of plates needed
%% max is used to avoid division-by-zero errors
int: numplates = max(ceil(total_wells/inner_plate_size),1);         

% TODO: temporal fix. This makes the problem too difficult for the borderline case in large plates.
% Probably because it tries to prove if that optimal solution exists.
%force_spread_controls = (ceil(inner_plate_size/9)*numplates >= total_controls);

force_spread_controls = force_spread_controls(numcols, numrows, total_controls, numplates);


%force_spread_concentrations = if replicates_on_same_plate \/ replicates <= numplates then (ceil(inner_plate_size/9) >= replicates*max_compound_concentrations) else (ceil(inner_plate_size/9) >= max_compound_concentrations) endif ;


% TODO: There are many other cases here that could be considered, for example, individual compounds with fewer concentrations/replicates.
force_spread_concentrations = if replicates_on_different_plates /\ replicates <= numplates then ((inner_plate_size div 5) >= max_compound_concentrations) else (ceil(inner_plate_size/5) >= replicates*max_compound_concentrations) endif ;


%% All types of experiments (excluding controls): compounds and combinations
int: experiments = compounds*max_compound_concentrations*replicates;  

%% Used in redundant constraints
int: emptywells = numplates*inner_plate_size - total_wells;   



%%%%%%% Detecting some unsatisfiable cases & Data validation %%%%%%%%%%
constraint assert(ceil(sum(compound_replicates)/numplates)*min(compound_concentrations++[infinity]) + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E01)");  
                                                                
constraint assert((floor(sum(compound_replicates)/numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E02)");  

constraint assert(emptywells == 0 \/ allow_empty_wells,"Warning: There are \(emptywells) empty wells. You can allow the plates to have inner empty wells or modify other quantities to fill them up.");

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

%% TODO: check if other compounds have fewer concentrations and can be spread around more
%% TODO: change to include replicates
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1,l in 1..experiments where ((l mod max_compound_concentrations) == 1)) (1>=among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],(l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)))):: domain endif;


% constraint if force_spread_concentrations then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1,l in 1..experiments where ((l mod max_compound_concentrations) == 1)) (
%   let{
%     var 0..1: num_concentrations;
%    } in among(num_concentrations,[plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],(l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)))::domain) endif;

% Replaced by constraint below for clarity (less index math)
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1,l in 1..experiments where ((l mod max_compound_concentrations) == 1)) (
%2 > among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],(l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)))::domain) endif;


% Replaced by constraint below to in order to include replicates
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1, c in 1..compounds, r in 0..replicates-1) (
%2 > among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],((c-1)*r*max_compound_concentrations+1..c*r*max_compound_concentrations))::domain) endif;   

      
constraint if force_spread_concentrations then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1, c in 1..compounds) (
2 > among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],
array_union([(c-1)*max_compound_concentrations+1 + r*compounds*max_compound_concentrations..c*max_compound_concentrations + r*compounds*max_compound_concentrations | r in 0..replicates-1]) 
)::domain) endif;   

                                    
%%array [1..experiments] of var {0} union Plates: experiment_plate; array_union
      
array [1..replicates] of set of int: tt = [r*max_compound_concentrations+1..r*max_compound_concentrations | r in 0..replicates-1]   ;   
      
%compound_replicates[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]  


% constraint forall(i in Plates, j in Rows) (
%   let{
%     var int: exprInRow;
%     constraint exprInRow = among(plates[i,j,..], 1..experiments)::domain; 
%     %sum([experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments]);
%   } in exprInRow <= numcols-among([plates[i,j,k] | k in Columns], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));
  
  
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

%% Spreads compounds over different rows. It's been replaced by the more restrictive alldifferent version below.
%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],numrows-2*size_empty_edge), [experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
alldifferent([experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numrows)]):: domain
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
alldifferent([experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numrows)]):: domain
))) endif; 

%% HERE!!! NOW!
%% 
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1, c in 1..compounds) (
%2 > among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],
%array_union([(c-1)*max_compound_concentrations+1 + r*compounds*max_compound_concentrations..c*max_compound_concentrations + r*compounds*max_compound_concentrations | r in 0..replicates-1]) 
%)::domain) endif;   


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% WORK AREA!!!
%balanced_alldifferent(array[int] of var int: x, int: mid_val, int: mid_count)

%array [Vertical, 1..experiments] of var {0} union Rows: ul_experiment_row;

%constraint forall(v in Vertical, e in 1..experiments) (if experiment_row[e] <= (numrows div 2) then ul_experiment_row[upper,e]==experiment_row[e] /\ ul_experiment_row[lower,e]==0  else ul_experiment_row[lower,e]==experiment_row[e] /\ ul_experiment_row[upper,e]==0  endif);

% constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (if compound_concentrations[l]>1 then
% forall(v in Vertical, r in 0..(compound_replicates[l]-1)) (
% let{
%   var int: compounds_in_row;
%   constraint compounds_in_row <= ((compound_concentrations[l]+1) div 2)+1;
%   constraint compounds_in_row >= (compound_concentrations[l] div 2)+1;
%   }
%   in nvalue(compounds_in_row,[ul_experiment_row[v,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numrows)])
% )endif)) endif; 



%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
%nvalue((compound_concentrations[l] div 2)+1,[ul_experiment_row[lower,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numrows)])
%))) endif; 

%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
%nvalue(ul_concentrations[r*compounds + l,lower]+1,[ul_experiment_row[lower,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in %1..min(compound_concentrations[l],numrows)])
%))) endif; 

%% MAYBE XXXX
%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(v in Vertical, r in 0..(compound_replicates[l]-1)) (
%alldifferent_except_0([ul_experiment_row[v,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numrows)])
%))) endif; 


%constraint count(e in 1..experiments)(ul_experiment_row[upper,e]>0) = sum(ul_half_plates[..,upper]);
%constraint count(e in 1..experiments)(ul_experiment_row[lower,e]>(numrows div 2)) = sum(ul_half_plates[..,lower]);
%%%%% END MAYBE


%% IDEA:
% Implied constraint: when there are more concentrations than rows, help the search to not fill in a row completely too soon by saving space in each row for the other compounds
% constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= numrows)) then 
% forall(i in Plates, j in Rows, r in 1..(compounds*replicates-1))
%   ((count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + sum([ experiment_plate_row[i,max_compound_concentrations*(k-1)+1]>0 /\ (compound_concentrations[((k-1) mod compounds)+1]>=numrows) | k in (r+1) .. compounds*replicates])) <= experiments_in_plate_row[i,j]) endif;

%int: temp = (compounds*replicates-1);

% NOT WORKING!!
% constraint if concentrations_on_different_rows then 
% forall(i in Plates, j in Rows, r in temp..(compounds*replicates-1))(
%   let{
%     array [Plates,Rows] of var 0..1: temp_array;
%      %compound_concentrations[(r mod compounds)+1];
%   } in sum(temp_array[i,..])==1 /\ count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + temp_array[i,j] == experiments_in_plate_row[i,j]
% ) endif;

%% array [Plates,Rows] of var 0..numcols: experiments_in_plate_row;
%% constraint forall(i in Plates, j in Rows)(count(experiment_plate_row[i,..],j,experiments_in_plate_row[i,j]));


 %int: temp = (compounds*replicates-1);
%%% array [Plates,Rows] of var 0..1: temp_array2;
%%% constraint forall(i in Plates, j in Rows, r in temp..(compounds*replicates-1))(
%%% count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + temp_array2[i,j] == experiments_in_plate_row[i,j]
%%% );

%%%constraint forall(i in Plates)(if experiment_plate_row[i,max_compound_concentrations*compounds*replicates] == 0 then sum(temp_array2[i,..])==0 else sum(temp_array2[i,..])==2 %%%endif);

array [Plates,1..compounds*replicates,Rows] of var 0..1: compounds_in_plate_row;
array [Plates,1..compounds*replicates,Columns] of var 0..1: compounds_in_plate_column;

%% Constraints for compounds_in_plate_row
constraint forall(j in Rows, e in 1..experiments)(
  (experiment_row[e]==j)  -> (compounds_in_plate_row[experiment_plate[e], (((e-1) div max_compound_concentrations)+1),j] == 1)
);
 
constraint forall(i in Plates, j in Rows, e in 1..compounds*replicates)(
  (experiment_plate[(e-1)*max_compound_concentrations+1] != i)  -> (compounds_in_plate_row[i,e,j] == 0)
);

constraint if concentrations_on_different_rows then forall(i in Plates, e in 1..compounds*replicates)(
  if experiment_plate[(e-1)*max_compound_concentrations+1]==i then sum(compounds_in_plate_row[i,e,..]) == min(compound_concentrations[((e-1) mod compounds)+1],numrows)
  else sum(compounds_in_plate_row[i,e,..]) == 0 endif
) endif;
 
 
constraint if max_compound_concentrations<=numrows /\ concentrations_on_different_rows then forall(i in Plates, j in Rows)(
  sum(compounds_in_plate_row[i,..,j]) == experiments_in_plate_row[i,j]
) 
else forall(i in Plates, j in Rows)(
  sum(compounds_in_plate_row[i,..,j]) <= experiments_in_plate_row[i,j]
)
endif;

constraint if max_compound_concentrations<=numrows /\ concentrations_on_different_rows then
  sum(compounds_in_plate_row[..,..,..]) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) else
  sum(compounds_in_plate_row[..,..,..]) <= sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])  
endif;


constraint if concentrations_on_different_rows /\ max_compound_concentrations<=numrows /\ even_rows then (forall(l in 1..compounds,r in 0..(replicates-1)) (
 ul_concentrations[r*compounds + l,upper] == sum(compounds_in_plate_row[..,r*compounds + l, 1..(numrows div 2)])
)) endif; 

constraint if concentrations_on_different_rows /\ max_compound_concentrations<=numrows /\ even_rows then (forall(l in 1..compounds,r in 0..(replicates-1)) (
 ul_concentrations[r*compounds + l,lower] == sum(compounds_in_plate_row[..,r*compounds + l, (numrows div 2)+1..numrows])
)) endif; 

constraint if concentrations_on_different_rows /\ even_rows then 
  forall(i in Plates)(sum(experiments_in_plate_row[i,1..(numrows div 2)]) == ul_half_plates[i, upper]) endif;
  
constraint if concentrations_on_different_rows /\ even_rows then 
  forall(i in Plates)(sum(experiments_in_plate_row[i,(numrows div 2)+1..numrows]) == ul_half_plates[i, lower]) endif;


%% Constraints for compounds_in_plate_column
constraint forall(k in Columns, e in 1..experiments)(
  (experiment_column[e]==k)  -> (compounds_in_plate_column[experiment_plate[e], (((e-1) div max_compound_concentrations)+1),k] == 1)
);
 
constraint forall(i in Plates, k in Columns, e in 1..compounds*replicates)(
  (experiment_plate[(e-1)*max_compound_concentrations+1] != i)  -> (compounds_in_plate_column[i,e,k] == 0)
);

constraint if concentrations_on_different_columns then forall(i in Plates, e in 1..compounds*replicates)(
  if experiment_plate[(e-1)*max_compound_concentrations+1]==i then sum(compounds_in_plate_column[i,e,..]) == min(compound_concentrations[((e-1) mod compounds)+1],numcols)
  else sum(compounds_in_plate_column[i,e,..]) == 0 endif
) endif;
 
 
constraint if max_compound_concentrations<=numcols /\ concentrations_on_different_columns then forall(i in Plates, k in Columns)(
  sum(compounds_in_plate_column[i,..,k]) == experiments_in_plate_column[i,k]
) 
else forall(i in Plates, k in Columns)(
  sum(compounds_in_plate_column[i,..,k]) <= experiments_in_plate_column[i,k]
)
endif;

constraint if max_compound_concentrations<=numcols /\ concentrations_on_different_columns then
  sum(compounds_in_plate_column[..,..,..]) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) else
  sum(compounds_in_plate_column[..,..,..]) <= sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])  
endif;

constraint if concentrations_on_different_columns /\ max_compound_concentrations<=numcols /\ even_columns then (forall(l in 1..compounds,r in 0..(replicates-1)) (
  lr_concentrations[r*compounds + l,left] == sum(compounds_in_plate_column[..,r*compounds + l, 1..(numcols div 2)])
)) endif; 

constraint if concentrations_on_different_columns /\ max_compound_concentrations<=numcols /\ even_columns then (forall(l in 1..compounds,r in 0..(replicates-1)) (
  lr_concentrations[r*compounds + l,right] == sum(compounds_in_plate_column[..,r*compounds + l, (numcols div 2)+1..numcols])
)) endif; 

%% HERE
constraint if concentrations_on_different_columns /\ even_columns then 
  forall(i in Plates)(sum(experiments_in_plate_column[i,1..(numcols div 2)]) == lr_half_plates[i, left]) endif;
  
constraint if concentrations_on_different_columns /\ even_columns then 
  forall(i in Plates)(sum(experiments_in_plate_column[i,(numcols div 2)+1..numcols]) == lr_half_plates[i, right]) endif;


% array [Rows] of var 0..2: temp_array2;
% constraint forall(i in Plates, j in Rows, r in temp-1..(compounds*replicates-2))(
% count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + temp_array2[j]<= experiments_in_plate_row[i,j]
% );

% constraint sum(temp_array2) ==20;


% array [Rows] of var 0..3: temp_array3;
% constraint forall(i in Plates, j in Rows, r in temp-2..(compounds*replicates-3))(
% count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + temp_array3[j]<= experiments_in_plate_row[i,j]
% );

% constraint sum(temp_array3) ==30;


% array [Rows] of var 0..4: temp_array4;
% constraint forall(i in Plates, j in Rows, r in temp-3..(compounds*replicates-4))(
% count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + temp_array4[j]<= experiments_in_plate_row[i,j]
% );

% constraint sum(temp_array4) ==40;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Split concentrations between upper and lower halves of the plate

array [1..compounds*replicates,Vertical] of var 0..max_compound_concentrations: ul_concentrations;

%% For each replicate of each compound, count how many concentrations are in the upper/lower parts of the plate
%% Definition of ul_concentrations (upper)  

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,upper],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..(numrows div 2))::domain
))) endif; 

%% Definition of ul_concentrations (lower)
constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], (numrows div 2)+1..numrows)::domain
))) endif; 

%% Constraints to divide the concentrations into top and bottom 
constraint if concentrations_on_different_rows then (forall(l in 1..compounds, v in Vertical) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,v] <= ((compound_concentrations[l]+1) div 2)
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds, v in Vertical) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,v] >= (compound_concentrations[l] div 2)
))) endif; 

constraint forall(l in 1..compounds, v in Vertical) (forall(r in compound_replicates[l]..replicates-1) (
 ul_concentrations[r*compounds + l,v] == 0
)); 

%% Implied constraints:
constraint if concentrations_on_different_rows then (sum(ul_concentrations)=sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) endif;

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,upper] + ul_concentrations[r*compounds + l,lower] == compound_concentrations[l]
))) endif; 

constraint if concentrations_on_different_rows then forall(v in Vertical) (sum(ul_concentrations[..,v]) == sum(ul_half_plates[..,v]) ) endif;

constraint if concentrations_on_different_rows then forall(i in Plates, v in Vertical) (sum(c in 0..compounds*replicates-1)(ul_concentrations[c+1,v]*(experiment_plate[c*max_compound_concentrations+1]==i)) == ul_half_plates[i,v] ) endif;


%% Defining ul_concentrations in terms of plates
constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,upper], plates[..,1..(numrows div 2),..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower], plates[..,(numrows div 2)+1..numrows,..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
))) endif; 

%% Defining lr_concentrations in terms of plates
constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,left], plates[..,..,1..(numcols div 2)], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,right], plates[..,..,(numcols div 2)+1..numcols], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
))) endif; 


%% Implied constraint
constraint if concentrations_on_different_rows /\ even_rows then (forall(v in Vertical)(	
sum(ul_half_plates[..,v]) == sum(ul_concentrations[..,v])	
))endif;


%% Definition of experiments_in_plate_row
constraint forall(i in Plates, j in Rows)(count(experiment_plate_row[i,..],j,experiments_in_plate_row[i,j]));

constraint forall(i in Plates, j in Rows)(among(experiments_in_plate_row[i,j],plates[i,j,..],1..experiments)::domain);

constraint forall(i in Plates, k in Columns)(among(experiments_in_plate_column[i,k],plates[i,..,k],1..experiments)::domain);

%constraint forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k] == sum(j in Rows)(0<plates[i,j,k] /\ plates[i,j,k]<=experiments));

constraint forall(i in Plates, k in Columns)(count_eq([plates[i,j,k] in 1..experiments | j in Rows], 1, experiments_in_plate_column[i,k]));
constraint forall(i in Plates, j in Rows)(count_eq([plates[i,j,k] in 1..experiments | k in Columns], 1, experiments_in_plate_row[i,j]));

constraint forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j] + controls_in_plate_row[i,j] == numcols);
constraint forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j] + controls_only_in_plate_row[i,j] + emptywells_in_plate_row[i,j] == numcols);

constraint forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k] + controls_in_plate_column[i,k] == numrows);
constraint forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k] + controls_only_in_plate_column[i,k] + emptywells_in_plate_column[i,k] == numrows);

constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= numrows)) then forall(i in Plates, j in Rows, r in 1..compounds*replicates)(
count(experiment_plate_row[i,1..experiments-max_compound_concentrations*r],j) + count(experiment_plate_row[i,experiments-max_compound_concentrations*r+1..experiments],j) == experiments_in_plate_row[i,j]
) endif;

constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= numrows)) then forall(i in Plates, j in Rows, r in 1..compounds*replicates-1)(
if(compound_concentrations[((r-1) mod compounds)+1]>=numrows) then
(count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + count(experiment_plate_row[i,max_compound_concentrations*(r+1)+1..experiments],j) + (experiment_plate_row[i,max_compound_concentrations*r+1]>0)) <= experiments_in_plate_row[i,j]
endif
) endif;

% Implied constraint: when there are more concentrations than rows, help the search to not fill in a row completely too soon by saving space in each row for the other compounds
constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= numrows)) then forall(i in Plates, j in Rows, r in 1..(compounds*replicates-1))((count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + sum([ experiment_plate_row[i,max_compound_concentrations*(k-1)+1]>0 /\ (compound_concentrations[((k-1) mod compounds)+1]>=numrows) | k in (r+1) .. compounds*replicates])) <= experiments_in_plate_row[i,j]) endif;



%%%%%%
%% Spreads concentrations over different columns. It's been replaced by the more restrictive alldifferent version below.
%constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],numcols-2*size_empty_edge), [experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) ( 
alldifferent( [experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numcols)]):: domain
))) endif; 

% Split concentrations between left-most and right-most halves of the plate

array [1..compounds*replicates,Horizontal] of var 0..max_compound_concentrations: lr_concentrations;

%% For each replicate of each compound, count how many concentrations there are in the left/right halves of the plate
%% Definition of lr_concentrations (left)  
constraint if concentrations_on_different_columns /\ even_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,left],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..(numcols div 2))::domain
))) endif; 


%% Definition of lr_concentrations (right)  
constraint if concentrations_on_different_columns /\ even_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,right],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], (numcols div 2)+1..numcols)::domain
))) endif; 

%% Balancing left and right concentrations
constraint if concentrations_on_different_columns /\ even_columns then (forall(l in 1..compounds, h in Horizontal) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,h] <= ((compound_concentrations[l] + 1) div 2)
))) endif; 

constraint if concentrations_on_different_columns /\ even_columns then (forall(l in 1..compounds, h in Horizontal) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,h] >= (compound_concentrations[l] div 2)
))) endif; 

constraint forall(l in 1..compounds, h in Horizontal) (forall(r in compound_replicates[l]..replicates-1) (
 lr_concentrations[r*compounds + l,h] == 0
)); 

%% Implied constraints:
constraint if concentrations_on_different_columns /\ even_columns then (sum(lr_concentrations) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) endif;

constraint if concentrations_on_different_columns /\ even_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,left] + lr_concentrations[r*compounds + l,right] == compound_concentrations[l]
))) endif; 

constraint if concentrations_on_different_columns /\ even_columns then (forall(h in Horizontal)(
sum(lr_half_plates[..,h]) == sum(lr_concentrations[..,h])
))endif;

constraint if concentrations_on_different_columns then forall(i in Plates, h in Horizontal) (sum(c in 0..compounds*replicates-1)(lr_concentrations[c+1,h]*(experiment_plate[c*max_compound_concentrations+1]==i)) == lr_half_plates[i,h] ) endif;

%% Implied constraints 
%% These constraints are too weak when the number of plates increases
%% Right now it looks like it's mostly taking time 

constraint global_cardinality_low_up(experiment_row, [i | i in Rows], [0| i in Rows], [numplates*numcols| i in Rows]);
constraint global_cardinality_low_up(experiment_column, [i | i in Columns], [0| i in Columns], [numplates*numrows| i in Columns]);   

array [Plates,Rows] of var 0..numcols: controls_in_plate_row;
array [Plates,Columns] of var 0..numrows: controls_in_plate_column;
array [Plates,Rows] of var 0..numcols: controls_only_in_plate_row;
array [Plates,Columns] of var 0..numrows: controls_only_in_plate_column;
array [Plates,Rows] of var 0..numcols: emptywells_in_plate_row;
array [Plates,Columns] of var 0..numrows: emptywells_in_plate_column;
array [Rows] of var 0..numplates*numcols: controls_in_row;
array [Columns] of var 0..numplates*numrows: controls_in_column;

constraint (sum(controls_in_plate_row) == total_controls + emptywells);
constraint (sum(controls_in_plate_column) == total_controls + emptywells);

constraint (sum(controls_only_in_plate_row) == total_controls);
constraint (sum(controls_only_in_plate_column) == total_controls);

constraint (sum(controls_in_row) == total_controls + emptywells);
constraint (sum(controls_in_column) == total_controls + emptywells);

constraint (sum(emptywells_in_plate_row) == emptywells);
constraint (sum(emptywells_in_plate_column) == emptywells);

%% Redundant
%constraint (sum(controls_in_plate_row) ==  sum(controls_only_in_plate_row) + emptywells);
%constraint (sum(controls_in_plate_column) == sum(controls_only_in_plate_column) + emptywells);

constraint forall(i in Plates, j in Rows)(count(plates[i,j,..],0,emptywells_in_plate_row[i,j]));
constraint forall(i in Plates, k in Columns)(count(plates[i,..,k],0,emptywells_in_plate_column[i,k]));

%Among: Requires exactly n variables in x to take one of the values in v .
constraint forall(i in Plates, j in Rows) (among(controls_in_plate_row[i,j], plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);
constraint forall(i in Plates, j in Rows) (among(controls_only_in_plate_row[i,j], plates[i,j,..], (experiments+1..experiments+num_controls*max_control_concentrations))::domain);


constraint forall(i in Plates, k in Columns) (among(controls_in_plate_column[i,k], plates[i,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);
constraint forall(i in Plates, k in Columns) (among(controls_only_in_plate_column[i,k], plates[i,..,k], (experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, j in Rows) (among(emptywells_in_plate_row[i,j], plates[i,j,..], {0})::domain);
constraint forall(i in Plates, k in Columns) (among(emptywells_in_plate_column[i,k], plates[i,..,k], {0})::domain);

constraint forall(j in Rows) (among(controls_in_row[j], plates[..,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(k in Columns) (among(controls_in_column[k], plates[..,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(j in Rows) (controls_in_row[j] == sum(controls_in_plate_row[..,j]));

constraint forall(j in Rows) (controls_in_row[j] == sum(emptywells_in_plate_row[..,j]) + sum(controls_only_in_plate_row[..,j]));

constraint forall(k in Columns) (controls_in_column[k] == sum(controls_in_plate_column[..,k]));

constraint forall(k in Columns) (controls_in_column[k] == sum(emptywells_in_plate_column[..,k]) + sum(controls_only_in_plate_column[..,k]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,upper,..]) + count(plates[i,1..(numrows div 2),..],0) == sum(controls_in_plate_row[i,1..(numrows div 2)]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,lower,..]) + count(plates[i,(numrows div 2)+1..numrows,..],0) == sum(controls_in_plate_row[i,(numrows div 2)+1..numrows]));

constraint forall(i in Plates) ( sum(controls_lr_plates[i,left,..]) + count(plates[i,..,1..(numcols div 2)],0) == sum(controls_in_plate_column[i,1..(numcols div 2)]));
constraint forall(i in Plates) ( sum(controls_lr_plates[i,right,..]) + count(plates[i,..,(numcols div 2)+1..numcols],0) == sum(controls_in_plate_column[i,(numcols div 2)+1..numcols]));

constraint forall(i in Plates) ( sum_controls_ul_plates[i,upper] == sum(j in 1..(numrows div 2))(controls_only_in_plate_row[i,j]));
constraint forall(i in Plates) ( sum_controls_ul_plates[i,lower] == sum(j in (numrows div 2)+1..numrows)(controls_only_in_plate_row[i,j]));

constraint forall(i in Plates) ( sum_controls_lr_plates[i,left] == sum(j in 1..(numcols div 2))(controls_only_in_plate_column[i,j]));
constraint forall(i in Plates) ( sum_controls_lr_plates[i,right] == sum(j in (numcols div 2)+1..numcols)(controls_only_in_plate_column[i,j]));


constraint sum(sum_controls_ul_plates) == total_controls;
constraint sum(sum_controls_lr_plates) == total_controls;

constraint forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) >= (total_controls div 2));
constraint forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) <= ((total_controls + 1) div 2));

constraint forall(h in Horizontal) (sum(sum_controls_lr_plates[..,h]) >= (total_controls div 2));
constraint forall(h in Horizontal) (sum(sum_controls_lr_plates[..,h]) <= ((total_controls + 1) div 2));

%% Implied constraint: how many experiments can be in a row/column? It's too weak and needs to be done per plate.
constraint global_cardinality(experiment_row, [j | j in Rows], [(numplates*numcols)-controls_in_row[j]| j in Rows]);
constraint global_cardinality(experiment_column, [k | k in Columns], [(numplates*numrows)-controls_in_column[k]| k in Columns]);




%% Implied constraint: how many experiments can there be in a row? Per plate
% I think this still works...
%% TODO: Redundant 
constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])== numcols-among(plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])== numcols-controls_in_plate_row[i,j]);

constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])== numcols-emptywells_in_plate_row[i,j]-controls_only_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])== numrows-among(plates[i,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])== numrows-controls_in_plate_column[i,k]);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])== numrows-emptywells_in_plate_column[i,k]-controls_only_in_plate_column[i,k]);


constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments]) == experiments_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments]) == experiments_in_plate_column[i,k]);

%experiments_in_plate_row
constraint forall(i in Plates, j in Rows) (experiments_in_plate_row[i,j] == numcols-among(plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, j in Rows) (experiments_in_plate_row[i,j] == numcols-controls_in_plate_row[i,j]);

constraint forall(i in Plates, j in Rows) (experiments_in_plate_row[i,j] == numcols-emptywells_in_plate_row[i,j]-controls_only_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (experiments_in_plate_column[i,k] == numrows-among(plates[i,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, k in Columns) (experiments_in_plate_column[i,k] == numrows-controls_in_plate_column[i,k]);

constraint forall(i in Plates, k in Columns) (experiments_in_plate_column[i,k] == numrows-emptywells_in_plate_column[i,k]-controls_only_in_plate_column[i,k]);


%% Below suggested by Gustav. It improves flattening but worsens the solving time
% constraint forall(i in Plates, j in Rows) (
%   let{
%     var int: exprInRow;
%     constraint exprInRow = among(plates[i,j,..], 1..experiments)::domain; 
%     %sum([experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments]);
%   } in exprInRow <= numcols-among([plates[i,j,k] | k in Columns], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));
  

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


constraint if concentrations_on_different_rows then forall(l in 1..compounds) (forall(conc in 0..compound_concentrations[l]-1)(alldifferent([experiment_row[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + conc + 1] | i in 0..(min(compound_replicates[l],numrows)-1)])::domain)) endif; 

constraint if concentrations_on_different_columns then forall(l in 1..compounds) (forall(conc in 0..compound_concentrations[l]-1)(alldifferent([experiment_column[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + conc + 1] | i in 0..(min(compound_replicates[l],numcols)-1)])::domain)) endif; 


%% Balancing compounds between plates: distribute the compounds equitatively among all the plates


%% TODO: could this lower bound be stronger?
%% FIX ME!
%int: at_least = max_compound_concentrations*floor((compounds*replicates)/numplates) + combination_concentrations*floor((combinations*replicates)/numplates); 

%int: at_least = min(compound_concentrations)*floor(sum(compound_replicates)/numplates) + combination_concentrations*floor((combinations*replicates)/numplates); 

%% FIX ME!
%% FIX ME!!
%int: at_most = min(max_compound_concentrations*ceil(sum(compound_replicates)/numplates) + combination_concentrations*ceil((combinations*replicates)/numplates), 
 %   inner_plate_size -(sum([floor(control_replicates[i]*control_concentrations[i]/numplates) | i in 1..num_controls])));



%% This implied constraint is very important for propagation!

%constraint global_cardinality_low_up(experiment_plate,[ i | i in Plates],[ at_least | i in Plates],[ at_most | i in Plates]);

% I could make this stronger, but I would need to assume that experiments can indeed be distrubuted in a balanced way.
% I would like to do something that allows example-02 and example-03 to work
% This constraint assumes that there is no plate with only controls, but that seems fair enough


int: min_compounds_plate = max(floor((sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]))/inner_plate_size), min(compound_concentrations++[0]) );

constraint global_cardinality_low_up(experiment_plate,[ i | i in Plates],[ min_compounds_plate | i in Plates],[ inner_plate_size | i in Plates]);

%% TODO: Think more about this constraint and what happens when there are replicates, etc.
%% This option is possibly going to make some plans unfeasible (when there are different numbers of replicates, etc).
constraint if sorted_compounds == true /\ max(compound_replicates)<=1 then sorted_except_0(experiment_plate,numplates,at_least_compounds,at_most_compounds) endif;

%% WARNING! This option assumes that compounds can be placed on plates in order! This works, for example, when all compounds have the same 
% number of replicates and concentrations and one would like to generate several layouts with the compounds on the exact same plates.
% Otherwise, experiment_plate needs to be given as input
constraint if sorted_compounds == true /\ replicates_on_same_plate == true  then sorted_except_0(experiment_plate[1..compounds*max(compound_concentrations)],numplates,at_least_compounds div numplates,at_most_compounds div numplates) endif;



% Balancing compounds between plates (and knowing that there must be some sort of balance!)
%% FIX ME!
%% I want to remove these constraints but I need to make other stronger first
int: at_least_compounds = min(compound_concentrations++[infinity])*(sum(compound_replicates) div numplates);
int: at_most_compounds = max_compound_concentrations*ceil(sum(compound_replicates)/numplates);

%% TODO: False only if there are no compounds
constraint if at_least_compounds <= at_most_compounds then global_cardinality_low_up([experiment_plate[i] | i in 1..experiments] ,[ i | i in Plates],[ at_least_compounds | i in Plates],[ at_most_compounds | i in Plates]) endif;




%% Implied constraints 
%% These constraints are too weak when the number of plates increases
constraint if concentrations_on_different_rows then global_cardinality_low_up(experiment_row, [i | i in Rows], [numplates*(at_least_compounds div numrows)| i in Rows], [numplates*numcols| i in Rows]) endif;

constraint if concentrations_on_different_columns then global_cardinality_low_up(experiment_column, [i | i in Columns], [numplates*floor(at_least_compounds/numcols)| i in Columns], [numplates*numrows| i in Columns]) endif;

constraint if concentrations_on_different_rows then forall(j in Rows)(sum(experiments_in_plate_row[..,j])>=numplates*(at_least_compounds div numrows)) endif;
constraint if concentrations_on_different_columns then forall(k in Columns)(sum(experiments_in_plate_column[..,k])>=numplates*(at_least_compounds div numcols)) endif;

constraint if concentrations_on_different_rows then forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j]>=(at_least_compounds div numrows)) endif;
constraint if concentrations_on_different_columns then forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k]>=(at_least_compounds div numcols)) endif;

constraint if concentrations_on_different_rows then forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j]<=ceil(at_most_compounds / numrows)) endif;
constraint if concentrations_on_different_columns then forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k]<=ceil(at_most_compounds / numcols)) endif;

array [Plates,Vertical] of var 0..numcols*(numrows div 2): ul_half_plates;
array [Plates,Horizontal] of var 0..numrows*(numcols div 2): lr_half_plates;

%%% Counting the number of experiments in the upper and lower half-plates
constraint forall(i in Plates) (among(ul_half_plates[i,upper],plates[i,1..numrows div 2,..], 1..experiments)::domain);
constraint forall(i in Plates) (among(ul_half_plates[i,lower],plates[i,(numrows div 2)+1..numrows,..], 1..experiments)::domain);

constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_row[e] <= (numrows div 2)) == ul_half_plates[i,upper]);
constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_row[e] > (numrows div 2)) == ul_half_plates[i,lower]);

constraint (sum(e in 1..experiments)(experiment_row[e] in 1..(numrows div 2)) == sum(i in Plates)(ul_half_plates[i,upper]));
constraint (sum(e in 1..experiments)(experiment_row[e] in (numrows div 2)+1..numrows) == sum(i in Plates)(ul_half_plates[i,lower]));

constraint if concentrations_on_different_rows /\ even_rows then forall(i in Plates)(sum(experiments_in_plate_row[i,1..(numrows div 2)]) == ul_half_plates[i,upper]) endif;
constraint if concentrations_on_different_rows /\ even_rows then forall(i in Plates)(sum(experiments_in_plate_row[i,(numrows div 2)+1..numrows]) == ul_half_plates[i,lower]) endif;

%% The following constraints should be redundant when using the two constraints above.
constraint (among(sum(i in Plates)(ul_half_plates[i,upper]),experiment_row, 1..(numrows div 2))::domain);
constraint (among(sum(i in Plates)(ul_half_plates[i,lower]),experiment_row, (numrows div 2)+1..numrows)::domain);

constraint (among(sum(i in Plates)(lr_half_plates[i,left]),experiment_column, 1..(numcols div 2))::domain);
constraint (among(sum(i in Plates)(lr_half_plates[i,right]),experiment_column, (numcols div 2)+1..numcols)::domain);


constraint if even_rows then (forall(i in Plates)(abs(ul_half_plates[i,upper]-ul_half_plates[i,lower])<=1)) endif;

%%% Counting the number of experiments in the left and right half-plates
constraint forall(i in Plates) (among(lr_half_plates[i,left],plates[i,..,1..numcols div 2], 1..experiments)::domain);
constraint forall(i in Plates) (among(lr_half_plates[i,right],plates[i,..,(numcols div 2)+1..numcols], 1..experiments)::domain);

constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_column[e] <= (numcols div 2)) == lr_half_plates[i,left]);
constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_column[e] > (numcols div 2)) == lr_half_plates[i,right]);

constraint if even_columns then (sum(e in 1..experiments)(experiment_column[e] in 1..(numcols div 2)) == sum(i in Plates)(lr_half_plates[i,left])) endif;
constraint if even_columns then (sum(e in 1..experiments)(experiment_column[e] in (numcols div 2)+1..numcols) == sum(i in Plates)(lr_half_plates[i,right])) endif;

constraint if even_columns then (forall(i in Plates)(abs(lr_half_plates[i,left]-lr_half_plates[i,right])<=1)) endif;




%%% Counting the number of experiments in left-most and right-most half-plates
%constraint forall(i in Plates) (among(lr_half_plates[i,left],[plates[i,j,k] | j in Rows, k in 1..ceil(numcols/2)], 1..experiments)); %%::domain
%constraint forall(i in Plates) (among(lr_half_plates[i,right],[plates[i,j,k] | j in Rows, k in floor(numcols/2)+1..numcols], 1..experiments));

%constraint forall(i in Plates) (sum(e in 1..experiments, k in 1..ceil(numcols/2))(experiment_plate[e] == i /\ experiment_column[e] == k) == lr_half_plates[i,left]);
%constraint forall(i in Plates) (sum(e in 1..experiments, k in ceil(numcols/2)+1..numcols)(experiment_plate[e] == i /\ experiment_column[e] == k) == lr_half_plates[i,right]);


%array [Plates,Vertical,Horizontal] of var floor((at_least_compounds+at_least_combinations)/4)..ceil((at_most_compounds+at_most_combinations)/4): quarter_plates;

% Balancing experiments in quarters
%constraint forall(i in Plates) (among(quarter_plates[i,upper,left],[plates[i,j,k] | j in 1..ceil(numrows/2), k in 1..ceil(numcols/2)], 1..experiments));
%constraint forall(i in Plates) (among(quarter_plates[i,upper,right],[plates[i,j,k] | j in 1..ceil(numrows/2), k in floor(numcols/2)+1..numcols], 1..experiments));

%constraint forall(i in Plates) (among(quarter_plates[i,lower,left],[plates[i,j,k] | j in floor(numrows/2)+1..numrows, k in 1..ceil(numcols/2)], 1..experiments));
%constraint forall(i in Plates) (among(quarter_plates[i,lower,right],[plates[i,j,k] | j in floor(numrows/2)+1..numrows, k in floor(numcols/2)+1..numcols], 1..experiments));


%%% NEW %%%
%% Implied constraints %%
constraint sum(ul_half_plates) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]);
constraint sum(lr_half_plates) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]);

constraint forall(v in Vertical)(sum(ul_half_plates[..,v]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])/2));
constraint forall(h in Horizontal)(sum(lr_half_plates[..,h]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])/2));

constraint forall(i in Plates, v in Vertical) ( ul_half_plates[i,v] <= ((sum([ (experiment_plate[c] == i) | c in 1..experiments]) + 1) div 2));
constraint forall(i in Plates, h in Horizontal) ( lr_half_plates[i,h] <= ((sum([ (experiment_plate[c] == i) | c in 1..experiments]) + 1) div 2));
                                                                
constraint forall(i in Plates, v in Vertical)(((floor(sum(compound_replicates)/numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations) div 2 <= ul_half_plates[i,v]);  

constraint forall(i in Plates, h in Horizontal)(((floor(sum(compound_replicates)/numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations) div 2 <= lr_half_plates[i,h]); 

%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v])<=at_most);
%constraint forall(i in Plates)(sum(h in Horizontal)(lr_half_plates[i,h])<=at_most);

%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v])>=at_least);
%constraint forall(i in Plates)(sum(h in Horizontal)(lr_half_plates[i,h])>=at_least);


%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v]) == sum(h in Horizontal)(lr_half_plates[i,h]));

%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v]) == sum(v in Vertical, h in Horizontal)(quarter_plates[i,v,h]));

%% Implied constraints
constraint forall(i in Plates)(count_eq(experiment_plate, i, sum(h in Horizontal)(lr_half_plates[i,h])));
constraint forall(i in Plates)(count_eq(experiment_plate, i, sum(v in Vertical)(ul_half_plates[i,v])));

%constraint forall(i in Plates)(count_eq(experiment_plate, i, sum(v in Vertical, h in Horizontal)(quarter_plates[i,v,h])));





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Balancing controls between plates.
%% Asumes controls can be balanced this way, that is, that the designer expects about the same number of controls on each plate and not a plate full of controls.

% array[int] of float: controls_per_plate = [control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])/numplates | i in 1..num_controls*max_control_concentrations];

% array[int] of int: min_controls_per_plate = [floor(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

% array[int] of int: max_controls_per_plate = [ceil(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

% constraint forall(i in Plates)(global_cardinality_low_up([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], min_controls_per_plate, max_controls_per_plate));


%% V2
array[Plates, 1..num_controls*max_control_concentrations] of var int: controls_per_plate;

constraint forall(p in Plates, i in 1..num_controls*max_control_concentrations) (controls_per_plate[p,i] >= floor(control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])/numplates));

constraint forall(p in Plates, i in 1..num_controls*max_control_concentrations) (controls_per_plate[p,i] <= ceil(control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])/numplates));

constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], controls_per_plate[i,..]));


% Implied constraint
constraint forall(i in 1..num_controls*max_control_concentrations) (sum(p in Plates)(controls_per_plate[p,i]) == control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]));


%% I need to work on these constraints, which seem to be quite ineficient. I believe I need to use variables for halves plates, maybe even for quarter plates.

array [Plates,Vertical,1..num_controls*max_control_concentrations] of var int: controls_ul_plates;

array [Plates,Vertical] of var int: sum_controls_ul_plates;

array [Plates,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_lr_plates;

array [Plates,Horizontal] of var int: sum_controls_lr_plates;

array [Plates,Vertical,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_quarter_plates;

array [Plates,Rows,{0} union 1..num_controls*max_control_concentrations] of var int: controls_row_plates;

array [Plates,Columns,{0} union 1..num_controls*max_control_concentrations] of var int: controls_column_plates;


constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..(numrows div 2), k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,upper,d] | d in 1..num_controls*max_control_concentrations]));

constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in (numrows div 2)+1..numrows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,lower,d] | d in 1..num_controls*max_control_concentrations]));


%% Defining controls_quarter_plates variables
constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..(numrows div 2), k in 1..(numcols div 2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,upper,left,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..(numrows div 2), k in (numcols div 2)+1..numcols], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,upper,right,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in (numrows div 2)+1..numrows, k in 1..(numcols div 2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,lower,left,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in (numrows div 2)+1..numrows, k in (numcols div 2)+1..numcols], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,lower,right,d] | d in 1..num_controls*max_control_concentrations])) endif;


constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]<=ceil(controls_per_plate[i,d]/2));
constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]>=floor(controls_per_plate[i,d]/2));

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(abs(controls_ul_plates[i,upper,d] - controls_ul_plates[i,lower,d]) <= 1);

constraint if even_columns then forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(abs(controls_quarter_plates[i,v,left,d] - controls_quarter_plates[i,v,right,d]) <= 1) endif;

constraint if even_columns then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(abs(controls_quarter_plates[i,upper,h,d] - controls_quarter_plates[i,lower,h,d]) <= 1) endif;


constraint if even_columns then forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,v,left,d] + controls_quarter_plates[i,v,right,d] == controls_ul_plates[i,v,d]) endif;

constraint if even_columns then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,upper,h,d] + controls_quarter_plates[i,lower,h,d] == controls_lr_plates[i,h,d]) endif;

constraint if even_columns then forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,v,left,d] >= (controls_ul_plates[i,v,d] div 2)) endif;

constraint if even_columns then forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,v,right,d] >= (controls_ul_plates[i,v,d] div 2)) endif;

constraint if even_columns then forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,v,left,d] <= ((controls_ul_plates[i,v,d]+1) div 2)) endif;

constraint if even_columns then forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,v,right,d] <= ((controls_ul_plates[i,v,d]+1) div 2)) endif;

constraint if even_rows then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,upper,h,d] >= (controls_lr_plates[i,h,d] div 2)) endif;

constraint if even_rows then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,lower,h,d] >= (controls_lr_plates[i,h,d] div 2)) endif;

constraint if even_rows then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,upper,h,d] <= ((controls_lr_plates[i,h,d]+1) div 2)) endif;

constraint if even_rows then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_quarter_plates[i,lower,h,d] <= ((controls_lr_plates[i,h,d]+1) div 2)) endif;

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)( sum(v in Vertical)(controls_ul_plates[i,v,d]) = controls_per_plate[i,d]);

constraint forall(i in 1..num_controls*max_control_concentrations)(sum([controls_ul_plates[p,v,i] | v in Vertical, p in Plates]) == control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) );


% Implied constraint:
constraint forall(i in Plates, v in Vertical)(sum(controls_ul_plates[i,v,..]) == sum_controls_ul_plates[i,v]);
constraint forall(i in Plates, h in Horizontal)(sum(controls_lr_plates[i,h,..]) == sum_controls_lr_plates[i,h]);

%% Half of the total number of controls on each upper/lower half of the plate.
constraint forall(i in Plates)(abs(sum(controls_ul_plates[i,upper,..]) - sum(controls_ul_plates[i,lower,..])) <= 1);
constraint forall(i in Plates)(abs(sum_controls_ul_plates[i,upper] - sum_controls_ul_plates[i,lower]) <= 1);

constraint forall(i in Plates)(abs(sum(controls_lr_plates[i,left,..]) - sum(controls_lr_plates[i,right,..])) <= 1);
constraint forall(i in Plates)(abs(sum_controls_lr_plates[i,left] - sum_controls_lr_plates[i,right]) <= 1);


%% Balancing the total of controls between left and right half-plates

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in 1..floor(numcols/2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,left,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in floor(numcols/2)+1..numcols], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,right,d] | d in 1..num_controls*max_control_concentrations])) endif;


constraint if even_columns then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]<=ceil(controls_per_plate[i,d]/2)) endif;

constraint if even_columns then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]>= (controls_per_plate[i,d] div 2)) endif;

constraint if even_columns then forall(i in Plates, d in 1..num_controls*max_control_concentrations)(abs(controls_lr_plates[i,left,d] - controls_lr_plates[i,right,d]) <= 1) endif;

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,left,d] + controls_lr_plates[i,right,d] = controls_per_plate[i,d]);

constraint forall(i in 1..num_controls*max_control_concentrations)(sum([controls_lr_plates[p,h,i] | h in Horizontal, p in Plates]) == control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) );

%% Half of the total number of controls on each upper/lower half of the plate.
constraint forall(i in Plates)(abs(sum(d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,left,d]) - sum(d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,right,d])) <= 1);

constraint forall(i in Plates)(abs(sum(d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d]) - sum(d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,lower,d])) <= 1);

%% Implied constraints for half plates
%% Connecting the number of compounds with the number of controls for each half plate.
%% TODO: this is only correct for plates with an even number of rows
constraint forall(i in Plates, v in Vertical)(ul_half_plates[i,v] + sum(controls_ul_plates[i,v,..]) <= (inner_plate_size div 2));
constraint forall(i in Plates, h in Horizontal)(lr_half_plates[i,h] + sum(controls_lr_plates[i,h,..]) <= (inner_plate_size div 2));

constraint forall(i in Plates)( sum(ul_half_plates[i,..]) + sum(sum_controls_ul_plates[i,..]) + count(plates[i,..,..], 0) == inner_plate_size);
constraint forall(i in Plates)( sum(lr_half_plates[i,..]) + sum(sum_controls_lr_plates[i,..]) + count(plates[i,..,..], 0) == inner_plate_size);

constraint forall(i in Plates)( ul_half_plates[i,upper] + sum_controls_ul_plates[i,upper] + count(plates[i,1..numrows div 2,..], 0) == inner_plate_size div 2);
constraint forall(i in Plates)( ul_half_plates[i,lower] + sum_controls_ul_plates[i,lower] + count(plates[i,(numrows div 2)+1..numrows,..], 0) == inner_plate_size div 2);

constraint forall(i in Plates)( lr_half_plates[i,left] + sum_controls_lr_plates[i,left] + count(plates[i,..,1..numcols div 2], 0) == inner_plate_size div 2);
constraint forall(i in Plates)( lr_half_plates[i,right] + sum_controls_lr_plates[i,right] + count(plates[i,..,(numcols div 2)+1..numcols], 0) == inner_plate_size div 2);


%%->constraint forall(i in Plates)(ul_half_plates[i,upper] + ul_half_plates[i,lower] + sum(d in 1..num_controls*max_control_concentrations,v in Vertical)(controls_ul_plates[i,v,d]) + count([plates[i,j,k] | j in 1+size_empty_edge..numrows-size_empty_edge, k in 1+size_empty_edge..numcols-size_empty_edge], 0) == inner_plate_size);


%% Is this helping? I don't think so! But it might...
%%->constraint (sum(r in 1..floor(numrows/2))(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d]) + count([plates[i,j,k] | i in Plates, j in 1..floor(numrows/2), k in Columns], 0));

%%->constraint (sum(r in floor(numrows/2)+1..numrows)(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,lower,d]) + count([plates[i,j,k] | i in Plates, j in floor(numrows/2)+1..numrows, k in Columns], 0));


%%% Controls per row across plates%%%

%constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_row[j] <= ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;

%constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_row[j] >= floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] <= ceil((total_controls+emptywells)/numrows) ) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] >= floor((total_controls+emptywells)/numrows) ) endif;

constraint forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] <= ceil(ceil(total_controls/numplates)/numrows) ) ;

constraint forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] >= ((total_controls div numplates) div numrows) ) ;


%%% Controls per row per plate %%%
%% Relaxing this constraint a bit...
%constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] <= control_slack + ceil(ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge))/numplates) + 2*size_empty_edge);

%constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] >= floor(floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge))/numplates) + 2*size_empty_edge);


constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] <= control_slack + ceil(ceil((total_controls+emptywells)/numplates)/numrows) );

constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] >= floor(floor((total_controls+emptywells)/numplates)/numrows) );

%constraint if emptywells == 0 then forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] == controls_in_plate_row[i,j] ) else forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] <= controls_in_plate_row[i,j] ) endif;

constraint forall(i in Plates, j in Rows)(controls_in_plate_row[i,j] == controls_only_in_plate_row[i,j] + emptywells_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (controls_in_plate_column[i,k] <= control_slack + ceil(ceil((total_controls+emptywells)/numplates)/numcols) );

constraint forall(i in Plates, k in Columns) (controls_in_plate_column[i,k] >= floor(floor((total_controls+emptywells)/numplates)/numcols) );

constraint forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] + emptywells_in_plate_column[i,k] == controls_in_plate_column[i,k] );

constraint if interconnected_plates then forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] >= (controls_in_row[j] div numplates)) endif;

constraint if interconnected_plates then forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] <= ((controls_in_row[j]+numplates-1)div numplates)) endif;


%%% Controls per column across plates%%%
 
constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] <= ceil((total_controls+emptywells)/numcols) ) endif;

constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] >= floor((total_controls+emptywells)/numcols) ) endif;

constraint forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] <= ceil(ceil(total_controls / numplates) / numcols) );

constraint forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] >= ((total_controls div numplates) div numcols) );


%%% Attempting to balance controls inside each row

array [Rows,Horizontal] of var 0..numplates*numcols: controls_in_half_row;

array [Rows,Horizontal] of var 0..numplates*numcols: controls_only_in_half_row; %Does NOT include empty wells

array [Plates,Rows,Horizontal] of var 0..ceil(numcols/2): controls_in_half_row_plate; %includes empty wells

array [Plates,Rows,Horizontal] of var 0..ceil(numcols/2): controls_only_in_half_row_plate; %Does NOT include empty wells

constraint forall(j in Rows) ( among(controls_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..floor(numcols/2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(j in Rows) ( among(controls_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in floor(numcols/2)+1..numcols], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..floor(numcols/2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,right],[plates[i,j,k] | k in floor(numcols/2)+1..numcols], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_only_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..(numcols div 2)], ( experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_only_in_half_row_plate[i,j,right],[plates[i,j,k] | k in (numcols div 2)+1..numcols], (experiments+1..experiments+num_controls*max_control_concentrations))::domain);

% controls_only_in_half_row
constraint forall(j in Rows) ( among(controls_only_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..(numcols div 2)], ( experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(j in Rows) ( among(controls_only_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in (numcols div 2)+1..numcols], ( experiments+1..experiments+num_controls*max_control_concentrations))::domain);


constraint forall(j in Rows, h in Horizontal) (controls_in_half_row[j,h] == sum(i in Plates)(controls_in_half_row_plate[i,j,h]));

constraint forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] == sum(h in Horizontal)(controls_only_in_half_row_plate[i,j,h]));

constraint forall(i in Plates, j in Rows, h in Horizontal) (controls_only_in_plate_row[i,j] div 2 <= controls_only_in_half_row_plate[i,j,h]);

constraint forall(i in Plates, j in Rows, h in Horizontal) ((controls_only_in_plate_row[i,j]+1) div 2 >= controls_only_in_half_row_plate[i,j,h]);

constraint forall(i in Plates, k in Columns, v in Vertical) (controls_only_in_plate_column[i,k] div 2 <= controls_only_in_half_column_plate[i,k,v]);

constraint forall(i in Plates, k in Columns, v in Vertical) ((controls_only_in_plate_column[i,k]+1) div 2 >= controls_only_in_half_column_plate[i,k,v]);


%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,right]);

constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/numrows) div 2) <= controls_in_half_row[j,left]) endif;

constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/numrows) div 2) <= controls_in_half_row[j,right]) endif;

constraint if interconnected_plates then forall(j in Rows, h in Horizontal) ( (controls_in_row[j] div 2) <= controls_in_half_row[j,h] ) endif;
% constraint if interconnected_plates then forall(j in Rows, h in Horizontal) ( ((controls_in_row[j]+1) div 2) >= controls_in_half_row[j,h] ) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] == (controls_in_half_row[j,left] + controls_in_half_row[j,right])) endif;


%constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( floor(floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/numplates)/(numrows-2*size_empty_edge))/2) + size_empty_edge) <= controls_in_half_row_plate[i,j,h]) endif;

constraint forall(i in Plates, h in Horizontal, j in Rows) ( (controls_in_plate_row[i,j] div 2) <= controls_in_half_row_plate[i,j,h]);

constraint forall(i in Plates, h in Horizontal, j in Rows) ( ( (controls_in_plate_row[i,j]+1) div 2) >= controls_in_half_row_plate[i,j,h]);


%constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil(ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/numplates)/(numrows-2*size_empty_edge))/2) + size_empty_edge) >= controls_in_half_row_plate[i,j,h]) endif;


constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] == controls_in_half_row_plate[i,j,left] + controls_in_half_row_plate[i,j,right]);


%%% Attempting to balance controls inside each column

array [Columns,Vertical] of var 0..numplates*numrows: controls_in_half_column;

array [Columns,Vertical] of var 0..numplates*numrows: controls_only_in_half_column; %Does NOT include empty wells

array [Plates,Columns,Vertical] of var 0..(numrows div 2): controls_in_half_column_plate; %includes empty wells

array [Plates,Columns,Vertical] of var 0..(numrows div 2): controls_only_in_half_column_plate; %Does NOT include empty wells

array [Plates,Rows,Horizontal] of var 0..ceil(numcols/2): experiments_in_half_row_plate;

array [Plates,Columns,Vertical] of var 0..ceil(numrows/2): experiments_in_half_column_plate;  

constraint forall(k in Columns) ( among(controls_in_half_column[k,upper],[plates[i,j,k] | i in Plates, j in 1..(numrows div 2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(k in Columns) ( among(controls_in_half_column[k,lower],[plates[i,j,k] | i in Plates, j in (numrows div 2)+1..numrows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, k in Columns) ( among(controls_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(numrows div 2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, k in Columns) ( among(controls_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (numrows div 2)+1..numrows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(i in Plates, k in Columns) ( among(controls_only_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(numrows div 2)], ( experiments+1..experiments+num_controls*max_control_concentrations))::domain);
	
constraint forall(i in Plates, k in Columns) ( among(controls_only_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (numrows div 2)+1..numrows], ( experiments+1..experiments+num_controls*max_control_concentrations))::domain);

%% controls_only_in_half_column
constraint forall(k in Columns) ( among(controls_only_in_half_column[k,upper],[plates[i,j,k] | i in Plates, j in 1..(numrows div 2)], ( experiments+1..experiments+num_controls*max_control_concentrations))::domain);

constraint forall(k in Columns) ( among(controls_only_in_half_column[k,lower],[plates[i,j,k] | i in Plates, j in (numrows div 2)+1..numrows], ( experiments+1..experiments+num_controls*max_control_concentrations))::domain);


constraint forall(k in Columns, v in Vertical) (controls_in_half_column[k,v] == sum(i in Plates)(controls_in_half_column_plate[i,k,v]));

constraint forall(i in Plates, k in Columns) ( controls_in_plate_column[i,k] == sum(v in Vertical)(controls_in_half_column_plate[i,k,v]));

constraint forall(i in Plates, v in Vertical, k in Columns) ( controls_in_plate_column[i,k] <= 2*controls_in_half_column_plate[i,k,v] + 1);

constraint forall(i in Plates, v in Vertical, k in Columns) ( controls_in_plate_column[i,k] >= 2*controls_in_half_column_plate[i,k,v] - 1);

constraint forall(i in Plates, v in Vertical, k in Columns) ( (controls_in_plate_column[i,k] div 2) <= controls_in_half_column_plate[i,k,v]);

constraint forall(i in Plates, v in Vertical, k in Columns) ( ( (controls_in_plate_column[i,k]+1) div 2) >= controls_in_half_column_plate[i,k,v]);

constraint forall(i in Plates, k in Columns) ( controls_only_in_plate_column[i,k] == sum(v in Vertical)(controls_only_in_half_column_plate[i,k,v]));

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,right]);

constraint if interconnected_plates then forall(k in Columns, v in Vertical) ( (floor((total_controls+emptywells)/numcols)  div 2) <= controls_in_half_column[k,v]) endif;

constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] == (controls_in_half_column[k,upper] + controls_in_half_column[k,lower])) endif;

%%% Defining experiments_in_half_row_plate /\ experiments_in_half_column_plate

constraint forall(i in Plates, j in Rows) ( among(experiments_in_half_row_plate[i,j,left], [plates[i,j,k] | k in 1..(numcols div 2)], (1..experiments))::domain);
constraint forall(i in Plates, j in Rows) ( among(experiments_in_half_row_plate[i,j,right],[plates[i,j,k] | k in (numcols div 2)+1..numcols], (1..experiments))::domain);

constraint forall(i in Plates, k in Columns) ( among(experiments_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(numrows div 2)], (1..experiments))::domain);
constraint forall(i in Plates, k in Columns) ( among(experiments_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (numrows div 2)+1..numrows], (1..experiments))::domain);

constraint forall(i in Plates, j in Rows) ( sum(h in Horizontal)(experiments_in_half_row_plate[i,j,h]) == experiments_in_plate_row[i,j]);
constraint forall(i in Plates, k in Columns) ( sum(v in Vertical)(experiments_in_half_column_plate[i,k,v]) == experiments_in_plate_column[i,k]);

constraint forall(i in Plates, j in Rows, h in Horizontal) ( experiments_in_half_row_plate[i,j,h] >= (experiments_in_plate_row[i,j] div 2)); %%WARNING
constraint forall(i in Plates, j in Rows, h in Horizontal) ( experiments_in_half_row_plate[i,j,h] <= ((experiments_in_plate_row[i,j] + 1) div 2)); %%WARNING

constraint forall(i in Plates, k in Columns, v in Vertical) (experiments_in_half_column_plate[i,k,v] >= (experiments_in_plate_column[i,k] div 2)); %%WARNING
constraint forall(i in Plates, k in Columns, v in Vertical) (experiments_in_half_column_plate[i,k,v] <= ((experiments_in_plate_column[i,k] + 1) div 2)); %%WARNING

constraint forall(i in Plates, h in Horizontal, j in Rows) ( (numcols div 2) == controls_in_half_row_plate[i,j,h] + experiments_in_half_row_plate[i,j,h]);
constraint forall(i in Plates, v in Vertical, k in Columns) ( (numrows div 2) == controls_in_half_column_plate[i,k,v] + experiments_in_half_column_plate[i,k,v]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Balancing each type of control per row

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (count_eq(plates[i,j,..],experiments+c,controls_row_plates[i,j,c]));

constraint forall(i in Plates, j in Rows) (count_eq(plates[i,j,..],0,controls_row_plates[i,j,0]));

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (controls_row_plates[i,j,c] <= ceil(controls_per_plate[i,c]/numrows));

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (controls_row_plates[i,j,c] >= (controls_per_plate[i,c] div numrows));

constraint forall(i in Plates, j in Rows) (sum(controls_row_plates[i,j,..]) + experiments_in_plate_row[i,j] == numcols);

constraint sum(controls_row_plates[..,..,0]) == emptywells;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Balancing each type of control per column

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (count_eq(plates[i,..,k],experiments+c,controls_column_plates[i,k,c]));

constraint forall(i in Plates, k in Columns) (count_eq(plates[i,..,k],0,controls_column_plates[i,k,0]));

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (controls_column_plates[i,k,c] <= ceil(controls_per_plate[i,c]/numcols));

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (controls_column_plates[i,k,c] >= (controls_per_plate[i,c] div numcols));

constraint forall(i in Plates, k in Columns) (sum(controls_column_plates[i,k,..]) + experiments_in_plate_column[i,k] == numrows);

constraint sum(controls_column_plates[..,..,0]) == emptywells;

%% Implied all_different constraint: all experiments/compounds are different
constraint alldifferent_except(plates, {0} union experiments+1..experiments+num_controls*max_control_concentrations):: domain;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Distributing controls

%% Controls of the same type are not located next to each other

%% Implied constraint
constraint if (min(spread_control++[1])>0) then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1) (alldifferent_except_0([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]])):: domain endif;

array [Plates,Rows,Columns] of var 0..1: controls_layout; 

constraint forall(i in Plates, j in Rows, k in Columns)(controls_layout[i,j,k] == ((plates[i,j,k] > experiments)) );


constraint if spread_controls then forall(i in Plates, j in Rows)(regular(controls_layout[i,j,..], "(0|(1 0))* (1?)")::domain) endif;

constraint if spread_controls then forall(i in Plates, k in Columns)(regular(controls_layout[i,..,k], "(0|(1 0))* (1?)")::domain) endif;

constraint if force_spread_controls then forall(i in Plates, j in Rows)(regular(controls_layout[i,j,..], "(0|(1 0 0))* (1|(1 0))?")::domain) endif;

constraint if force_spread_controls then forall(i in Plates, k in Columns)(regular(controls_layout[i,..,k], "(0|(1 0 0))* (1|(1 0))?")::domain) endif;


constraint if spread_controls then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_plate_row[i,j],controls_layout[i,j,..])::domain) endif;

constraint if spread_controls /\ numcols > 2  then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_half_row_plate[i,j,left],controls_layout[i,j,1..(numcols div 2)])::domain) endif;

constraint if spread_controls /\ numcols > 2  then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_half_row_plate[i,j,right],[controls_layout[i,j,k] | k in ((numcols div 2)+1)..numcols])::domain) endif;

constraint if spread_controls /\ numcols > 2 then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_half_row_plate[i,j,right],reverse([controls_layout[i,j,k] | k in ((numcols div 2)+1)..numcols]))::domain) endif;

constraint if spread_controls then forall(i in Plates, k in Columns)(my_implied_cost(controls_only_in_plate_column[i,k],controls_layout[i,..,k])::domain) endif;

constraint if spread_controls /\ numrows > 2 then forall(i in Plates, k in Columns)(my_implied_cost(controls_only_in_half_column_plate[i,k,upper],[controls_layout[i,j,k] | j in 1..(numrows div 2)])::domain) endif; 

constraint if spread_controls /\ numrows > 2 then forall(i in Plates, k in Columns)(my_implied_cost(controls_only_in_half_column_plate[i,k,lower],[controls_layout[i,j,k] | j in (numrows div 2)+1..numrows])::domain) endif;

constraint if force_spread_controls then forall(i in Plates, j in Rows)(my_implied_cost_ext(controls_only_in_plate_row[i,j],controls_layout[i,j,..])::domain) endif;

constraint if force_spread_controls then forall(i in Plates, k in Columns)(my_implied_cost_ext(controls_only_in_plate_column[i,k],controls_layout[i,..,k])::domain) endif;


constraint forall(i in Plates) (sum_controls_ul_plates[i,upper] == sum(controls_layout[i,1..(numrows div 2),..]));
constraint forall(i in Plates) (sum_controls_ul_plates[i,lower] == sum(controls_layout[i,(numrows div 2)+1..numrows,..]));

constraint forall(i in Plates) (sum_controls_lr_plates[i,left] == sum(controls_layout[i,..,1..(numcols div 2)]));
constraint forall(i in Plates) (sum_controls_lr_plates[i,right] == sum(controls_layout[i,..,(numcols div 2)+1..numcols]));



constraint forall(i in Plates, j in Rows) (global_cardinality(controls_layout[i,j,..],
                             [0,1],
                             [numcols-controls_only_in_plate_row[i,j],controls_only_in_plate_row[i,j]]));

constraint forall(i in Plates, k in Columns) (global_cardinality(controls_layout[i,..,k],
                             [0,1],
                             [numrows-controls_only_in_plate_column[i,k],controls_only_in_plate_column[i,k]]));

                          

constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] == experiments+ctr -> 
            plates[i,j-1,k-1]!= plates[i,j,k] /\ 
            plates[i,j-1,k]  != plates[i,j,k] /\
            plates[i,j-1,k+1]!= plates[i,j,k] /\
            plates[i,j,k-1]  != plates[i,j,k] /\
            plates[i,j,k+1]  != plates[i,j,k] /\
            plates[i,j+1,k-1]!= plates[i,j,k] /\
            plates[i,j+1,k]  != plates[i,j,k] /\
            plates[i,j+1,k+1]!= plates[i,j,k]
            ) 
    else forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] == experiments+ctr -> 
            plates[i,j-1,k]  != plates[i,j,k] /\
            plates[i,j,k-1]  != plates[i,j,k] /\
            plates[i,j,k+1]  != plates[i,j,k] /\
            plates[i,j+1,k]  != plates[i,j,k]
            )
            endif );

%% Upper row j=1
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, k in 2..numcols-1)(plates[i,1,k] == experiments+ctr -> 
            plates[i,1,k-1]  != plates[i,1,k] /\
            plates[i,1,k+1]  != plates[i,1,k] /\
            plates[i,2,k-1]!= plates[i,1,k] /\
            plates[i,2,k]  != plates[i,1,k] /\
            plates[i,2,k+1]!= plates[i,1,k]
            ) 
    else forall(i in Plates, k in 2..numcols-1)(plates[i,1,k] == experiments+ctr -> 
            plates[i,1,k-1] != plates[i,1,k] /\
            plates[i,1,k+1] != plates[i,1,k] /\
            plates[i,2,k]   != plates[i,1,k]
            )
            endif );

%% Lower row j = numrows
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, k in 2..numcols-1)(plates[i,numrows,k] == experiments+ctr -> 
            plates[i,numrows-1,k-1]!= plates[i,numrows,k] /\ 
            plates[i,numrows-1,k]  != plates[i,numrows,k] /\
            plates[i,numrows-1,k+1]!= plates[i,numrows,k] /\
            plates[i,numrows,k-1]  != plates[i,numrows,k] /\
            plates[i,numrows,k+1]  != plates[i,numrows,k]
            ) 
    else forall(i in Plates, k in 2..numcols-1)(plates[i,numrows,k] == experiments+ctr -> 
            plates[i,numrows-1,k]  != plates[i,numrows,k] /\
            plates[i,numrows,k-1]  != plates[i,numrows,k] /\
            plates[i,numrows,k+1]  != plates[i,numrows,k]
            )
            endif );         
            
%% Left column k=1 
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, j in 2..numrows-1)(plates[i,j,1] == experiments+ctr -> 
            plates[i,j-1,1]  != plates[i,j,1] /\
            plates[i,j-1,2]!= plates[i,j,1] /\
            plates[i,j,2]  != plates[i,j,1] /\
            plates[i,j+1,1]  != plates[i,j,1] /\
            plates[i,j+1,2]!= plates[i,j,1]
            ) 
    else forall(i in Plates, j in 2..numrows-1)(plates[i,j,1] == experiments+ctr -> 
            plates[i,j-1,1] != plates[i,j,1] /\
            plates[i,j,2]   != plates[i,j,1] /\
            plates[i,j+1,1] != plates[i,j,1]
            )
            endif );                                                   
                                        
%% Right column k=numcols
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, j in 2..numrows-1)(plates[i,j,numcols] == experiments+ctr -> 
            plates[i,j-1,numcols-1]!= plates[i,j,numcols] /\ 
            plates[i,j-1,numcols]  != plates[i,j,numcols] /\
            plates[i,j,numcols-1]  != plates[i,j,numcols] /\
            plates[i,j+1,numcols-1]!= plates[i,j,numcols] /\
            plates[i,j+1,numcols]  != plates[i,j,numcols]
            ) 
    else forall(i in Plates, j in 2..numrows-1)(plates[i,j,numcols] == experiments+ctr -> 
            plates[i,j-1,numcols]  != plates[i,j,numcols] /\
            plates[i,j,numcols-1]  != plates[i,j,numcols] /\
            plates[i,j+1,numcols]  != plates[i,j,numcols]
            )
            endif );                                                                                                            
                                                                    
%% Upper left corner j=1 k=1
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates)(plates[i,1,1] == experiments+ctr -> 
            plates[i,1,2] != plates[i,1,1] /\
            plates[i,2,1] != plates[i,1,1] /\
            plates[i,2,2] != plates[i,1,1]
            ) 
    else forall(i in Plates)(plates[i,1,1] == experiments+ctr -> 
            plates[i,1,2] != plates[i,1,1] /\
            plates[i,2,1]   != plates[i,1,1]
            )
            endif );

                                                                                                            
%% Upper right corner j=1 k=numcols
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates)(plates[i,1,numcols] == experiments+ctr -> 
            plates[i,1,numcols-1] != plates[i,1,numcols] /\
            plates[i,2,numcols-1] != plates[i,1,numcols] /\
            plates[i,2,numcols]   != plates[i,1,numcols]
            ) 
    else forall(i in Plates)(plates[i,1,numcols] == experiments+ctr -> 
            plates[i,1,numcols-1] != plates[i,1,numcols] /\
            plates[i,2,numcols]   != plates[i,1,numcols]
            )
            endif );                                                                                                                                                                                                                                                                                                                                    
            

%% Lower left corner j = numrows k=1
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates)(plates[i,numrows,1] == experiments+ctr -> 
            plates[i,numrows-1,1]  != plates[i,numrows,1] /\
            plates[i,numrows-1,2]!= plates[i,numrows,1] /\
            plates[i,numrows,2]  != plates[i,numrows,1]
            ) 
    else forall(i in Plates)(plates[i,numrows,1] == experiments+ctr -> 
            plates[i,numrows-1,1]  != plates[i,numrows,1] /\
            plates[i,numrows,2]  != plates[i,numrows,1]
            )
            endif );     
            
%% Lower right corner j = numrows k = numcols
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, k in 2..numcols-1)(plates[i,numrows,numcols] == experiments+ctr -> 
            plates[i,numrows-1,numcols-1]!= plates[i,numrows,numcols] /\ 
            plates[i,numrows-1,numcols]  != plates[i,numrows,numcols] /\
            plates[i,numrows,numcols-1]  != plates[i,numrows,numcols]
            ) 
    else forall(i in Plates)(plates[i,numrows,numcols] == experiments+ctr -> 
            plates[i,numrows-1,numcols]  != plates[i,numrows,numcols] /\
            plates[i,numrows,numcols-1]  != plates[i,numrows,numcols]
            )
            endif );   
            
            
constraint if spread_controls then forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] > experiments -> 
            %plates[i,j-1,k-1] <= experiments /\ 
            plates[i,j-1,k]   <= experiments /\
            %plates[i,j-1,k+1] <= experiments /\
            plates[i,j,k-1]   <= experiments /\
            plates[i,j,k+1]   <= experiments /\
            %plates[i,j+1,k-1] <= experiments /\
            plates[i,j+1,k]   <= experiments %/\
            %plates[i,j+1,k+1] <= experiments
            ) endif;

constraint if spread_controls then forall(i in Plates, j in 2..numrows-1)(plates[i,j,1] > experiments -> 
            plates[i,j-1,1]   <= experiments /\
            plates[i,j,2]   <= experiments /\
            plates[i,j+1,1]   <= experiments
            ) endif;

constraint if spread_controls then forall(i in Plates, j in 2..numrows-1)(plates[i,j,numcols] > experiments -> 
            plates[i,j-1,numcols]   <= experiments /\
            plates[i,j,numcols-1]   <= experiments /\
            plates[i,j+1,numcols]   <= experiments
            ) endif;


constraint if spread_controls then forall(i in Plates, k in 2..numcols-1)(plates[i,1,k] > experiments -> 
            plates[i,1,k-1]   <= experiments /\
            plates[i,1,k+1]   <= experiments /\
            plates[i,2,k]   <= experiments
            ) endif;               

constraint if spread_controls then forall(i in Plates, k in 2..numcols-1)(plates[i,numrows,k] > experiments -> 
            plates[i,numrows-1,k]   <= experiments /\
            plates[i,numrows,k-1]   <= experiments /\
            plates[i,numrows,k+1]   <= experiments
            ) endif;               


constraint if spread_controls then forall(i in Plates)(plates[i,1,1] > experiments -> 
            plates[i,1,2]   <= experiments /\
            plates[i,2,1]   <= experiments
            ) endif;

constraint if spread_controls then forall(i in Plates)(plates[i,1,numcols] > experiments -> 
            plates[i,1,numcols-1]   <= experiments /\
            plates[i,2,numcols]   <= experiments
            ) endif;

constraint if spread_controls then forall(i in Plates)(plates[i,numrows,1] > experiments -> 
            plates[i,numrows-1,1]   <= experiments /\
            plates[i,numrows,2]   <= experiments
            ) endif;

constraint if spread_controls then forall(i in Plates)(plates[i,numrows,numcols] > experiments -> 
            plates[i,numrows-1,numcols]   <= experiments /\
            plates[i,numrows,numcols-1]   <= experiments
            ) endif;                                                                                                                                                                                                                                                                                                                                                            
%% TODO: Add spread_controls contraint in borders!


% j=1
constraint forall(i in Plates, k in 2..numcols-1)(plates[i,1,k] > experiments -> 
           plates[i,1,k-1]  != plates[i,1,k] /\
           plates[i,1,k+1]  != plates[i,1,k] /\
           plates[i,2,k-1]!= plates[i,1,k] /\
           plates[i,2,k]  != plates[i,1,k] /\
           plates[i,2,k+1]!= plates[i,1,k]
           );

% j = numrows
constraint forall(i in Plates, k in 2..numcols-1)(plates[i,numrows,k] > experiments -> 
           plates[i,numrows-1,k-1]!= plates[i,numrows,k] /\ 
           plates[i,numrows-1,k]  != plates[i,numrows,k] /\
           plates[i,numrows-1,k+1]!= plates[i,numrows,k] /\
           plates[i,numrows,k-1]  != plates[i,numrows,k] /\
           plates[i,numrows,k+1]  != plates[i,numrows,k]
           );
           
% First column: k = 1           
constraint forall(i in Plates, j in 2..numrows-1)(plates[i,j,1] > experiments -> 
           plates[i,j-1,1]  != plates[i,j,1] /\
           plates[i,j-1,2]!= plates[i,j,1] /\
           plates[i,j,2]  != plates[i,j,1] /\
           plates[i,j+1,1]  != plates[i,j,1] /\
           plates[i,j+1,2]!= plates[i,j,1]
           );

% Last column: k = numcols
constraint forall(i in Plates, j in 2..numrows-1)(plates[i,j,numcols] > experiments -> 
           plates[i,j-1,numcols-1]!= plates[i,j,numcols] /\ 
           plates[i,j-1,numcols]  != plates[i,j,numcols] /\
           plates[i,j,numcols-1]  != plates[i,j,numcols] /\
           plates[i,j+1,numcols-1]!= plates[i,j,numcols] /\
           plates[i,j+1,numcols]  != plates[i,j,numcols]
           );            
  
    
%% All especial cases for the corners are now covered by the alldifferent_except_0 constraints

                                               
constraint if force_spread_controls then (forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] > experiments -> 
           plates[i,j-1,k]  <= experiments /\
           plates[i,j,k-1]  <= experiments /\
           plates[i,j,k+1]  <= experiments /\
           plates[i,j+1,k]  <= experiments
           ) 
           ) endif ;
           
constraint if force_spread_controls then forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] > experiments -> 
            plates[i,j-1,k-1] <= experiments /\ 
            plates[i,j-1,k]  <= experiments /\
            plates[i,j-1,k+1] <= experiments /\
            plates[i,j,k-1]  <= experiments /\
            plates[i,j,k+1]  <= experiments /\
            plates[i,j+1,k-1] <= experiments /\
            plates[i,j+1,k]  <= experiments /\
            plates[i,j+1,k+1] <= experiments 
            ) endif;
                                               
constraint if force_spread_controls then forall(i in Plates, j in 3..numrows-2, k in 3..numcols-2)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-2] <= experiments /\ 
            plates[i,j-2,k-1]  <= experiments /\
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-2,k+1]  <= experiments /\
            plates[i,j-2,k+2] <= experiments /\
            plates[i,j-1,k-2]  <= experiments /\
            plates[i,j-1,k+2]  <= experiments /\
            plates[i,j,k-2]  <= experiments /\
            plates[i,j,k+2]  <= experiments /\
            plates[i,j+1,k-2]  <= experiments /\
            plates[i,j+1,k+2]  <= experiments /\
            plates[i,j+2,k-2] <= experiments /\
            plates[i,j+2,k-1]  <= experiments /\
            plates[i,j+2,k]  <= experiments /\
            plates[i,j+2,k+1]  <= experiments /\
            plates[i,j+2,k+2] <= experiments
            ) endif;
                      
            
constraint if force_spread_controls then forall(i in Plates, j = 2, k in 3..numcols-2)(plates[i,j,k] > experiments -> 
            plates[i,j,k-2]  <= experiments /\
            plates[i,j,k+2]  <= experiments /\
            plates[i,j-1,k-2]  <= experiments /\
            plates[i,j-1,k+2]  <= experiments /\
            plates[i,j+1,k-2]  <= experiments /\
            plates[i,j+1,k+2]  <= experiments /\
            plates[i,j+2,k-2] <= experiments /\
            plates[i,j+2,k-1]  <= experiments /\
            plates[i,j+2,k]  <= experiments /\
            plates[i,j+2,k+1]  <= experiments /\
            plates[i,j+2,k+2] <= experiments
            ) endif;
                      

constraint if force_spread_controls then forall(i in Plates, j = 1, k in 3..numcols-2)(plates[i,j,k] > experiments -> 
            plates[i,j,k-2]  <= experiments /\
            plates[i,j,k+2]  <= experiments /\
            plates[i,j+1,k-2]  <= experiments /\
            plates[i,j+1,k+2]  <= experiments /\
            plates[i,j+2,k-2] <= experiments /\
            plates[i,j+2,k-1]  <= experiments /\
            plates[i,j+2,k]  <= experiments /\
            plates[i,j+2,k+1]  <= experiments /\
            plates[i,j+2,k+2] <= experiments
            ) endif;
                        
                                               
constraint if force_spread_controls then forall(i in Plates, j = numrows-1, k in 3..numcols-2)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-2] <= experiments /\ 
            plates[i,j-2,k-1]  <= experiments /\
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-2,k+1]  <= experiments /\
            plates[i,j-2,k+2] <= experiments /\
            plates[i,j,k-2]  <= experiments /\
            plates[i,j,k+2]  <= experiments /\
            plates[i,j+1,k-2]  <= experiments /\
            plates[i,j+1,k+2]  <= experiments
            ) endif;    
            
constraint if force_spread_controls then forall(i in Plates, j = numrows, k in 3..numcols-2)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-2] <= experiments /\ 
            plates[i,j-2,k-1]  <= experiments /\
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-2,k+1]  <= experiments /\
            plates[i,j-2,k+2] <= experiments /\
            plates[i,j,k-2]  <= experiments /\
            plates[i,j,k+2]  <= experiments
            ) endif;             
                    

%% Border cases for k: k=1
constraint if force_spread_controls then forall(i in Plates, j in 3..numrows-2, k=1)(plates[i,j,1] > experiments -> 
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-2,k+1]  <= experiments /\
            plates[i,j-2,k+2] <= experiments /\
            plates[i,j-1,k+2]  <= experiments /\
            plates[i,j,k+2]  <= experiments /\
            plates[i,j+1,k+2]  <= experiments /\
            plates[i,j+2,k]  <= experiments /\
            plates[i,j+2,k+1]  <= experiments /\
            plates[i,j+2,k+2] <= experiments
            ) endif;                                                                                                                                                                                                                                                                                    

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
%% Border cases for k                                                                                                                                          
constraint if force_spread_controls then forall(i in Plates, j in 3..numrows-2, k = 2)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-1]  <= experiments /\
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-2,k+1]  <= experiments /\
            plates[i,j-2,k+2] <= experiments /\
            plates[i,j-1,k+2]  <= experiments /\
            plates[i,j,k+2]  <= experiments /\
            plates[i,j+1,k+2]  <= experiments /\
            plates[i,j+2,k-1]  <= experiments /\
            plates[i,j+2,k]  <= experiments /\
            plates[i,j+2,k+1]  <= experiments /\
            plates[i,j+2,k+2] <= experiments
            ) endif;                                                                                                                                                                                                                                                                                    
         
%% Border cases for k                                                                                                                                          
constraint if force_spread_controls then forall(i in Plates, j in 3..numrows-2, k = numcols-1)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-2] <= experiments /\ 
            plates[i,j-2,k-1]  <= experiments /\
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-2,k+1]  <= experiments /\
            plates[i,j-1,k-2]  <= experiments /\
            plates[i,j,k-2]  <= experiments /\
            plates[i,j+1,k-2]  <= experiments /\
            plates[i,j+2,k-2] <= experiments /\
            plates[i,j+2,k-1]  <= experiments /\
            plates[i,j+2,k]  <= experiments /\
            plates[i,j+2,k+1]  <= experiments
            ) endif;                                                                                                                                                                                                                                                                                    
            
%% Border cases for k                                                                                                                                          
constraint if force_spread_controls then forall(i in Plates, j in 3..numrows-2, k = numcols)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-2] <= experiments /\ 
            plates[i,j-2,k-1]  <= experiments /\
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-1,k-2]  <= experiments /\
            plates[i,j,k-2]  <= experiments /\
            plates[i,j+1,k-2]  <= experiments /\
            plates[i,j+2,k-2] <= experiments /\
            plates[i,j+2,k-1]  <= experiments /\
            plates[i,j+2,k]  <= experiments
            ) endif;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            
                        
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




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Pretty printing %%%

output [if testing \/ debugging then "\(numplates) plates\n" else "" endif];

output [if debugging /\ (emptywells>0) then "warning: there are \(emptywells) empty wells. Consider using them as controls.\n\n" else "" endif];

output [if debugging then "Spreading controls... \(spread_control) ... \(min(spread_control++[1])>0)\n" else "" endif];

output [if debugging then "Force spread controls... \(force_spread_controls)\n" else "" endif];

output [if debugging then "Random value... \(normal(0,1))\n" else "" endif];

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

output [if debugging then "\nSpread controls:\n" else "" endif, if debugging then show(spread_controls) else "" endif];

output [if debugging then "\nControls' layout:\n" else "" endif, if debugging then show(controls_layout) else "" endif];

output [if debugging then "\nExperiment's row:\n" else "" endif, if debugging then show(experiment_row) else "" endif];

output [if debugging then "\nexperiment_plate_row:\n" else "" endif, if debugging then show(experiment_plate_row) else "" endif];

output [if debugging then "\nExperiments in row (per plate):\n" else "" endif, if debugging then show(experiments_in_plate_row) else "" endif];

output [if debugging then "\nExperiments in column (per plate):\n" else "" endif, if debugging then show(experiments_in_plate_column) else "" endif];

output [if debugging then "\nControls per row:\n" else "" endif, if debugging then show(controls_in_row) else "" endif];

output [if debugging then "\nControls per row per plate:\n" else "" endif, if debugging then show(controls_in_plate_row) else "" endif];

output [if debugging then "\nControls per column:\n" else "" endif, if debugging then show(controls_in_column) else "" endif];

output [if debugging then "\nControls per half row:\n" else "" endif, if debugging then show(controls_in_half_row) else "" endif];

output [if debugging then "\nControls per upper/lower half plates:\n" else "" endif, if debugging then show(controls_ul_plates) else "" endif];

output [if debugging then "\nControls per left/right half plates:\n" else "" endif, if debugging then show(controls_lr_plates) else "" endif];

output [if debugging then "\nControls per quarter plates (controls_quarter_plates):\n" else "" endif, if debugging then show(controls_quarter_plates) else "" endif];

output [if debugging then "\nTotal Compounds:\n" else "" endif, if debugging then show(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) else "" endif];

output [if debugging then "\nCompounds per half plates:\n" else "" endif, if debugging then show(ul_half_plates) else "" endif];

output [if debugging then "\nConcentrations per half plates:\n" else "" endif, if debugging then show(ul_concentrations) else "" endif];

output [if debugging then "\nConcentrations upper half plates:\n" else "" endif, if debugging then show(sum(ul_concentrations[..,upper])) else "" endif];

output [if debugging then "\nConcentrations lower half plates:\n" else "" endif, if debugging then show(sum(ul_concentrations[..,lower])) else "" endif];

output [if debugging then "\nCompounds per left-right half plates:\n" else "" endif, if debugging then show(lr_half_plates) else "" endif];

output [if debugging then "\nControls per plate per row:\n" else "" endif, if debugging then show(controls_row_plates) else "" endif];

output [if debugging then "\nControls per plate per column:\n" else "" endif, if debugging then show(controls_column_plates) else "" endif];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CSV Ouput %%%

array[int] of string: letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
         
output ["plateID,well,cmpdname,CONCuM,cmpdnum,VOLuL\n"];

%%% TODO: FIX ME!!!!!! Problem when there are vertical/horizontal cell lines!
output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[size_empty_edge+j+(numrows+(1+inner_empty_edge)*size_empty_edge)*(h-1)] ++ if (size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1)) < 10 then "0" else "" endif ++ "\(size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1))," ++ 

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

++ "\n" else "" endif|
        i in Plates, j in Rows, k in Columns, v in 1..vertical_cell_lines, h in 1..horizontal_cell_lines];       
         

       