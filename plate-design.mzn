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
% Description:  A microplate layout designer
%
% Authors: Maria Andreina FRANCISCO RODRIGUEZ (maria.andreina.francisco@farmbio.uu.se)
% Version: 1.0
% Last Revision: August 2021
%

include "globals.mzn";
include "gecode.mzn";
%include "chuffed.mzn";
 

%%%% Input Data  %%%%

%% Information about constraints %%
int: horizontal_cell_lines;
int: vertical_cell_lines;
bool: allow_empty_wells;
bool: replicates_on_different_plates;
bool: replicates_on_same_plate;
bool: concentrations_on_different_rows;
bool: concentrations_on_different_columns;
bool: force_spread_controls;

int: size_empty_edge;

%% Compounds %%
int: compounds; %% number of drugs/compounds
array [1..compounds] of int: compound_replicates;
int: replicates = max(compound_replicates++[0]);
array [1..compounds] of int: compound_concentrations;
int: max_compound_concentrations = max(compound_concentrations++[0]);

array[1..compounds] of string: compound_names;

%% Combinations %%
%% Deprecated
int: combinations;
int: combination_concentrations;
array[1..combinations] of string: combination_names;


%% Information about controls %%
int: num_controls;

array [1..num_controls] of int: control_replicates; 
array [1..num_controls] of int: control_concentrations;
int: max_control_concentrations = max(control_concentrations++[0]);
array[1..num_controls,1..max_control_concentrations] of string: control_concentration_names;
array[1..num_controls] of string: control_names;

int: total_controls = sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]);

%%% Potentially new parameters
bool: balance_controls_inside_plate = true;
bool: interconnected_plates = (numplates==1); %Makes the problem much harder by connecting the plates
int: control_slack = 1; %Makes the problem harder/easier by constraining how equally should we distribute controls
bool: spread_controls = (ceil(inner_plate_size/2)*numplates >= total_controls); %Makes the problem harder when there are many controls

%% Which individual controls can be spread more than others?
array[1..num_controls*max_control_concentrations] of bool: spread_control = [ ceil(inner_plate_size/4)*numplates >= (control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])) | i in 1..num_controls*max_control_concentrations];


%%% Testing %%%

opt bool: testing;
bool: debugging = false;


%% Plate size / number of wells
int: num_rows;
int: num_cols;

%% TODO: this could be problematic when there are multiple cell lines
int: numrows = floor(num_rows/horizontal_cell_lines)-2*size_empty_edge;
int: numcols = floor(num_cols/vertical_cell_lines)-2*size_empty_edge;


%% TODO: add latex indicators for combinations and controls 
%array[1..compound_concentrations] of string: compound_concentration_names;

array[1..compounds,int] of string: compound_concentration_names;


%% FIX ME!
array[1..max_compound_concentrations] of string: compound_concentration_indicators;
array[1..combination_concentrations] of string: combination_concentration_names;



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



%% Number of wells needed. Note that plates might not be full
int: total_wells = sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + total_controls; 

constraint assert(total_wells > 0, "Invalid datafile: the plates cannot be completely empty.");


int: inner_plate_size = numcols*numrows;

constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, "Invalid datafile: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");

%% Number of plates needed
%% max is used to avoid division-by-zero errors
int: numplates = max(ceil(total_wells/inner_plate_size),1);         
              
force_spread_controls = (ceil(inner_plate_size/9)*numplates >= total_controls) ;
%force_spread_controls = false;

%% Detecting some unsatisfiable cases.
constraint assert(ceil(sum(compound_replicates)/numplates)*min(compound_concentrations++[infinity]) + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E01)");  
                                                                
constraint assert((floor(sum(compound_replicates)/numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E02)");  

                            
                                                                                   
set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: Plates = 1..numplates;


%% All types of experiments (excluding controls): compounds and combinations
int: experiments = compounds*max_compound_concentrations*replicates;  

%% Used in redundant constraints
int: emptywells = numplates*inner_plate_size - total_wells;   


constraint assert(emptywells == 0 \/ allow_empty_wells,"Warning: There are \(emptywells) empty wells. You can allow the plates to have inner empty wells or modify other quantities to fill them up.");

constraint assert(emptywells >= 0,"Model ERROR! Inner empty wells is negative. This should never happen!");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plates (our solution!!) %%%%%%%%%%%
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
%% Edge effect: Leave upper and lower rows of every plate empty
%% Not needed anymore. Modelled by changing the output
%constraint forall(b in 0..size_empty_edge-1) (forall(i in Plates, j in {1+b,numrows-b}, k in Columns)(plates[i,j,k] = 0));


%% Edge effect: Leave left-most and right-most columns of every plate empty
%constraint forall(b in 0..size_empty_edge-1) (forall(i in Plates, j in Rows, k in {1+b,numcols-b})(plates[i,j,k] = 0));


%% Exactly the total number of empty wells (zeros) and controls (experiments+i)
% merged with another constraint. See below.
%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[0]++[experiments+i | i in 1..num_controls*control_concentrations],[emptywells]++[control_replicates[((i-1) mod num_controls)+1] | i in 1..num_controls*control_concentrations]);

%% Exactly the number of replicates for each compound*compound_concentrations and combination*combination_concentrations
%% merged with another constraint. See below.
%% Exactly one (right now).
%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates), [ i | i in 1..experiments] , [ 1 | i in 1..experiments]);



constraint global_cardinality(plates,[0]++[ i | i in 1..experiments]++[i | i in experiments+1..experiments]++[ experiments+i | i in 1..num_controls*max_control_concentrations],[emptywells]++[(floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1] | i in 1..experiments]++[1|i in experiments+1..experiments]++[control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..num_controls*max_control_concentrations]);

%% Experiments that do not exists are not located in any plate, row, or column.
constraint forall(i in 1..experiments)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) == false <-> (experiment_plate[i] = 0 /\ experiment_row[i] = 0 /\ experiment_column[i] = 0));

constraint forall(i in 1..experiments)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) <-> (experiment_plate[i] != 0 /\ experiment_row[i] != 0 /\ experiment_column[i] != 0));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channelling constraints: an experiment/compound is located at a given plate

constraint forall(l in 1..experiments, i in Plates)(count_eq([plates[i,j,k] | j in Rows, k in Columns],l,(experiment_plate[l] == i)));

constraint forall(l in 1..experiments)(count_eq([plates[i,j,k] | i in Plates, j in Rows, k in Columns],l,(experiment_plate[l] != 0)));


%% For experiments that do not exist = plate 0 (redundant)
%constraint forall(l in 1..experiments)(
 % count_eq(array1d(1..numplates*inner_plate_size, plates),l,0) <-> (experiment_plate[l] = 0 /\ experiment_row[l] = 0 /\ experiment_column[l] = 0)
%);

%constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*inner_plate_size, plates),l,0) <-> (experiment_row[l] == 0));
%constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*inner_plate_size, plates),l,0) <-> (experiment_column[l] == 0));



%% There might be a stronger version of these constraints...
%constraint forall(l in 1..experiments, i in Plates, j in Rows, k in Columns)((experiment_row[l] == j) <- (plates[i,j,k] == l));

constraint forall(l in 1..experiments, i in Plates)(count_eq([plates[i,j,k] | j in Rows, k in Columns],l,(experiment_plate[l] == i)));

%constraint forall(i in Plates, j in Rows, k in Columns)(plates[i,j,k] in 1..experiments -> experiment_row[plates[i,j,k]] = j);

%constraint forall(l in 1..experiments, i in Plates, j in Rows, k in Columns)((experiment_column[l] == k) <- (plates[i,j,k] == l));

%constraint forall(i in Plates, j in Rows, k in Columns)((experiment_column[plates[i,j,k]] = k) <- (plates[i,j,k] in 1..experiments));

constraint forall(l in 1..experiments, j in Rows)(count_eq([plates[i,j,k] | i in Plates, k in Columns],l,(experiment_row[l] == j)));

%constraint forall(l in 1..experiments, j in Rows)(experiment_row[l] != j <-> count_eq([plates[experiment_plate[l],j,k] | k in Columns],l,0));

%constraint forall(l in 1..experiments, k in Columns)(experiment_column[l] == k <-> count_eq([plates[experiment_plate[l],j,k] | j in Rows],l,1));

%constraint forall(l in 1..experiments, k in Columns)(count_eq([plates[experiment_plate[l],j,k] | j in Rows],l,(experiment_column[l] == k)));

constraint forall(l in 1..experiments, k in Columns)(count_eq([plates[i,j,k] | i in Plates, j in Rows],l,(experiment_column[l] == k)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A compound with all concentrations must appear on a single plate

constraint forall(l in 1..experiments where ((l mod max_compound_concentrations) == 1))(all_equal([experiment_plate[i] | i in l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)]));

%% A combination with all concentrations must appear on a single plate
%constraint forall(l in (experiments+1)..experiments where ((l-experiments) mod combination_concentrations == 1))(all_equal([experiment_plate[i] | i in l..(l+combination_concentrations-1)]));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Spreading concentrations of a compound across different rows and columns 
%%% (consider balancing the number of rows and columns!)
% Consider making it different on different plates



% Might be redundant != implied
%constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_plate[plates[i,j,k]]==i);
%constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_row[plates[i,j,k]]==j);
%constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_column[plates[i,j,k]]==k);


% Spread the compounds across different rows and columns

%% Spreads compounds over different rows. It's been replaced by the more restrictive alldifferent version below.
%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],numrows-2*size_empty_edge), [experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
alldifferent([experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numrows)]):: domain
))) endif; 



% Split concentrations between upper and lower halves of the plate

array [1..compounds*replicates,Vertical] of var 0..max_compound_concentrations: ul_concentrations;

%% For each replicate of each compound, count how many concentrations are in the upper/lower parts of the plate
%% Definition of ul_concentrations (upper)  

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,upper],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..floor(numrows/2))
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,upper] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,upper] >= floor(compound_concentrations[l]/2)
))) endif; 


%% Definition of ul_concentrations (lower)
constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], floor(numrows/2)+1..numrows)
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,lower] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,lower] >= floor(compound_concentrations[l]/2)
))) endif; 


%% Implied constraints:
constraint if concentrations_on_different_rows then (sum(ul_concentrations)=sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) endif;

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,upper] + ul_concentrations[r*compounds + l,lower] == compound_concentrations[l]
))) endif; 

constraint if concentrations_on_different_rows then (sum(ul_concentrations[..,lower]) == sum(ul_half_plates[..,lower]) ) endif;

constraint if concentrations_on_different_rows then (sum(ul_concentrations[..,upper]) == sum(ul_half_plates[..,upper]) ) endif;

constraint if concentrations_on_different_rows then forall(i in Plates) (sum(c in 0..compounds*replicates-1)(ul_concentrations[c+1,lower]*(experiment_plate[c*max_compound_concentrations+1]==i)) == ul_half_plates[i,lower] ) endif;

constraint if concentrations_on_different_rows then forall(i in Plates) (sum(c in 0..compounds*replicates-1)(ul_concentrations[c+1,upper]*(experiment_plate[c*max_compound_concentrations+1]==i)) == ul_half_plates[i,upper] ) endif;


%% Defining ul_concentrations in terms of plates
constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,upper], plates[..,1..floor(numrows/2),..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]}, )
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower], plates[..,floor(numrows/2)+1..numrows,..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]}, )
))) endif; 


array [Plates,Rows] of var 0..numplates*numcols: experiments_in_plate_row;

%% Definition of experiments_in_plate_row
constraint forall(i in Plates, j in Rows)(count(experiment_plate_row[i,..],j,experiments_in_plate_row[i,j]));

constraint forall(i in Plates, j in Rows)(among(experiments_in_plate_row[i,j],plates[i,j,..],1..experiments));

constraint forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j] + controls_in_plate_row[i,j] == numcols);

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
constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,left],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..floor(numcols/2))
))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,left] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,left] >= floor(compound_concentrations[l]/2)
))) endif; 


%% Definition of lr_concentrations (right)  
constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,right],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], floor(numcols/2)+1..numcols)
))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,right] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,right] >= floor(compound_concentrations[l]/2)
))) endif; 

%% Implied constraints:
constraint if concentrations_on_different_columns then (sum(lr_concentrations) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) endif;

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,left] + lr_concentrations[r*compounds + l,right] == compound_concentrations[l]
))) endif; 


%% Implied constraints 
%% These constraints are too weak when the number of plates increases
%% Right now it looks like it's mostly taking time 

constraint global_cardinality_low_up(experiment_row, [i | i in Rows], [0| i in Rows], [numplates*numcols| i in Rows]);

array [Plates,Rows] of var 0..numcols: controls_in_plate_row;
array [Plates,Columns] of var 0..numrows: controls_in_plate_column;
array [Plates,Rows] of var 0..numcols: controls_only_in_plate_row;
array [Plates,Columns] of var 0..numrows: controls_only_in_plate_column;
array [Rows] of var 0..numplates*numcols: controls_in_row;
array [Columns] of var 0..numplates*numrows: controls_in_column;

constraint (sum(controls_in_plate_row) == total_controls + emptywells);
constraint (sum(controls_in_plate_column) == total_controls + emptywells);

constraint (sum(controls_only_in_plate_row) == total_controls);
constraint (sum(controls_only_in_plate_column) == total_controls);

constraint (sum(controls_in_row) == total_controls + emptywells);
constraint (sum(controls_in_column) == total_controls + emptywells);

%% Redundant
%constraint (sum(controls_in_plate_row) ==  sum(controls_only_in_plate_row) + emptywells);
%constraint (sum(controls_in_plate_column) == sum(controls_only_in_plate_column) + emptywells);


%Among: Requires exactly n variables in x to take one of the values in v .
constraint forall(i in Plates, j in Rows) (among(controls_in_plate_row[i,j], plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));
constraint forall(i in Plates, j in Rows) (among(controls_only_in_plate_row[i,j], plates[i,j,..], (experiments+1..experiments+num_controls*max_control_concentrations)));


constraint forall(i in Plates, k in Columns) (among(controls_in_plate_column[i,k], plates[i,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));
constraint forall(i in Plates, k in Columns) (among(controls_only_in_plate_column[i,k], plates[i,..,k], (experiments+1..experiments+num_controls*max_control_concentrations)));


constraint forall(j in Rows) (among(controls_in_row[j], plates[..,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(k in Columns) (among(controls_in_column[k], plates[..,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(j in Rows) (controls_in_row[j] == sum(controls_in_plate_row[..,j]));

constraint forall(j in Rows) (controls_in_row[j] >= sum(controls_only_in_plate_row[..,j]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,upper,..]) + count(plates[i,1..floor(numrows/2),..],0) == sum(controls_in_plate_row[i,1..floor(numrows/2)]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,lower,..]) + count(plates[i,floor(numrows/2)+1..numrows,..],0) == sum(controls_in_plate_row[i,floor(numrows/2)+1..numrows]));

%% HERE BACK :-(
%constraint forall(i in Plates, v in Vertical) ( sum(controls_ul_plates[i,v,..]) == sum_controls_ul_plates[i,v]);

%constraint forall(i in Plates) ( sum_controls_ul_plates[i,upper] == sum(j in 1..(numrows div 2))(controls_only_in_plate_row[i,j]));

%constraint forall(i in Plates) ( sum_controls_ul_plates[i,lower] == sum(j in (numrows div 2)+1..numrows)(controls_only_in_plate_row[i,j]));


%constraint global_cardinality_closed(experiment_row, [j | j in 2..(numrows-1)],
%[numplates*(numcols-2)-among([plates[i,j,k] | i in Plates, k in 2..(numcols-1)], (experiments+1..experiments+num_controls*control_concentrations))| j in 2..(numrows-1)]);

%% Implied constraint: how many experiments can be in a row/column? It's too weak and needs to be done per plate.
constraint global_cardinality(experiment_row, [j | j in Rows], [(numplates*numcols)-controls_in_row[j]| j in Rows]);
constraint global_cardinality(experiment_column, [k | k in Columns], [(numplates*numrows)-controls_in_column[k]| k in Columns]);




%% Implied constraint: how many experiments can there be in a row? Per plate
% I think this still works...
%% TODO: Redundant
constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])<= numcols-among(plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])<= numcols-controls_in_plate_row[i,j]);

constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])<= numcols-controls_only_in_plate_row[i,j]);

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
(among(plates[i,..,k], 1..experiments)::domain) <= numrows-among([plates[i,j,k] | j in Rows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));


%% Implied constraint:

                                                                                      
%%KEEP%% constraint global_cardinality_low_up_closed(experiment_column, [i | i in 2..numcols-1], [0| i in 2..numcols-1], [numplates*(numrows-2)| i in 2..numcols-1]);                                         


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Different compound replicas must appear on different plates (if possible)
int: min_plates = min(numplates,replicates);

constraint if replicates_on_different_plates then forall(l in 1..compounds) (nvalue(min(numplates,compound_replicates[l]), [experiment_plate[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])) endif; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Different compound replicas must appear on the same plate
constraint if replicates_on_same_plate then forall(l in 1..compounds) (all_equal([experiment_plate[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])) endif;


%% Different combination replicas must appear on different plates (if possible!)
%% Taking away combinations.
%constraint forall(l in 0..(combinations-1)) (nvalue(min_plates, [experiment_plate[experiments + l*combination_concentrations + i*combinations*combination_concentrations + 1] | i in 0..(replicates-1)])); 


%constraint if try_new then forall(l in 1..compounds) (alldifferent_except_0([experiment_column[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])::domain) endif; 

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




% Balancing compounds between plates (and knowing that there must be some sort of balance!)
%% FIX ME!
%% I want to remove these constraints but I need to make other stronger first
int: at_least_compounds = min(compound_concentrations++[0])*floor(sum(compound_replicates)/numplates);
int: at_most_compounds = max_compound_concentrations*ceil(sum(compound_replicates)/numplates);

constraint global_cardinality_low_up([experiment_plate[i] | i in 1..experiments] ,[ i | i in Plates],[ at_least_compounds | i in Plates],[ at_most_compounds | i in Plates]);




%% Implied constraints 
%% These constraints are too weak when the number of plates increases
%%% CHECK! This constraints came from the another version of the code. Need to check that these are still valid.

%???constraint global_cardinality_low_up(experiment_row, [i | i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*floor(at_least_compounds/(numrows-2*size_empty_edge))| i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*(numcols-2*size_empty_edge)| i in 1+size_empty_edge..(numrows-size_empty_edge)]);


%constraint global_cardinality_low_up(experiment_row, [i | i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*floor(at_least/(numrows-2*size_empty_edge))| i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*(numcols-2*size_empty_edge)| i in 1+size_empty_edge..(numrows-size_empty_edge)]);

%constraint global_cardinality_low_up(experiment_column, [i | i in 1+size_empty_edge..(numcols-size_empty_edge)], [numplates*floor(at_least/(numcols-2*size_empty_edge))| i in 1+size_empty_edge..(numcols-size_empty_edge)], [numplates*(numrows-2*size_empty_edge)| i in 1+size_empty_edge..(numcols-size_empty_edge)]);



enum Vertical = {upper,lower};
enum Horizontal = {left,right};


array [Plates,Vertical] of var 0..ceil(inner_plate_size/2): ul_half_plates;
%array [Plates,Horizontal] of var 0..ceil(inner_plate_size/2): lr_half_plates;

%%% Counting the number of experiments in the upper and lower half-plates
constraint forall(i in Plates) (among(ul_half_plates[i,upper],plates[i,1..floor(numrows/2),..], 1..experiments));
constraint forall(i in Plates) (among(ul_half_plates[i,lower],plates[i,floor(numrows/2)+1..numrows,..], 1..experiments));

constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_row[e] <= floor(numrows/2)) == ul_half_plates[i,upper]);
constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_row[e] > floor(numrows/2)) == ul_half_plates[i,lower]);

constraint (sum(e in 1..experiments)(experiment_row[e] in 1..floor(numrows/2)) == sum(i in Plates)(ul_half_plates[i,upper]));
constraint (sum(e in 1..experiments)(experiment_row[e] in floor(numrows/2)+1..numrows) == sum(i in Plates)(ul_half_plates[i,lower]));

%% The following constraints should be redundant when using the two constraints above.
%% CHECK LATER!
constraint (among(sum(i in Plates)(ul_half_plates[i,upper]),experiment_row, 1..floor(numrows/2)));
constraint (among(sum(i in Plates)(ul_half_plates[i,lower]),experiment_row, floor(numrows/2)+1..numrows));


constraint if ((numrows mod 2) == 0) then (forall(i in Plates)(abs(ul_half_plates[i,upper]-ul_half_plates[i,lower])<=1)) endif;



%%% Counting the number of experiments in left-most and right-most half-plates
%constraint forall(i in Plates) (among(lr_half_plates[i,left],[plates[i,j,k] | j in Rows, k in 1..ceil(numcols/2)], 1..experiments));
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

constraint (sum(ul_half_plates[..,upper]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])/2));

constraint (sum(ul_half_plates[..,lower]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])/2));

%% HERE A experiment_plate


constraint forall(i in Plates, v in Vertical) ( ul_half_plates[i,v] <= ((sum([ (experiment_plate[c] == i) | c in 1..experiments]) + 1) div 2));
                                                                
constraint forall(i in Plates, v in Vertical)(((floor(sum(compound_replicates)/numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations) div 2 <= ul_half_plates[i,v]);  

%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v])<=at_most);
%constraint forall(i in Plates)(sum(h in Horizontal)(lr_half_plates[i,h])<=at_most);

%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v])>=at_least);
%constraint forall(i in Plates)(sum(h in Horizontal)(lr_half_plates[i,h])>=at_least);


%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v]) == sum(h in Horizontal)(lr_half_plates[i,h]));

%constraint forall(i in Plates)(sum(v in Vertical)(ul_half_plates[i,v]) == sum(v in Vertical, h in Horizontal)(quarter_plates[i,v,h]));

%% Implied constraints
%constraint forall(i in Plates)(count_eq(experiment_plate, i, sum(h in Horizontal)(lr_half_plates[i,h])));

constraint forall(i in Plates)(count_eq(experiment_plate, i, sum(v in Vertical)(ul_half_plates[i,v])));

%constraint forall(i in Plates)(count_eq(experiment_plate, i, sum(v in Vertical, h in Horizontal)(quarter_plates[i,v,h])));





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Balancing controls between plates.
%% Asumes controls can be balanced this way, that is, that the designer expects about the same number of controls on each plate and not a plate full of controls.

array[int] of float: controls_per_plate = [control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])/numplates | i in 1..num_controls*max_control_concentrations];

array[int] of int: min_controls_per_plate = [floor(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

array[int] of int: max_controls_per_plate = [ceil(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

constraint forall(i in Plates)(global_cardinality_low_up([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], min_controls_per_plate, max_controls_per_plate));



%% HERE!!!
%% I need to work on these constraints, which seem to be quite ineficient. I believe I need to use variables for halves plates, maybe even for quarter plates.

array [Plates,Vertical,1..num_controls*max_control_concentrations] of var int: controls_ul_plates;

array [Plates,Vertical] of var int: sum_controls_ul_plates;

array [Plates,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_lr_plates;

array [Plates,Rows,{0} union 1..num_controls*max_control_concentrations] of var int: controls_row_plates;

array [Plates,Columns,{0} union 1..num_controls*max_control_concentrations] of var int: controls_column_plates;


constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..floor(numrows/2), k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,upper,d] | d in 1..num_controls*max_control_concentrations]));

constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in floor(numrows/2)+1..numrows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,lower,d] | d in 1..num_controls*max_control_concentrations]));

constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]<=ceil(max_controls_per_plate[d]/2));
constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]>=floor(min_controls_per_plate[d]/2));

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(abs(controls_ul_plates[i,upper,d] - controls_ul_plates[i,lower,d]) <= 1);

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d] + controls_ul_plates[i,lower,d] <= max_controls_per_plate[d]);

constraint forall(i in 1..num_controls*max_control_concentrations)(sum([controls_ul_plates[p,v,i] | v in Vertical, p in Plates]) == control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) );


%% NOW :(

%constraint forall(i in Plates, v in Vertical)(sum(controls_ul_plates[i,v,..]) == sum_controls_ul_plates[i,v]);

%% Half of the total number of controls on each upper/lower half of the plate.
constraint forall(i in Plates)(abs(sum(controls_ul_plates[i,upper,..]) - sum(controls_ul_plates[i,lower,..])) <= 1);

constraint forall(i in Plates)(abs(sum_controls_ul_plates[i,upper] - sum_controls_ul_plates[i,lower]) <= 1);


% %% Balancing the total of controls between left and right half-plates

constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in 1..floor(numcols/2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,left,d] | d in 1..num_controls*max_control_concentrations]));

constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in floor(numcols/2)+1..numcols], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,right,d] | d in 1..num_controls*max_control_concentrations]));


constraint forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]<=ceil(max_controls_per_plate[d]/2));
constraint forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]>=floor(min_controls_per_plate[d]/2));

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(abs(controls_lr_plates[i,left,d] - controls_lr_plates[i,right,d]) <= 1);

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,left,d] + controls_lr_plates[i,right,d] <= max_controls_per_plate[d]);

constraint forall(i in 1..num_controls*max_control_concentrations)(sum([controls_lr_plates[p,h,i] | h in Horizontal, p in Plates]) == control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) );

%% Half of the total number of controls on each upper/lower half of the plate.
constraint forall(i in Plates)(abs(sum(d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,left,d]) - sum(d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,right,d])) <= 1);



%% Implied constraints for half plates
%% Connecting the number of compounds with the number of controls for each half plate.
%% TODO: revise the right hand side
constraint forall(i in Plates, v in Vertical)(ul_half_plates[i,v] + sum(controls_ul_plates[i,v,..]) <= ((inner_plate_size+1) div 2));

constraint forall(i in Plates)(ul_half_plates[i,upper] + ul_half_plates[i,lower] + sum(controls_ul_plates[i,..,..]) + count(plates[i,..,..], 0) == inner_plate_size);

constraint forall(i in Plates)( sum(ul_half_plates[i,..]) + sum(sum_controls_ul_plates[i,..]) + count(plates[i,..,..], 0) == inner_plate_size);

%%->constraint forall(i in Plates)(ul_half_plates[i,upper] + ul_half_plates[i,lower] + sum(d in 1..num_controls*max_control_concentrations,v in Vertical)(controls_ul_plates[i,v,d]) + count([plates[i,j,k] | j in 1+size_empty_edge..numrows-size_empty_edge, k in 1+size_empty_edge..numcols-size_empty_edge], 0) == inner_plate_size);


%% Is this helping? I don't think so! But it might...
%%->constraint (sum(r in 1..floor(numrows/2))(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d]) + count([plates[i,j,k] | i in Plates, j in 1..floor(numrows/2), k in Columns], 0));

%%->constraint (sum(r in floor(numrows/2)+1..numrows)(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,lower,d]) + count([plates[i,j,k] | i in Plates, j in floor(numrows/2)+1..numrows, k in Columns], 0));


%%% Controls per row across plates%%%

%constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_row[j] <= ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;

%constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_row[j] >= floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] <= ceil((total_controls+emptywells)/numrows) ) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] >= floor((total_controls+emptywells)/numrows) ) endif;


%%% Controls per row per plate %%%
%% Relaxing this constraint a bit...
%constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] <= control_slack + ceil(ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge))/numplates) + 2*size_empty_edge);

%constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] >= floor(floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge))/numplates) + 2*size_empty_edge);


constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] <= control_slack + ceil(ceil((total_controls+emptywells)/numrows)/numplates) );

constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] >= floor(floor((total_controls+emptywells)/numrows)/numplates) );


constraint if emptywells == 0 then forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] == controls_in_plate_row[i,j] ) else forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] <= controls_in_plate_row[i,j] ) endif;


constraint forall(i in Plates, k in Columns) (controls_in_plate_column[i,k] <= control_slack + ceil(ceil((total_controls+emptywells)/numcols)/numplates) );

constraint forall(i in Plates, k in Columns) (controls_in_plate_column[i,k] >= floor(floor((total_controls+emptywells)/numcols)/numplates) );

constraint if emptywells == 0 then forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] == controls_in_plate_column[i,k] ) else forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] <= controls_in_plate_column[i,k] ) endif;

constraint if interconnected_plates then forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] >= (controls_in_row[j] div numplates)) endif;

constraint if interconnected_plates then forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] <= ((controls_in_row[j]+numplates-1)div numplates)) endif;


%%% Controls per column across plates%%%
 
constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] <= ceil((total_controls+emptywells)/numcols) ) endif;

constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] >= floor((total_controls+emptywells)/numcols) ) endif;


%%% Attempting to balance controls inside each row

array [Rows,Horizontal] of var 0..numplates*numcols: controls_in_half_row;

array [Plates,Rows,Horizontal] of var 0..ceil(numcols/2): controls_in_half_row_plate; %includes empty wells

constraint forall(j in Rows) ( among(controls_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..floor(numcols/2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(j in Rows) ( among(controls_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in floor(numcols/2)+1..numcols], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..floor(numcols/2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,right],[plates[i,j,k] | k in floor(numcols/2)+1..numcols], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(j in Rows, h in Horizontal) (controls_in_half_row[j,h] == sum(i in Plates)(controls_in_half_row_plate[i,j,h]));

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,right]);

constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/numrows) div 2) <= controls_in_half_row[j,left]) endif;

constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/numrows) div 2) <= controls_in_half_row[j,right]) endif;

constraint if interconnected_plates then forall(j in Rows, h in Horizontal) ( (controls_in_row[j] div 2) <= controls_in_half_row[j,h] ) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] == (controls_in_half_row[j,left] + controls_in_half_row[j,right])) endif;


%constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( floor(floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/numplates)/(numrows-2*size_empty_edge))/2) + size_empty_edge) <= controls_in_half_row_plate[i,j,h]) endif;

constraint forall(i in Plates, h in Horizontal, j in Rows) ( (controls_in_plate_row[i,j] div 2) <= controls_in_half_row_plate[i,j,h]);

constraint forall(i in Plates, h in Horizontal, j in Rows) ( ( (controls_in_plate_row[i,j]+1) div 2) >= controls_in_half_row_plate[i,j,h]);


%constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil(ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/numplates)/(numrows-2*size_empty_edge))/2) + size_empty_edge) >= controls_in_half_row_plate[i,j,h]) endif;


constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] == controls_in_half_row_plate[i,j,left] + controls_in_half_row_plate[i,j,right]);


%%% Attempting to balance controls inside each column

array [Columns,Vertical] of var 0..numplates*numrows: controls_in_half_column;

array [Plates,Columns,Vertical] of var 0..floor(numrows/2): controls_in_half_column_plate; %includes empty wells

constraint forall(k in Columns) ( among(controls_in_half_column[k,upper],[plates[i,j,k] | i in Plates, j in 1..(numrows div 2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(k in Columns) ( among(controls_in_half_column[k,lower],[plates[i,j,k] | i in Plates, j in (numrows div 2)+1..numrows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, k in Columns) ( among(controls_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(numrows div 2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, k in Columns) ( among(controls_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (numrows div 2)+1..numrows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(k in Columns, v in Vertical) (controls_in_half_column[k,v] == sum(i in Plates)(controls_in_half_column_plate[i,k,v]));


constraint forall(i in Plates, v in Vertical, k in Columns) ( (controls_in_plate_column[i,k] div 2) <= controls_in_half_column_plate[i,k,v]);

constraint forall(i in Plates, v in Vertical, k in Columns) ( ( (controls_in_plate_column[i,k]+1) div 2) >= controls_in_half_column_plate[i,k,v]);


%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,right]);

constraint if interconnected_plates then forall(k in Columns) ( (floor((total_controls+emptywells)/numcols)  div 2) <= controls_in_half_column[k,upper]) endif;

constraint if interconnected_plates then forall(k in Columns) ( (floor((total_controls+emptywells)/numcols) div 2) <= controls_in_half_column[k,lower]) endif;

constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] == (controls_in_half_column[k,upper] + controls_in_half_column[k,lower])) endif;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Balancing each type of control per row

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (count_eq(plates[i,j,..],experiments+c,controls_row_plates[i,j,c]));

constraint forall(i in Plates, j in Rows) (count_eq(plates[i,j,..],0,controls_row_plates[i,j,0]));

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (controls_row_plates[i,j,c] <= ceil(max_controls_per_plate[c]/numrows));

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (controls_row_plates[i,j,c] >= floor(min_controls_per_plate[c]/numrows));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Balancing each type of control per column

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (count_eq(plates[i,..,k],experiments+c,controls_column_plates[i,k,c]));

constraint forall(i in Plates, k in Columns) (count_eq(plates[i,..,k],0,controls_column_plates[i,k,0]));

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (controls_column_plates[i,k,c] <= ceil(max_controls_per_plate[c]/numcols));

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (controls_column_plates[i,k,c] >= floor(min_controls_per_plate[c]/numcols));


%% Implied all_different constraint
%% Using alldifferent_except_0 seems to be faster than alldifferent_except... but it could be due to less propagation?
% array [Plates,Rows,Columns] of var 0..experiments: exp_plates; 

% constraint forall(i in Plates, j in Rows, k in Columns)((plates[i,j,k] <= experiments) -> (exp_plates[i,j,k] == plates[i,j,k]));

% constraint forall(i in Plates, j in Rows, k in Columns)((plates[i,j,k] > experiments) -> (exp_plates[i,j,k] == 0));

% constraint forall(i in Plates, j in Rows, k in Columns)((exp_plates[i,j,k] > 0) -> (exp_plates[i,j,k] == plates[i,j,k]));

% constraint alldifferent_except_0(exp_plates):: domain;

constraint alldifferent_except(plates, {0} union experiments+1..experiments+num_controls*max_control_concentrations):: domain;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Distributing controls

%% Controls of the same type are not located next to each other

%% Implied constraint
%% TODO: RELAX for the case with too many replicates of the same control
constraint if (min(spread_control++[1])>0) then forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1) (alldifferent_except_0([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]])):: domain endif;

array [Plates,Rows,Columns] of var bool: controls_layout; 

constraint forall(i in Plates, j in Rows, k in Columns)(controls_layout[i,j,k] == ((plates[i,j,k] > experiments)) );

constraint if spread_controls then forall(i in Plates, j in Rows)(regular(controls_layout[i,j,..], "(0|1)(0|01)*")::domain) endif;

constraint if spread_controls then forall(i in Plates, k in Columns)(regular(controls_layout[i,..,k], "(0|1)(0|01)*")::domain) endif;

constraint if spread_controls then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_plate_row[i,j],controls_layout[i,j,..])::domain) endif;

constraint if spread_controls then forall(i in Plates, k in Columns)(my_implied_cost(controls_only_in_plate_column[i,k],controls_layout[i,..,k])::domain) endif;

constraint forall(i in Plates) (sum_controls_ul_plates[i,upper] == sum(controls_layout[i,1..(numrows div 2),..]));
constraint forall(i in Plates) (sum_controls_ul_plates[i,lower] == sum(controls_layout[i,(numrows div 2)+1..numrows,..]));


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
  forall(i in 2..L-1)(c_cost[i+1] <= 1 + c_cost[i-1]);


%function array[int] of int: stupidcost(int: length, int: cost)

%                        int: S,
%                        array [int,int] of int: d,
%                        int: q0,
%                        set of int: F,
%                        array [int,int] of int: c,
%                        var int: C)                       
                

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


constraint if force_spread_controls then forall(i in Plates, j in 3..numrows-2, k in 3..numcols-2)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-2]!= plates[i,j,k] /\ 
            plates[i,j-2,k-1]  != plates[i,j,k] /\
            plates[i,j-2,k]  != plates[i,j,k] /\
            plates[i,j-2,k+1]  != plates[i,j,k] /\
            plates[i,j-2,k+2]!= plates[i,j,k] /\
            plates[i,j-1,k-2]  != plates[i,j,k] /\
            plates[i,j-1,k+2]  != plates[i,j,k] /\
            plates[i,j,k-2]  != plates[i,j,k] /\
            plates[i,j,k+2]  != plates[i,j,k] /\
            plates[i,j+1,k-2]  != plates[i,j,k] /\
            plates[i,j+1,k+2]  != plates[i,j,k] /\
            plates[i,j+2,k-2]!= plates[i,j,k] /\
            plates[i,j+2,k-1]  != plates[i,j,k] /\
            plates[i,j+2,k]  != plates[i,j,k] /\
            plates[i,j+2,k+1]  != plates[i,j,k] /\
            plates[i,j+2,k+2]!= plates[i,j,k]
            ) endif;

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
                                                                               
%%%% Find a solution! %%%%

array [Plates,1..experiments] of var {0} union Rows: experiment_plate_row;

constraint forall(l in 1..experiments)((experiment_plate[l] > 0) -> (experiment_plate_row[experiment_plate[l],l] == experiment_row[l]));

constraint forall(i in Plates, l in 1..experiments)((experiment_plate[l] != i) -> (experiment_plate_row[i,l] == 0));

%array [1..numplates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..numplates*2];
 
%array [1..numplates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..numplates*2];

array [1..numplates*3] of ann: search_order = [if (i mod 3)==1 then int_search(experiment_plate_row[(i+2)div 3,..], first_fail, indomain_random) elseif (i mod 3) == 2 then bool_search(controls_layout[(i+1)div 3,..,..], random, indomain_max) else int_search(plates[i div 3,..,..], first_fail, indomain_random) endif | i in 1..numplates*3]; 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 

%% Best for Gecode
%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(plates, first_fail, indomain_random)]) satisfy; 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++[int_search(plates[i,..,..], random, indomain_max) | i in Plates])
%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_plate_row, first_fail, indomain_random)]++[ 
%int_search(plates[i,..,..], first_fail, indomain_random) | i in Plates]) 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++[
%int_search(experiment_plate_row[1,..], first_fail, indomain_random),int_search(plates[1,..,..], random, indomain_max)]++[ 
%int_search(experiment_plate_row[2,..], first_fail, indomain_random),int_search(plates[2,..,..], random, indomain_max)])



solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++
search_order)
      :: restart_geometric(1.1,5*experiments)
     satisfy ; 


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



%%KEEP%% solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(experiment_column, random, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Pretty printing %%%

output [if testing \/ debugging then "\(numplates) plates\n" else "" endif];

output [if debugging /\ (emptywells>0) then "warning: there are \(emptywells) empty wells. Consider using them as controls.\n\n" else "" endif];

output [if debugging then "Spreading controls... \(spread_control) ... \(min(spread_control++[1])>0)\n" else "" endif];

output [if debugging then "Plate 1:\n" else "" endif];
         
%%TODO: change output so symbols are not hardcoded
output [if debugging then if fix(plates[i,j,k]) == 0  then "  ." 
        elseif fix(plates[i,j,k]) == (experiments+1)  then "  +" %% Positive controls
        elseif fix(plates[i,j,k]) == (experiments+2)  then "  -" %% Negative controls
        elseif fix(plates[i,j,k]) == (experiments+3)  then "  x" %% Blank controls
        elseif fix(plates[i,j,k]) == (experiments+4)  then "  ~" %% DMSO controls
        elseif fix(plates[i,j,k]) == (experiments+5)  then "  *" %% other
        elseif fix(plates[i,j,k]) == (experiments+6)  then "  #" %% other
        elseif fix(plates[i,j,k]) == (experiments+7)  then "  &" %% other
        elseif fix(plates[i,j,k]) == (experiments+8)  then "  @" %% other
        elseif fix(plates[i,j,k]) == (experiments+9)  then "  <" %% other
        elseif fix(plates[i,j,k]) == (experiments+10) then "  >" %% other
        elseif fix(plates[i,j,k]) == (experiments+11) then "  $" %% other
        elseif fix(plates[i,j,k]) ==  (experiments+12) then "  X" %% other      
        elseif fix(plates[i,j,k]) >  (experiments+12) then "  Y" %% other        
        %else show_int(3,plates[i,j,k]) endif ++
        else "   " endif ++
        %% Adding lines and headings %%
        if j== numrows /\ k == numcols /\ i<numplates then "\n\n Plate \(i+1):\n" 
        elseif k == numcols then "\n" else " " endif else "" endif|
        i in Plates, j in Rows, k in Columns];        
         
output [if debugging then "Plate:" else "" endif, if debugging then show(plates) else "" endif];

output [if debugging then "\nCompound's plate:\n" else "" endif, if debugging then show(experiment_plate) else "" endif];

output [if debugging then "\nSpread constrols:\n" else "" endif, if debugging then show(spread_controls) else "" endif];

output [if debugging then "\nControls' layout:\n" else "" endif, if debugging then show(controls_layout) else "" endif];

output [if debugging then "\nExperiment's row:\n" else "" endif, if debugging then show(experiment_row) else "" endif];

output [if debugging then "\nExperiment's row (per plate):\n" else "" endif, if debugging then show(experiments_in_plate_row) else "" endif];

 %output ["\nExperiment's column:\n", show(experiment_column)];

output [if debugging then "\nControls per row:\n" else "" endif, if debugging then show(controls_in_row) else "" endif];

output [if debugging then "\nControls per row per plate:\n" else "" endif, if debugging then show(controls_in_plate_row) else "" endif];

output [if debugging then "\nControls per column:\n" else "" endif, if debugging then show(controls_in_column) else "" endif];

output [if debugging then "\nControls per half row:\n" else "" endif, if debugging then show(controls_in_half_row) else "" endif];

output [if debugging then "\nControls per upper/lower half plates:\n" else "" endif, if debugging then show(controls_ul_plates) else "" endif];

output [if debugging then "\nControls per left/right half plates:\n" else "" endif, if debugging then show(controls_lr_plates) else "" endif];

output [if debugging then "\nTotal Compounds:\n" else "" endif, if debugging then show(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) else "" endif];

output [if debugging then "\nCompounds per half plates:\n" else "" endif, if debugging then show(ul_half_plates) else "" endif];

output [if debugging then "\nConcentrations per half plates:\n" else "" endif, if debugging then show(ul_concentrations) else "" endif];

output [if debugging then "\nConcentrations upper half plates:\n" else "" endif, if debugging then show(sum(ul_concentrations[..,upper])) else "" endif];

output [if debugging then "\nConcentrations lower half plates:\n" else "" endif, if debugging then show(sum(ul_concentrations[..,lower])) else "" endif];

output [if debugging then "\nControls per plate per row:\n" else "" endif, if debugging then show(controls_row_plates) else "" endif];

output [if debugging then "\nControls per plate per column:\n" else "" endif, if debugging then show(controls_column_plates) else "" endif];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CSV Ouput %%%

array[int] of string: letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
         
output ["plateID,well,cmpdname,CONCuM,cmpdnum,VOLuL\n"];

%%% TODO: FIX ME!!!!!! Problem when there are vertical/horizontal cell lines!
output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[size_empty_edge+j+(numrows+2*size_empty_edge)*(h-1)] ++ if (size_empty_edge+k+(2*size_empty_edge+numcols)*(v-1)) < 10 then "0" else "" endif ++ "\(size_empty_edge+k+(2*size_empty_edge+numcols)*(v-1))," ++ 

%%%% Compounds %%%%

if fix(plates[i,j,k]) <= experiments then compound_names[(floor((fix(plates[i,j,k])-1)/max_compound_concentrations) mod compounds)+1] 
++ "," ++ compound_concentration_names[(floor((fix(plates[i,j,k])-1)/max_compound_concentrations) mod compounds)+1,(((fix(plates[i,j,k])-1)) mod max_compound_concentrations)+1] ++ "," ++

compound_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1] ++ "_" ++ compound_concentration_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1,((floor((fix(plates[i,j,k])-1))) mod max_compound_concentrations)+1]

elseif fix(plates[i,j,k]) <= experiments then
combination_names[floor(((fix(plates[i,j,k])-experiments-1)/(replicates*combination_concentrations)))+1] ++
"," ++ combination_concentration_names[(((fix(plates[i,j,k])-experiments-1)) mod combination_concentrations)+1] ++ "," ++


%% Latex name of the combination
%compound_concentration_indicators[((floor((fix(plates[i,j,k])-1))) mod compound_concentrations)+1] ++ "{" ++ compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*compound_concentrations)))+1] ++ "}"

%"{"++combination_names[floor(((fix(plates[i,j,k])-experiments-1)/(replicates*combination_concentrations)))+1]++"}"

combination_names[floor(((fix(plates[i,j,k])-experiments-1)/(replicates*combination_concentrations)))+1]++"_"++combination_concentration_names[(((fix(plates[i,j,k])-experiments-1)) mod combination_concentrations)+1]

%%%% Controls %%%%
%% 
%control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])

elseif fix(plates[i,j,k]) <= experiments + num_controls*max_control_concentrations then
%control_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1] ++ "," ++  
control_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1] ++ "," ++  
%% control concentration

%control_concentration_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1,floor((fix(plates[i,j,k])-experiments-1)/num_controls)+1] 
control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1] 

++ "," ++ 
%% Latex name of the control. Right now it's the same name
control_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1] ++ "_" ++ control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1]  

%% Blanks
else 

"blank" ++ "," ++  
%% blank concentration

"" 

++ "," ++ 
%% Latex name of the control. Right now it's the same name
"blank"  


endif 

% ++ "," ++ compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*concentrations)))+1] ++ "-" ++ concentration_names[((floor((fix(plates[i,j,k])-1)/(replicates))) mod concentrations)+1] 
++ "\n" else "" endif|
        i in Plates, j in Rows, k in Columns, v in 1..vertical_cell_lines, h in 1..horizontal_cell_lines];       
         

       