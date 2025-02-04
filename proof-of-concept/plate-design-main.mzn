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
% Version: 0.1
% Last Revision: January 2025
%

include "globals.mzn";
include "gecode.mzn";
 
%%%%%%%%%%%%%%%%%%%%%
%%%% Input Data  %%%%
%%%%%%%%%%%%%%%%%%%%%


%% Information about the plate size / number of wells %%
int: num_rows;
int: num_cols;


%% Information about the layout %%
int: horizontal_cell_lines; 
int: vertical_cell_lines;
int: size_empty_edge;
bool: inner_empty_edge = true;


%% Information about constraints %%
bool: replicates_on_different_plates;
bool: replicates_on_same_plate;
bool: concentrations_on_different_rows;
bool: concentrations_on_different_columns;
%bool: force_spread_controls;
%bool: force_spread_concentrations;

%% Optional constraints %%
opt bool: force_empty_corners;


%%% Information about compounds %%%

%% Number of drugs/compounds 
int: num_compounds = length(compound_replicates); 

%% Number of times each compound-concentration should appear (copies)
array [int] of int: compound_replicates;

int: replicates = max(compound_replicates++[0]); % TODO DELETE

%% Number of concentrations for each compound
array [1..num_compounds] of int: compound_concentrations;

%% Maximum number of concentrations among all compounds
int: max_compound_concentrations = max(compound_concentrations++[0]);

%% Names of the compounds and their corresponding concentrations. Used to generate the resulting file
%% This could be done outside of MiniZinc as post-processing
array[1..num_compounds] of string: compound_names;
array[1..num_compounds,1..max_compound_concentrations] of string: compound_concentration_names;


%%% Information about controls %%%
int: num_controls = length(control_replicates);
array [int] of int: control_replicates; 
array [1..num_controls] of int: control_concentrations;
int: max_control_concentrations = max(control_concentrations++[0]);

%% Names of the controls and their corresponding concentrations. Used to generate the resulting file
%% This could be done outside of MiniZinc as post-processing
array[1..num_controls,1..max_control_concentrations] of string: control_concentration_names;
array[1..num_controls] of string: control_names;

int: total_controls = sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]);

opt bool: opt_equal_controls;

bool: equal_controls  = opt_equal_controls \/ (numplates == 1) default false;

%%%%%%%%%%%%%%%%%%%%%%
%%%% DERIVED DATA %%%%
%%%%%%%%%%%%%%%%%%%%%%

bool: empty_corners = force_empty_corners default false;
bool: spread_controls = (ceil(inner_plate_size/2)*numplates >= total_controls); %Makes the problem harder when there are many controls

% Some implied constraints only work when the plates have an even number of columns
bool: even_columns = ((numcols mod 2) == 0);

% Some implied constraints only work when the plates have an even number of rows
bool: even_rows = ((numrows mod 2) == 0);

%% Which individual controls can be spread more than others?
array[1..num_controls*max_control_concentrations] of bool: spread_control = [ ceil(inner_plate_size/4)*numplates >= (control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])) | i in 1..num_controls*max_control_concentrations];


%%% Testing %%%
opt bool: testing;% = true;
opt bool: print_all;% = true;
bool: debugging = print_all default false;
%bool: enforce = false;
opt int: swap_search;
opt bool: sorted_compounds;




%% TODO: this could be problematic when there are multiple cell lines (This is indeed wrong) FIX 
int: numrows = if inner_empty_edge then floor(num_rows/horizontal_cell_lines)-2*size_empty_edge else floor((num_rows-2*size_empty_edge)/horizontal_cell_lines) endif;
int: numcols = if inner_empty_edge then floor(num_cols/vertical_cell_lines)-2*size_empty_edge else floor((num_cols-2*size_empty_edge)/vertical_cell_lines) endif;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Datafile validation %%%
constraint assert(num_compounds >= 0,"Invalid datafile: Number of compounds cannot be less than zero.");
constraint assert(num_controls >= 0,"Invalid datafile: Number of controls should not be less than zero.");
constraint assert(vertical_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero.");
constraint assert(horizontal_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero.");
constraint assert(numrows > 0,"Invalid datafile: Number of rows should be larger than zero.");
constraint assert(numcols > 0,"Invalid datafile: Number of columns should be larger than zero.");
constraint assert(num_compounds==0 \/ min(compound_replicates) > 0,"Invalid datafile: Number of replicates should be larger than zero.");
constraint assert(num_compounds==0 \/ min(compound_concentrations) > 0,"Invalid datafile: Number of concentrations should be larger than zero.");
constraint assert((replicates_on_different_plates /\ replicates_on_same_plate) == false,"Invalid datafile: replicates cannot be both on the same plate and on different plates");
constraint assert(not (num_controls == 1 /\ spread_control[1]==false),"Invalid datafile: There are too many controls of only one kind. This is not allowed at the moment. If you believe this is a mistake, please contact the developers.");
constraint assert(total_wells > 0, "Invalid data: the plates cannot be completely empty.");
constraint assert(inner_plate_size > 4*empty_corners, "Invalid data: There are no wells on the plate.");
constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, "Invalid data: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Number of wells needed. Note that plates might not be full
int: total_wells = sum([compound_concentrations[i]*compound_replicates[i] | i in 1..num_compounds]) + total_controls; 

set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: InnerRows = 2..numrows-1; % Used when forcing empty corners
set of int: InnerColumns = 2..numcols-1; % Used when forcing empty corners
set of int: OuterRows = {1,numrows}; % Used when forcing empty corners
set of int: OuterColumns = {1,numcols}; % Used when forcing empty corners

set of int: Plates = 1..numplates;

enum Vertical = {upper,lower};
enum Horizontal = {left,right};

int: inner_plate_size = numcols*numrows;



%% Number of plates needed
%% max is used to ensure there is always at least 1 plate
int: numplates = max(ceil(total_wells/(inner_plate_size-4*empty_corners)),1);         





%% Experiments that represent compounds (excluding controls)
int: experiments = num_compounds*max_compound_concentrations;  

%% Used in redundant constraints
int: emptywells = numplates*inner_plate_size - total_wells;   



%%%%%%% Detecting some unsatisfiable cases & Data validation %%%%%%%%%%
constraint assert(ceil(sum(compound_replicates)/numplates)*min(compound_concentrations++[infinity]) + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E01)");  
                                                                
constraint assert((floor(sum(compound_replicates)/numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E02)");  

constraint assert(emptywells >= 0,"Model ERROR! Inner empty wells is negative. This should never happen!");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plates (Main variable model) %%%%%%%%%%%

% Values represent the following:
% 0 = empty wells
% 1... experiments = compounds -> compound1-conc1, comp2-conc2...compn-conc-y
% experiments+1... = controls -> control1_conc1, control1_conc2, ...control2_conc1, control2_conc2...

array [Plates,Rows,Columns] of var 0..(experiments+num_controls*max_control_concentrations): plates; 

array [Plates,Rows,Columns] of var bool: controls_layout; 





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CHANELLING CONSTRAINTS %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constraint forall(i in Plates, j in Rows, k in Columns)(controls_layout[i,j,k] == ((plates[i,j,k] > experiments)) );




%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% MODEL CONSTRAINTS %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Place only the exact amount of compounds, controls, and empty wells.
constraint global_cardinality(plates, [ i | i in 0..experiments+num_controls*max_control_concentrations],
                              [emptywells]++
                              [compound_replicates[floor((i-1)/max_compound_concentrations)+1]*(((i-1) mod max_compound_concentrations)<compound_concentrations[floor((i-1)/max_compound_concentrations)+1]) | i in 1..experiments]++
                              [control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..num_controls*max_control_concentrations]);



%% Keep the four corners of all plates empty when empty_corners = true
%% This is used, for example, when a particular piece of equipment has issues with corner wells.
constraint if empty_corners then forall(i in Plates, j in {1,numrows}, k in {1,numcols})(plates[i,j,k] = 0) endif;



solve satisfy;