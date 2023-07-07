% Copyright 2023 PLAID Authors.
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
% Description:  Plate Layouts using Artificial Intelligence Design (PLAID).
%               New constraint model for PLAID.
%
% Changes:
% * Now the user specifies controls PER PLATE instead of the total for the experiment.
%
%
% Authors: Maria Andreina FRANCISCO RODRIGUEZ (maria.andreina.francisco@it.uu.se)
% Version: 0.1
% Last Revision: July 2023
%

include "globals.mzn";
include "gecode.mzn";
 
 

%%%% Input Data  %%%%

%% Information about constraints on the layout %%
bool: replicates_on_different_plates;
bool: replicates_on_same_plate;
bool: concentrations_on_different_rows;
bool: concentrations_on_different_columns;
bool: force_spread_controls;
bool: force_spread_concentrations;

%% Plate size / number of wells
int: num_rows;
int: num_cols;

%% Information about the layout %%
int: horizontal_cell_lines;
int: vertical_cell_lines;
int: size_empty_edge;
bool: inner_empty_edge = true; % Unused at the moment

%% Compounds %%
int: num_compounds; %% number of drugs/compounds
array[1..num_compounds] of string: compound_names;
array [1..num_compounds] of int: compound_concentrations;
array [1..num_compounds] of int: compound_replicates;
int: max_compound_concentrations = max(compound_concentrations++[0]);
array[1..num_compounds,1..max_compound_concentrations] of string: compound_concentration_names;

%% Information about controls %%
int: num_controls;
array[1..num_controls] of string: control_names;
array [1..num_controls] of int: control_replicates; 
array [1..num_controls] of int: control_concentrations;
int: max_control_concentrations = max(control_concentrations++[0]);
array[1..num_controls,1..max_control_concentrations] of string: control_concentration_names;

int: total_compounds = sum([compound_concentrations[i]*compound_replicates[i] | i in 1..num_compounds]);
int: total_controls_per_plate = sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]);




%% Potentially new parameters %%
bool: balance_controls_inside_plate = true;
bool: spread_controls = (ceil(inner_plate_size/2) >= total_controls_per_plate); %Makes the problem harder when there are many controls

%% Which individual controls can be spread more than others?
array[1..num_controls*max_control_concentrations] of bool: spread_control = [ ceil(inner_plate_size/4) >= (control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])) | i in 1..num_controls*max_control_concentrations];


%%% Testing %%%
opt bool: testing;% = true;
opt bool: print_all;% = true;
bool: debugging = print_all \/ false;
%bool: enforce = false;
opt bool: sorted_compounds;








%% Number of wells needed. Note that plates might not be full
int: total_wells = total_compounds + total_controls; 









%%%%% Data validation %%%%%
constraint assert(total_wells > 0, "Invalid data: the plates cannot be completely empty.");
constraint assert(inner_plate_size>0, "Invalid data: There are no wells on the plate.");
constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, "Invalid data: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


