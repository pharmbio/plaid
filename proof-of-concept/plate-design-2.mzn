% Copyright 2025 PLAID Authors.
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

include "globals.mzn";

%%% Plate size / number of wells %%%
int: num_rows;
int: num_cols;


%%% Constraints %%%
bool: empty_corners = false; %TODO Make optional



%%% Information about controls %%%
int: num_controls = length(control_replicates);
array [int] of int: control_replicates; 
array [1..num_controls] of int: control_concentrations;
int: max_control_concentrations = max(control_concentrations++[0]);
array[1..num_controls,1..max_control_concentrations] of string: control_concentration_names;
array[1..num_controls] of string: control_names;

int: total_controls = sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]);


%%% TODO %%% Temporal things to fix later

int: numrows = num_rows;
int: numcols = num_cols;



%%% Pre-calculations %%%
% Minimal number of wells needed
int: total_wells = total_controls; % TODO Fix me! Change input to controls per plate eventually
int: total_controls_per_plate = total_controls;

int: inner_plate_size = numcols*numrows;

int: numplates = max(ceil(total_wells/(inner_plate_size-4*empty_corners)),1); % max is needed to tell the model this is always at least 1

int: emptywells = numplates*inner_plate_size - total_wells; %% Number of wells that will remain empty



%%% Balancing calculations %%%
bool: spread_controls = (ceil(inner_plate_size/2)*numplates >= total_controls); %Makes the problem harder when there are many controls


%%% Definining useful sets %%%
set of int: Plates = 1..numplates;
set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;




%%% Defining the main variable %%%
array [Plates,Rows,Columns] of var 0..(num_controls*max_control_concentrations): plates;

array [Plates,Rows,Columns] of var bool: controls_layout;
%array [Plates,Rows,Columns] of var int: compound_layer;


%%%% MODEL CONSTRAINTS %%%%
%% Place only the exact amount of compounds, controls, and empty wells.
%constraint global_cardinality(plates,[0]++[1],[emptywells]++[(floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1] | i in 1..experiments]);

int: experiments = 0; % TODO Fix me!

constraint global_cardinality(plates,[0]++[ i | i in 1..experiments]++[i | i in experiments+1..experiments]++[ experiments+i | i in 1..num_controls*max_control_concentrations],[emptywells]++[1|i in experiments+1..experiments]++[control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..num_controls*max_control_concentrations]);



%%% Channelling Constraints %%%
constraint forall(i in Plates, j in Rows, k in Columns)(controls_layout[i,j,k] == ((plates[i,j,k] > experiments)) );

%%% Constraints over controls %%%

% Place the exact number of controls in the control layer %
constraint global_cardinality(controls_layout,[false,true],[inner_plate_size*numplates-total_controls,total_controls]);


constraint if spread_controls then forall(i in Plates, j in Rows)(regular(controls_layout[i,j,..], "(0|(1 0))* (1?)")::domain) endif;

constraint if spread_controls then forall(i in Plates, k in Columns)(regular(controls_layout[i,..,k], "(0|(1 0))* (1?)")::domain) endif;




%%% Solving %%%
solve satisfy;

output show(controls_layout);