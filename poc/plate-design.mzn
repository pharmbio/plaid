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


%%%% Plate dimetions %%%%
int: num_rows;
int: num_cols;

%% Information about the layout %%
int: horizontal_cell_lines;
int: vertical_cell_lines;
int: size_empty_edge;
bool: inner_empty_edge = true; % Unused at the moment

%% Information about compounds %%
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
int: control_ids = num_controls*max_control_concentrations;



%% Potentially new parameters %%
bool: spread_controls = (ceil(inner_plate_size/2) >= total_controls_per_plate); %Makes the problem harder when there are many controls

%% Which individual controls can be spread more than others?
array[1..control_ids] of bool: spread_control = [ ceil(inner_plate_size/4) >= (control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])) | i in 1..control_ids];


%% TODO: this could be problematic when there are multiple cell lines
int: numrows = if inner_empty_edge then floor(num_rows/horizontal_cell_lines)-2*size_empty_edge else floor((num_rows-2*size_empty_edge)/horizontal_cell_lines) endif;
int: numcols = if inner_empty_edge then floor(num_cols/vertical_cell_lines)-2*size_empty_edge else floor((num_cols-2*size_empty_edge)/vertical_cell_lines) endif;



%%%% Minimum number of plates needed %%%%
int: inner_plate_size = numcols*numrows;
int: wells_for_compounds = inner_plate_size - total_controls_per_plate;
int: numplates = ceil(total_compounds / wells_for_compounds);

%%%% Number of wells used %%%%
int: total_used_wells = total_compounds + numplates*total_controls_per_plate; 



%%% Definitions %%%
set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: Plates = 1..numplates;

enum Vertical = {upper,lower};
enum Horizontal = {left,right};



%bool: force_spread_controls;
%bool: force_spread_concentrations;



%%%% Testing & Debugging %%%%
opt bool: testing;% = true;
opt bool: print_all;% = true;
bool: debugging = print_all \/ false;
%bool: enforce = false;
opt bool: sorted_compounds;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Data validation %%%%%
constraint assert(total_controls_per_plate <= inner_plate_size, "Invalid data: the number of controls does not fit inside a plate.");
constraint assert(total_used_wells > 0, "Invalid data: the plates cannot be completely empty.");
constraint assert(inner_plate_size > 0, "Invalid data: There are no wells on the plate.");
constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, "Invalid data: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Used in redundant constraints
int: emptywells = numplates*inner_plate_size - total_used_wells;   



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plates (Main variable representation) %%%%%%%%%%%
array [Plates,Rows,Columns] of var 0..(control_ids + num_compounds*max_compound_concentrations): plates; 



%%%% Main constraint model %%%%
%% Place only the exact amount of compounds, controls, and empty wells.


constraint global_cardinality(plates,[ i | i in 0..control_ids+num_compounds*max_compound_concentrations],[emptywells]++[numplates*control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..control_ids]++[compound_replicates[floor((i-1)/max_compound_concentrations)+1]*(((i-1) mod max_compound_concentrations)<compound_concentrations[floor((i-1)/max_compound_concentrations)+1]) | i in 1..num_compounds*max_compound_concentrations]);






solve satisfy;

%output ["\(numrows),\(numcols),\(numplates)"];




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CSV Ouput %%%

array[int] of string: letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
         
output ["plateID, well, cmpdname, CONCuM, cmpdnum, VOLuL\n"];


output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[size_empty_edge+j+(numrows+(1+inner_empty_edge)*size_empty_edge)*(h-1)] ++ if (size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1)) < 10 then "0" else "" endif ++ "\(size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1))," ++ 

%%%% Controls %%%%

if fix(plates[i,j,k]) <= control_ids then 
control_names[floor((fix(plates[i,j,k])-1)/max_control_concentrations)+1] ++ "," ++  
%% control concentration

control_concentration_names[floor((fix(plates[i,j,k])-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-1) mod max_control_concentrations)+1] 

++ "," ++ 
%% Latex name of the control. Right now it's the same as the name and concentration
control_names[floor((fix(plates[i,j,k])-1)/max_control_concentrations)+1] ++ "_" ++ control_concentration_names[floor((fix(plates[i,j,k])-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-1) mod max_control_concentrations)+1]   


%%%% Compounds %%%%

else
compound_names[floor((fix(plates[i,j,k])-control_ids-1)/max_compound_concentrations)+1] ++ "," ++  
%% control concentration

compound_concentration_names[floor((fix(plates[i,j,k])-control_ids-1)/max_compound_concentrations)+1,((fix(plates[i,j,k])-control_ids-1) mod max_compound_concentrations)+1] 

++ "," ++ 
%% Latex name of the control. Right now it's the same as the name
compound_names[floor((fix(plates[i,j,k])-control_ids-1)/max_compound_concentrations)+1] ++ "_" ++ compound_concentration_names[floor((fix(plates[i,j,k])-control_ids-1)/max_compound_concentrations)+1,((fix(plates[i,j,k])-control_ids-1) mod max_compound_concentrations)+1]  

endif 

++ "\n" else "" endif|
        i in Plates, j in Rows, k in Columns, v in 1..vertical_cell_lines, h in 1..horizontal_cell_lines];    
        
        


