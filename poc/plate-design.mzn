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
int: max_compound_replicates = max(compound_replicates++[0]);
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

int: compound_ids = num_compounds*max_compound_concentrations;
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
int: min_numplates = ceil(total_compounds / wells_for_compounds);

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
bool: debugging = print_all \/ true;
%bool: enforce = false;
opt bool: sorted_compounds;


int: numplates = min_numplates + ((ceil(sum(compound_replicates)/min_numplates)*min(compound_concentrations++[infinity]) > wells_for_compounds) \/
              (((sum(compound_replicates) div min_numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations > wells_for_compounds)) ;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Datafile validation %%%%
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
constraint assert(numrows mod 2 == 0, "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of rows.");
constraint assert(numcols mod 2 == 0, "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of columns." );
constraint assert(total_controls_per_plate <= inner_plate_size, "Invalid data: the number of controls does not fit inside a plate.");
constraint assert(total_used_wells > 0, "Invalid data: the plates cannot be completely empty.");
constraint assert(inner_plate_size > 0, "Invalid data: There are no wells on the plate.");
constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, "Invalid data: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");


%%%%%%% Detecting some unsatisfiable cases & Data validation %%%%%%%%%%
%constraint assert(ceil(sum(compound_replicates)/numplates)*min(compound_concentrations++[infinity]) <= wells_for_compounds, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds across the plates. (E01)");  
                                                                
%constraint assert(((sum(compound_replicates) div numplates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations <= wells_for_compounds, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds across the plates. (E02)");  

constraint assert(emptywells >= 0,"Model ERROR! Inner empty wells is negative. This should never happen!");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Used in redundant constraints
int: emptywells = numplates*inner_plate_size - total_used_wells;   



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plates (Main variable representation) %%%%%%%%%%%
array [Plates,Rows,Columns] of var 0..(control_ids + compound_ids): plates; 

%% Alternative (extra) model
array [1..num_compounds, Plates] of var 0..max_compound_replicates: compound_plate;



%%%% Main constraint model %%%%
%% Place only the exact amount of compounds, controls, and empty wells across all plates
constraint global_cardinality(plates,[ i | i in 0..control_ids+num_compounds*max_compound_concentrations],[emptywells]++[numplates*control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..control_ids]++[compound_replicates[floor((i-1)/max_compound_concentrations)+1]*(((i-1) mod max_compound_concentrations)<compound_concentrations[floor((i-1)/max_compound_concentrations)+1]) | i in 1..num_compounds*max_compound_concentrations]);

%% Place the exact amount of controls on each plate 
%% Leave the exact amount of empty wells on each plate (needed for propagation)
constraint forall (p in Plates) (global_cardinality(plates[p,..,..],[ i | i in 0..control_ids],[wells_for_compounds-sum([compound_plate[c,p]*compound_concentrations[c]|c in 1..num_compounds])]++[control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..control_ids]));



% The plate(s) where each compound is
constraint forall (c in 1..num_compounds) (
        sum(compound_plate[c,..]) == compound_replicates[c]
    );

% The number of compounds per plate is limited by the available space
constraint forall (p in Plates) (
        sum([compound_concentrations[c]*compound_plate[c,p] | c in 1..num_compounds]) <= wells_for_compounds
    );

%% Place the exact amount of compounds on each plate 
constraint forall (p in Plates) (
        global_cardinality(plates[p,..,..],
                          [ i + control_ids | i in 1..compound_ids],
                          [compound_plate[floor((i-1)/max_compound_concentrations)+1,p]*(((i-1) mod max_compound_concentrations)<compound_concentrations[floor((i-1)/max_compound_concentrations)+1]) | i in 1..compound_ids]
        ));


% Replicates on the same plate, different plates, or does not matter?
constraint if replicates_on_different_plates then
        forall (c in 1..num_compounds, p in Plates) (compound_plate[c,p] <= 1)
    elseif replicates_on_same_plate then 
        forall (c in 1..num_compounds) (global_cardinality(compound_plate[c,..],[0,compound_replicates[c]],[numplates-1,1]))
    else 
      true
    endif
    ;



%all_different_except(array [$X] of var int: vs,
 %                              set of int: S)

solve satisfy;

%output ["\(numrows),\(numcols),\(numplates)"];




output [if debugging then "\nCompound's plate:\n" else "" endif, if debugging then show(compound_plate) else "" endif];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CSV Ouput %%%%

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
%% Latex name of the compound. Right now it's the same as the name and concentration
compound_names[floor((fix(plates[i,j,k])-control_ids-1)/max_compound_concentrations)+1] ++ "_" ++ compound_concentration_names[floor((fix(plates[i,j,k])-control_ids-1)/max_compound_concentrations)+1,((fix(plates[i,j,k])-control_ids-1) mod max_compound_concentrations)+1]  

endif 

++ "\n" else "" endif|
        i in Plates, j in Rows, k in Columns, v in 1..vertical_cell_lines, h in 1..horizontal_cell_lines];    
        
        


