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
% Last Revision: October 2020


include "globals.mzn";
include "gecode.mzn";


%%%% Input Data  %%%%

int: compounds; %% number of drugs/compounds
int: combinations; %% combinations of compounds. Used as more compounds.
int: cell_lines;
int: plates_type;
int: culture_media_compositions;
%% int: treatment_hr; %% Unclear right now if needed
int: seeding_density;
int: replicates;
int: compound_concentrations;
int: combination_concentrations;
int: control_concentrations;
bool: allow_empty_wells;

int: num_controls;
array [1..num_controls] of int: control_replicates; 


%% Plate size / number of wells
int: numrows;
int: numcols;

%% Compound names. Used for writing CSV file
array[1..compounds] of string: compound_names;

array[1..combinations] of string: combination_names;

array[1..num_controls] of string: control_names;

%% TODO: add latex indicators for combinations and controls 
array[1..compound_concentrations] of string: compound_concentration_names;

array[1..compound_concentrations] of string: compound_concentration_indicators;

array[1..combination_concentrations] of string: combination_concentration_names;

array[1..control_concentrations] of string: control_concentration_names;


%%% Datafile validation %%%
constraint assert(compounds >= 0,"Invalid datafile: Number of compounds should not be less than zero");
constraint assert(combinations >= 0,"Invalid datafile: Number of compounds should not be less than zero");
constraint assert(num_controls >= 0,"Invalid datafile: Number of controls should not be less than zero");
constraint assert(cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero");
constraint assert(numrows > 0,"Invalid datafile: Number of rows should be larger than zero");
constraint assert(numcols > 0,"Invalid datafile: Number of columns should be larger than zero");
constraint assert(plates_type > 0,"Invalid datafile: Plate types should be larger than zero");
constraint assert(culture_media_compositions > 0,"Invalid datafile: culture_media_compositions should be larger than zero");
constraint assert(seeding_density > 0,"Invalid datafile: seeding_density should be larger than zero");


%%%% Do not change %%%%%%%%%%%%
cell_lines	 = 1;
plates_type	 = 1;
culture_media_compositions	= 1;
seeding_density	 = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Number of wells needed. Note that plates might not be full

int: total_wells = (compounds*compound_concentrations + combinations*combination_concentrations)*cell_lines*plates_type*culture_media_compositions*seeding_density*replicates + sum(control_replicates)*control_concentrations; 


%% Number of plates needed
% At least 1 plate so we can produce an empty plate. Another option would be a warning that the plate is empty
int: numplates = max(ceil(total_wells/((numcols-2)*(numrows-2))),1);         
                                          
set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: Plates = 1..numplates;


%% All types of experiments (excluding controls)
int: experiments = (compounds*compound_concentrations+combinations*combination_concentrations)*cell_lines*plates_type*culture_media_compositions*seeding_density*replicates;  

int: compound_ids = (compounds*compound_concentrations)*cell_lines*plates_type*culture_media_compositions*seeding_density*replicates;  % number of compounds 

%% To be used as a redundant constraint
int: emptywells = numplates*numcols*numrows - experiments - sum(control_replicates)*control_concentrations;   


%% To be used as a redundant constraint
int: inner_emptywells = numplates*(numcols-2)*(numrows-2) - experiments - sum(control_replicates)*control_concentrations; 

constraint assert(inner_emptywells = 0 \/ allow_empty_wells,"Warning: " ++
                  "There are \(inner_emptywells) empty wells. You can allow them or modify other quantities");

%% Plates (our solution!!)
array [Plates,Rows,Columns] of var 0..(experiments+num_controls*control_concentrations): plates; 

% 0 = empty wells
% 1... compounds*compound_concentrations*replicates = compounds
% compounds*compound_concentrations*replicates+1 ... experiments = combinations
% experiments+1... = controls -> control1_conc1, control1_conc2, ...control2_conc1, control2_conc2...

%% Alternative model
array [1..experiments] of var Plates: compound_location;


%%%% CONSTRAINTS %%%%
%% TODO: make edge effect optional!
%% Edge effect: Leave first and last rows of every plate empty
constraint forall(i in Plates, j in {1,numrows}, k in Columns)(plates[i,j,k] = 0);


%% Edge effect: Leave first and last columns of every plate empty
constraint forall(i in Plates, j in Rows, k in {1,numcols})(plates[i,j,k] = 0);


%% Exactly the total number of empty wells (zeros) and controls (experiments+i)
% merged with another constraint. See below.
%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[0]++[experiments+i | i in 1..num_controls*control_concentrations],[emptywells]++[control_replicates[((i-1) mod num_controls)+1] | i in 1..num_controls*control_concentrations]);

%% Exactly the number of replicates for each compound*compound_concentrations and combination*combination_concentrations
%% merged with another constraint. See below.
%% Exactly one (right now).
%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates), [ i | i in 1..experiments] , [ 1 | i in 1..experiments]);


% Merging both global_cardinality constraints
constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[ i | i in 0..experiments]++[experiments+i | i in 1..num_controls*control_concentrations],[emptywells]++[ 1 | i in 1..experiments]++[control_replicates[((i-1) mod num_controls)+1] | i in 1..num_controls*control_concentrations]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channelling constraints: an experiment/compound is located at a given plate
constraint forall(l in 1..experiments, i in Plates)(compound_location[l] == i <-> count_eq([array1d(1..numplates*numcols*numrows,plates)[p] | p in (i-1)*numcols*numrows+1..i*numcols*numrows],l,1));


constraint forall(l in 1..experiments, i in Plates)(compound_location[l] == i <-> count_eq([array1d(1..numplates*numcols*numrows,plates)[p] | p in 1..numplates*numcols*numrows where
 (p < (i-1)*numcols*numrows+1 \/ p > i*numcols*numrows)],l,0));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A compound with all concentrations must appear on a single plate
%constraint forall(l in 1..experiments where (l mod compound_concentrations = 1))(all_equal([compound_location[i] | i in l..l+compound_concentrations-1]));

constraint forall(l in 1..compound_ids where ((l mod compound_concentrations) = 1))(all_equal([compound_location[i] | i in l..(l+compound_concentrations-1)]));

%% A combination with all concentrations must appear on a single plate
constraint forall(l in compound_ids+1..experiments where ((l-compound_ids) mod combination_concentrations = 1))(all_equal([compound_location[i] | i in l..l+combination_concentrations-1]));




%% Different compound replicas must appear on different plates (if possible!)
int: min_plates = min(numplates,replicates);

constraint forall(l in 0..(compounds-1)) (nvalue(min_plates, [compound_location[l*compound_concentrations + i*compounds*compound_concentrations + 1] | i in 0..(replicates-1)])); 

%% Different combination replicas must appear on different plates (if possible!)
constraint forall(l in 0..(combinations-1)) (nvalue(min_plates, [compound_location[compound_ids + l*combination_concentrations + i*combinations*combination_concentrations + 1] | i in 0..(replicates-1)])); 


%%% Put empty wells in the corners (as much as possible)
int: num_empty_corners = min(4*numplates,inner_emptywells);
constraint count_eq([plates[i,j,k] | i in Plates, j in {2,numrows-1}, k in {2,numcols-1}],0,num_empty_corners);


%%% If there are still more empty wells, place them near the corners
int: lower_2 = min(2*numplates, inner_emptywells - num_empty_corners);

constraint count_eq([plates[i,j,k] | i in Plates, j in {2,numrows-1}, k in {3}],0,lower_2);


%%% If there are still more empty wells, place them near the corners (as much as possible)
%% CHEAT! I need to put the empty wells in order. It improves propagation A LOT!
%% On the other hand, we're not expecting many empty wells (if any).
int: lower_3 = min(2*card({i | i in {numplates-1,numplates} where (i>0)}), inner_emptywells - num_empty_corners - lower_2);
constraint count_eq([plates[i,j,k] | i in {numplates-1,numplates} where (i>0), j in {2,numrows-1}, k in {numcols-2}],0,lower_3);


%% If there are still empty wells, make the outer columns empty.
int: empty_wells_in_columns = min(2*numplates*(numrows-4), inner_emptywells-num_empty_corners);
%constraint count_eq([plates[i,j,k] | i in Plates, j in 3..numrows-2, k in {2,numcols-1}],0,empty_wells_in_columns);


%% If there are still empty wells, make the outer columns empty.
%int: empty_wells_in_rows = min(2*numplates*(numcols-4), inner_emptywells-num_empty_corners-empty_wells_in_columns);
%constraint count_eq([plates[i,j,k] | i in Plates, j in {2,numrows-1}, k in 3..numcols-2],0,empty_wells_in_rows);



%% Balancing compounds between plates: distribute the compounds equitatively among all the plates
%%%% This is only interesting when there are inner empty wells. Otherwise, it's redundant
int: at_least = compound_concentrations*floor((compounds*replicates)/numplates) + combination_concentrations*floor((combinations*replicates)/numplates); 
int: at_most = min(compound_concentrations*ceil((compounds*replicates)/numplates) + combination_concentrations*ceil((combinations*replicates)/numplates), (numrows-2)*(numcols-2)); 


%% This implied constraint is very important for propagation!

int: platesize = (numrows-2)*(numcols-2);

constraint global_cardinality_low_up(compound_location,[ i | i in Plates],[ at_least | i in Plates],[ at_most | i in Plates]);

%% write one for indices for compounds
%% write one for indices for combinations
%% Maybe one that mixes both! (for example, the maximum might be stronger for the whole array


%%% Balancing controls between plates.

array[int] of float: controls_per_plate = [control_replicates[i]/numplates | i in 1..num_controls];

array[int] of int: min_controls_per_plate = [floor(controls_per_plate[i]) | i in 1..num_controls];

array[int] of int: max_controls_per_plate = [ceil(controls_per_plate[i]) | i in 1..num_controls];

constraint forall(i in Plates)(global_cardinality_low_up([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls], min_controls_per_plate, max_controls_per_plate));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Distributing controls
%% Controls of the same type are NOT located next to each other
% TODO: relax in case there are too many controls per plate.

constraint forall(c in 1..num_controls) (

forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] == experiments+c -> 
           plates[i,j-1,k-1]!= experiments+c /\ 
           plates[i,j-1,k]  != experiments+c /\
           plates[i,j-1,k+1]!= experiments+c /\
           plates[i,j,k-1]  != experiments+c /\
           plates[i,j,k+1]  != experiments+c /\
           plates[i,j+1,k-1]!= experiments+c /\
           plates[i,j+1,k]  != experiments+c /\
           plates[i,j+1,k+1]!= experiments+c)
           );

           
%% Find a solution!

solve::seq_search([int_search(compound_location, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 

%solve::int_search(compound_location, first_fail, indomain_random) satisfy; 

%solve::int_search(plates, random, indomain_max) satisfy; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Pretty printing %%%

output [if (inner_emptywells>0) then "warning: there are \(inner_emptywells) empty wells. Consider using them as controls.\n\n" else "" endif];

output ["Plate 1:\n"];
         
%%TODO: change output so symbols are not hardcoded
output [if fix(plates[i,j,k]) == 0 then "  ." 
        elseif fix(plates[i,j,k]) == (experiments+1) then "  +" %% Positive controls
        elseif fix(plates[i,j,k]) == (experiments+2) then "  -" %% Negative controls
        elseif fix(plates[i,j,k]) == (experiments+3) then "  x" %% Blank controls
        elseif fix(plates[i,j,k]) == (experiments+4) then "  ~" %% DMSO controls
        elseif fix(plates[i,j,k]) == (experiments+5) then "  *" %% other
        elseif fix(plates[i,j,k]) == (experiments+5) then "  #" %% other
        else show_int(3,plates[i,j,k]) endif ++
        %% Adding lines and headings %%
        if j== numrows /\ k == numcols /\ i<numplates then "\n\n Plate \(i+1):\n" 
        elseif k == numcols then "\n" else " " endif |
        i in Plates, j in Rows, k in Columns];        
         
%output ["Compound's plate:\n", show(compound_location)];
         

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CSV Ouput %%%

array[int] of string: letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
         
output ["\nplateID,well,cmpdname,CONCuM,cmpdnum,VOLuL\n"];

output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[j] ++ if k < 10 then "0" else "" endif ++ "\(k)," ++ 


%%%% Compounds %%%%
if fix(plates[i,j,k]) <= compound_ids then compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*compound_concentrations)))+1] 
++ "," ++ compound_concentration_names[((floor((fix(plates[i,j,k])-1))) mod compound_concentrations)+1] ++ "," ++

%% Latex name of the compound
%% FIX! WHY do I have a floor here with mod?
%compound_concentration_indicators[((floor((fix(plates[i,j,k])-1))) mod compound_concentrations)+1] ++ "{" ++ compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*compound_concentrations)))+1] ++ "}"

compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*compound_concentrations)))+1] ++ "_" ++ compound_concentration_names[((floor((fix(plates[i,j,k])-1))) mod compound_concentrations)+1]

elseif fix(plates[i,j,k]) <= experiments then %"comb, comb" ++
combination_names[floor(((fix(plates[i,j,k])-compound_ids-1)/(replicates*combination_concentrations)))+1] ++
"," ++ combination_concentration_names[(((fix(plates[i,j,k])-compound_ids-1)) mod combination_concentrations)+1] ++ "," ++


%% Latex name of the combination
%compound_concentration_indicators[((floor((fix(plates[i,j,k])-1))) mod compound_concentrations)+1] ++ "{" ++ compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*compound_concentrations)))+1] ++ "}"

%"{"++combination_names[floor(((fix(plates[i,j,k])-compound_ids-1)/(replicates*combination_concentrations)))+1]++"}"

combination_names[floor(((fix(plates[i,j,k])-compound_ids-1)/(replicates*combination_concentrations)))+1]++"_"++combination_concentration_names[(((fix(plates[i,j,k])-compound_ids-1)) mod combination_concentrations)+1]

%%%% Controls %%%%
else control_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1] ++ "," ++  
%% control concentration

control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/num_controls)+1] 

++ "," ++ 
%% Latex name of the control. Right now it's the same name
control_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1] ++ "_" ++ control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/num_controls)+1]  
endif 

% ++ "," ++ compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*concentrations)))+1] ++ "-" ++ concentration_names[((floor((fix(plates[i,j,k])-1)/(replicates))) mod concentrations)+1] 
++ "\n" else "" endif|
        i in Plates, j in Rows, k in Columns];       
         

%output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[j] ++ if k < 10 then "0" else "" endif ++ "\(k)," ++ show_int(3,floor(((fix(plates[i,j,k])-1)/(replicates*concentrations)))) ++ "," ++ "conc," ++ "cmpdnum" ++ show_int(3,plates[i,j,k]) ++ "\n" else "" endif|
 %       i in Plates, j in Rows, k in Columns];            
         
         