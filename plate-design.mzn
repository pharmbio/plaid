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
% Last Revision: January 2021


include "globals.mzn";
include "gecode.mzn";


%%%% Input Data  %%%%


%% Information about constraints %%
int: horizontal_cell_lines;
int: vertical_cell_lines;
bool: allow_empty_wells;
bool: replicates_on_different_plates;
bool: replicates_on_same_plate;
bool: concentrations_on_different_rows = true;
bool: concentrations_on_different_columns = true;

int: size_empty_edge;

%% Compounds %%
int: compounds; %% number of drugs/compounds
array [1..compounds] of int: compound_replicates;
int: replicates = max(compound_replicates);
array [1..compounds] of int: compound_concentrations;
int: max_compound_concentrations = max(compound_concentrations);

array[1..compounds] of string: compound_names;

%% Combinations %%
%% Might go away...
int: combinations; %% combinations of compounds. Used as more compounds.
int: combination_concentrations;
%array [1..combinations] of int: combination_replicates;
array[1..combinations] of string: combination_names;


%% Information about controls %%
int: num_controls;

array [1..num_controls] of int: control_replicates; 
array [1..num_controls] of int: control_concentrations;
int: max_control_concentrations = max(control_concentrations);
array[1..num_controls,1..max_control_concentrations] of string: control_concentration_names;
array[1..num_controls] of string: control_names;


int: blanks = 0;
string: blanks_name = "";

%%% Testing %%%

bool: testing = false;


%% Plate size / number of wells

int: num_rows;
int: num_cols;

int: numrows = floor(num_rows/horizontal_cell_lines);
int: numcols = floor(num_cols/vertical_cell_lines);




%% TODO: add latex indicators for combinations and controls 
%array[1..compound_concentrations] of string: compound_concentration_names;

array[1..compounds,int] of string: compound_concentration_names;

%% FIX ME!
array[1..max_compound_concentrations] of string: compound_concentration_indicators;

array[1..combination_concentrations] of string: combination_concentration_names;



%%% Datafile validation %%%
constraint assert(compounds >= 0,"Invalid datafile: Number of compounds should not be less than zero");
constraint assert(combinations >= 0,"Invalid datafile: Number of compounds should not be less than zero");
constraint assert(num_controls >= 0,"Invalid datafile: Number of controls should not be less than zero");
constraint assert(vertical_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero");
constraint assert(horizontal_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero");
constraint assert(numrows > 0,"Invalid datafile: Number of rows should be larger than zero");
constraint assert(numcols > 0,"Invalid datafile: Number of columns should be larger than zero");
constraint assert(min(compound_replicates) > 0,"Invalid datafile: Number of replicates should be larger than zero");
constraint assert(min(compound_concentrations) > 0,"Invalid datafile: Number of concentrations should be larger than zero");

%% Number of wells needed. Note that plates might not be full
int: total_wells = sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + combinations*combination_concentrations*replicates + sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]); 

int: inner_plate_size = (numcols-2*size_empty_edge)*(numrows-2*size_empty_edge);

%% Number of plates needed
int: numplates = ceil(total_wells/inner_plate_size);         
              
%% Detecting some unsatisfiable cases                                                                      
%% FIX ME!
%% FIX ME!!
constraint assert(ceil(sum(compound_replicates)/numplates)*max_compound_concentrations + floor(combinations*replicates/numplates)*combination_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds, combinations, and controls equally accross the plates. (E01) - warning! validation needs to be fixed!");  

%% FIX ME!!
%% FIX ME!!
constraint assert(floor(sum(compound_replicates)/numplates)*max_compound_concentrations + ceil(combinations*replicates/numplates)*combination_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds, combinations, and controls equally accross the plates. (E02) - warning! validation needs to be fixed!!");                               
                            
                            
set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: Plates = 1..numplates;


%% All types of experiments (excluding controls): compounds and combinations
int: experiments = compounds*max_compound_concentrations*replicates + combinations*combination_concentrations*replicates;  

int: compound_ids = compounds*max_compound_concentrations*replicates;  % number of compounds 

%% To be used in redundant constraints
int: emptywells = numplates*numcols*numrows - total_wells;   


%% To be used as in redundant constraints: empty wells other than those in the outermost border
%% Also used as a validation/warning
int: inner_emptywells = numplates*inner_plate_size - total_wells; 

constraint assert(inner_emptywells == 0 \/ allow_empty_wells,"Warning: There are \(inner_emptywells) empty wells. You can allow them or modify other quantities to fill them up.");

constraint assert(inner_emptywells >= 0,"Model ERROR! Inner empty wells is negative. This should never happen!");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plates (our solution!!) %%%%%%%%%%%
% +1 = blanks
%% FIX ME!!
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
constraint forall(b in 0..size_empty_edge-1) (forall(i in Plates, j in {1+b,numrows-b}, k in Columns)(plates[i,j,k] = 0));


%% Edge effect: Leave left-most and right-most columns of every plate empty
constraint forall(b in 0..size_empty_edge-1) (forall(i in Plates, j in Rows, k in {1+b,numcols-b})(plates[i,j,k] = 0));


%% Exactly the total number of empty wells (zeros) and controls (experiments+i)
% merged with another constraint. See below.
%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[0]++[experiments+i | i in 1..num_controls*control_concentrations],[emptywells]++[control_replicates[((i-1) mod num_controls)+1] | i in 1..num_controls*control_concentrations]);

%% Exactly the number of replicates for each compound*compound_concentrations and combination*combination_concentrations
%% merged with another constraint. See below.
%% Exactly one (right now).
%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates), [ i | i in 1..experiments] , [ 1 | i in 1..experiments]);

%% FIX ME!!
%% FIX ME!!
%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[ i | i in 0..experiments + num_controls*max_control_concentrations + 1],[emptywells]++[ 1 | i in 1..experiments]++[control_replicates[((i-1) mod num_controls)+1] | i in 1..num_controls*max_control_concentrations]++[blanks]);

%constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[ i | i in 0..experiments + num_controls*max_control_concentrations + 1],[emptywells]++[ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[floor(((i-1)/(replicates*max_compound_concentrations)))+1] | i in 1..compound_ids]++[1 | i in compound_ids+1..experiments]++[control_replicates[((i-1) mod num_controls)+1] | i in 1..num_controls*max_control_concentrations]++[blanks]);

constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[0]++[ experiments+i | i in 1..num_controls*max_control_concentrations],[emptywells]++[control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..num_controls*max_control_concentrations]);

constraint global_cardinality(array1d(1..numplates*numcols*numrows, plates),[ i | i in 1..compound_ids], [(floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1] | i in 1..compound_ids]);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channelling constraints: an experiment/compound is located at a given plate
%constraint forall(l in 1..experiments, i in Plates)(experiment_plate[l] == i <-> count_eq([array1d(1..numplates*numcols*numrows,plates)[p] | p in (i-1)*numcols*numrows+1..i*numcols*numrows],l,1));

%constraint forall(l in 1..experiments, i in Plates)(count_eq([array1d(1..numplates*numcols*numrows,plates)[p] | p in (i-1)*numcols*numrows+1..i*numcols*numrows],l,(experiment_plate[l] == i)));

constraint forall(l in 1..experiments, i in Plates)(count_eq([plates[i,j,k] | j in Rows, k in Columns],l,(experiment_plate[l] == i)));

%% This constraint does not hold now that not all numbers for experiments must appear
%constraint forall(l in 1..experiments, i in Plates)(count_eq([array1d(1..numplates*numcols*numrows,plates)[p] | p in 1..numplates*numcols*numrows where
 %(p < (i-1)*numcols*numrows+1 \/ p > i*numcols*numrows)],l,(experiment_plate[l] != i)));


%% For experiments that do not exist = plate 0 (redundant)
constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*numcols*numrows, plates),l,0) <-> (experiment_plate[l] == 0));
constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*numcols*numrows, plates),l,0) <-> (experiment_row[l] == 0));
constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*numcols*numrows, plates),l,0) <-> (experiment_column[l] == 0));

%% There might be a stronger version of these constraints...
constraint forall(l in 1..experiments, i in Plates, j in Rows, k in Columns)((experiment_row[l] == j) <- (plates[i,j,k] == l));

constraint forall(l in 1..experiments, i in Plates, j in Rows, k in Columns)((experiment_column[l] == k) <- (plates[i,j,k] == l));

constraint forall(l in 1..experiments, j in Rows)(count_eq([plates[i,j,k] | i in Plates, k in Columns],l,(experiment_row[l] == j)));

%constraint forall(l in 1..experiments, j in Rows)(experiment_row[l] != j <-> count_eq([plates[experiment_plate[l],j,k] | k in Columns],l,0));

%constraint forall(l in 1..experiments, k in Columns)(experiment_column[l] == k <-> count_eq([plates[experiment_plate[l],j,k] | j in Rows],l,1));

%constraint forall(l in 1..experiments, k in Columns)(count_eq([plates[experiment_plate[l],j,k] | j in Rows],l,(experiment_column[l] == k)));

constraint forall(l in 1..experiments, k in Columns)(count_eq([plates[i,j,k] | i in Plates, j in Rows],l,(experiment_column[l] == k)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A compound with all concentrations must appear on a single plate

%% CHANGED
%% Bring back because it's super important! But it's so weird right now! See next version!
%constraint forall(l in 1..compound_ids where ((l mod max_compound_concentrations) == 1))(all_equal([experiment_plate[i] | i in l..(l+compound_concentrations[floor(((l-1)/(replicates*max_compound_concentrations)))+1]-1)]));

constraint forall(l in 1..compound_ids where ((l mod max_compound_concentrations) == 1))(all_equal([experiment_plate[i] | i in l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)]));

%% A combination with all concentrations must appear on a single plate
constraint forall(l in (compound_ids+1)..experiments where ((l-compound_ids) mod combination_concentrations == 1))(all_equal([experiment_plate[i] | i in l..(l+combination_concentrations-1)]));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Spreading concentrations of a compound across different rows and columns 
%%% (consider balancing the number of rows and columns!)
% Consider making it different on different plates


int: min_rows_combination = min(combination_concentrations,numrows-2*size_empty_edge);
int: min_columns_combination = min(combination_concentrations,numcols-2*size_empty_edge);

% True but possibly rendundant
%constraint forall(l in 1..experiments) (plates[experiment_plate[l],experiment_row[l],experiment_column[l]] == l);

% Might be redundant != implied
%%KEEP%% constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_plate[plates[i,j,k]]==i);
constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_row[plates[i,j,k]]==j);
%%KEEP%% constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_column[plates[i,j,k]]==k);


% Spread the compounds across different rows and columns
%% FIX ME! (twice) -- fixed??
%constraint if concentrations_on_different_rows then forall(l in 0..(compounds*replicates-1)) (nvalue(min(compound_concentrations[floor(l/replicates)+1],numrows-2*size_empty_edge), [experiment_row[l*max_compound_concentrations+i]| i in 1..compound_concentrations[floor(l/replicates)+1]])) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],numrows-2*size_empty_edge), [experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

%% FIX ME! (twice) -- fixed??
%constraint if concentrations_on_different_columns then forall(l in 0..(compounds*replicates-1)) (nvalue(min(compound_concentrations[floor(l/replicates)+1],numcols-2*size_empty_edge), [experiment_column[l*max_compound_concentrations+i]| i in 1..compound_concentrations[floor(l/replicates)+1]])) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],numcols-2*size_empty_edge), [experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

% Spread the combinations across different rows and columns
constraint forall(l in 0..combinations*replicates-1) (nvalue(min_rows_combination, [experiment_row[compound_ids+l*combination_concentrations+i]| i in 1..combination_concentrations])); 

constraint forall(l in 0..combinations*replicates-1) (nvalue(min_columns_combination, [experiment_column[compound_ids+l*combination_concentrations+i]| i in 1.. combination_concentrations])); 


%% Implied constraints 
%% These constraints are too weak when the number of plates increases


%constraint global_cardinality_low_up(experiment_row, [i | i in 2..(numrows-1)], [0| i in 2..(numrows-1)], [numplates*(numcols-2)| i in 2..(numrows-1)]);


array [Rows] of var 0..numplates*numcols: controls_in_row;
array [Columns] of var 0..numplates*numrows: controls_in_column;

constraint (sum(controls_in_row) == sum([control_replicates[i]*control_concentrations[i] | i in 1..num_controls]) + emptywells);
constraint (sum(controls_in_column) == sum([control_replicates[i]*control_concentrations[i] | i in 1..num_controls]) + emptywells);

%Among: Requires exactly n variables in x to take one of the values in v .
%% FIX ME!!
constraint forall(j in Rows) (among(controls_in_row[j], [plates[i,j,k] | i in Plates, k in Columns], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

%% FIX ME!!
constraint forall(k in Columns) (among(controls_in_column[k], [plates[i,j,k] | i in Plates, j in Rows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));



%constraint global_cardinality_closed(experiment_row, [j | j in 2..(numrows-1)],
%[numplates*(numcols-2)-among([plates[i,j,k] | i in Plates, k in 2..(numcols-1)], (experiments+1..experiments+num_controls*control_concentrations))| j in 2..(numrows-1)]);

%% Implied constraint: how many experiments can be in a row/column? It's too weak and needs to be done per plate.
%% HINT! I also know how many should be 0!!!! ADD IT! FIX ME! CHECK ME!
constraint global_cardinality(experiment_row, [j | j in Rows], [numplates*(numcols)-controls_in_row[j]| j in Rows]);
constraint global_cardinality(experiment_column, [k | k in Columns], [numplates*(numrows)-controls_in_column[k]| k in Columns]);



%% Implied constraint: how many experiments can there be in a row? Per plate
% I think this still works...
constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])<= numcols-among([plates[i,j,k] | k in Columns], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])<= numrows-among([plates[i,j,k] | j in Rows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));


%% Implied constraint:

                                                                                      
%%KEEP%% constraint global_cardinality_low_up_closed(experiment_column, [i | i in 2..numcols-1], [0| i in 2..numcols-1], [numplates*(numrows-2)| i in 2..numcols-1]);                                         


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Different compound replicas must appear on different plates (if possible!)
int: min_plates = min(numplates,replicates);


%% FIX ME! (twice)
%% MODIFIED!
constraint forall(l in 1..compounds) (nvalue(min(numplates,compound_replicates[l]), [experiment_plate[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])); 


%% Different combination replicas must appear on different plates (if possible!)
constraint forall(l in 0..(combinations-1)) (nvalue(min_plates, [experiment_plate[compound_ids + l*combination_concentrations + i*combinations*combination_concentrations + 1] | i in 0..(replicates-1)])); 


%%% Put empty wells in the corners (as much as possible)
int: num_empty_corners = min(4*numplates,inner_emptywells);

constraint count_eq([plates[i,j,k] | i in Plates, j in {1+size_empty_edge,numrows-size_empty_edge}, k in {1+size_empty_edge,numcols-size_empty_edge}],0,num_empty_corners);


%%% If there are still more empty wells, place them near the corners
int: lower_2 = min(2*numplates, inner_emptywells - num_empty_corners);

constraint count_eq([plates[i,j,k] | i in Plates, j in {1+size_empty_edge,numrows-size_empty_edge}, k in {2+size_empty_edge}],0,lower_2);


%%% If there are still more empty wells, place them near the corners (as much as possible)
%% CHEAT! I need to put the empty wells in order. It improves propagation A LOT!
%% On the other hand, we're not expecting many empty wells (if any).
int: lower_3 = min(2*card({i | i in {numplates-1,numplates} where (i>0)}), inner_emptywells - num_empty_corners - lower_2);
constraint count_eq([plates[i,j,k] | i in {numplates-1,numplates} where (i>0), j in {size_empty_edge+1,numrows-size_empty_edge}, k in {numcols-size_empty_edge-1}],0,lower_3);


%% If there are still empty wells, make the outer columns empty.
%int: empty_wells_in_columns = min(2*numplates*(numrows-4), inner_emptywells-num_empty_corners);
%constraint count_eq([plates[i,j,k] | i in Plates, j in 3..numrows-2, k in {2,numcols-1}],0,empty_wells_in_columns);


%% If there are still empty wells, make the outer columns empty.
%int: empty_wells_in_rows = min(2*numplates*(numcols-4), inner_emptywells-num_empty_corners-empty_wells_in_columns);
%constraint count_eq([plates[i,j,k] | i in Plates, j in {2,numrows-1}, k in 3..numcols-2],0,empty_wells_in_rows);


%% Balancing compounds between plates: distribute the compounds equitatively among all the plates


%% TODO: could this lower bound be stronger?
%% FIX ME!
%int: at_least = max_compound_concentrations*floor((compounds*replicates)/numplates) + combination_concentrations*floor((combinations*replicates)/numplates); 

int: at_least = min(compound_concentrations)*floor(sum(compound_replicates)/numplates) + combination_concentrations*floor((combinations*replicates)/numplates); 

%% FIX ME!
%% FIX ME!!
int: at_most = min(max_compound_concentrations*ceil(sum(compound_replicates)/numplates) + combination_concentrations*ceil((combinations*replicates)/numplates), 
    inner_plate_size -(sum([floor(control_replicates[i]*control_concentrations[i]/numplates) | i in 1..num_controls])));



%% This implied constraint is very important for propagation!

constraint global_cardinality_low_up(experiment_plate,[ i | i in Plates],[ at_least | i in Plates],[ at_most | i in Plates]);


% Balancing compounds between plates (and knowing that there must be some sort of balance!)
%% FIX ME!
int: at_least_compounds = min(compound_concentrations)*floor(sum(compound_replicates)/numplates);
int: at_most_compounds = max_compound_concentrations*ceil(sum(compound_replicates)/numplates);

constraint global_cardinality_low_up([experiment_plate[i] | i in 1..compound_ids] ,[ i | i in Plates],[ at_least_compounds | i in Plates],[ at_most_compounds | i in Plates]);


% Balancing combinations between plates (and knowing that there must be some sort of balance!)
int: at_least_combinations = combination_concentrations*floor((combinations*replicates)/numplates); 
int: at_most_combinations = combination_concentrations*ceil((combinations*replicates)/numplates);

constraint global_cardinality_low_up([experiment_plate[i] | i in compound_ids+1..experiments] ,[ i | i in Plates],[ at_least_combinations | i in Plates],[ at_most_combinations | i in Plates]);







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Balancing controls between plates.
%% Asumes controls can be balanced this way, that is, that the designer expects about the same number of controls on each plate and not a plate full of controls.
%% FIX ME!!
%array[int] of float: controls_per_plate = [control_replicates[((i-1) mod num_controls)+1]/numplates | i in 1..num_controls*max_control_concentrations];

array[int] of float: controls_per_plate = [control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])/numplates | i in 1..num_controls*max_control_concentrations];

array[int] of int: min_controls_per_plate = [floor(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

array[int] of int: max_controls_per_plate = [ceil(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

constraint forall(i in Plates)(global_cardinality_low_up([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], min_controls_per_plate, max_controls_per_plate));




%[control_replicates[((i-1) mod num_controls)+1] | i in 1..num_controls*control_concentrations]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Distributing controls

%% Controls of the same type are not located next to each other
%% TODO: relax this constraint when there are too many controls in the plate
%% Strong version
constraint forall(c in 1..num_controls*max_control_concentrations) (
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

% j=1
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates, k in 2..numcols-1)(plates[i,1,k] == experiments+c -> 
           plates[i,1,k-1]  != experiments+c /\
           plates[i,1,k+1]  != experiments+c /\
           plates[i,2,k-1]!= experiments+c /\
           plates[i,2,k]  != experiments+c /\
           plates[i,2,k+1]!= experiments+c)
           );

% j = numrows
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates, k in 2..numcols-1)(plates[i,numrows,k] == experiments+c -> 
           plates[i,numrows-1,k-1]!= experiments+c /\ 
           plates[i,numrows-1,k]  != experiments+c /\
           plates[i,numrows-1,k+1]!= experiments+c /\
           plates[i,numrows,k-1]  != experiments+c /\
           plates[i,numrows,k+1]  != experiments+c)
           );
           
% First column: k = 1           
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates, j in 2..numrows-1)(plates[i,j,1] == experiments+c -> 
           plates[i,j-1,1]  != experiments+c /\
           plates[i,j-1,2]!= experiments+c /\
           plates[i,j,2]  != experiments+c /\
           plates[i,j+1,1]  != experiments+c /\
           plates[i,j+1,2]!= experiments+c)
           );

% Last column: k = numcols
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates, j in 2..numrows-1)(plates[i,j,numcols] == experiments+c -> 
           plates[i,j-1,numcols-1]!= experiments+c /\ 
           plates[i,j-1,numcols]  != experiments+c /\
           plates[i,j,numcols-1]  != experiments+c /\
           plates[i,j+1,numcols-1]!= experiments+c /\
           plates[i,j+1,numcols]  != experiments+c)
           );            
  
  
% Upper left corner
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates, k in 2..numcols-1)(plates[i,1,1] == experiments+c -> 
           plates[i,1,2]  != experiments+c /\
           plates[i,2,1]  != experiments+c /\
           plates[i,2,2]!= experiments+c)
           );
 
% Lower left corder: j = numrows
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates)(plates[i,numrows,1] == experiments+c -> 
           plates[i,numrows-1,1]  != experiments+c /\
           plates[i,numrows-1,2]!= experiments+c /\
           plates[i,numrows,2]  != experiments+c)
           );
          
% Upper right corner: j=1 k = numcols
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates)(plates[i,1,numcols] == experiments+c -> 
           plates[i,1,numcols-1]  != experiments+c /\
           plates[i,2,numcols-1]!= experiments+c /\
           plates[i,2,numcols]  != experiments+c)
           );
            
% Lower right corner: j = numrows k = numcols
constraint forall(c in 1..num_controls*max_control_concentrations) (
forall(i in Plates)(plates[i,numrows,numcols] == experiments+c -> 
           plates[i,numrows-1,numcols-1]!= experiments+c /\ 
           plates[i,numrows-1,numcols]  != experiments+c /\
           plates[i,numrows,numcols-1]  != experiments+c)
           );
                                               
%% Softer version
%constraint forall(c in 1..num_controls) (
%forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] == experiments+c -> 
 %          plates[i,j-1,k]  != experiments+c /\
  %         plates[i,j,k-1]  != experiments+c /\
   %        plates[i,j,k+1]  != experiments+c /\
    %       plates[i,j+1,k]  != experiments+c)
     %      );
           
           
           
%%%% Find a solution! %%%%

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 

solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(plates, first_fail, indomain_random)]) satisfy; 

%solve satisfy;



%%KEEP%% solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(experiment_column, random, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 


%solve::int_search(experiment_plate, first_fail, indomain_random) satisfy; 

%solve::int_search(plates, random, indomain_max) satisfy; 



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Pretty printing %%%

output [if testing then "\(numplates) plates\n" else "" endif];

% output [if (inner_emptywells>0) then "warning: there are \(inner_emptywells) empty wells. Consider using them as controls.\n\n" else "" endif];

% output ["Plate 1:\n"];
         
% %%TODO: change output so symbols are not hardcoded
% output [if fix(plates[i,j,k]) == 0 then "  ." 
%         elseif fix(plates[i,j,k]) == (experiments+1)  then "  +" %% Positive controls
%         elseif fix(plates[i,j,k]) == (experiments+2)  then "  -" %% Negative controls
%         elseif fix(plates[i,j,k]) == (experiments+3)  then "  x" %% Blank controls
%         elseif fix(plates[i,j,k]) == (experiments+4)  then "  ~" %% DMSO controls
%         elseif fix(plates[i,j,k]) == (experiments+5)  then "  *" %% other
%         elseif fix(plates[i,j,k]) == (experiments+6)  then "  #" %% other
%         elseif fix(plates[i,j,k]) == (experiments+7)  then "  &" %% other
%         elseif fix(plates[i,j,k]) == (experiments+8)  then "  @" %% other
%         elseif fix(plates[i,j,k]) == (experiments+9)  then "  <" %% other
%         elseif fix(plates[i,j,k]) == (experiments+10) then "  >" %% other
%         elseif fix(plates[i,j,k]) == (experiments+11) then "  $" %% other
%         else show_int(3,plates[i,j,k]) endif ++
%         %% Adding lines and headings %%
%         if j== numrows /\ k == numcols /\ i<numplates then "\n\n Plate \(i+1):\n" 
%         elseif k == numcols then "\n" else " " endif |
%         i in Plates, j in Rows, k in Columns];        
         
%  output ["Compound's plate:\n", show(experiment_plate)];

% output ["Compound's plate:\n", show(compound_concentration_names)];    
     
% output ["\nExperiment's row:\n", show(experiment_row)];

% output ["\nExperiment's column:\n", show(experiment_column)];

% output ["\nControls per row:\n", show(controls_in_row)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CSV Ouput %%%

array[int] of string: letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
         
output ["\nplateID,well,cmpdname,CONCuM,cmpdnum,VOLuL\n"];

output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[j+numrows*(h-1)] ++ if (k+numcols*(v-1)) < 10 then "0" else "" endif ++ "\(k+numcols*(v-1))," ++ 

%, v in 1..vertical_cell_lines

%%%% Compounds %%%%
%% FIX ME!!

if fix(plates[i,j,k]) <= compound_ids then compound_names[(floor((fix(plates[i,j,k])-1)/max_compound_concentrations) mod compounds)+1] 
++ "," ++ compound_concentration_names[(floor((fix(plates[i,j,k])-1)/max_compound_concentrations) mod compounds)+1,(((fix(plates[i,j,k])-1)) mod max_compound_concentrations)+1] ++ "," ++

%% Latex name of the compound
%% FIX! WHY do I have a floor here with mod?
%compound_concentration_indicators[((floor((fix(plates[i,j,k])-1))) mod compound_concentrations)+1] ++ "{" ++ compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*compound_concentrations)))+1] ++ "}"

%% FIX ME! (twice) 
compound_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1] ++ "_" ++ compound_concentration_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1,((floor((fix(plates[i,j,k])-1))) mod max_compound_concentrations)+1]

elseif fix(plates[i,j,k]) <= experiments then %"comb, comb" ++
combination_names[floor(((fix(plates[i,j,k])-compound_ids-1)/(replicates*combination_concentrations)))+1] ++
"," ++ combination_concentration_names[(((fix(plates[i,j,k])-compound_ids-1)) mod combination_concentrations)+1] ++ "," ++


%% Latex name of the combination
%compound_concentration_indicators[((floor((fix(plates[i,j,k])-1))) mod compound_concentrations)+1] ++ "{" ++ compound_names[floor(((fix(plates[i,j,k])-1)/(replicates*compound_concentrations)))+1] ++ "}"

%"{"++combination_names[floor(((fix(plates[i,j,k])-compound_ids-1)/(replicates*combination_concentrations)))+1]++"}"

combination_names[floor(((fix(plates[i,j,k])-compound_ids-1)/(replicates*combination_concentrations)))+1]++"_"++combination_concentration_names[(((fix(plates[i,j,k])-compound_ids-1)) mod combination_concentrations)+1]

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
         

       