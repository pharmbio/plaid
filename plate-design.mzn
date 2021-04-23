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
% Last Revision: April 2021


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
%% Might go away...
int: combinations; %% combinations of compounds. Used as more compounds.
int: combination_concentrations;
%array [1..combinations] of int: combination_replicates;
array[1..combinations] of string: combination_names;


%% Information about controls %%
int: num_controls;

array [1..num_controls] of int: control_replicates; 
array [1..num_controls] of int: control_concentrations;
int: max_control_concentrations = max(control_concentrations++[0]);
array[1..num_controls,1..max_control_concentrations] of string: control_concentration_names;
array[1..num_controls] of string: control_names;


%%% Potentially new parameters
bool: balance_controls_inside_plate = true;
bool: interconnected_plates = false; %Makes the problem much harder by connecting the plates
int: control_slack = 1; %Makes the problem harder/easier by constraining how equally should we distribute controls
bool: spread_controls = false;


%%% Testing %%%

bool: testing = true;
bool: debugging = false;

bool: temp_bool = true; 

bool: ok_bool = true; %This ones actually help a lot!

%bool: fixing_bool = false; %%Not too bad

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
constraint assert(compounds >= 0,"Invalid datafile: Number of compounds should not be less than zero.");
constraint assert(combinations >= 0,"Invalid datafile: Number of compounds should not be less than zero.");
constraint assert(num_controls >= 0,"Invalid datafile: Number of controls should not be less than zero.");
constraint assert(vertical_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero.");
constraint assert(horizontal_cell_lines > 0,"Invalid datafile: Number of cell lines should be larger than zero.");
constraint assert(numrows > 0,"Invalid datafile: Number of rows should be larger than zero.");
constraint assert(numcols > 0,"Invalid datafile: Number of columns should be larger than zero.");
constraint assert(compounds==0 \/ min(compound_replicates) > 0,"Invalid datafile: Number of replicates should be larger than zero.");
constraint assert(compounds==0 \/ min(compound_concentrations) > 0,"Invalid datafile: Number of concentrations should be larger than zero.");
constraint assert((replicates_on_different_plates /\ replicates_on_same_plate) == false,"Invalid datafile: replicates cannot be both on the same plate and on different plates");



%% Number of wells needed. Note that plates might not be full
int: total_wells = sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + combinations*combination_concentrations*replicates + sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]); 

constraint assert(total_wells > 0, "Invalid datafile: the plates cannot be completely empty.");

int: inner_plate_size = (numcols-2*size_empty_edge)*(numrows-2*size_empty_edge);

constraint assert(min(compound_concentrations) <= inner_plate_size, "Invalid datafile: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");

%% Number of plates needed
%% max is used to avoid division-by-zero errors
int: numplates = max(ceil(total_wells/inner_plate_size),1);         
              
force_spread_controls = 2*sum([control_concentrations[i]*control_replicates[i] | i in 1..num_controls]) < inner_plate_size*numplates;

%% Detecting some unsatisfiable cases.
%% Conditions E01 and E02 need to be updated or the model needs to be more flexible.                                                                      
%% FIX ME!!
%constraint assert(ceil(sum(compound_replicates)/numplates)*max_compound_concentrations + floor(combinations*replicates/numplates)*combination_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds, combinations, and controls equally accross the plates. (E01) - warning! validation needs to be fixed!");  

%% FIX ME!!
%constraint assert(floor(sum(compound_replicates)/numplates)*max_compound_concentrations + ceil(combinations*replicates/numplates)*combination_concentrations + sum([floor(control_concentrations[i]*control_replicates[i]/numplates) | i in 1..num_controls]) <= inner_plate_size, "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds, combinations, and controls equally accross the plates. (E02) - warning! validation needs to be fixed!!");                               
                            
                            
set of int: Rows = 1..numrows;
set of int: Columns = 1..numcols;
set of int: Plates = 1..numplates;


%% All types of experiments (excluding controls): compounds and combinations
int: experiments = compounds*max_compound_concentrations*replicates + combinations*combination_concentrations*replicates;  

int: compound_ids = compounds*max_compound_concentrations*replicates;  % number of compounds 

%% Used in redundant constraints
int: emptywells = numplates*numcols*numrows - total_wells;   


%% To be used as in redundant constraints: empty wells other than those in the outermost border
%% Also used as a validation/warning
int: inner_emptywells = numplates*inner_plate_size - total_wells; 


constraint assert(inner_emptywells == 0 \/ allow_empty_wells,"Warning: There are \(inner_emptywells) empty wells. You can allow the plates to have inner empty wells or modify other quantities to fill them up.");

constraint assert(inner_emptywells >= 0,"Model ERROR! Inner empty wells is negative. This should never happen!");


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



constraint global_cardinality(plates,[0]++[ i | i in 1..compound_ids]++[i | i in compound_ids+1..experiments]++[ experiments+i | i in 1..num_controls*max_control_concentrations],[emptywells]++[(floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1] | i in 1..compound_ids]++[1|i in compound_ids+1..experiments]++[control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..num_controls*max_control_concentrations]);

%% Experiments that do not exists are not located in any plate, row, or column.
constraint forall(i in 1..compound_ids)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) == false <-> (experiment_plate[i] = 0 /\ experiment_row[i] = 0 /\ experiment_column[i] = 0));

constraint forall(i in 1..compound_ids)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) <-> (experiment_plate[i] != 0 /\ experiment_row[i] != 0 /\ experiment_column[i] != 0));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channelling constraints: an experiment/compound is located at a given plate

constraint forall(l in 1..experiments, i in Plates)(count_eq([plates[i,j,k] | j in Rows, k in Columns],l,(experiment_plate[l] == i)));

constraint forall(l in 1..experiments)(count_eq([plates[i,j,k] | i in Plates, j in Rows, k in Columns],l,(experiment_plate[l] != 0)));


%% For experiments that do not exist = plate 0 (redundant)
%constraint forall(l in 1..experiments)(
 % count_eq(array1d(1..numplates*numcols*numrows, plates),l,0) <-> (experiment_plate[l] = 0 /\ experiment_row[l] = 0 /\ experiment_column[l] = 0)
%);

%constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*numcols*numrows, plates),l,0) <-> (experiment_row[l] == 0));
%constraint forall(l in 1..experiments)(count_eq(array1d(1..numplates*numcols*numrows, plates),l,0) <-> (experiment_column[l] == 0));





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


% Might be redundant != implied
%constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_plate[plates[i,j,k]]==i);
%constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_row[plates[i,j,k]]==j);
%constraint forall(i in Plates, j in Rows, k in Columns) ((0 < plates[i,j,k] /\ plates[i,j,k] <= experiments) <-> experiment_column[plates[i,j,k]]==k);


% Spread the compounds across different rows and columns

%% Spreads compounds over different rows. It's been replaced by the more restrictive alldifferent version below.
%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],numrows-2*size_empty_edge), [experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
alldifferent([experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numrows-2*size_empty_edge)]):: domain
))) endif; 



% Split concentrations between upper and lower halves of the plate

array [1..compounds*replicates,Vertical] of var 0..max_compound_concentrations: ul_concentrations;

%% For each replicate of each compound, count how many concentrations are in the upper/lower parts of the plate
%% Definition of ul_concentrations (upper)  

constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,upper],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..floor(numrows/2))
))) endif; 

constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,upper] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,upper] >= floor(compound_concentrations[l]/2)
))) endif; 


%% Definition of ul_concentrations (lower)
constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], floor(numrows/2)+1..numrows)
))) endif; 

constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,lower] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,lower] >= floor(compound_concentrations[l]/2)
))) endif; 


%% Implied constraints:
constraint if concentrations_on_different_rows /\ ok_bool then (sum(ul_concentrations)=sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) endif;

constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 ul_concentrations[r*compounds + l,upper] + ul_concentrations[r*compounds + l,lower] == compound_concentrations[l]
))) endif; 

constraint if concentrations_on_different_rows /\ ok_bool then (sum(ul_concentrations[..,lower]) == sum(ul_half_plates[..,lower]) ) endif;

constraint if concentrations_on_different_rows /\ ok_bool then (sum(ul_concentrations[..,upper]) == sum(ul_half_plates[..,upper]) ) endif;

constraint if concentrations_on_different_rows /\ ok_bool then forall(i in Plates) (sum(c in 0..compounds*replicates-1)(ul_concentrations[c+1,lower]*(experiment_plate[c*max_compound_concentrations+1]==i)) == ul_half_plates[i,lower] ) endif;

constraint if concentrations_on_different_rows /\ ok_bool then forall(i in Plates) (sum(c in 0..compounds*replicates-1)(ul_concentrations[c+1,upper]*(experiment_plate[c*max_compound_concentrations+1]==i)) == ul_half_plates[i,upper] ) endif;


%% Defining ul_concentrations in terms of plates
constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,upper], plates[..,1..floor(numrows/2),..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]}, )
))) endif; 

constraint if concentrations_on_different_rows /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower], plates[..,floor(numrows/2)+1..numrows,..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]}, )
))) endif; 


array [Plates,Rows] of var 0..numplates*numcols: experiments_in_plate_row;

%% Definition of experiments_in_plate_row
constraint forall(i in Plates, j in Rows)(count(experiment_plate_row[i,..],j,experiments_in_plate_row[i,j]));

constraint forall(i in Plates, j in Rows)(among(experiments_in_plate_row[i,j],plates[i,j,..],1..experiments));

constraint forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j] + controls_in_plate_row[i,j] == numcols);

constraint if concentrations_on_different_rows /\ (max_compound_concentrations >= (numrows-2*size_empty_edge)) then forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge, r in 1..compounds*replicates)(
count(experiment_plate_row[i,1..experiments-max_compound_concentrations*r],j) + count(experiment_plate_row[i,experiments-max_compound_concentrations*r+1..experiments],j) == experiments_in_plate_row[i,j]
) endif;

%% Good!
constraint if concentrations_on_different_rows /\ (max_compound_concentrations >= (numrows-2*size_empty_edge)) then forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge, r in 1..compounds*replicates-1)(
if(compound_concentrations[((r-1) mod compounds)+1]>=(numrows-2*size_empty_edge)) then
(count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + count(experiment_plate_row[i,max_compound_concentrations*(r+1)+1..experiments],j) + (experiment_plate_row[i,max_compound_concentrations*r+1]>0)) <= experiments_in_plate_row[i,j]
endif
) endif;

% Implied constraint: when there are more concentrations than rows, help the search to not fill in a row completely too soon by saving space in each row for the other compounds
constraint if concentrations_on_different_rows /\ (max_compound_concentrations >= (numrows-2*size_empty_edge)) then forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge, r in 1..(compounds*replicates-1))((count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + sum([ experiment_plate_row[i,max_compound_concentrations*(k-1)+1]>0 /\ (compound_concentrations[((k-1) mod compounds)+1]>=(numrows-2*size_empty_edge)) | k in (r+1) .. compounds*replicates])) <= experiments_in_plate_row[i,j]) endif;



%%%%%%
%% Spreads concentrations over different columns. It's been replaced by the more restrictive alldifferent version below.
%constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],numcols-2*size_empty_edge), [experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) ( 
alldifferent( [experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],numcols-2*size_empty_edge)]):: domain
))) endif; 

% Split concentrations between left-most and right-most halves of the plate

array [1..compounds*replicates,Horizontal] of var 0..max_compound_concentrations: lr_concentrations;

%% For each replicate of each compound, count how many concentrations there are in the left/right halves of the plate
%% Definition of lr_concentrations (left)  
constraint if concentrations_on_different_columns /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,left],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..floor(numcols/2))
))) endif; 

constraint if concentrations_on_different_columns /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,left] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_columns /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,left] >= floor(compound_concentrations[l]/2)
))) endif; 


%% Definition of lr_concentrations (right)  
constraint if concentrations_on_different_columns /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,right],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], floor(numcols/2)+1..numcols)
))) endif; 

constraint if concentrations_on_different_columns /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,right] <= ceil(compound_concentrations[l]/2)
))) endif; 

constraint if concentrations_on_different_columns /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,right] >= floor(compound_concentrations[l]/2)
))) endif; 

%% Implied constraints:
constraint if concentrations_on_different_columns /\ ok_bool then (sum(lr_concentrations) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])) endif;

constraint if concentrations_on_different_columns /\ ok_bool then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
 lr_concentrations[r*compounds + l,left] + lr_concentrations[r*compounds + l,right] == compound_concentrations[l]
))) endif; 


%% Implied constraints 
%% These constraints are too weak when the number of plates increases
%% Right now it looks like it's mostly taking time 

constraint global_cardinality_low_up(experiment_row, [i | i in 1+size_empty_edge..(numrows-size_empty_edge)], [0| i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*(numcols-2*size_empty_edge)| i in 1+size_empty_edge..(numrows-size_empty_edge)]);

array [Plates,Rows] of var 0..numplates*numcols: controls_in_plate_row;
array [Plates,Columns] of var 0..numplates*numcols: controls_in_plate_column;
array [Rows] of var 0..numplates*numcols: controls_in_row;
array [Columns] of var 0..numplates*numrows: controls_in_column;

constraint (sum(controls_in_plate_row) == sum([control_replicates[i]*control_concentrations[i] | i in 1..num_controls]) + emptywells);
constraint (sum(controls_in_row) == sum([control_replicates[i]*control_concentrations[i] | i in 1..num_controls]) + emptywells);
constraint (sum(controls_in_column) == sum([control_replicates[i]*control_concentrations[i] | i in 1..num_controls]) + emptywells);


%Among: Requires exactly n variables in x to take one of the values in v .
constraint forall(i in Plates, j in Rows) (among(controls_in_plate_row[i,j], plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(j in Rows) (among(controls_in_row[j], plates[..,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(k in Columns) (among(controls_in_column[k], plates[..,..,k], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(j in Rows) (controls_in_row[j] == sum(controls_in_plate_row[..,j]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,upper,..]) + count(plates[i,1..floor(numrows/2),..],0) == sum(controls_in_plate_row[i,1..floor(numrows/2)]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,lower,..]) + count(plates[i,floor(numrows/2)+1..numrows,..],0) == sum(controls_in_plate_row[i,floor(numrows/2)+1..numrows]));

%constraint global_cardinality_closed(experiment_row, [j | j in 2..(numrows-1)],
%[numplates*(numcols-2)-among([plates[i,j,k] | i in Plates, k in 2..(numcols-1)], (experiments+1..experiments+num_controls*control_concentrations))| j in 2..(numrows-1)]);

%% Implied constraint: how many experiments can be in a row/column? It's too weak and needs to be done per plate.
constraint global_cardinality(experiment_row, [j | j in Rows], [numplates*(numcols)-controls_in_row[j]| j in Rows]);
constraint global_cardinality(experiment_column, [k | k in Columns], [numplates*(numrows)-controls_in_column[k]| k in Columns]);




%% Implied constraint: how many experiments can there be in a row? Per plate
% I think this still works...
%% TODO: Redundant
constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])<= numcols-among(plates[i,j,..], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])<= numcols-controls_in_plate_row[i,j]);

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
%constraint forall(l in 0..(combinations-1)) (nvalue(min_plates, [experiment_plate[compound_ids + l*combination_concentrations + i*combinations*combination_concentrations + 1] | i in 0..(replicates-1)])); 




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


int: min_compounds_plate = max(floor((sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + combinations*combination_concentrations*replicates)/inner_plate_size), min(compound_concentrations) );
constraint global_cardinality_low_up(experiment_plate,[ i | i in Plates],[ min_compounds_plate | i in Plates],[ inner_plate_size | i in Plates]);




% Balancing compounds between plates (and knowing that there must be some sort of balance!)
%% FIX ME!
%% I want to remove these constraints but I need to make other stronger first
int: at_least_compounds = min(compound_concentrations)*floor(sum(compound_replicates)/numplates);
int: at_most_compounds = max_compound_concentrations*ceil(sum(compound_replicates)/numplates);

constraint global_cardinality_low_up([experiment_plate[i] | i in 1..compound_ids] ,[ i | i in Plates],[ at_least_compounds | i in Plates],[ at_most_compounds | i in Plates]);




%% Implied constraints 
%% These constraints are too weak when the number of plates increases
%%% CHECK! This constraints came from the another version of the code. Need to check that these are still valid.

%???constraint global_cardinality_low_up(experiment_row, [i | i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*floor(at_least_compounds/(numrows-2*size_empty_edge))| i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*(numcols-2*size_empty_edge)| i in 1+size_empty_edge..(numrows-size_empty_edge)]);


%constraint global_cardinality_low_up(experiment_row, [i | i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*floor(at_least/(numrows-2*size_empty_edge))| i in 1+size_empty_edge..(numrows-size_empty_edge)], [numplates*(numcols-2*size_empty_edge)| i in 1+size_empty_edge..(numrows-size_empty_edge)]);

%constraint global_cardinality_low_up(experiment_column, [i | i in 1+size_empty_edge..(numcols-size_empty_edge)], [numplates*floor(at_least/(numcols-2*size_empty_edge))| i in 1+size_empty_edge..(numcols-size_empty_edge)], [numplates*(numrows-2*size_empty_edge)| i in 1+size_empty_edge..(numcols-size_empty_edge)]);



enum Vertical = {upper,lower};
enum Horizontal = {left,right};


%% TODO: Are there better lower limits?
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


constraint if (((numrows-2*size_empty_edge) mod 2) == 0) then (forall(i in Plates)(abs(ul_half_plates[i,upper]-ul_half_plates[i,lower])<=1)) endif;



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
constraint (sum(ul_half_plates) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + combinations*combination_concentrations*replicates);

constraint (sum(ul_half_plates[..,upper]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + combinations*combination_concentrations*replicates)/2);

constraint (sum(ul_half_plates[..,lower]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) + combinations*combination_concentrations*replicates)/2);



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

array [Plates,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_lr_plates;


constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..floor(numrows/2), k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,upper,d] | d in 1..num_controls*max_control_concentrations]));

constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in floor(numrows/2)+1..numrows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,lower,d] | d in 1..num_controls*max_control_concentrations]));

constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]<=ceil(max_controls_per_plate[d]/2));
constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]>=floor(min_controls_per_plate[d]/2));

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(abs(controls_ul_plates[i,upper,d] - controls_ul_plates[i,lower,d]) <= 1);

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d] + controls_ul_plates[i,lower,d] <= max_controls_per_plate[d]);

constraint forall(i in 1..num_controls*max_control_concentrations)(sum([controls_ul_plates[p,v,i] | v in Vertical, p in Plates]) == control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) );

%% Half of the total number of controls on each upper/lower half of the plate.
constraint if ok_bool then forall(i in Plates)(abs(sum(d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d]) - sum(d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,lower,d])) <= 1) endif;


% %%% NEW %%%
% %% Balancing the total of controls between left and right half-plates

constraint if ok_bool then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in 1..floor(numcols/2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,left,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if ok_bool then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in floor(numcols/2)+1..numcols], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,right,d] | d in 1..num_controls*max_control_concentrations])) endif;


constraint if ok_bool then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]<=ceil(max_controls_per_plate[d]/2)) endif;
constraint if ok_bool then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]>=floor(min_controls_per_plate[d]/2)) endif;

constraint if ok_bool then forall(i in Plates, d in 1..num_controls*max_control_concentrations)(abs(controls_lr_plates[i,left,d] - controls_lr_plates[i,right,d]) <= 1) endif;

constraint if ok_bool then forall(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,left,d] + controls_lr_plates[i,right,d] <= max_controls_per_plate[d]) endif;

constraint if ok_bool then forall(i in 1..num_controls*max_control_concentrations)(sum([controls_lr_plates[p,h,i] | h in Horizontal, p in Plates]) == control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) ) endif;

%% Half of the total number of controls on each upper/lower half of the plate.
constraint if ok_bool then forall(i in Plates)(abs(sum(d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,left,d]) - sum(d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,right,d])) <= 1) endif;



%% Implied constraints for half plates
%% Connecting the number of compounds with the number of controls for each half plate.
constraint forall(i in Plates, v in Vertical)(ul_half_plates[i,v] + sum(d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]) <= ceil(inner_plate_size/2));

constraint forall(i in Plates)(ul_half_plates[i,upper] + ul_half_plates[i,lower] + sum(d in 1..num_controls*max_control_concentrations,v in Vertical)(controls_ul_plates[i,v,d]) + count([plates[i,j,k] | j in Rows, k in Columns], 0) == numcols*numrows);


%%->constraint forall(i in Plates)(ul_half_plates[i,upper] + ul_half_plates[i,lower] + sum(d in 1..num_controls*max_control_concentrations,v in Vertical)(controls_ul_plates[i,v,d]) + count([plates[i,j,k] | j in 1+size_empty_edge..numrows-size_empty_edge, k in 1+size_empty_edge..numcols-size_empty_edge], 0) == inner_plate_size);


%% Is this helping? I don't think so! But it might...
%%->constraint (sum(r in 1..floor(numrows/2))(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d]) + count([plates[i,j,k] | i in Plates, j in 1..floor(numrows/2), k in Columns], 0));

%%->constraint (sum(r in floor(numrows/2)+1..numrows)(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,lower,d]) + count([plates[i,j,k] | i in Plates, j in floor(numrows/2)+1..numrows, k in Columns], 0));


%%% Controls per row across plates%%%

constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_row[j] <= ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;

constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_row[j] >= floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;


%%% Controls per row per plate %%%
%% Relaxing this constraint a bit...
constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] <= control_slack + ceil(ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge))/numplates) + 2*size_empty_edge);

constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] >= floor(floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge))/numplates) + 2*size_empty_edge);


constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] >= (controls_in_row[j] div numplates));

constraint forall(i in Plates, j in 1+size_empty_edge..numrows-size_empty_edge) (controls_in_plate_row[i,j] <= ((controls_in_row[j]+numplates-1)div numplates));


%%% Controls per column across plates%%%
 
constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numcols-size_empty_edge) (controls_in_column[j] <= ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;

constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numcols-size_empty_edge) (controls_in_column[j] >= floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates) endif;


%%% Attempting to balance controls inside each row

array [Rows,Horizontal] of var 0..numplates*numcols: controls_in_half_row;

array [Plates,Rows,Horizontal] of var 0..numplates*numcols: controls_in_half_row_plate;

constraint forall(j in Rows) ( among(controls_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..floor(numcols/2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(j in Rows) ( among(controls_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in floor(numcols/2)+1..numcols], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..floor(numcols/2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,right],[plates[i,j,k] | k in floor(numcols/2)+1..numcols], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(j in Rows, h in Horizontal) (controls_in_half_row[j,h] == sum(i in Plates)(controls_in_half_row_plate[i,j,h]));

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,right]);

constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) <= controls_in_half_row[j,left]) endif;

constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) <= controls_in_half_row[j,right]) endif;

constraint if temp_bool /\ interconnected_plates then forall(j in Rows, h in Horizontal) ( (controls_in_row[j] div 2) <= controls_in_half_row[j,h] ) endif;

constraint if temp_bool /\ interconnected_plates then forall(j in Rows) (controls_in_row[j] == (controls_in_half_row[j,left] + controls_in_half_row[j,right])) endif;


constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( floor(floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/numplates)/(numrows-2*size_empty_edge))/2) + size_empty_edge) <= controls_in_half_row_plate[i,j,h]) endif;

constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( (controls_in_plate_row[i,j] div 2) <= controls_in_half_row_plate[i,j,h]) endif;

constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( ( (controls_in_plate_row[i,j]+1) div 2) >= controls_in_half_row_plate[i,j,h]) endif;


%constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil(ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/numplates)/(numrows-2*size_empty_edge))/2) + size_empty_edge) >= controls_in_half_row_plate[i,j,h]) endif;


constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] == controls_in_half_row_plate[i,j,left] + controls_in_half_row_plate[i,j,right]);

%%% Attempting to balance controls inside each column

array [Columns,Vertical] of var 0..numplates*numrows: controls_in_half_column;

constraint forall(k in Columns) ( among(controls_in_half_column[k,upper],[plates[i,j,k] | i in Plates, j in 1..floor(numrows/2)], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

constraint forall(k in Columns) ( among(controls_in_half_column[k,lower],[plates[i,j,k] | i in Plates, j in floor(numrows/2)+1..numrows], ({0} union experiments+1..experiments+num_controls*max_control_concentrations)));

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numrows-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..numrows-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) >= controls_in_half_row[j,right]);

constraint if temp_bool /\ interconnected_plates then forall(k in 1+size_empty_edge..numcols-size_empty_edge) ( floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) <= controls_in_half_column[k,upper]) endif;

constraint if temp_bool /\ interconnected_plates then forall(k in 1+size_empty_edge..numcols-size_empty_edge) ( floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+inner_emptywells)/(numcols-2*size_empty_edge)) + 2*size_empty_edge*numplates)/2) <= controls_in_half_column[k,lower]) endif;

constraint if temp_bool /\ interconnected_plates then forall(k in Columns) (controls_in_column[k] == (controls_in_half_column[k,upper] + controls_in_half_column[k,lower])) endif;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Distributing controls

%% Controls of the same type are not located next to each other

%% Implied constraint
constraint forall(i in Plates, j in 1..numrows-1, k in 1..numcols-1) (alldifferent_except_0([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]])):: domain;

%% TODO: relax this constraint when there are too many controls in the plate
%% Strong version
 constraint forall(i in Plates, j in 2..numrows-1, k in 2..numcols-1)(plates[i,j,k] > experiments -> 
            plates[i,j-1,k-1]!= plates[i,j,k] /\ 
            plates[i,j-1,k]  != plates[i,j,k] /\
            plates[i,j-1,k+1]!= plates[i,j,k] /\
            plates[i,j,k-1]  != plates[i,j,k] /\
            plates[i,j,k+1]  != plates[i,j,k] /\
            plates[i,j+1,k-1]!= plates[i,j,k] /\
            plates[i,j+1,k]  != plates[i,j,k] /\
            plates[i,j+1,k+1]!= plates[i,j,k]
            );


 constraint if spread_controls then forall(i in Plates, j in 3..numrows-2, k in 3..numcols-2)(plates[i,j,k] > experiments -> 
            plates[i,j-2,k-2]!= plates[i,j,k] /\ 
            plates[i,j-2,k-1]  != plates[i,j,k] /\
            plates[i,j-2,k]  != plates[i,j,k] /\
            plates[i,j-2,k+1]  != plates[i,j,k] /\
            plates[i,j-2,k+2]!= plates[i,j,k] /\
            plates[i,j,k-2]  != plates[i,j,k] /\
            plates[i,j,k+2]  != plates[i,j,k] /\
            plates[i,j+1,k-2]  != plates[i,j,k] /\
            plates[i,j+1,k+2]  != plates[i,j,k] /\
            plates[i,j+2,k-2]!= plates[i,j,k] /\
            plates[i,j+2,k-1]  != plates[i,j,k] /\
            plates[i,j+2,k]  != plates[i,j,k] /\
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
           
           
           
%%%% Find a solution! %%%%

array [Plates,1..experiments] of var {0} union Rows: experiment_plate_row;

constraint forall(l in 1..experiments)((experiment_plate[l] > 0) -> (experiment_plate_row[experiment_plate[l],l] == experiment_row[l]));

constraint forall(i in Plates, l in 1..experiments)((experiment_plate[l] != i) -> (experiment_plate_row[i,l] == 0));

%array [1..numplates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..numplates*2];
 
array [1..numplates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..numplates*2];
 
%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 

%% Best for Gecode
%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(plates, first_fail, indomain_random)]) satisfy; 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++[int_search(plates[i,..,..], random, indomain_max) | i in Plates])
%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_plate_row, first_fail, indomain_random)]++[ 
%int_search(plates[i,..,..], first_fail, indomain_random) | i in Plates]) 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++[
%int_search(experiment_plate_row[1,..], first_fail, indomain_random),int_search(plates[1,..,..], random, indomain_max)]++[ 
%int_search(experiment_plate_row[2,..], first_fail, indomain_random),int_search(plates[2,..,..], random, indomain_max)])

%int_search(plates, first_fail, indomain_random)]) 
%int_search(plates[i,..,..], random, indomain_max) | i in Plates ]) 

solve::seq_search([int_search(experiment_plate, first_fail, indomain_random)]++
search_order)
%[int_search(experiment_plate_row, first_fail, indomain_random)]++[int_search(plates[i,..,..], random, indomain_max) | i in Plates]) 
%[int_search(plates[i,..,..], random, indomain_max) | i in Plates ]) 
%      :: restart_geometric(1.1,1000)
      :: restart_geometric(1.1,10*experiments)
     %:: restart_linear(1000)
     %:: restart_constant(10*experiments)
     satisfy; 



%% Chuffed
% solve::seq_search([int_search(experiment_plate, random_order, indomain_min), int_search(experiment_row, random_order, indomain_min), int_search(plates, random_order, indomain_max)]) 
%      :: restart_geometric(1.5,100)
%      satisfy; 


%solve::seq_search([int_search(experiment_plate, random_order, indomain_min), priority_search(experiment_plate,[int_search([experiment_row[i]],input_order,indomain_split_random)| i in 1..experiments],smallest,complete)]) satisfy;


%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, random, indomain_split), int_search(plates, first_fail, indomain_random)]) satisfy; 

%solve satisfy;



%%KEEP%% solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(experiment_column, random, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 

%solve::seq_search([int_search(experiment_plate, first_fail, indomain_random), int_search(experiment_row, first_fail, indomain_random), int_search(plates, random, indomain_max)]) satisfy; 


%solve::int_search(experiment_plate, first_fail, indomain_random) satisfy; 

%solve::int_search(plates, random, indomain_max) satisfy; 



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Pretty printing %%%

output [if testing \/ debugging then "\(numplates) plates\n" else "" endif];

output [if debugging /\ (inner_emptywells>0) then "warning: there are \(inner_emptywells) empty wells. Consider using them as controls.\n\n" else "" endif];

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
        elseif fix(plates[i,j,k]) >  (experiments+11) then "  X" %% other        
        %else show_int(3,plates[i,j,k]) endif ++
        else "   " endif ++
        %% Adding lines and headings %%
        if j== numrows /\ k == numcols /\ i<numplates then "\n\n Plate \(i+1):\n" 
        elseif k == numcols then "\n" else " " endif else "" endif|
        i in Plates, j in Rows, k in Columns];        
         
output [if debugging then "Compound's plate:\n" else "" endif, if debugging then show(experiment_plate) else "" endif];

%output ["Compound's plate:\n", show(compound_concentration_names)];    

output [if debugging then "\nExperiment's row:\n" else "" endif, if debugging then show(experiment_row) else "" endif];

output [if debugging then "\nExperiment's row (per plate):\n" else "" endif, if debugging then show(experiments_in_plate_row) else "" endif];

 %output ["\nExperiment's column:\n", show(experiment_column)];

output [if debugging then "\nControls per row:\n" else "" endif, if debugging then show(controls_in_row) else "" endif];

output [if debugging then "\nControls per row per plate:\n" else "" endif, if debugging then show(controls_in_plate_row) else "" endif];

output [if debugging then "\nControls per column:\n" else "" endif, if debugging then show(controls_in_column) else "" endif];

output [if debugging then "\nControls per half row:\n" else "" endif, if debugging then show(controls_in_half_row) else "" endif];

output [if debugging then "\nControls per upper/lower half plates:\n" else "" endif, if debugging then show(controls_ul_plates) else "" endif];

output [if debugging then "\nControls per left/right half plates:\n" else "" endif, if debugging then show(controls_lr_plates) else "" endif];

output [if debugging then "\nCompounds per half plates:\n" else "" endif, if debugging then show(ul_half_plates) else "" endif];

output [if debugging then "\nConcentrations per half plates:\n" else "" endif, if debugging then show(ul_concentrations) else "" endif];

output [if debugging then "\nConcentrations upper half plates:\n" else "" endif, if debugging then show(sum(ul_concentrations[..,upper])) else "" endif];

output [if debugging then "\nConcentrations lower half plates:\n" else "" endif, if debugging then show(sum(ul_concentrations[..,lower])) else "" endif];


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
         

       