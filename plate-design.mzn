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
% Description: A highly effective microplate layout designer using constraint
%              programming to optimize the distribution of compounds, controls,
%              and replicates across microplate wells while minimizing systematic
%              errors and edge effects.
%
% Authors: Maria Andreina FRANCISCO RODRIGUEZ (maria.andreina.francisco@farmbio.uu.se)
% Version: 1.1
% Last Revision: November 2025
%
% ==============================================================================

include "globals.mzn";
include "gecode.mzn";
include "layout_predicates.mzn";


% ==============================================================================
% SECTION 1: INPUT PARAMETERS - CONSTRAINT CONFIGURATION
% ==============================================================================
%
% Boolean flags controlling which constraints to apply. These determine the
% experimental design requirements that must be satisfied.
%
% ==============================================================================

% Well Configuration
bool: allow_empty_wells;  % Allow inner wells to remain empty if needed

% Replicate Placement Constraints (mutually exclusive)
bool: replicates_on_different_plates;  % Force replicates across different plates
bool: replicates_on_same_plate;        % Force all replicates on same plate

% Concentration Spreading Constraints
bool: concentrations_on_different_rows;     % Force concentrations to different rows
bool: concentrations_on_different_columns;  % Force concentrations to different columns

% Advanced Distribution Controls
bool: force_spread_controls;       % Enforce control spreading (auto-calculated if not set)
bool: force_spread_concentrations; % Enforce concentration spreading (auto-calculated if not set)


% ==============================================================================
% SECTION 2: INPUT PARAMETERS - PLATE GEOMETRY
% ==============================================================================
%
% Physical layout parameters defining plate dimensions and edge handling.
%
% ==============================================================================

% Cell Line Configuration
% These parameters support multi-cell-line experiments on the same plate
int: horizontal_cell_lines;  % Number of horizontal cell line divisions
int: vertical_cell_lines;    % Number of vertical cell line divisions

% Edge Handling
int: size_empty_edge;  % Number of outermost rows/columns to leave empty (edge effect mitigation)

opt bool: inner_empty_edge_input;
bool: inner_empty_edge = inner_empty_edge_input default true;  % Whether to create inner empty edges between cell lines

% Plate Dimensions (in wells)
int: num_rows;  % Total number of rows in physical plate
int: num_cols;  % Total number of columns in physical plate

opt int: size_corner_empty_wells;
int: num_corner_empty_wells = size_corner_empty_wells default 0;  % Currently unused; reserved for future use


% ==============================================================================
% SECTION 3: INPUT PARAMETERS - COMPOUNDS
% ==============================================================================
%
% Compound definitions: the experimental substances being tested across
% multiple concentrations and replicates.
%
% ==============================================================================

int: compounds;  % Number of distinct compounds/drugs to test

% Per-compound configuration
array[1..compounds] of int: compound_replicates;      % Number of replicates per compound
array[1..compounds] of int: compound_concentrations;  % Number of concentration levels per compound

% Derived compound parameters
int: replicates = max(compound_replicates++[0]);  % Maximum replicates across all compounds
int: max_compound_concentrations = max(compound_concentrations++[0]);  % Maximum concentrations

% Compound naming (for output generation)
array[1..compounds] of string: compound_names;
array[1..compounds, 1..max_compound_concentrations] of string: compound_concentration_names;


% ==============================================================================
% SECTION 4: INPUT PARAMETERS - COMBINATIONS (DEPRECATED)
% ==============================================================================
%
% Legacy support for compound combinations. This functionality is deprecated
% and maintained only for backward compatibility with older data files.
%
% ==============================================================================

int: combinations;                    % Number of compound combinations (set to 0 for new designs)
int: combination_concentrations;      % Concentration levels per combination

array[1..combinations] of string: combination_names;
array[1..combination_concentrations] of string: combination_concentration_names;

array[int] of string: compound_concentration_indicators;  % LaTeX indicators for output


% ==============================================================================
% SECTION 5: INPUT PARAMETERS - CONTROLS
% ==============================================================================
%
% Control definitions: reference samples (positive/negative controls, blanks)
% distributed across plates to enable normalization and quality assessment.
%
% ==============================================================================

int: num_controls;  % Number of distinct control types

% Per-control configuration
array[1..num_controls] of int: control_replicates;      % Replicates per control type
array[1..num_controls] of int: control_concentrations;  % Concentration levels per control

% Derived control parameters
int: max_control_concentrations = max(control_concentrations++[0]);
int: total_controls = sum([control_concentrations[i] * control_replicates[i] | i in 1..num_controls]);

% Control naming (for output generation)
array[1..num_controls] of string: control_names;
array[1..num_controls, 1..max_control_concentrations] of string: control_concentration_names;


% ==============================================================================
% SECTION 6: ADVANCED CONFIGURATION PARAMETERS
% ==============================================================================
%
% Fine-tuning parameters for advanced users. These control constraint tightness
% and solver behavior.
%
% ==============================================================================

% Control Distribution Parameters
bool: balance_controls_inside_plate = true;  % Balance controls within each plate
bool: interconnected_plates = true;          % Connect plates for better balance (harder to solve)
int: control_slack = 0;                      % Tolerance for control distribution inequality (0 = strict)

% Automatic Spreading Threshold Calculation
% Note: These are calculated based on plate size and control count
bool: spread_controls = (ceil(inner_plate_size/2) * num_plates >= total_controls);

% Individual control spreading flags (auto-calculated per control type)
% Controls with few replicates can be spread more aggressively
array[1..num_controls * max_control_concentrations] of bool: spread_control = [
  ceil(inner_plate_size/4) * num_plates >= 
  (control_replicates[floor((i-1)/max_control_concentrations)+1] * 
   (((i-1) mod max_control_concentrations) < control_concentrations[floor((i-1)/max_control_concentrations)+1]))
  | i in 1..num_controls * max_control_concentrations
];


% ==============================================================================
% SECTION 7: DEBUGGING AND TESTING PARAMETERS
% ==============================================================================
%
% Optional parameters for development, testing, and debugging.
%
% ==============================================================================

opt bool: testing = false;      % Enable test mode
opt bool: print_all = false;    % Verbose output for debugging
bool: debugging = print_all \/ false;  % Combined debug flag

opt int: swap_search;           % Alternative search strategy selector
opt bool: sorted_compounds;     % Force compounds into sorted order (special cases only)


% ==============================================================================
% SECTION 8: DERIVED GEOMETRY PARAMETERS
% ==============================================================================
%
% Calculated plate geometry values based on input dimensions and edge settings.
% These determine the usable "inner" area of each plate.
%
% ==============================================================================

% Calculate inner plate dimensions (excluding edges and between cell lines)
int: num_rows_line = if inner_empty_edge then 
                       floor(num_rows/horizontal_cell_lines) - 2*size_empty_edge 
                     else 
                       floor((num_rows - 2*size_empty_edge)/horizontal_cell_lines) 
                     endif;

int: num_cols_line = if inner_empty_edge then 
                       floor(num_cols/vertical_cell_lines) - 2*size_empty_edge 
                     else 
                       floor((num_cols - 2*size_empty_edge)/vertical_cell_lines) 
                     endif;

% Sets for iteration
set of int: Rows = 1..num_rows_line;
set of int: Columns = 1..num_cols_line;

% Enums for plate region identification
enum Vertical = {upper, lower};
enum Horizontal = {left, right};

% Decision variables for tracking experiments per row/column
array [Plates,Rows] of var 0..num_cols_line: experiments_in_plate_row;
array [Plates,Columns] of var 0..num_rows_line: experiments_in_plate_column;

% Plate characteristics
bool: even_columns = ((num_cols_line mod 2) == 0);
bool: even_rows = ((num_rows_line mod 2) == 0);

int: inner_plate_size = num_cols_line * num_rows_line;  % Wells available per plate


% ==============================================================================
% SECTION 9: INPUT VALIDATION CONSTRAINTS
% ==============================================================================
%
% Data file validation to catch configuration errors early and provide
% meaningful error messages before solver runs.
%
% ==============================================================================

constraint assert(compounds >= 0, 
  "Invalid datafile: Number of compounds cannot be less than zero.");
  
constraint assert(combinations >= 0, 
  "Invalid datafile: Number of combinations cannot be less than zero.");
  
constraint assert(num_controls >= 0, 
  "Invalid datafile: Number of controls should not be less than zero.");
  
constraint assert(vertical_cell_lines > 0, 
  "Invalid datafile: Number of cell lines should be larger than zero.");
  
constraint assert(horizontal_cell_lines > 0, 
  "Invalid datafile: Number of cell lines should be larger than zero.");
  
constraint assert(num_rows_line > 0, 
  "Invalid datafile: Number of rows should be larger than zero.");
  
constraint assert(num_cols_line > 0, 
  "Invalid datafile: Number of columns should be larger than zero.");
  
constraint assert(compounds == 0 \/ min(compound_replicates) > 0, 
  "Invalid datafile: Number of replicates should be larger than zero.");
  
constraint assert(compounds == 0 \/ min(compound_concentrations) > 0, 
  "Invalid datafile: Number of concentrations should be larger than zero.");
  
constraint assert((replicates_on_different_plates /\ replicates_on_same_plate) == false, 
  "Invalid datafile: replicates cannot be both on the same plate and on different plates");
  
constraint assert(not (num_controls == 1 /\ spread_control[1] == false), 
  "Invalid datafile: There are too many controls of only one kind. This is not allowed at the moment. If you believe this is a mistake, please contact the developers.");
  
constraint assert(num_rows_line mod 2 == 0, 
  "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of rows.");
  
constraint assert(num_cols_line mod 2 == 0, 
  "Invalid datafile: Currently we only support plate sizes that have an inner area with an even number of columns.");


% ==============================================================================
% SECTION 10: DERIVED EXPERIMENTAL PARAMETERS
% ==============================================================================
%
% Calculated values describing the total experimental requirements.
%
% ==============================================================================

% Total wells needed across all plates
int: total_wells = sum([compound_concentrations[i] * compound_replicates[i] | i in 1..compounds]) + total_controls;

% Number of plates required (rounded up to fit all content)
int: num_plates = max(ceil(total_wells/inner_plate_size), 1);

% Set for plate iteration
set of int: Plates = 1..num_plates;

% Total experiment count (all compound-concentration-replicate combinations)
int: experiments = compounds * max_compound_concentrations * replicates;

% Empty wells to distribute (may be 0 if plates are exactly full)
int: emptywells = num_plates * inner_plate_size - total_wells;

% Value range constants for improved readability
set of int: ExperimentValues = 1..experiments;
set of int: ControlValues = experiments+1..experiments+num_controls*max_control_concentrations;
set of int: ControlValuesWithEmpty = {0} union ControlValues;
set of int: InnerRows = 1+size_empty_edge..num_rows_line-size_empty_edge;
set of int: InnerColumns = 1+size_empty_edge..num_cols_line-size_empty_edge;


% Minimum compounds per plate for balanced distribution
int: min_compounds_plate = max(
  floor((sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]))/inner_plate_size), 
  min(compound_concentrations++[0])
);

% Minimum plate count for replicate distribution
int: min_plates = min(num_plates, replicates);

% ==============================================================================
% SECTION 11: AUTOMATIC THRESHOLD CALCULATION
% ==============================================================================
%
% Automatically determine if spreading constraints should be enforced based on
% plate size and content count. Uses empirically validated thresholds.
%
% ==============================================================================

% Override force_spread_controls with calculated value based on plate configuration
force_spread_controls = force_spread_controls(num_cols_line, num_rows_line, total_controls, num_plates);

% Override force_spread_concentrations based on replicate placement strategy
force_spread_concentrations = if replicates_on_different_plates /\ replicates <= num_plates then 
                                ((inner_plate_size div 5) >= max_compound_concentrations) 
                              else 
                                (ceil(inner_plate_size/5) >= replicates * max_compound_concentrations) 
                              endif;


% ==============================================================================
% SECTION 12: FEASIBILITY VALIDATION
% ==============================================================================
%
% Pre-solve checks to detect unsatisfiable configurations before wasting
% solver time. Provides specific error codes for diagnosis.
%
% ==============================================================================

constraint assert(total_wells > 0, 
  "Invalid data: the plates cannot be completely empty.");
  
constraint assert(inner_plate_size > 0, 
  "Invalid data: There are no wells on the plate.");
  
constraint assert(min(compound_concentrations++[0]) <= inner_plate_size, 
  "Invalid data: Number of concentrations does not fit in one plate. If you think this is a mistake, please contact the development team.");

% Check if balanced distribution is mathematically possible
constraint assert(
  ceil(sum(compound_replicates)/num_plates) * min(compound_concentrations++[infinity]) + 
  sum([floor(control_concentrations[i] * control_replicates[i]/num_plates) | i in 1..num_controls]) <= inner_plate_size, 
  "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E01)");

constraint assert(
  (floor(sum(compound_replicates)/num_plates) - 1) * min(compound_concentrations++[infinity]) + 
  max_compound_concentrations + 
  sum([floor(control_concentrations[i] * control_replicates[i]/num_plates) | i in 1..num_controls]) <= inner_plate_size, 
  "Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E02)");

constraint assert(emptywells == 0 \/ allow_empty_wells, 
  "Warning: There are \(emptywells) empty wells. You can allow the plates to have inner empty wells or modify other quantities to fill them up.");

constraint assert(emptywells >= 0, 
  "Model ERROR! Inner empty wells is negative. This should never happen!");


% ==============================================================================
% SECTION 13: DECISION VARIABLES - PLATE LAYOUT
% ==============================================================================
%
% Main decision variables representing the complete plate layout solution.
% 
% Encoding scheme for well contents:
%   0 = empty well
%   1 to experiments = compound experiments (indexed by compound/replicate/concentration)
%   experiments+1 to experiments+num_controls*max_control_concentrations = controls
%
% ==============================================================================

% Primary plate model: 3D array representing all plates
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
constraint global_cardinality(plates,[i | i in 0..(experiments + num_controls*max_control_concentrations)],[emptywells]++[(floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((floor((i-1))) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1] | i in 1..experiments]++[1|i in experiments+1..experiments]++[control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1]) | i in 1..num_controls*max_control_concentrations]);

%% Experiments that do not exist are not located in any plate, row, or column.
constraint forall(i in 1..experiments)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((i-1) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) == false <-> (experiment_plate[i] = 0 /\ experiment_row[i] = 0 /\ experiment_column[i] = 0));

constraint forall(i in 1..experiments)(((floor((i-1)/(compounds*max_compound_concentrations))<compound_replicates[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) /\ ((i-1) mod max_compound_concentrations) < compound_concentrations[(floor(((i-1)/max_compound_concentrations)) mod compounds)+1]) <-> (experiment_plate[i] != 0 /\ experiment_row[i] != 0 /\ experiment_column[i] != 0));




% ==============================================================================
% SECTION 14: CORE CONSTRAINTS - PLATE CONTENT
% ==============================================================================
%
% Global cardinality and existence constraints that ensure the correct
% number of experiments, controls, and empty wells are placed on the plates.
%
% ==============================================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channelling constraints: an experiment/compound is located at a given plate

% Channeling: synchronize the two model representations (Priority 5)
constraint forall(l in 1..experiments)(
  channel_experiment(l, experiment_plate[l], experiment_row[l], 
                    experiment_column[l], plates)
);


%% For experiments that do not exist = plate 0 (redundant)
%constraint forall(l in 1..experiments)(
 % count_eq(array1d(1..num_plates*inner_plate_size, plates),l,0) <-> (experiment_plate[l] = 0 /\ experiment_row[l] = 0 /\ experiment_column[l] = 0)
%);

%constraint forall(l in 1..experiments)(count_eq(array1d(1..num_plates*inner_plate_size, plates),l,0) <-> (experiment_row[l] == 0));
%constraint forall(l in 1..experiments)(count_eq(array1d(1..num_plates*inner_plate_size, plates),l,0) <-> (experiment_column[l] == 0));



%% Channeling constraints between the two models


constraint forall(l in 1..experiments)(count_eq(plates,l,(experiment_row[l] != 0)));
constraint forall(l in 1..experiments)(count_eq(plates,l,(experiment_column[l] != 0)));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% All concentrations of a given compound replica must appear on a single plate

constraint forall(l in 1..experiments where ((l mod max_compound_concentrations) == 1))(all_equal([experiment_plate[i] | i in l..(l+compound_concentrations[((((l-1) div max_compound_concentrations)) mod compounds)+1]-1)]));

%% TODO: check if other compounds have fewer concentrations and can be spread around more
%% TODO: change to include replicates
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..num_rows_line-1, k in 1..num_cols_line-1,l in 1..experiments where ((l mod max_compound_concentrations) == 1)) (1>=among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],(l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)))):: domain endif;


% constraint if force_spread_concentrations then forall(i in Plates, j in 1..num_rows_line-1, k in 1..num_cols_line-1,l in 1..experiments where ((l mod max_compound_concentrations) == 1)) (
%   let{
%     var 0..1: num_concentrations;
%    } in among(num_concentrations,[plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],(l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)))::domain) endif;

% Replaced by constraint below for clarity (less index math)
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..num_rows_line-1, k in 1..num_cols_line-1,l in 1..experiments where ((l mod max_compound_concentrations) == 1)) (
%2 > among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],(l..(l+compound_concentrations[(floor(((l-1)/max_compound_concentrations)) mod compounds)+1]-1)))::domain) endif;


% Replaced by constraint below to in order to include replicates
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..num_rows_line-1, k in 1..num_cols_line-1, c in 1..compounds, r in 0..replicates-1) (
%2 > among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],((c-1)*r*max_compound_concentrations+1..c*r*max_compound_concentrations))::domain) endif;   

      
constraint if force_spread_concentrations then forall(i in Plates, j in 1..num_rows_line-1, k in 1..num_cols_line-1, c in 1..compounds) (
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
%   } in exprInRow <= num_cols_line-among([plates[i,j,k] | k in Columns], (ControlValuesWithEmpty)));
  
  
constraint forall(i in Plates, j in Rows) (
(among(plates[i,j,..], 1..experiments)::domain) <= num_cols_line-among([plates[i,j,k] | k in Columns], (ControlValuesWithEmpty))::domain);



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
%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],num_rows_line-2*size_empty_edge), [experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
alldifferent([experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],num_rows_line)]):: domain
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
alldifferent([experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],num_rows_line)]):: domain
))) endif; 

%% HERE!!! NOW!
%% 
%constraint if force_spread_concentrations then forall(i in Plates, j in 1..num_rows_line-1, k in 1..num_cols_line-1, c in 1..compounds) (
%2 > among([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]],
%array_union([(c-1)*max_compound_concentrations+1 + r*compounds*max_compound_concentrations..c*max_compound_concentrations + r*compounds*max_compound_concentrations | r in 0..replicates-1]) 
%)::domain) endif;   


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% WORK AREA!!!
%balanced_alldifferent(array[int] of var int: x, int: mid_val, int: mid_count)

%array [Vertical, 1..experiments] of var {0} union Rows: ul_experiment_row;

%constraint forall(v in Vertical, e in 1..experiments) (if experiment_row[e] <= (num_rows_line div 2) then ul_experiment_row[upper,e]==experiment_row[e] /\ ul_experiment_row[lower,e]==0  else ul_experiment_row[lower,e]==experiment_row[e] /\ ul_experiment_row[upper,e]==0  endif);

% constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (if compound_concentrations[l]>1 then
% forall(v in Vertical, r in 0..(compound_replicates[l]-1)) (
% let{
%   var int: compounds_in_row;
%   constraint compounds_in_row <= ((compound_concentrations[l]+1) div 2)+1;
%   constraint compounds_in_row >= (compound_concentrations[l] div 2)+1;
%   }
%   in nvalue(compounds_in_row,[ul_experiment_row[v,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],num_rows_line)])
% )endif)) endif; 



%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
%nvalue((compound_concentrations[l] div 2)+1,[ul_experiment_row[lower,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],num_rows_line)])
%))) endif; 

%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (
%nvalue(ul_concentrations[r*compounds + l,lower]+1,[ul_experiment_row[lower,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in %1..min(compound_concentrations[l],num_rows_line)])
%))) endif; 

%% MAYBE XXXX
%constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(v in Vertical, r in 0..(compound_replicates[l]-1)) (
%alldifferent_except_0([ul_experiment_row[v,r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],num_rows_line)])
%))) endif; 


%constraint count(e in 1..experiments)(ul_experiment_row[upper,e]>0) = sum(ul_half_plates[..,upper]);
%constraint count(e in 1..experiments)(ul_experiment_row[lower,e]>(num_rows_line div 2)) = sum(ul_half_plates[..,lower]);
%%%%% END MAYBE


%% IDEA:
% Implied constraint: when there are more concentrations than rows, help the search to not fill in a row completely too soon by saving space in each row for the other compounds
% constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= num_rows_line)) then 
% forall(i in Plates, j in Rows, r in 1..(compounds*replicates-1))
%   ((count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + sum([ experiment_plate_row[i,max_compound_concentrations*(k-1)+1]>0 /\ (compound_concentrations[((k-1) mod compounds)+1]>=num_rows_line) | k in (r+1) .. compounds*replicates])) <= experiments_in_plate_row[i,j]) endif;

%int: temp = (compounds*replicates-1);

% NOT WORKING!!
% constraint if concentrations_on_different_rows then 
% forall(i in Plates, j in Rows, r in temp..(compounds*replicates-1))(
%   let{
%     array [Plates,Rows] of var 0..1: temp_array;
%      %compound_concentrations[(r mod compounds)+1];
%   } in sum(temp_array[i,..])==1 /\ count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + temp_array[i,j] == experiments_in_plate_row[i,j]
% ) endif;

%% array [Plates,Rows] of var 0..num_cols_line: experiments_in_plate_row;
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
  if experiment_plate[(e-1)*max_compound_concentrations+1]==i then sum(compounds_in_plate_row[i,e,..]) == min(compound_concentrations[((e-1) mod compounds)+1],num_rows_line)
  else sum(compounds_in_plate_row[i,e,..]) == 0 endif
) endif;
 
 
constraint if max_compound_concentrations<=num_rows_line /\ concentrations_on_different_rows then forall(i in Plates, j in Rows)(
  sum(compounds_in_plate_row[i,..,j]) == experiments_in_plate_row[i,j]
) 
else forall(i in Plates, j in Rows)(
  sum(compounds_in_plate_row[i,..,j]) <= experiments_in_plate_row[i,j]
)
endif;

constraint if max_compound_concentrations<=num_rows_line /\ concentrations_on_different_rows then
  sum(compounds_in_plate_row[..,..,..]) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) else
  sum(compounds_in_plate_row[..,..,..]) <= sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])  
endif;


constraint if concentrations_on_different_rows /\ max_compound_concentrations<=num_rows_line /\ even_rows then (forall(l in 1..compounds,r in 0..(replicates-1)) (
 ul_concentrations[r*compounds + l,upper] == sum(compounds_in_plate_row[..,r*compounds + l, 1..(num_rows_line div 2)])
)) endif; 

constraint if concentrations_on_different_rows /\ max_compound_concentrations<=num_rows_line /\ even_rows then (forall(l in 1..compounds,r in 0..(replicates-1)) (
 ul_concentrations[r*compounds + l,lower] == sum(compounds_in_plate_row[..,r*compounds + l, (num_rows_line div 2)+1..num_rows_line])
)) endif; 

constraint if concentrations_on_different_rows /\ even_rows then 
  forall(i in Plates)(sum(experiments_in_plate_row[i,1..(num_rows_line div 2)]) == ul_half_plates[i, upper]) endif;
  
constraint if concentrations_on_different_rows /\ even_rows then 
  forall(i in Plates)(sum(experiments_in_plate_row[i,(num_rows_line div 2)+1..num_rows_line]) == ul_half_plates[i, lower]) endif;


%% Constraints for compounds_in_plate_column
constraint forall(k in Columns, e in 1..experiments)(
  (experiment_column[e]==k)  -> (compounds_in_plate_column[experiment_plate[e], (((e-1) div max_compound_concentrations)+1),k] == 1)
);
 
constraint forall(i in Plates, k in Columns, e in 1..compounds*replicates)(
  (experiment_plate[(e-1)*max_compound_concentrations+1] != i)  -> (compounds_in_plate_column[i,e,k] == 0)
);

constraint if concentrations_on_different_columns then forall(i in Plates, e in 1..compounds*replicates)(
  if experiment_plate[(e-1)*max_compound_concentrations+1]==i then sum(compounds_in_plate_column[i,e,..]) == min(compound_concentrations[((e-1) mod compounds)+1],num_cols_line)
  else sum(compounds_in_plate_column[i,e,..]) == 0 endif
) endif;
 
 
constraint if max_compound_concentrations<=num_cols_line /\ concentrations_on_different_columns then forall(i in Plates, k in Columns)(
  sum(compounds_in_plate_column[i,..,k]) == experiments_in_plate_column[i,k]
) 
else forall(i in Plates, k in Columns)(
  sum(compounds_in_plate_column[i,..,k]) <= experiments_in_plate_column[i,k]
)
endif;

constraint if max_compound_concentrations<=num_cols_line /\ concentrations_on_different_columns then
  sum(compounds_in_plate_column[..,..,..]) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]) else
  sum(compounds_in_plate_column[..,..,..]) <= sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])  
endif;

constraint if concentrations_on_different_columns /\ max_compound_concentrations<=num_cols_line /\ even_columns then (forall(l in 1..compounds,r in 0..(replicates-1)) (
  lr_concentrations[r*compounds + l,left] == sum(compounds_in_plate_column[..,r*compounds + l, 1..(num_cols_line div 2)])
)) endif; 

constraint if concentrations_on_different_columns /\ max_compound_concentrations<=num_cols_line /\ even_columns then (forall(l in 1..compounds,r in 0..(replicates-1)) (
  lr_concentrations[r*compounds + l,right] == sum(compounds_in_plate_column[..,r*compounds + l, (num_cols_line div 2)+1..num_cols_line])
)) endif; 

%% HERE
constraint if concentrations_on_different_columns /\ even_columns then 
  forall(i in Plates)(sum(experiments_in_plate_column[i,1..(num_cols_line div 2)]) == lr_half_plates[i, left]) endif;
  
constraint if concentrations_on_different_columns /\ even_columns then 
  forall(i in Plates)(sum(experiments_in_plate_column[i,(num_cols_line div 2)+1..num_cols_line]) == lr_half_plates[i, right]) endif;


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
 among(ul_concentrations[r*compounds + l,upper],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..(num_rows_line div 2))::domain
))) endif; 

%% Definition of ul_concentrations (lower)
constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower],[experiment_row[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], (num_rows_line div 2)+1..num_rows_line)::domain
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
 among(ul_concentrations[r*compounds + l,upper], plates[..,1..(num_rows_line div 2),..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
))) endif; 

constraint if concentrations_on_different_rows then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(ul_concentrations[r*compounds + l,lower], plates[..,(num_rows_line div 2)+1..num_rows_line,..], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
))) endif; 

%% Defining lr_concentrations in terms of plates
constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,left], plates[..,..,1..(num_cols_line div 2)], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,right], plates[..,..,(num_cols_line div 2)+1..num_cols_line], {r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i| i in 1..compound_concentrations[l]})::domain
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

constraint forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j] + controls_in_plate_row[i,j] == num_cols_line);
constraint forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j] + controls_only_in_plate_row[i,j] + emptywells_in_plate_row[i,j] == num_cols_line);

constraint forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k] + controls_in_plate_column[i,k] == num_rows_line);
constraint forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k] + controls_only_in_plate_column[i,k] + emptywells_in_plate_column[i,k] == num_rows_line);

constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= num_rows_line)) then forall(i in Plates, j in Rows, r in 1..compounds*replicates)(
count(experiment_plate_row[i,1..experiments-max_compound_concentrations*r],j) + count(experiment_plate_row[i,experiments-max_compound_concentrations*r+1..experiments],j) == experiments_in_plate_row[i,j]
) endif;

constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= num_rows_line)) then forall(i in Plates, j in Rows, r in 1..compounds*replicates-1)(
if(compound_concentrations[((r-1) mod compounds)+1]>=num_rows_line) then
(count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + count(experiment_plate_row[i,max_compound_concentrations*(r+1)+1..experiments],j) + (experiment_plate_row[i,max_compound_concentrations*r+1]>0)) <= experiments_in_plate_row[i,j]
endif
) endif;

% Implied constraint: when there are more concentrations than rows, help the search to not fill in a row completely too soon by saving space in each row for the other compounds
constraint if (concentrations_on_different_rows /\ (max_compound_concentrations >= num_rows_line)) then forall(i in Plates, j in Rows, r in 1..(compounds*replicates-1))((count(experiment_plate_row[i,1..max_compound_concentrations*r],j) + sum([ experiment_plate_row[i,max_compound_concentrations*(k-1)+1]>0 /\ (compound_concentrations[((k-1) mod compounds)+1]>=num_rows_line) | k in (r+1) .. compounds*replicates])) <= experiments_in_plate_row[i,j]) endif;



%%%%%%
%% Spreads concentrations over different columns. It's been replaced by the more restrictive alldifferent version below.
%constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) (nvalue(min(compound_concentrations[l],num_cols_line-2*size_empty_edge), [experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]])))) endif; 

constraint if concentrations_on_different_columns then (forall(l in 1..compounds) (forall(r in 0..(compound_replicates[l]-1)) ( 
alldifferent( [experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..min(compound_concentrations[l],num_cols_line)]):: domain
))) endif; 

% Split concentrations between left-most and right-most halves of the plate

array [1..compounds*replicates,Horizontal] of var 0..max_compound_concentrations: lr_concentrations;

%% For each replicate of each compound, count how many concentrations there are in the left/right halves of the plate
%% Definition of lr_concentrations (left)  
constraint if concentrations_on_different_columns /\ even_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,left],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], 1..(num_cols_line div 2))::domain
))) endif; 


%% Definition of lr_concentrations (right)  
constraint if concentrations_on_different_columns /\ even_columns then (forall(l in 1..compounds) (forall(r in 0..(replicates-1)) (
 among(lr_concentrations[r*compounds + l,right],[experiment_column[r*compounds*max_compound_concentrations+(l-1)*max_compound_concentrations+i]| i in 1..compound_concentrations[l]], (num_cols_line div 2)+1..num_cols_line)::domain
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

constraint global_cardinality_low_up(experiment_row, [i | i in Rows], [0| i in Rows], [num_plates*num_cols_line| i in Rows]);
constraint global_cardinality_low_up(experiment_column, [i | i in Columns], [0| i in Columns], [num_plates*num_rows_line| i in Columns]);   

array [Plates,Rows] of var 0..num_cols_line: controls_in_plate_row;
array [Plates,Columns] of var 0..num_rows_line: controls_in_plate_column;
array [Plates,Rows] of var 0..num_cols_line: controls_only_in_plate_row;
array [Plates,Columns] of var 0..num_rows_line: controls_only_in_plate_column;
array [Plates,Rows] of var 0..num_cols_line: emptywells_in_plate_row;
array [Plates,Columns] of var 0..num_rows_line: emptywells_in_plate_column;
array [Rows] of var 0..num_plates*num_cols_line: controls_in_row;
array [Columns] of var 0..num_plates*num_rows_line: controls_in_column;

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
constraint forall(i in Plates, j in Rows) (among(controls_in_plate_row[i,j], plates[i,j,..], (ControlValuesWithEmpty))::domain);
constraint forall(i in Plates, j in Rows) (among(controls_only_in_plate_row[i,j], plates[i,j,..], (ControlValues))::domain);


constraint forall(i in Plates, k in Columns) (among(controls_in_plate_column[i,k], plates[i,..,k], (ControlValuesWithEmpty))::domain);
constraint forall(i in Plates, k in Columns) (among(controls_only_in_plate_column[i,k], plates[i,..,k], (ControlValues))::domain);

constraint forall(i in Plates, j in Rows) (among(emptywells_in_plate_row[i,j], plates[i,j,..], {0})::domain);
constraint forall(i in Plates, k in Columns) (among(emptywells_in_plate_column[i,k], plates[i,..,k], {0})::domain);

constraint forall(j in Rows) (among(controls_in_row[j], plates[..,j,..], (ControlValuesWithEmpty))::domain);

constraint forall(k in Columns) (among(controls_in_column[k], plates[..,..,k], (ControlValuesWithEmpty))::domain);

constraint forall(j in Rows) (controls_in_row[j] == sum(controls_in_plate_row[..,j]));

constraint forall(j in Rows) (controls_in_row[j] == sum(emptywells_in_plate_row[..,j]) + sum(controls_only_in_plate_row[..,j]));

constraint forall(k in Columns) (controls_in_column[k] == sum(controls_in_plate_column[..,k]));

constraint forall(k in Columns) (controls_in_column[k] == sum(emptywells_in_plate_column[..,k]) + sum(controls_only_in_plate_column[..,k]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,upper,..]) + count(plates[i,1..(num_rows_line div 2),..],0) == sum(controls_in_plate_row[i,1..(num_rows_line div 2)]));

constraint forall(i in Plates) ( sum(controls_ul_plates[i,lower,..]) + count(plates[i,(num_rows_line div 2)+1..num_rows_line,..],0) == sum(controls_in_plate_row[i,(num_rows_line div 2)+1..num_rows_line]));

constraint forall(i in Plates) ( sum(controls_lr_plates[i,left,..]) + count(plates[i,..,1..(num_cols_line div 2)],0) == sum(controls_in_plate_column[i,1..(num_cols_line div 2)]));
constraint forall(i in Plates) ( sum(controls_lr_plates[i,right,..]) + count(plates[i,..,(num_cols_line div 2)+1..num_cols_line],0) == sum(controls_in_plate_column[i,(num_cols_line div 2)+1..num_cols_line]));

constraint forall(i in Plates) ( sum_controls_ul_plates[i,upper] == sum(j in 1..(num_rows_line div 2))(controls_only_in_plate_row[i,j]));
constraint forall(i in Plates) ( sum_controls_ul_plates[i,lower] == sum(j in (num_rows_line div 2)+1..num_rows_line)(controls_only_in_plate_row[i,j]));

constraint forall(i in Plates) ( sum_controls_lr_plates[i,left] == sum(j in 1..(num_cols_line div 2))(controls_only_in_plate_column[i,j]));
constraint forall(i in Plates) ( sum_controls_lr_plates[i,right] == sum(j in (num_cols_line div 2)+1..num_cols_line)(controls_only_in_plate_column[i,j]));


constraint sum(sum_controls_ul_plates) == total_controls;
constraint sum(sum_controls_lr_plates) == total_controls;

constraint forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) >= (total_controls div 2));
constraint forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) <= ((total_controls + 1) div 2));

constraint forall(h in Horizontal) (sum(sum_controls_lr_plates[..,h]) >= (total_controls div 2));
constraint forall(h in Horizontal) (sum(sum_controls_lr_plates[..,h]) <= ((total_controls + 1) div 2));

%% Implied constraint: how many experiments can be in a row/column? It's too weak and needs to be done per plate.
constraint global_cardinality(experiment_row, [j | j in Rows], [(num_plates*num_cols_line)-controls_in_row[j]| j in Rows]);
constraint global_cardinality(experiment_column, [k | k in Columns], [(num_plates*num_rows_line)-controls_in_column[k]| k in Columns]);




%% Implied constraint: how many experiments can there be in a row? Per plate
% I think this still works...
%% TODO: Redundant 
constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])== num_cols_line-among(plates[i,j,..], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])== num_cols_line-controls_in_plate_row[i,j]);

constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments])== num_cols_line-emptywells_in_plate_row[i,j]-controls_only_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])== num_rows_line-among(plates[i,..,k], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])== num_rows_line-controls_in_plate_column[i,k]);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])== num_rows_line-emptywells_in_plate_column[i,k]-controls_only_in_plate_column[i,k]);


constraint forall(i in Plates, j in Rows) (sum([ experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments]) == experiments_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments]) == experiments_in_plate_column[i,k]);

%experiments_in_plate_row
constraint forall(i in Plates, j in Rows) (experiments_in_plate_row[i,j] == num_cols_line-among(plates[i,j,..], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, j in Rows) (experiments_in_plate_row[i,j] == num_cols_line-controls_in_plate_row[i,j]);

constraint forall(i in Plates, j in Rows) (experiments_in_plate_row[i,j] == num_cols_line-emptywells_in_plate_row[i,j]-controls_only_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (experiments_in_plate_column[i,k] == num_rows_line-among(plates[i,..,k], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, k in Columns) (experiments_in_plate_column[i,k] == num_rows_line-controls_in_plate_column[i,k]);

constraint forall(i in Plates, k in Columns) (experiments_in_plate_column[i,k] == num_rows_line-emptywells_in_plate_column[i,k]-controls_only_in_plate_column[i,k]);


%% Below suggested by Gustav. It improves flattening but worsens the solving time
% constraint forall(i in Plates, j in Rows) (
%   let{
%     var int: exprInRow;
%     constraint exprInRow = among(plates[i,j,..], 1..experiments)::domain; 
%     %sum([experiment_row[l] == j /\ experiment_plate[l] == i| l in 1..experiments]);
%   } in exprInRow <= num_cols_line-among([plates[i,j,k] | k in Columns], (ControlValuesWithEmpty)));
  

%% Replaced by constraint below (suggested by Gustav)
%constraint forall(i in Plates, k in Columns) (sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])<= num_rows_line-among(plates[i,..,k], (ControlValuesWithEmpty)));

%% Gustav:
constraint forall(i in Plates, k in Columns) (
% sum([ experiment_column[l] == k /\ experiment_plate[l] == i| l in 1..experiments])
(among(plates[i,..,k], 1..experiments)::domain) <= num_rows_line-among([plates[i,j,k] | j in Rows], (ControlValuesWithEmpty))::domain);

                                      


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Different compound replicas must appear on different plates (if possible)
constraint if replicates_on_different_plates then forall(l in 1..compounds) (nvalue(min(num_plates,compound_replicates[l]), [experiment_plate[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])) endif; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Different compound replicas must appear on the same plate
constraint if replicates_on_same_plate then forall(l in 1..compounds) (all_equal([experiment_plate[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + 1] | i in 0..(compound_replicates[l]-1)])) endif;


constraint if concentrations_on_different_rows then forall(l in 1..compounds) (forall(conc in 0..compound_concentrations[l]-1)(alldifferent([experiment_row[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + conc + 1] | i in 0..(min(compound_replicates[l],num_rows_line)-1)])::domain)) endif; 

constraint if concentrations_on_different_columns then forall(l in 1..compounds) (forall(conc in 0..compound_concentrations[l]-1)(alldifferent([experiment_column[(l-1)*max_compound_concentrations + i*compounds*max_compound_concentrations + conc + 1] | i in 0..(min(compound_replicates[l],num_cols_line)-1)])::domain)) endif; 


%% Balancing compounds between plates: distribute the compounds equitatively among all the plates


%% TODO: could this lower bound be stronger?
%% FIX ME!
%int: at_least = max_compound_concentrations*floor((compounds*replicates)/num_plates) + combination_concentrations*floor((combinations*replicates)/num_plates); 

%int: at_least = min(compound_concentrations)*floor(sum(compound_replicates)/num_plates) + combination_concentrations*floor((combinations*replicates)/num_plates); 

%% FIX ME!
%% FIX ME!!
%int: at_most = min(max_compound_concentrations*ceil(sum(compound_replicates)/num_plates) + combination_concentrations*ceil((combinations*replicates)/num_plates), 
 %   inner_plate_size -(sum([floor(control_replicates[i]*control_concentrations[i]/num_plates) | i in 1..num_controls])));



%% This implied constraint is very important for propagation!

%constraint global_cardinality_low_up(experiment_plate,[ i | i in Plates],[ at_least | i in Plates],[ at_most | i in Plates]);

% I could make this stronger, but I would need to assume that experiments can indeed be distrubuted in a balanced way.
% I would like to do something that allows example-02 and example-03 to work
% This constraint assumes that there is no plate with only controls, but that seems fair enough


constraint global_cardinality_low_up(experiment_plate,[ i | i in Plates],[ min_compounds_plate | i in Plates],[ inner_plate_size | i in Plates]);

%% TODO: Think more about this constraint and what happens when there are replicates, etc.
%% This option is possibly going to make some plans unfeasible (when there are different numbers of replicates, etc).
constraint if sorted_compounds == true /\ max(compound_replicates)<=1 then sorted_except_0(experiment_plate,num_plates,at_least_compounds,at_most_compounds) endif;

%% WARNING! This option assumes that compounds can be placed on plates in order! This works, for example, when all compounds have the same 
% number of replicates and concentrations and one would like to generate several layouts with the compounds on the exact same plates.
% Otherwise, experiment_plate needs to be given as input
constraint if sorted_compounds == true /\ replicates_on_same_plate == true  then sorted_except_0(experiment_plate[1..compounds*max(compound_concentrations)],num_plates,at_least_compounds div num_plates,at_most_compounds div num_plates) endif;



% Balancing compounds between plates (and knowing that there must be some sort of balance!)
%% FIX ME!
%% I want to remove these constraints but I need to make other stronger first
int: at_least_compounds = min(compound_concentrations++[infinity])*(sum(compound_replicates) div num_plates);
int: at_most_compounds = max_compound_concentrations*ceil(sum(compound_replicates)/num_plates);

%% TODO: False only if there are no compounds
constraint if at_least_compounds <= at_most_compounds then global_cardinality_low_up([experiment_plate[i] | i in 1..experiments] ,[ i | i in Plates],[ at_least_compounds | i in Plates],[ at_most_compounds | i in Plates]) endif;




%% Implied constraints 
%% These constraints are too weak when the number of plates increases
constraint if concentrations_on_different_rows then global_cardinality_low_up(experiment_row, [i | i in Rows], [num_plates*(at_least_compounds div num_rows_line)| i in Rows], [num_plates*num_cols_line| i in Rows]) endif;

constraint if concentrations_on_different_columns then global_cardinality_low_up(experiment_column, [i | i in Columns], [num_plates*floor(at_least_compounds/num_cols_line)| i in Columns], [num_plates*num_rows_line| i in Columns]) endif;

constraint if concentrations_on_different_rows then forall(j in Rows)(sum(experiments_in_plate_row[..,j])>=num_plates*(at_least_compounds div num_rows_line)) endif;
constraint if concentrations_on_different_columns then forall(k in Columns)(sum(experiments_in_plate_column[..,k])>=num_plates*(at_least_compounds div num_cols_line)) endif;

constraint if concentrations_on_different_rows then forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j]>=(at_least_compounds div num_rows_line)) endif;
constraint if concentrations_on_different_columns then forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k]>=(at_least_compounds div num_cols_line)) endif;

constraint if concentrations_on_different_rows then forall(i in Plates, j in Rows)(experiments_in_plate_row[i,j]<=ceil(at_most_compounds / num_rows_line)) endif;
constraint if concentrations_on_different_columns then forall(i in Plates, k in Columns)(experiments_in_plate_column[i,k]<=ceil(at_most_compounds / num_cols_line)) endif;

array [Plates,Vertical] of var 0..num_cols_line*(num_rows_line div 2): ul_half_plates;
array [Plates,Horizontal] of var 0..num_rows_line*(num_cols_line div 2): lr_half_plates;

%%% Counting the number of experiments in the upper and lower half-plates
constraint forall(i in Plates) (among(ul_half_plates[i,upper],plates[i,1..num_rows_line div 2,..], 1..experiments)::domain);
constraint forall(i in Plates) (among(ul_half_plates[i,lower],plates[i,(num_rows_line div 2)+1..num_rows_line,..], 1..experiments)::domain);

constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_row[e] <= (num_rows_line div 2)) == ul_half_plates[i,upper]);
constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_row[e] > (num_rows_line div 2)) == ul_half_plates[i,lower]);

constraint (sum(e in 1..experiments)(experiment_row[e] in 1..(num_rows_line div 2)) == sum(i in Plates)(ul_half_plates[i,upper]));
constraint (sum(e in 1..experiments)(experiment_row[e] in (num_rows_line div 2)+1..num_rows_line) == sum(i in Plates)(ul_half_plates[i,lower]));

constraint if concentrations_on_different_rows /\ even_rows then forall(i in Plates)(sum(experiments_in_plate_row[i,1..(num_rows_line div 2)]) == ul_half_plates[i,upper]) endif;
constraint if concentrations_on_different_rows /\ even_rows then forall(i in Plates)(sum(experiments_in_plate_row[i,(num_rows_line div 2)+1..num_rows_line]) == ul_half_plates[i,lower]) endif;

%% The following constraints should be redundant when using the two constraints above.
constraint (among(sum(i in Plates)(ul_half_plates[i,upper]),experiment_row, 1..(num_rows_line div 2))::domain);
constraint (among(sum(i in Plates)(ul_half_plates[i,lower]),experiment_row, (num_rows_line div 2)+1..num_rows_line)::domain);

constraint (among(sum(i in Plates)(lr_half_plates[i,left]),experiment_column, 1..(num_cols_line div 2))::domain);
constraint (among(sum(i in Plates)(lr_half_plates[i,right]),experiment_column, (num_cols_line div 2)+1..num_cols_line)::domain);


constraint if even_rows then (forall(i in Plates)(abs(ul_half_plates[i,upper]-ul_half_plates[i,lower])<=1)) endif;

%%% Counting the number of experiments in the left and right half-plates
constraint forall(i in Plates) (among(lr_half_plates[i,left],plates[i,..,1..num_cols_line div 2], 1..experiments)::domain);
constraint forall(i in Plates) (among(lr_half_plates[i,right],plates[i,..,(num_cols_line div 2)+1..num_cols_line], 1..experiments)::domain);

constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_column[e] <= (num_cols_line div 2)) == lr_half_plates[i,left]);
constraint forall(i in Plates) (sum(e in 1..experiments)(experiment_plate[e] == i /\ experiment_column[e] > (num_cols_line div 2)) == lr_half_plates[i,right]);

constraint if even_columns then (sum(e in 1..experiments)(experiment_column[e] in 1..(num_cols_line div 2)) == sum(i in Plates)(lr_half_plates[i,left])) endif;
constraint if even_columns then (sum(e in 1..experiments)(experiment_column[e] in (num_cols_line div 2)+1..num_cols_line) == sum(i in Plates)(lr_half_plates[i,right])) endif;

constraint if even_columns then (forall(i in Plates)(abs(lr_half_plates[i,left]-lr_half_plates[i,right])<=1)) endif;




%%% Counting the number of experiments in left-most and right-most half-plates
%constraint forall(i in Plates) (among(lr_half_plates[i,left],[plates[i,j,k] | j in Rows, k in 1..ceil(num_cols_line/2)], 1..experiments)); %%::domain
%constraint forall(i in Plates) (among(lr_half_plates[i,right],[plates[i,j,k] | j in Rows, k in floor(num_cols_line/2)+1..num_cols_line], 1..experiments));

%constraint forall(i in Plates) (sum(e in 1..experiments, k in 1..ceil(num_cols_line/2))(experiment_plate[e] == i /\ experiment_column[e] == k) == lr_half_plates[i,left]);
%constraint forall(i in Plates) (sum(e in 1..experiments, k in ceil(num_cols_line/2)+1..num_cols_line)(experiment_plate[e] == i /\ experiment_column[e] == k) == lr_half_plates[i,right]);


%array [Plates,Vertical,Horizontal] of var floor((at_least_compounds+at_least_combinations)/4)..ceil((at_most_compounds+at_most_combinations)/4): quarter_plates;

% Balancing experiments in quarters
%constraint forall(i in Plates) (among(quarter_plates[i,upper,left],[plates[i,j,k] | j in 1..ceil(num_rows_line/2), k in 1..ceil(num_cols_line/2)], 1..experiments));
%constraint forall(i in Plates) (among(quarter_plates[i,upper,right],[plates[i,j,k] | j in 1..ceil(num_rows_line/2), k in floor(num_cols_line/2)+1..num_cols_line], 1..experiments));

%constraint forall(i in Plates) (among(quarter_plates[i,lower,left],[plates[i,j,k] | j in floor(num_rows_line/2)+1..num_rows_line, k in 1..ceil(num_cols_line/2)], 1..experiments));
%constraint forall(i in Plates) (among(quarter_plates[i,lower,right],[plates[i,j,k] | j in floor(num_rows_line/2)+1..num_rows_line, k in floor(num_cols_line/2)+1..num_cols_line], 1..experiments));


%%% NEW %%%
%% Implied constraints %%
constraint sum(ul_half_plates) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]);
constraint sum(lr_half_plates) == sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds]);

constraint forall(v in Vertical)(sum(ul_half_plates[..,v]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])/2));
constraint forall(h in Horizontal)(sum(lr_half_plates[..,h]) <= ceil(sum([compound_concentrations[i]*compound_replicates[i] | i in 1..compounds])/2));

constraint forall(i in Plates, v in Vertical) ( ul_half_plates[i,v] <= ((sum([ (experiment_plate[c] == i) | c in 1..experiments]) + 1) div 2));
constraint forall(i in Plates, h in Horizontal) ( lr_half_plates[i,h] <= ((sum([ (experiment_plate[c] == i) | c in 1..experiments]) + 1) div 2));
                                                                
constraint forall(i in Plates, v in Vertical)(((floor(sum(compound_replicates)/num_plates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations) div 2 <= ul_half_plates[i,v]);  

constraint forall(i in Plates, h in Horizontal)(((floor(sum(compound_replicates)/num_plates)-1)*min(compound_concentrations++[infinity]) + max_compound_concentrations) div 2 <= lr_half_plates[i,h]); 

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

array[int] of float: controls_per_plate = [control_replicates[floor((i-1)/max_control_concentrations)+1]*(((i-1) mod max_control_concentrations)<control_concentrations[floor((i-1)/max_control_concentrations)+1])/num_plates | i in 1..num_controls*max_control_concentrations];

array[int] of int: min_controls_per_plate = [floor(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

array[int] of int: max_controls_per_plate = [ceil(controls_per_plate[i]) | i in 1..num_controls*max_control_concentrations];

constraint forall(i in Plates)(global_cardinality_low_up([plates[i,j,k] | j in Rows, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], min_controls_per_plate, max_controls_per_plate));



%% I need to work on these constraints, which seem to be quite ineficient. I believe I need to use variables for halves plates, maybe even for quarter plates.

array [Plates,Vertical,1..num_controls*max_control_concentrations] of var int: controls_ul_plates;

array [Plates,Vertical] of var int: sum_controls_ul_plates;

array [Plates,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_lr_plates;

array [Plates,Horizontal] of var int: sum_controls_lr_plates;

array [Plates,Vertical,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_quarter_plates;

array [Plates,Rows,{0} union 1..num_controls*max_control_concentrations] of var int: controls_row_plates;

array [Plates,Columns,{0} union 1..num_controls*max_control_concentrations] of var int: controls_column_plates;


constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..(num_rows_line div 2), k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,upper,d] | d in 1..num_controls*max_control_concentrations]));

constraint forall(i in Plates)(global_cardinality([plates[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line, k in Columns], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_ul_plates[i,lower,d] | d in 1..num_controls*max_control_concentrations]));


%% Defining controls_quarter_plates variables
constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..(num_rows_line div 2), k in 1..(num_cols_line div 2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,upper,left,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in 1..(num_rows_line div 2), k in (num_cols_line div 2)+1..num_cols_line], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,upper,right,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line, k in 1..(num_cols_line div 2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,lower,left,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line, k in (num_cols_line div 2)+1..num_cols_line], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_quarter_plates[i,lower,right,d] | d in 1..num_controls*max_control_concentrations])) endif;


constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]<=ceil(max_controls_per_plate[d]/2));
constraint forall(i in Plates, v in Vertical, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,v,d]>=floor(min_controls_per_plate[d]/2));

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

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)( sum(v in Vertical)(controls_ul_plates[i,v,d]) <= max_controls_per_plate[d]);

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

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in 1..floor(num_cols_line/2)], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,left,d] | d in 1..num_controls*max_control_concentrations])) endif;

constraint if even_columns then forall(i in Plates)(global_cardinality([plates[i,j,k] | j in Rows, k in floor(num_cols_line/2)+1..num_cols_line], [experiments+d | d in 1..num_controls*max_control_concentrations], [controls_lr_plates[i,right,d] | d in 1..num_controls*max_control_concentrations])) endif;


constraint if even_columns then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]<=ceil(max_controls_per_plate[d]/2)) endif;

constraint if even_columns then forall(i in Plates, h in Horizontal, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,h,d]>= (min_controls_per_plate[d] div 2)) endif;

constraint if even_columns then forall(i in Plates, d in 1..num_controls*max_control_concentrations)(abs(controls_lr_plates[i,left,d] - controls_lr_plates[i,right,d]) <= 1) endif;

constraint forall(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_lr_plates[i,left,d] + controls_lr_plates[i,right,d] <= max_controls_per_plate[d]);

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

constraint forall(i in Plates)( ul_half_plates[i,upper] + sum_controls_ul_plates[i,upper] + count(plates[i,1..num_rows_line div 2,..], 0) == inner_plate_size div 2);
constraint forall(i in Plates)( ul_half_plates[i,lower] + sum_controls_ul_plates[i,lower] + count(plates[i,(num_rows_line div 2)+1..num_rows_line,..], 0) == inner_plate_size div 2);

constraint forall(i in Plates)( lr_half_plates[i,left] + sum_controls_lr_plates[i,left] + count(plates[i,..,1..num_cols_line div 2], 0) == inner_plate_size div 2);
constraint forall(i in Plates)( lr_half_plates[i,right] + sum_controls_lr_plates[i,right] + count(plates[i,..,(num_cols_line div 2)+1..num_cols_line], 0) == inner_plate_size div 2);


%%->constraint forall(i in Plates)(ul_half_plates[i,upper] + ul_half_plates[i,lower] + sum(d in 1..num_controls*max_control_concentrations,v in Vertical)(controls_ul_plates[i,v,d]) + count([plates[i,j,k] | j in 1+size_empty_edge..num_rows_line-size_empty_edge, k in 1+size_empty_edge..num_cols_line-size_empty_edge], 0) == inner_plate_size);


%% Is this helping? I don't think so! But it might...
%%->constraint (sum(r in 1..floor(num_rows_line/2))(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,upper,d]) + count([plates[i,j,k] | i in Plates, j in 1..floor(num_rows_line/2), k in Columns], 0));

%%->constraint (sum(r in floor(num_rows_line/2)+1..num_rows_line)(controls_in_row[r]) == sum(i in Plates, d in 1..num_controls*max_control_concentrations)(controls_ul_plates[i,lower,d]) + count([plates[i,j,k] | i in Plates, j in floor(num_rows_line/2)+1..num_rows_line, k in Columns], 0));


%%% Controls per row across plates%%%

%constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..num_rows_line-size_empty_edge) (controls_in_row[j] <= ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_rows_line-2*size_empty_edge)) + 2*size_empty_edge*num_plates) endif;

%constraint if temp_bool /\ interconnected_plates then forall(j in 1+size_empty_edge..num_rows_line-size_empty_edge) (controls_in_row[j] >= floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_rows_line-2*size_empty_edge)) + 2*size_empty_edge*num_plates) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] <= ceil((total_controls+emptywells)/num_rows_line) ) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] >= floor((total_controls+emptywells)/num_rows_line) ) endif;

constraint forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] <= ceil(ceil(total_controls/num_plates)/num_rows_line) ) ;

constraint forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] >= ((total_controls div num_plates) div num_rows_line) ) ;


%%% Controls per row per plate %%%
%% Relaxing this constraint a bit...
%constraint forall(i in Plates, j in 1+size_empty_edge..num_rows_line-size_empty_edge) (controls_in_plate_row[i,j] <= control_slack + ceil(ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_rows_line-2*size_empty_edge))/num_plates) + 2*size_empty_edge);

%constraint forall(i in Plates, j in 1+size_empty_edge..num_rows_line-size_empty_edge) (controls_in_plate_row[i,j] >= floor(floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_rows_line-2*size_empty_edge))/num_plates) + 2*size_empty_edge);


constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] <= control_slack + ceil(ceil((total_controls+emptywells)/num_plates)/num_rows_line) );

constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] >= floor(floor((total_controls+emptywells)/num_plates)/num_rows_line) );

%constraint if emptywells == 0 then forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] == controls_in_plate_row[i,j] ) else forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] <= controls_in_plate_row[i,j] ) endif;

constraint forall(i in Plates, j in Rows)(controls_in_plate_row[i,j] == controls_only_in_plate_row[i,j] + emptywells_in_plate_row[i,j]);

constraint forall(i in Plates, k in Columns) (controls_in_plate_column[i,k] <= control_slack + ceil(ceil((total_controls+emptywells)/num_plates)/num_cols_line) );

constraint forall(i in Plates, k in Columns) (controls_in_plate_column[i,k] >= floor(floor((total_controls+emptywells)/num_plates)/num_cols_line) );

constraint forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] + emptywells_in_plate_column[i,k] == controls_in_plate_column[i,k] );

constraint if interconnected_plates then forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] >= (controls_in_row[j] div num_plates)) endif;

constraint if interconnected_plates then forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] <= ((controls_in_row[j]+num_plates-1)div num_plates)) endif;


%%% Controls per column across plates%%%
 
constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] <= ceil((total_controls+emptywells)/num_cols_line) ) endif;

constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] >= floor((total_controls+emptywells)/num_cols_line) ) endif;

constraint forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] <= ceil(ceil(total_controls / num_plates) / num_cols_line) );

constraint forall(i in Plates, k in Columns) (controls_only_in_plate_column[i,k] >= ((total_controls div num_plates) div num_cols_line) );



% ==============================================================================
% SECTION 15: CONTROL DISTRIBUTION - ROW & COLUMN BALANCE
% ==============================================================================
%
% Constraints ensuring controls are evenly distributed across rows and
% columns within each plate, including half-row and half-column balancing.
%
% ==============================================================================

%%% Attempting to balance controls inside each row

array [Rows,Horizontal] of var 0..num_plates*num_cols_line: controls_in_half_row;

array [Rows,Horizontal] of var 0..num_plates*num_cols_line: controls_only_in_half_row; %Does NOT include empty wells

array [Plates,Rows,Horizontal] of var 0..ceil(num_cols_line/2): controls_in_half_row_plate; %includes empty wells

array [Plates,Rows,Horizontal] of var 0..ceil(num_cols_line/2): controls_only_in_half_row_plate; %Does NOT include empty wells

constraint forall(j in Rows) ( among(controls_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..floor(num_cols_line/2)], (ControlValuesWithEmpty))::domain);

constraint forall(j in Rows) ( among(controls_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in floor(num_cols_line/2)+1..num_cols_line], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..floor(num_cols_line/2)], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,right],[plates[i,j,k] | k in floor(num_cols_line/2)+1..num_cols_line], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_only_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..(num_cols_line div 2)], ( ControlValues))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_only_in_half_row_plate[i,j,right],[plates[i,j,k] | k in (num_cols_line div 2)+1..num_cols_line], (ControlValues))::domain);

% controls_only_in_half_row
constraint forall(j in Rows) ( among(controls_only_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..(num_cols_line div 2)], ( ControlValues))::domain);

constraint forall(j in Rows) ( among(controls_only_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in (num_cols_line div 2)+1..num_cols_line], ( ControlValues))::domain);


constraint forall(j in Rows, h in Horizontal) (controls_in_half_row[j,h] == sum(i in Plates)(controls_in_half_row_plate[i,j,h]));

constraint forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] == sum(h in Horizontal)(controls_only_in_half_row_plate[i,j,h]));

constraint forall(i in Plates, j in Rows)(
  balance_in_half_rows(controls_only_in_half_row_plate, i, j)
);

constraint forall(i in Plates, k in Columns)(
  balance_in_half_columns(controls_only_in_half_column_plate, i, k)
);


%constraint forall(j in 1+size_empty_edge..num_rows_line-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_rows_line-2*size_empty_edge)) + 2*size_empty_edge*num_plates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..num_rows_line-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_cols_line-2*size_empty_edge)) + 2*size_empty_edge*num_plates)/2) >= controls_in_half_row[j,right]);

constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/num_rows_line) div 2) <= controls_in_half_row[j,left]) endif;

constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/num_rows_line) div 2) <= controls_in_half_row[j,right]) endif;

constraint if interconnected_plates then forall(j in Rows, h in Horizontal) ( (controls_in_row[j] div 2) <= controls_in_half_row[j,h] ) endif;
% constraint if interconnected_plates then forall(j in Rows, h in Horizontal) ( ((controls_in_row[j]+1) div 2) >= controls_in_half_row[j,h] ) endif;

constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] == (controls_in_half_row[j,left] + controls_in_half_row[j,right])) endif;


%constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..num_rows_line-size_empty_edge) ( floor(floor((floor((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/num_plates)/(num_rows_line-2*size_empty_edge))/2) + size_empty_edge) <= controls_in_half_row_plate[i,j,h]) endif;

constraint forall(i in Plates, h in Horizontal, j in Rows) ( (controls_in_plate_row[i,j] div 2) <= controls_in_half_row_plate[i,j,h]);

constraint forall(i in Plates, h in Horizontal, j in Rows) ( ( (controls_in_plate_row[i,j]+1) div 2) >= controls_in_half_row_plate[i,j,h]);


%constraint if temp_bool then forall(i in Plates, h in Horizontal, j in 1+size_empty_edge..num_rows_line-size_empty_edge) ( ceil(ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/num_plates)/(num_rows_line-2*size_empty_edge))/2) + size_empty_edge) >= controls_in_half_row_plate[i,j,h]) endif;


constraint forall(i in Plates, j in Rows) (controls_in_plate_row[i,j] == controls_in_half_row_plate[i,j,left] + controls_in_half_row_plate[i,j,right]);

%%% Half-row and half-column control distribution variables
constraint forall(j in Rows) ( among(controls_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..floor(num_cols_line/2)], (ControlValuesWithEmpty))::domain);
constraint forall(j in Rows) ( among(controls_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in floor(num_cols_line/2)+1..num_cols_line], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..floor(num_cols_line/2)], (ControlValuesWithEmpty))::domain);
constraint forall(i in Plates, j in Rows) ( among(controls_in_half_row_plate[i,j,right],[plates[i,j,k] | k in floor(num_cols_line/2)+1..num_cols_line], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, j in Rows) ( among(controls_only_in_half_row_plate[i,j,left],[plates[i,j,k] | k in 1..(num_cols_line div 2)], ( ControlValues))::domain);
constraint forall(i in Plates, j in Rows) ( among(controls_only_in_half_row_plate[i,j,right],[plates[i,j,k] | k in (num_cols_line div 2)+1..num_cols_line], (ControlValues))::domain);

% controls_only_in_half_row
constraint forall(j in Rows) ( among(controls_only_in_half_row[j,left],[plates[i,j,k] | i in Plates, k in 1..(num_cols_line div 2)], ( ControlValues))::domain);
constraint forall(j in Rows) ( among(controls_only_in_half_row[j,right],[plates[i,j,k] | i in Plates, k in (num_cols_line div 2)+1..num_cols_line], ( ControlValues))::domain);

constraint forall(j in Rows, h in Horizontal) (controls_in_half_row[j,h] == sum(i in Plates)(controls_in_half_row_plate[i,j,h]));
constraint forall(i in Plates, j in Rows) (controls_only_in_plate_row[i,j] == sum(h in Horizontal)(controls_only_in_half_row_plate[i,j,h]));

constraint forall(i in Plates, j in Rows, h in Horizontal) (controls_only_in_plate_row[i,j] div 2 <= controls_only_in_half_row_plate[i,j,h]);
constraint forall(i in Plates, j in Rows, h in Horizontal) ((controls_only_in_plate_row[i,j]+1) div 2 >= controls_only_in_half_row_plate[i,j,h]);
constraint forall(i in Plates, k in Columns, v in Vertical) (controls_only_in_plate_column[i,k] div 2 <= controls_only_in_half_column_plate[i,k,v]);
constraint forall(i in Plates, k in Columns, v in Vertical) ((controls_only_in_plate_column[i,k]+1) div 2 >= controls_only_in_half_column_plate[i,k,v]);

constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/num_rows_line) div 2) <= controls_in_half_row[j,left]) endif;
constraint if interconnected_plates then forall(j in Rows) ( (floor((total_controls+emptywells)/num_rows_line) div 2) <= controls_in_half_row[j,right]) endif;
constraint if interconnected_plates then forall(j in Rows, h in Horizontal) ( (controls_in_row[j] div 2) <= controls_in_half_row[j,h] ) endif;
constraint if interconnected_plates then forall(j in Rows) (controls_in_row[j] == (controls_in_half_row[j,left] + controls_in_half_row[j,right])) endif;

constraint forall(i in Plates, h in Horizontal, j in Rows) ( (controls_in_plate_row[i,j] div 2) <= controls_in_half_row_plate[i,j,h]);
constraint forall(i in Plates, h in Horizontal, j in Rows) ( ( (controls_in_plate_row[i,j]+1) div 2) >= controls_in_half_row_plate[i,j,h]);




%%% Attempting to balance controls inside each column

array [Columns,Vertical] of var 0..num_plates*num_rows_line: controls_in_half_column;

array [Columns,Vertical] of var 0..num_plates*num_rows_line: controls_only_in_half_column; %Does NOT include empty wells

array [Plates,Columns,Vertical] of var 0..(num_rows_line div 2): controls_in_half_column_plate; %includes empty wells

array [Plates,Columns,Vertical] of var 0..(num_rows_line div 2): controls_only_in_half_column_plate; %Does NOT include empty wells

array [Plates,Rows,Horizontal] of var 0..ceil(num_cols_line/2): experiments_in_half_row_plate;

array [Plates,Columns,Vertical] of var 0..ceil(num_rows_line/2): experiments_in_half_column_plate;  

constraint forall(k in Columns) ( among(controls_in_half_column[k,upper],[plates[i,j,k] | i in Plates, j in 1..(num_rows_line div 2)], (ControlValuesWithEmpty))::domain);

constraint forall(k in Columns) ( among(controls_in_half_column[k,lower],[plates[i,j,k] | i in Plates, j in (num_rows_line div 2)+1..num_rows_line], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, k in Columns) ( among(controls_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(num_rows_line div 2)], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, k in Columns) ( among(controls_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line], (ControlValuesWithEmpty))::domain);

constraint forall(i in Plates, k in Columns) ( among(controls_only_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(num_rows_line div 2)], ( ControlValues))::domain);
	
constraint forall(i in Plates, k in Columns) ( among(controls_only_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line], ( ControlValues))::domain);

%% controls_only_in_half_column
constraint forall(k in Columns) ( among(controls_only_in_half_column[k,upper],[plates[i,j,k] | i in Plates, j in 1..(num_rows_line div 2)], ( ControlValues))::domain);

constraint forall(k in Columns) ( among(controls_only_in_half_column[k,lower],[plates[i,j,k] | i in Plates, j in (num_rows_line div 2)+1..num_rows_line], ( ControlValues))::domain);

%%% Defining experiments_in_half_row_plate and experiments_in_half_column_plate

constraint forall(i in Plates, j in Rows) ( among(experiments_in_half_row_plate[i,j,left], [plates[i,j,k] | k in 1..(num_cols_line div 2)], (1..experiments))::domain);
constraint forall(i in Plates, j in Rows) ( among(experiments_in_half_row_plate[i,j,right],[plates[i,j,k] | k in (num_cols_line div 2)+1..num_cols_line], (1..experiments))::domain);

constraint forall(i in Plates, k in Columns) ( among(experiments_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(num_rows_line div 2)], (1..experiments))::domain);
constraint forall(i in Plates, k in Columns) ( among(experiments_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line], (1..experiments))::domain);

constraint forall(i in Plates, j in Rows) ( sum(h in Horizontal)(experiments_in_half_row_plate[i,j,h]) == experiments_in_plate_row[i,j]);
constraint forall(i in Plates, k in Columns) ( sum(v in Vertical)(experiments_in_half_column_plate[i,k,v]) == experiments_in_plate_column[i,k]);

constraint forall(i in Plates, j in Rows)(
  balance_in_half_rows(experiments_in_half_row_plate, i, j)
);

constraint forall(i in Plates, k in Columns)(
  balance_in_half_columns(experiments_in_half_column_plate, i, k)
);

constraint forall(i in Plates, h in Horizontal, j in Rows) ( (num_cols_line div 2) == controls_in_half_row_plate[i,j,h] + experiments_in_half_row_plate[i,j,h]);
constraint forall(i in Plates, v in Vertical, k in Columns) ( (num_rows_line div 2) == controls_in_half_column_plate[i,k,v] + experiments_in_half_column_plate[i,k,v]);




constraint forall(k in Columns, v in Vertical) (controls_in_half_column[k,v] == sum(i in Plates)(controls_in_half_column_plate[i,k,v]));

constraint forall(i in Plates, k in Columns) ( controls_in_plate_column[i,k] == sum(v in Vertical)(controls_in_half_column_plate[i,k,v]));

constraint forall(i in Plates, v in Vertical, k in Columns) ( controls_in_plate_column[i,k] <= 2*controls_in_half_column_plate[i,k,v] + 1);

constraint forall(i in Plates, v in Vertical, k in Columns) ( controls_in_plate_column[i,k] >= 2*controls_in_half_column_plate[i,k,v] - 1);

constraint forall(i in Plates, v in Vertical, k in Columns) ( (controls_in_plate_column[i,k] div 2) <= controls_in_half_column_plate[i,k,v]);

constraint forall(i in Plates, v in Vertical, k in Columns) ( ( (controls_in_plate_column[i,k]+1) div 2) >= controls_in_half_column_plate[i,k,v]);

constraint forall(i in Plates, k in Columns) ( controls_only_in_plate_column[i,k] == sum(v in Vertical)(controls_only_in_half_column_plate[i,k,v]));

%constraint forall(j in 1+size_empty_edge..num_rows_line-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_rows_line-2*size_empty_edge)) + 2*size_empty_edge*num_plates)/2) >= controls_in_half_row[j,left]);

%constraint forall(j in 1+size_empty_edge..num_rows_line-size_empty_edge) ( ceil((ceil((sum(l in 1..num_controls)(control_replicates[l]*control_concentrations[l])+emptywells)/(num_cols_line-2*size_empty_edge)) + 2*size_empty_edge*num_plates)/2) >= controls_in_half_row[j,right]);

constraint if interconnected_plates then forall(k in Columns, v in Vertical) ( (floor((total_controls+emptywells)/num_cols_line)  div 2) <= controls_in_half_column[k,v]) endif;

constraint if interconnected_plates then forall(k in Columns) (controls_in_column[k] == (controls_in_half_column[k,upper] + controls_in_half_column[k,lower])) endif;

%%% Defining experiments_in_half_row_plate /\ experiments_in_half_column_plate

constraint forall(i in Plates, j in Rows) ( among(experiments_in_half_row_plate[i,j,left], [plates[i,j,k] | k in 1..(num_cols_line div 2)], (1..experiments))::domain);
constraint forall(i in Plates, j in Rows) ( among(experiments_in_half_row_plate[i,j,right],[plates[i,j,k] | k in (num_cols_line div 2)+1..num_cols_line], (1..experiments))::domain);

constraint forall(i in Plates, k in Columns) ( among(experiments_in_half_column_plate[i,k,upper],[plates[i,j,k] | j in 1..(num_rows_line div 2)], (1..experiments))::domain);
constraint forall(i in Plates, k in Columns) ( among(experiments_in_half_column_plate[i,k,lower],[plates[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line], (1..experiments))::domain);

constraint forall(i in Plates, j in Rows) ( sum(h in Horizontal)(experiments_in_half_row_plate[i,j,h]) == experiments_in_plate_row[i,j]);
constraint forall(i in Plates, k in Columns) ( sum(v in Vertical)(experiments_in_half_column_plate[i,k,v]) == experiments_in_plate_column[i,k]);

constraint forall(i in Plates, j in Rows, h in Horizontal) ( experiments_in_half_row_plate[i,j,h] >= (experiments_in_plate_row[i,j] div 2)); %%WARNING
constraint forall(i in Plates, j in Rows, h in Horizontal) ( experiments_in_half_row_plate[i,j,h] <= ((experiments_in_plate_row[i,j] + 1) div 2)); %%WARNING

constraint forall(i in Plates, k in Columns, v in Vertical) (experiments_in_half_column_plate[i,k,v] >= (experiments_in_plate_column[i,k] div 2)); %%WARNING
constraint forall(i in Plates, k in Columns, v in Vertical) (experiments_in_half_column_plate[i,k,v] <= ((experiments_in_plate_column[i,k] + 1) div 2)); %%WARNING

constraint forall(i in Plates, h in Horizontal, j in Rows) ( (num_cols_line div 2) == controls_in_half_row_plate[i,j,h] + experiments_in_half_row_plate[i,j,h]);
constraint forall(i in Plates, v in Vertical, k in Columns) ( (num_rows_line div 2) == controls_in_half_column_plate[i,k,v] + experiments_in_half_column_plate[i,k,v]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Balancing each type of control per row

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (count_eq(plates[i,j,..],experiments+c,controls_row_plates[i,j,c]));

constraint forall(i in Plates, j in Rows) (count_eq(plates[i,j,..],0,controls_row_plates[i,j,0]));

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (controls_row_plates[i,j,c] <= ceil(max_controls_per_plate[c]/num_rows_line));

constraint forall(i in Plates, j in Rows, c in 1..num_controls*max_control_concentrations) (controls_row_plates[i,j,c] >= (min_controls_per_plate[c] div num_rows_line));

constraint forall(i in Plates, j in Rows) (sum(controls_row_plates[i,j,..]) + experiments_in_plate_row[i,j] == num_cols_line);

constraint sum(controls_row_plates[..,..,0]) == emptywells;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Balancing each type of control per column

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (count_eq(plates[i,..,k],experiments+c,controls_column_plates[i,k,c]));

constraint forall(i in Plates, k in Columns) (count_eq(plates[i,..,k],0,controls_column_plates[i,k,0]));

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (controls_column_plates[i,k,c] <= ceil(max_controls_per_plate[c]/num_cols_line));

constraint forall(i in Plates, k in Columns, c in 1..num_controls*max_control_concentrations) (controls_column_plates[i,k,c] >= (min_controls_per_plate[c] div num_cols_line));

constraint forall(i in Plates, k in Columns) (sum(controls_column_plates[i,k,..]) + experiments_in_plate_column[i,k] == num_rows_line);

constraint sum(controls_column_plates[..,..,0]) == emptywells;

%% Implied all_different constraint: all experiments/compounds are different
constraint alldifferent_except(plates, ControlValuesWithEmpty):: domain;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ==============================================================================
% SECTION 16: CONTROL SPREADING CONSTRAINTS
% ==============================================================================
%
% Constraints preventing controls of the same type from being placed
% adjacent to each other, handling special cases for edges and corners.
%
% ==============================================================================

%%% Distributing controls

%% Controls of the same type are not located next to each other

%% Implied constraint
constraint if (min(spread_control++[1])>0) then forall(i in Plates, j in 1..num_rows_line-1, k in 1..num_cols_line-1) (alldifferent_except_0([plates[i,j,k],plates[i,j+1,k],plates[i,j,k+1],plates[i,j+1,k+1]])):: domain endif;


% ──────────────────────────────────────────────────────────────────────────────
% Section 16.1: Control Layout Helper Variable
% ──────────────────────────────────────────────────────────────────────────────

array [Plates,Rows,Columns] of var 0..1: controls_layout; 

constraint forall(i in Plates, j in Rows, k in Columns)(controls_layout[i,j,k] == ((is_control_cell(plates[i,j,k]))) );


constraint if spread_controls then forall(i in Plates, j in Rows)(regular(controls_layout[i,j,..], "(0|(1 0))* (1?)")::domain) endif;

constraint if spread_controls then forall(i in Plates, k in Columns)(regular(controls_layout[i,..,k], "(0|(1 0))* (1?)")::domain) endif;

constraint if force_spread_controls then forall(i in Plates, j in Rows)(regular(controls_layout[i,j,..], "(0|(1 0 0))* (1|(1 0))?")::domain) endif;

constraint if force_spread_controls then forall(i in Plates, k in Columns)(regular(controls_layout[i,..,k], "(0|(1 0 0))* (1|(1 0))?")::domain) endif;


constraint if spread_controls then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_plate_row[i,j],controls_layout[i,j,..])::domain) endif;

constraint if spread_controls /\ num_cols_line > 2  then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_half_row_plate[i,j,left],controls_layout[i,j,1..(num_cols_line div 2)])::domain) endif;

constraint if spread_controls /\ num_cols_line > 2  then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_half_row_plate[i,j,right],[controls_layout[i,j,k] | k in ((num_cols_line div 2)+1)..num_cols_line])::domain) endif;

constraint if spread_controls /\ num_cols_line > 2 then forall(i in Plates, j in Rows)(my_implied_cost(controls_only_in_half_row_plate[i,j,right],reverse([controls_layout[i,j,k] | k in ((num_cols_line div 2)+1)..num_cols_line]))::domain) endif;

constraint if spread_controls then forall(i in Plates, k in Columns)(my_implied_cost(controls_only_in_plate_column[i,k],controls_layout[i,..,k])::domain) endif;

constraint if spread_controls /\ num_rows_line > 2 then forall(i in Plates, k in Columns)(my_implied_cost(controls_only_in_half_column_plate[i,k,upper],[controls_layout[i,j,k] | j in 1..(num_rows_line div 2)])::domain) endif; 

constraint if spread_controls /\ num_rows_line > 2 then forall(i in Plates, k in Columns)(my_implied_cost(controls_only_in_half_column_plate[i,k,lower],[controls_layout[i,j,k] | j in (num_rows_line div 2)+1..num_rows_line])::domain) endif;

constraint if force_spread_controls then forall(i in Plates, j in Rows)(my_implied_cost_ext(controls_only_in_plate_row[i,j],controls_layout[i,j,..])::domain) endif;

constraint if force_spread_controls then forall(i in Plates, k in Columns)(my_implied_cost_ext(controls_only_in_plate_column[i,k],controls_layout[i,..,k])::domain) endif;


constraint forall(i in Plates) (sum_controls_ul_plates[i,upper] == sum(controls_layout[i,1..(num_rows_line div 2),..]));
constraint forall(i in Plates) (sum_controls_ul_plates[i,lower] == sum(controls_layout[i,(num_rows_line div 2)+1..num_rows_line,..]));

constraint forall(i in Plates) (sum_controls_lr_plates[i,left] == sum(controls_layout[i,..,1..(num_cols_line div 2)]));
constraint forall(i in Plates) (sum_controls_lr_plates[i,right] == sum(controls_layout[i,..,(num_cols_line div 2)+1..num_cols_line]));




% ──────────────────────────────────────────────────────────────────────────────
% Section 16.2: Global Cardinality Constraints  
% Ensure correct counts of control/non-control cells per row/column
% ──────────────────────────────────────────────────────────────────────────────

constraint forall(i in Plates, j in Rows) (global_cardinality(controls_layout[i,j,..],
                             [0,1],
                             [num_cols_line-controls_only_in_plate_row[i,j],controls_only_in_plate_row[i,j]]));

constraint forall(i in Plates, k in Columns) (global_cardinality(controls_layout[i,..,k],
                             [0,1],
                             [num_rows_line-controls_only_in_plate_column[i,k],controls_only_in_plate_column[i,k]]));

                          

constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, j in 2..num_rows_line-1, k in 2..num_cols_line-1)(
      no_adjacent_same_control_center(plates, i, j, k, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates, j in 2..num_rows_line-1, k in 2..num_cols_line-1)(plates[i,j,k] == experiments+ctr -> 
 %           plates[i,j-1,k]  != plates[i,j,k] /\
  %          plates[i,j,k-1]  != plates[i,j,k] /\
   %         plates[i,j,k+1]  != plates[i,j,k] /\
    %        plates[i,j+1,k]  != plates[i,j,k]
     %       )
            endif );

%% Upper row j=1
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, k in 2..num_cols_line-1)(
      no_adjacent_same_control_top_edge(plates, i, 1, k, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates, k in 2..num_cols_line-1)(plates[i,1,k] == experiments+ctr -> 
 %           plates[i,1,k-1] != plates[i,1,k] /\
  %          plates[i,1,k+1] != plates[i,1,k] /\
   %         plates[i,2,k]   != plates[i,1,k]
    %        )
            endif );

%% Lower row j = num_rows_line
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, k in 2..num_cols_line-1)(
      no_adjacent_same_control_bottom_edge(plates, i, num_rows_line, k, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates, k in 2..num_cols_line-1)(plates[i,num_rows_line,k] == experiments+ctr -> 
 %           plates[i,num_rows_line-1,k]  != plates[i,num_rows_line,k] /\
  %          plates[i,num_rows_line,k-1]  != plates[i,num_rows_line,k] /\
   %         plates[i,num_rows_line,k+1]  != plates[i,num_rows_line,k]
    %        )
            endif );         
            
%% Left column k=1 
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, j in 2..num_rows_line-1)(
      no_adjacent_same_control_left_edge(plates, i, j, 1, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
 %   else forall(i in Plates, j in 2..num_rows_line-1)(plates[i,j,1] == experiments+ctr -> 
  %          plates[i,j-1,1] != plates[i,j,1] /\
   %         plates[i,j,2]   != plates[i,j,1] /\
    %        plates[i,j+1,1] != plates[i,j,1]
     %       )
            endif );                                                   
                                        
%% Right column k=num_cols_line
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, j in 2..num_rows_line-1)(
      no_adjacent_same_control_right_edge(plates, i, j, num_cols_line, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates, j in 2..num_rows_line-1)(plates[i,j,num_cols_line] == experiments+ctr -> 
 %           plates[i,j-1,num_cols_line]  != plates[i,j,num_cols_line] /\
  %          plates[i,j,num_cols_line-1]  != plates[i,j,num_cols_line] /\
   %         plates[i,j+1,num_cols_line]  != plates[i,j,num_cols_line]
    %        )
            endif );                                                                                                            
                                                                    
%% Upper left corner j=1 k=1
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates)(
      no_adjacent_same_control_upper_left(plates, i, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates)(plates[i,1,1] == experiments+ctr -> 
 %           plates[i,1,2] != plates[i,1,1] /\
  %          plates[i,2,1]   != plates[i,1,1]
   %         )
            endif );

                                                                                                            
%% Upper right corner j=1 k=num_cols_line
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates)(
      no_adjacent_same_control_upper_right(plates, i, num_cols_line, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates)(plates[i,1,num_cols_line] == experiments+ctr -> 
 %           plates[i,1,num_cols_line-1] != plates[i,1,num_cols_line] /\
  %          plates[i,2,num_cols_line]   != plates[i,1,num_cols_line]
   %         )
            endif );                                                                                                                                                                                                                                                                                                                                    
            

%% Lower left corner j = num_rows_line k=1
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates)(
      no_adjacent_same_control_lower_left(plates, i, num_rows_line, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates)(plates[i,num_rows_line,1] == experiments+ctr -> 
 %           plates[i,num_rows_line-1,1]  != plates[i,num_rows_line,1] /\
  %          plates[i,num_rows_line,2]  != plates[i,num_rows_line,1]
   %         )
            endif );     
            
%% Lower right corner j = num_rows_line k = num_cols_line
constraint forall(ctr in 1..num_controls*max_control_concentrations)( if spread_control[ctr] then 
    forall(i in Plates, k in 2..num_cols_line-1)(
      no_adjacent_same_control_lower_right(plates, i, num_rows_line, num_cols_line, experiments+ctr)
    ) 
            %% This might be leading to unsat when there are too many of the same control
%    else forall(i in Plates)(plates[i,num_rows_line,num_cols_line] == experiments+ctr -> 
 %           plates[i,num_rows_line-1,num_cols_line]  != plates[i,num_rows_line,num_cols_line] /\
  %          plates[i,num_rows_line,num_cols_line-1]  != plates[i,num_rows_line,num_cols_line]
   %         )
            endif );   
            
            
constraint if spread_controls then forall(i in Plates, j in 2..num_rows_line-1, k in 2..num_cols_line-1)(
  no_controls_adjacent_center(plates, i, j, k)
) endif;

constraint if spread_controls then forall(i in Plates, j in 2..num_rows_line-1)(
  no_controls_adjacent_left_edge(plates, i, j)
) endif;

constraint if spread_controls then forall(i in Plates, j in 2..num_rows_line-1)(
  no_controls_adjacent_right_edge(plates, i, j, num_cols_line)
) endif;


constraint if spread_controls then forall(i in Plates, k in 2..num_cols_line-1)(
  no_controls_adjacent_top_edge(plates, i, k)
) endif;               

constraint if spread_controls then forall(i in Plates, k in 2..num_cols_line-1)(
  no_controls_adjacent_bottom_edge(plates, i, num_rows_line, k)
) endif;               


constraint if spread_controls then forall(i in Plates)(
  no_controls_adjacent_upper_left(plates, i)
) endif;

constraint if spread_controls then forall(i in Plates)(
  no_controls_adjacent_upper_right(plates, i, num_cols_line)
) endif;

constraint if spread_controls then forall(i in Plates)(
  no_controls_adjacent_lower_left(plates, i, num_rows_line)
) endif;

constraint if spread_controls then forall(i in Plates)(
  no_controls_adjacent_lower_right(plates, i, num_rows_line, num_cols_line)
) endif;                                                                                                                                                                                                                                                                                                                                                            
%% TODO: Add spread_controls contraint in borders!


% j=1
constraint if spread_controls then forall(i in Plates, k in 2..num_cols_line-1)(
  spread_any_control_top_edge(plates, i, k)
) endif;

% j = num_rows_line
constraint if spread_controls then forall(i in Plates, k in 2..num_cols_line-1)(
  spread_any_control_bottom_edge(plates, i, num_rows_line, k)
) endif;
           
% First column: k = 1           
constraint if spread_controls then forall(i in Plates, j in 2..num_rows_line-1)(
  spread_any_control_left_edge(plates, i, j)
) endif;

% Last column: k = num_cols_line
constraint if spread_controls then forall(i in Plates, j in 2..num_rows_line-1)(
  spread_any_control_right_edge(plates, i, j, num_cols_line)
) endif;
  
    
%% All especial cases for the corners are now covered by the alldifferent_except_0 constraints

                                               
constraint if force_spread_controls then (forall(i in Plates, j in 2..num_rows_line-1, k in 2..num_cols_line-1)(
  no_controls_adjacent_center(plates, i, j, k)
) ) endif ;
           
constraint if force_spread_controls then forall(i in Plates, j in 2..num_rows_line-1, k in 2..num_cols_line-1)(is_control_cell(plates[i,j,k]) -> 
            plates[i,j-1,k-1] <= experiments /\ 
            plates[i,j-1,k]  <= experiments /\
            plates[i,j-1,k+1] <= experiments /\
            plates[i,j,k-1]  <= experiments /\
            plates[i,j,k+1]  <= experiments /\
            plates[i,j+1,k-1] <= experiments /\
            plates[i,j+1,k]  <= experiments /\
            plates[i,j+1,k+1] <= experiments 
            ) endif;
                                               
constraint if force_spread_controls then forall(i in Plates, j in 3..num_rows_line-2, k in 3..num_cols_line-2)(is_control_cell(plates[i,j,k]) -> 
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
                      
            
constraint if force_spread_controls then forall(i in Plates, j = 2, k in 3..num_cols_line-2)(is_control_cell(plates[i,j,k]) -> 
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
                      

constraint if force_spread_controls then forall(i in Plates, j = 1, k in 3..num_cols_line-2)(is_control_cell(plates[i,j,k]) -> 
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
                        
                                               
constraint if force_spread_controls then forall(i in Plates, j = num_rows_line-1, k in 3..num_cols_line-2)(is_control_cell(plates[i,j,k]) -> 
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
            
constraint if force_spread_controls then forall(i in Plates, j = num_rows_line, k in 3..num_cols_line-2)(is_control_cell(plates[i,j,k]) -> 
            plates[i,j-2,k-2] <= experiments /\ 
            plates[i,j-2,k-1]  <= experiments /\
            plates[i,j-2,k]  <= experiments /\
            plates[i,j-2,k+1]  <= experiments /\
            plates[i,j-2,k+2] <= experiments /\
            plates[i,j,k-2]  <= experiments /\
            plates[i,j,k+2]  <= experiments
            ) endif;             
                    

%% Border cases for k: k=1
constraint if force_spread_controls then forall(i in Plates, j in 3..num_rows_line-2, k=1)(plates[i,j,1] > experiments -> 
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
constraint if force_spread_controls then forall(i in Plates, j in 3..num_rows_line-2, k = 2)(is_control_cell(plates[i,j,k]) -> 
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
constraint if force_spread_controls then forall(i in Plates, j in 3..num_rows_line-2, k = num_cols_line-1)(is_control_cell(plates[i,j,k]) -> 
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
constraint if force_spread_controls then forall(i in Plates, j in 3..num_rows_line-2, k = num_cols_line)(is_control_cell(plates[i,j,k]) -> 
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
            
                        

% ==============================================================================
% SECTION 17: SEARCH STRATEGY & SOLVER CONFIGURATION
% ==============================================================================
%
% Search annotations and solver configuration that guide the constraint
% solver to find solutions efficiently.
%
% ==============================================================================

%%%% Find a solution! %%%%

array [Plates,1..experiments] of var {0} union Rows: experiment_plate_row;

constraint forall(l in 1..experiments)((experiment_plate[l] > 0) -> (experiment_plate_row[experiment_plate[l],l] == experiment_row[l]));

constraint forall(i in Plates, l in 1..experiments)((experiment_plate[l] != i) -> (experiment_plate_row[i,l] == 0));



%array [1..num_plates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..num_plates*2];
 
%array [1..num_plates*2] of ann: search_order = [if (i mod 2)==1 then int_search(experiment_plate_row[(i+1)div 2,..], first_fail, indomain_random) else int_search(plates[i div 2,..,..], random, indomain_max) endif | i in 1..num_plates*2];

%% Strategy 1: good for example04 and example11 BACKUP, previous version!
%array [1..num_plates*3] of ann: search_order_a = [if (i mod 3)==1 then int_search(experiment_plate_row[(i+2) div 3,..], first_fail, indomain_random) elseif (i mod 3) == 2 then int_search(controls_layout[(i+2) div 3,..,..], random, indomain_max) else int_search(plates[i div 3,..,..], first_fail, indomain_random) endif | i in 1..num_plates*3]; 

array [1..num_plates*3] of ann: search_order_a = [if (i mod 3)==2 then int_search(experiment_plate_row[(i+2) div 3,..], first_fail, indomain_random) elseif (i mod 3) == 1 then int_search(controls_layout[(i+2) div 3,..,..], random, indomain_max) else int_search(plates[i div 3,..,..], first_fail, indomain_random) endif | i in 1..num_plates*3]; 


%% The one I normally use?? 
%array [1..num_plates*3] of ann: search_order = [if (i mod 3)==2 then int_search(experiment_plate_row[(i+2)div 3,..], first_fail, indomain_random) elseif (i mod 3) == 1 then int_search(controls_layout[(i+2)div 3,..,..], random, indomain_max) else int_search(plates[(i+2) div 3,..,..], first_fail, indomain_random) endif | i in 1..num_plates*3]; 

%% Strategy 2:
array [1..num_plates*2] of ann: search_order_b = [if (i mod 2) == 1 then int_search(controls_layout[(i+1) div 2,..,..], random, indomain_max) else int_search(plates[(i+1) div 2,..,..], first_fail, indomain_random) endif | i in 1..num_plates*2]; 


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

% ==============================================================================
% SECTION 18: OUTPUT GENERATION - VISUAL PLATE DISPLAY
% ==============================================================================
%
% Output statements for generating human-readable plate layouts with
% symbols representing different compounds and controls.
%
% ==============================================================================

% %%% Pretty printing %%%

output [if testing \/ debugging then "\(num_plates) plates\n" else "" endif];

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
        if j== num_rows_line /\ k == num_cols_line /\ i<num_plates then "\n\n Plate \(i+1):\n" 
        elseif k == num_cols_line then "\n" else " " endif else "" endif|
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

% ==============================================================================
% SECTION 19: OUTPUT GENERATION - CSV EXPORT
% ==============================================================================
%
% Output statements for generating CSV files with complete experimental
% data including plate IDs, well coordinates, compound names, and concentrations.
%
% ==============================================================================

%%% CSV Ouput %%%

array[int] of string: letters_capital = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
array[int] of string: letters_inline  = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"];
array[int] of string: letters = letters_capital ++ [letters_capital[i] ++ letters_inline[j] | i in 1..length(letters_capital), j in 1..length(letters_inline)];
         
output ["plateID,well,cmpdname,CONCuM,cmpdnum,VOLuL\n"];

%%% TODO: FIX ME!!!!!! Problem when there are vertical/horizontal cell lines!
output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[fix(calculate_line_index(inner_empty_edge, size_empty_edge, num_rows_line, h, j))] ++ if fix(calculate_line_index(inner_empty_edge, size_empty_edge, num_cols_line, v, k)) < 10 then "0" else "" endif ++ "\(fix(calculate_line_index(inner_empty_edge, size_empty_edge, num_cols_line, v, k)))," ++ 

%%%% Compounds %%%%

if fix(plates[i,j,k]) <= experiments then compound_names[(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) div max_compound_concentrations)+1] 
++ "," ++ compound_concentration_names[(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) div max_compound_concentrations)+1,(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) mod max_compound_concentrations)+1] ++ "," ++

compound_names[get_compound_id(fix(plates[i,j,k]))] ++ "_" ++ compound_concentration_names[get_compound_id(fix(plates[i,j,k])),((floor((fix(plates[i,j,k])-1))) mod max_compound_concentrations)+1]


%%%% Controls %%%%

else
%control_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1] ++ "," ++  
control_names[get_control_type_id(fix(plates[i,j,k]))] ++ "," ++  
%% control concentration

%control_concentration_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1,floor((fix(plates[i,j,k])-experiments-1)/num_controls)+1] 
control_concentration_names[get_control_type_id(fix(plates[i,j,k])),((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1] 

++ "," ++ 
%% Latex name of the control. Right now it's the same as the name
control_names[get_control_type_id(fix(plates[i,j,k]))] ++ "_" ++ control_concentration_names[get_control_type_id(fix(plates[i,j,k])),((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1]  

endif 

++ "\n" else "" endif|
        i in Plates, j in Rows, k in Columns, v in 1..vertical_cell_lines, h in 1..horizontal_cell_lines];       
         

       
