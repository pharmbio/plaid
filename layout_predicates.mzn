% ==============================================================================
% PLAID - Layout Predicates and Helper Functions
% ==============================================================================
%
% This file contains reusable predicates and helper functions for the PLAID
% microplate layout designer. All helpers are organized into logical sections
% for easy navigation and maintenance.
%
% Authors: Maria Andreina FRANCISCO RODRIGUEZ
% License: Apache 2.0
% Last Revision: November 2025
% ==============================================================================

include "globals.mzn";


% ==============================================================================
% SECTION 1: SORTING AND ORDERING PREDICATES
% ==============================================================================

% Predicate: sorted_except_0
% Purpose: Ensures array is sorted with groups of identical values, allowing zeros
% Parameters:
%   - x: Array to check
%   - num: Number of distinct non-zero values allowed
%   - min/max: Minimum and maximum occurrences of each value
% Mathematical basis: Uses regular expression automaton to enforce pattern
predicate sorted_except_0(array[int] of var int: x, int: num, int: min, int: max) = 
  regular(x, concat(["(0* \(i)){\(min),\(max)} " | i in 1..num]++["0*"]))::domain;


% ==============================================================================
% SECTION 2: COST OPTIMIZATION PREDICATES
% ==============================================================================

% Predicate: my_implied_cost
% Purpose: Enforces implied constraints for cost optimization based on group automaton
% Mathematical basis: Cumulative cost must be monotonically increasing with bounded
%                    differences between adjacent elements
% Usage: Called when optimizing control distribution costs
predicate my_implied_cost(var int: cost, array[int] of var int: x) = 
  let {
    int: L = length(x);
    array[1..L] of var int: cumulative_cost;
  } in
  cumulative_cost[1] = x[1] /\
  cumulative_cost[2] <= 1 /\
  cumulative_cost[L] = cost /\
  forall(i in 1..L-1)(x[i+1] = cumulative_cost[i+1] - cumulative_cost[i]) /\
  forall(i in 1..L-1)(cumulative_cost[i+1] = x[i+1] + cumulative_cost[i]) /\
  forall(i in 1..L-1)(cumulative_cost[i] <= cumulative_cost[i+1]) /\
  forall(i in 2..L-1)(cumulative_cost[i+1] <= 1 + cumulative_cost[i-1]) /\
  forall(i in 2..L-1)(cumulative_cost[i+1] - 1 <= cumulative_cost[i-1]);


% Predicate: my_implied_cost_ext (Extended version)
% Purpose: Extended cost constraint with lookahead of 2 positions
% Difference from my_implied_cost: Checks i+2 instead of i+1 for tighter bounds
predicate my_implied_cost_ext(var int: cost, array[int] of var int: x) = 
  let {
    int: L = length(x);
    array[1..L] of var int: cumulative_cost;
  } in
  cumulative_cost[1] = x[1] /\
  cumulative_cost[2] <= 1 /\
  cumulative_cost[L] = cost /\
  forall(i in 1..L-1)(x[i+1] = cumulative_cost[i+1] - cumulative_cost[i]) /\
  forall(i in 1..L-1)(cumulative_cost[i+1] = x[i+1] + cumulative_cost[i]) /\
  forall(i in 1..L-1)(cumulative_cost[i] <= cumulative_cost[i+1]) /\
  forall(i in 2..L-2)(cumulative_cost[i+2] <= 1 + cumulative_cost[i-1]) /\
  forall(i in 2..L-2)(cumulative_cost[i+2] - 1 <= cumulative_cost[i-1]);


% ==============================================================================
% SECTION 3: CONTROL SPREADING THRESHOLDS
% ==============================================================================

% Function: force_spread_controls
% Purpose: Determines if control spreading constraints should be enforced
% Returns: true if plate configuration allows for effective control spreading
%
% Mathematical basis: Empirically determined thresholds for different plate sizes
% - Values represent proven upper bounds for satisfiability
% - Spreading controls ensures no adjacent wells contain same control type
%
% Supported plate configurations:
%   - 96-well plates (8x12 or 12x8): 11 controls per plate
%   - 96-well with border (6x10 or 10x6): 7 controls per plate
%   - 384-well with border (14x22 or 22x14): 31 controls per plate
%   - 384-well plates (16x24 or 24x16): 37 controls per plate
%   - Generic small plates (<60 wells): ceil(wells/9) controls per plate
function bool: force_spread_controls(int: num_cols, int: num_rows, 
                                     int: total_controls, int: num_plates) = 
  % 96-well plate (proven upper bound)
  if (num_cols = 12 /\ num_rows = 8) \/ (num_cols = 8 /\ num_rows = 12) then 
    (11 * num_plates >= total_controls) 
  % 96-well plate with empty border (proven upper bound)
  elseif (num_cols = 10 /\ num_rows = 6) \/ (num_cols = 6 /\ num_rows = 10) then 
    (7 * num_plates >= total_controls) 
  % 384-well plate with empty border
  % Note: 31 is conservative; 32 is solvable but slow
  elseif (num_cols = 22 /\ num_rows = 14) \/ (num_cols = 14 /\ num_rows = 22) then 
    (31 * num_plates >= total_controls) 
  % 384-well plate
  % Note: 37 is conservative threshold
  elseif (num_cols = 24 /\ num_rows = 16) \/ (num_cols = 16 /\ num_rows = 24) then 
    (37 * num_plates >= total_controls) 
  % Generic heuristic for smaller plates
  elseif num_cols * num_rows < 60 then 
    (ceil(num_cols * num_rows / 9) * num_plates >= total_controls)
  else 
    false
  endif;


% ==============================================================================
% SECTION 4: COORDINATE CALCULATION HELPERS
% ==============================================================================

% Function: calculate_line_index
% Purpose: Calculates linear index for multi-line plate layouts with edge handling
% Parameters:
%   - edge_type: true if using inner empty edges
%   - edge_size: number of empty edge wells
%   - line_size: size of each cell line
%   - line: which line (row/column) we're in
%   - coordinate: position within the line
% Returns: Absolute index in the plate accounting for edges and line structure
function var int: calculate_line_index(bool: edge_type, int: edge_size, 
                                       int: line_size, var int: line, 
                                       var int: coordinate) =
  if edge_type then 
    (line - 1) * (2 * edge_size + line_size) + edge_size + coordinate
  else 
    (line - 1) * line_size + edge_size + coordinate 
  endif;


% ==============================================================================
% SECTION 5: INDEX CALCULATION HELPERS
% ==============================================================================

% Function: experiment_index
% Purpose: Calculates unique index for compound/replicate/concentration combination
% Usage: Used throughout the model for accessing experiment arrays
% Formula: replicate_offset + compound_offset + concentration
% Example: For compound 2, replicate 1, concentration 3 with 5 compounds and 
%          10 max concentrations: 1*5*10 + (2-1)*10 + 3 = 63
function int: experiment_index(int: compound, int: replicate, int: concentration, 
                               int: num_compounds, int: max_compound_concentrations) =
    replicate * num_compounds * max_compound_concentrations + 
    (compound - 1) * max_compound_concentrations + 
    concentration;


% Function: experiment_index_from_compound_replicate
% Purpose: Alternative indexing when compound_replicate index is already computed
% Simplifies: experiment_index when you already have the compound*replicate value
function int: experiment_index_from_compound_replicate(int: compound_replicate, 
                                                       int: concentration,
                                                       int: max_compound_concentrations) =
    compound_replicate * max_compound_concentrations + concentration;


% Function: control_index
% Purpose: Calculates index for control types (offset after all experiments)
% Formula: experiments_count + (control_type-1)*max_concentrations + concentration
function int: control_index(int: control_type, int: concentration,
                           int: num_experiments, int: max_control_concentrations) =
    num_experiments + (control_type - 1) * max_control_concentrations + concentration;


% ==============================================================================
% SECTION 6: WELL LOCATION HELPERS
% ==============================================================================

% Predicate: valid_well
% Purpose: Checks if a well location is within plate bounds
predicate valid_well(int: row, int: col, int: num_rows, int: num_cols) =
    row >= 1 /\ row <= num_rows /\
    col >= 1 /\ col <= num_cols;


% Predicate: in_quadrant
% Purpose: Determines if a well is in a specific quadrant (1=upper-left, 
%          2=upper-right, 3=lower-left, 4=lower-right)
predicate in_quadrant(int: row, int: col, int: quadrant, int: num_rows, int: num_cols) =
    let {
        bool: upper_half = row <= num_rows div 2;
        bool: left_half = col <= num_cols div 2;
    } in
    (quadrant = 1 /\ upper_half /\ left_half) \/
    (quadrant = 2 /\ upper_half /\ not left_half) \/
    (quadrant = 3 /\ not upper_half /\ left_half) \/
    (quadrant = 4 /\ not upper_half /\ not left_half);


% ==============================================================================
% SECTION 7: COUNTING HELPERS
% ==============================================================================

% Function: count_experiments_in_row
% Purpose: Counts how many experiments appear in a specific plate row
function var int: count_experiments_in_row(int: plate_id, int: row_id, 
                                           int: num_experiments,
                                           array[int] of var int: experiment_plates,
                                           array[int] of var int: experiment_rows) =
    sum(e in 1..num_experiments)(
      experiment_plates[e] = plate_id /\ experiment_rows[e] = row_id
    );


% Function: count_experiments_in_col
% Purpose: Counts how many experiments appear in a specific plate column
function var int: count_experiments_in_col(int: plate_id, int: col_id,
                                           int: num_experiments,
                                           array[int] of var int: experiment_plates, 
                                           array[int] of var int: experiment_cols) =
    sum(e in 1..num_experiments)(
      experiment_plates[e] = plate_id /\ experiment_cols[e] = col_id
    );


% Function: count_content_type
% Purpose: Counts occurrences of a specific content type in a well array
function var int: count_content_type(array[int] of var int: wells, int: content_type) =
    sum(w in index_set(wells))(wells[w] = content_type);


% ==============================================================================
% SECTION 8: CONTROL DISTRIBUTION HELPERS
% ==============================================================================

% Predicate: should_spread_control
% Purpose: Checks if a specific control type should use spreading constraints
predicate should_spread_control(int: control_idx, int: max_control_idx,
                               array[int] of bool: spread_flags) =
    control_idx >= 1 /\ 
    control_idx <= max_control_idx /\
    spread_flags[control_idx];


% Function: get_control_count
% Purpose: Retrieves the count for a specific control type and concentration
% Returns: Number of replicates for this control, or 0 if invalid indices
function int: get_control_count(int: control_type, int: concentration,
                               int: num_controls, int: max_control_concentrations,
                               array[int] of int: controls_per_plate_array) =
    if control_type >= 1 /\ control_type <= num_controls /\
       concentration >= 1 /\ concentration <= max_control_concentrations
    then controls_per_plate_array[
      (control_type - 1) * max_control_concentrations + concentration
    ]
    else 0
    endif;


% ==============================================================================
% SECTION 9: NEIGHBOR CHECKING HELPERS
% ==============================================================================

% Predicate: has_neighbor_with_value
% Purpose: Checks if any of the 8 neighbors of a well contains a specific value
% Returns: true if ANY neighbor (including diagonals) has the given value
% Note: Handles boundary conditions automatically
predicate has_neighbor_with_value(
    array[int, int, int] of var int: plates_array,
    int: plate, int: row, int: col,
    int: num_rows, int: num_cols,
    var int: control_value) =
    % Check all 8 directions with boundary guards
    (row > 1 /\ col > 1 /\ 
     plates_array[plate, row - 1, col - 1] = control_value) \/
    (row > 1 /\ 
     plates_array[plate, row - 1, col] = control_value) \/
    (row > 1 /\ col < num_cols /\ 
     plates_array[plate, row - 1, col + 1] = control_value) \/
    (col > 1 /\ 
     plates_array[plate, row, col - 1] = control_value) \/
    (col < num_cols /\ 
     plates_array[plate, row, col + 1] = control_value) \/
    (row < num_rows /\ col > 1 /\ 
     plates_array[plate, row + 1, col - 1] = control_value) \/
    (row < num_rows /\ 
     plates_array[plate, row + 1, col] = control_value) \/
    (row < num_rows /\ col < num_cols /\ 
     plates_array[plate, row + 1, col + 1] = control_value);


% Predicate: no_neighbor_with_value
% Purpose: Inverse of has_neighbor_with_value - ensures no neighbors have the value
predicate no_neighbor_with_value(
    array[int, int, int] of var int: plates_array,
    int: plate, int: row, int: col,
    int: num_rows, int: num_cols,
    var int: value) =
    not has_neighbor_with_value(plates_array, plate, row, col, 
                                num_rows, num_cols, value);


% ==============================================================================
% SECTION 10: PLATE REGION HELPERS
% ==============================================================================

% Function: get_mid_row
% Purpose: Returns the midpoint row for splitting plate into upper/lower halves
function int: get_mid_row(int: num_rows) = num_rows div 2;


% Function: get_mid_col
% Purpose: Returns the midpoint column for splitting plate into left/right halves
function int: get_mid_col(int: num_cols) = num_cols div 2;


% Predicate: in_upper_half
% Purpose: Checks if a row is in the upper half of the plate
predicate in_upper_half(int: row, int: num_rows) = 
  row <= get_mid_row(num_rows);


% Predicate: in_left_half
% Purpose: Checks if a column is in the left half of the plate
predicate in_left_half(int: col, int: num_cols) = 
  col <= get_mid_col(num_cols);


% Function: get_upper_rows
% Purpose: Returns set of row indices in the upper half of plate
function set of int: get_upper_rows(int: num_rows) = 
  1..get_mid_row(num_rows);


% Function: get_lower_rows
% Purpose: Returns set of row indices in the lower half of plate
function set of int: get_lower_rows(int: num_rows) = 
  get_mid_row(num_rows) + 1..num_rows;


% Function: get_left_cols
% Purpose: Returns set of column indices in the left half of plate
function set of int: get_left_cols(int: num_cols) = 
  1..get_mid_col(num_cols);


% Function: get_right_cols
% Purpose: Returns set of column indices in the right half of plate
function set of int: get_right_cols(int: num_cols) = 
  get_mid_col(num_cols) + 1..num_cols;


% ==============================================================================
% SECTION 11: WELL CONTENT TYPE HELPERS
% ==============================================================================

% Predicate: is_control
% Purpose: Checks if a well value represents a control (not experiment or empty)
predicate is_control(var int: well_value, int: num_experiments) =
    well_value > num_experiments;


% Predicate: is_experiment
% Purpose: Checks if a well value represents an experiment
predicate is_experiment(var int: well_value, int: num_experiments) =
    well_value > 0 /\ well_value <= num_experiments;


% Predicate: is_empty_well
% Purpose: Checks if a well is empty (value = 0)
predicate is_empty_well(int: well_value) = 
  well_value = 0;


% ==============================================================================
% SECTION 12: COMPOUND CONCENTRATION HELPERS
% ==============================================================================

% Predicate: should_force_spread_compound
% Purpose: Checks if concentration spreading should be enforced for a compound
predicate should_force_spread_compound(int: compound_id, 
                                       array[int] of bool: force_spread_array) =
    compound_id >= 1 /\ 
    compound_id <= length(force_spread_array) /\
    force_spread_array[compound_id];


% Predicate: valid_concentration
% Purpose: Validates if a concentration level is valid for a given compound
predicate valid_concentration(int: compound_id, int: concentration,
                              array[int] of int: compound_concentrations_array) =
    concentration >= 1 /\
    concentration <= compound_concentrations_array[compound_id];


% Function: get_concentration_range
% Purpose: Generates set of experiment indices for all concentrations of a 
%          compound/replicate combination
% Usage: Replaces complex array_union calculations (~8 times in plate-design.mzn)
% Benefit: 50-80 character reduction per use, crystal clear intent
%
% Example: get_concentration_range(2, 5, 3, 10) returns experiment indices for
%          compound 2 across all 3 replicates, each with up to 10 concentrations
function set of int: get_concentration_range(int: compound, int: num_compounds, 
                                            int: num_replicates, 
                                            int: max_concentrations) =
    array_union([
      (compound - 1) * max_concentrations + 1 + r * num_compounds * max_concentrations
      .. compound * max_concentrations + r * num_compounds * max_concentrations 
      | r in 0..num_replicates - 1
    ]);


% ==============================================================================
% SECTION 13: BOOLEAN CONFIGURATION HELPERS
% ==============================================================================

% Predicate: is_concentration_spread_enabled
% Purpose: Semantic wrapper for checking if concentration spreading is enabled
% Benefit: Consistent naming, clearer intent
predicate is_concentration_spread_enabled(bool: spread_flag) = 
  spread_flag;


% ==============================================================================
% SECTION 14: PLATE VALUE DECODING FUNCTIONS
% ==============================================================================
%
% These functions decode the integer values stored in the plates array to
% extract compound IDs, control IDs, and concentration indices.
%
% Encoding scheme:
%   0 = empty well
%   1..experiments = experiments (compound/replicate/concentration combos)
%   experiments+1..end = controls
%
% ==============================================================================

% Extract compound ID from an experiment plate value
% Returns: compound ID (1..compounds)
function int: get_compound_id(int: plate_value) =
  floor(((plate_value-1)/max_compound_concentrations)) mod compounds + 1;

% Extract concentration index from an experiment plate value
% Returns: concentration index (1..max_compound_concentrations)
function int: get_concentration_id(int: plate_value) =
  ((plate_value-1) mod max_compound_concentrations) + 1;

% Extract replicate ID from an experiment plate value
% Returns: replicate ID (0..replicates-1)
function int: get_replicate_id(int: plate_value) =
  floor((plate_value-1)/(compounds*max_compound_concentrations));

% Extract control type ID from a control plate value
% Returns: control type ID (1..num_controls)
function int: get_control_type_id(int: plate_value) =
  floor((plate_value-experiments-1)/max_control_concentrations) + 1;

% Extract control concentration index from a control plate value
% Returns: concentration index (0..max_control_concentrations-1)
function int: get_control_concentration_id(int: plate_value) =
  ((plate_value-experiments-1) mod max_control_concentrations);

% Convert compound, replicate, concentration to linear experiment index
% Returns: experiment index (1..experiments)
function int: get_experiment_index(int: compound_id, int: replicate_id, int: concentration_id) =
  replicate_id*compounds*max_compound_concentrations + 
  (compound_id-1)*max_compound_concentrations + concentration_id;


% ==============================================================================
% SECTION 15: CHANNELING PREDICATE
% ==============================================================================
%
% Maintains synchronization between the two model representations:
%   1. plates[Plates,Rows,Columns] - 3D spatial model
%   2. experiment_plate/row/column[experiments] - experiment-centric model
%
% ==============================================================================

% Channel a single experiment between the two model representations
predicate channel_experiment(
  int: experiment_id,
  var int: plate_id,
  var int: row_id,
  var int: col_id,
  array[Plates,Rows,Columns] of var int: plates_array
) = 
  % Experiment appears in plates array iff plate_id is non-zero
  count_eq(plates_array, experiment_id, (plate_id != 0)) /\

  % If experiment is placed, it must be on the specified plate
  forall(i in Plates)(
    count_eq(plates_array[i,..,..], experiment_id, (plate_id == i))
  ) /\

  % If experiment is placed, it must be in the specified row
  forall(j in Rows)(
    count_eq(plates_array[..,j,..], experiment_id, (row_id == j))
  ) /\

  % If experiment is placed, it must be in the specified column
  forall(k in Columns)(
    count_eq(plates_array[..,..,k], experiment_id, (col_id == k))
  );


% ==============================================================================
% SECTION 16: DISTRIBUTION BALANCE PREDICATES
% ==============================================================================
%
% These predicates enforce balanced distribution of items (controls/compounds)
% across half-rows and half-columns of plates. This ensures even spatial
% distribution to minimize systematic errors.
%
% ==============================================================================

% Balance items in the left/right halves of a row
% Ensures each half has approximately equal counts (differ by at most 1)
predicate balance_in_half_rows(
  array[Plates, Rows, Horizontal] of var int: counts,
  int: plate_id,
  int: row_id
) = 
  let {
    var int: total = counts[plate_id, row_id, left] + counts[plate_id, row_id, right];
  } in (
    % Each half gets at least floor(total/2)
    counts[plate_id, row_id, left] >= (total div 2) /\
    counts[plate_id, row_id, right] >= (total div 2) /\
    % Each half gets at most ceil(total/2)
    counts[plate_id, row_id, left] <= ((total + 1) div 2) /\
    counts[plate_id, row_id, right] <= ((total + 1) div 2)
  );

% Balance items in the upper/lower halves of a column
% Ensures each half has approximately equal counts (differ by at most 1)
predicate balance_in_half_columns(
  array[Plates, Columns, Vertical] of var int: counts,
  int: plate_id,
  int: col_id
) = 
  let {
    var int: total = counts[plate_id, col_id, upper] + counts[plate_id, col_id, lower];
  } in (
    % Each half gets at least floor(total/2)
    counts[plate_id, col_id, upper] >= (total div 2) /\
    counts[plate_id, col_id, lower] >= (total div 2) /\
    % Each half gets at most ceil(total/2)
    counts[plate_id, col_id, upper] <= ((total + 1) div 2) /\
    counts[plate_id, col_id, lower] <= ((total + 1) div 2)
  );

% ==============================================================================
% SECTION 16: CONTROL CELL IDENTIFICATION
% ==============================================================================

% Check if a cell contains a control (not experiment or empty)
function var bool: is_control_cell(var int: cell_value) = cell_value > experiments;


% ==============================================================================
% SECTION 17: CONTROL SPREADING PREDICATES
% ==============================================================================
%
% These predicates enforce that controls of the same type are not placed
% adjacent to each other (distance-1) or within distance-2 (force_spread).
%
% EXPERIMENTAL: May affect constraint propagation and performance!
%
% ==============================================================================

% Check if 8 neighbors (including diagonals) are different from center
% Used for center cells (not on edges)
predicate no_adjacent_same_control_center(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row, 
  int: col,
  int: control_value
) = 
  plate_layout[plate, row, col] == control_value ->
    (plate_layout[plate, row-1, col-1] != control_value /\
     plate_layout[plate, row-1, col]   != control_value /\
     plate_layout[plate, row-1, col+1] != control_value /\
     plate_layout[plate, row,   col-1] != control_value /\
     plate_layout[plate, row,   col+1] != control_value /\
     plate_layout[plate, row+1, col-1] != control_value /\
     plate_layout[plate, row+1, col]   != control_value /\
     plate_layout[plate, row+1, col+1] != control_value);

% Check neighbors for top edge (no row-1 neighbors)
predicate no_adjacent_same_control_top_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row,
  int: col,
  int: control_value
) = 
  plate_layout[plate, row, col] == control_value ->
    (plate_layout[plate, row,   col-1] != control_value /\
     plate_layout[plate, row,   col+1] != control_value /\
     plate_layout[plate, row+1, col-1] != control_value /\
     plate_layout[plate, row+1, col]   != control_value /\
     plate_layout[plate, row+1, col+1] != control_value);

% Check neighbors for bottom edge (no row+1 neighbors)
predicate no_adjacent_same_control_bottom_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row,
  int: col,
  int: control_value
) = 
  plate_layout[plate, row, col] == control_value ->
    (plate_layout[plate, row-1, col-1] != control_value /\
     plate_layout[plate, row-1, col]   != control_value /\
     plate_layout[plate, row-1, col+1] != control_value /\
     plate_layout[plate, row,   col-1] != control_value /\
     plate_layout[plate, row,   col+1] != control_value);

% Check neighbors for left edge (no col-1 neighbors)
predicate no_adjacent_same_control_left_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row,
  int: col,
  int: control_value
) = 
  plate_layout[plate, row, col] == control_value ->
    (plate_layout[plate, row-1, col]   != control_value /\
     plate_layout[plate, row-1, col+1] != control_value /\
     plate_layout[plate, row,   col+1] != control_value /\
     plate_layout[plate, row+1, col]   != control_value /\
     plate_layout[plate, row+1, col+1] != control_value);

% Check neighbors for right edge (no col+1 neighbors)
predicate no_adjacent_same_control_right_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row,
  int: col,
  int: control_value
) = 
  plate_layout[plate, row, col] == control_value ->
    (plate_layout[plate, row-1, col-1] != control_value /\
     plate_layout[plate, row-1, col]   != control_value /\
     plate_layout[plate, row,   col-1] != control_value /\
     plate_layout[plate, row+1, col-1] != control_value /\
     plate_layout[plate, row+1, col]   != control_value);
predicate spread_any_control_top_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: col
) = 
  plate_layout[plate, 1, col] > experiments ->
    (plate_layout[plate, 1, col-1]  != plate_layout[plate, 1, col] /\
     plate_layout[plate, 1, col+1]  != plate_layout[plate, 1, col] /\
     plate_layout[plate, 2, col-1]  != plate_layout[plate, 1, col] /\
     plate_layout[plate, 2, col]    != plate_layout[plate, 1, col] /\
     plate_layout[plate, 2, col+1]  != plate_layout[plate, 1, col]);

predicate spread_any_control_bottom_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_rows,
  int: col
) = 
  plate_layout[plate, num_rows, col] > experiments ->
    (plate_layout[plate, num_rows-1, col-1] != plate_layout[plate, num_rows, col] /\
     plate_layout[plate, num_rows-1, col]   != plate_layout[plate, num_rows, col] /\
     plate_layout[plate, num_rows-1, col+1] != plate_layout[plate, num_rows, col] /\
     plate_layout[plate, num_rows, col-1]   != plate_layout[plate, num_rows, col] /\
     plate_layout[plate, num_rows, col+1]   != plate_layout[plate, num_rows, col]);

predicate spread_any_control_left_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row
) = 
  plate_layout[plate, row, 1] > experiments ->
    (plate_layout[plate, row-1, 1] != plate_layout[plate, row, 1] /\
     plate_layout[plate, row-1, 2] != plate_layout[plate, row, 1] /\
     plate_layout[plate, row, 2]   != plate_layout[plate, row, 1] /\
     plate_layout[plate, row+1, 1] != plate_layout[plate, row, 1] /\
     plate_layout[plate, row+1, 2] != plate_layout[plate, row, 1]);

predicate spread_any_control_right_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row,
  int: num_cols
) = 
  plate_layout[plate, row, num_cols] > experiments ->
    (plate_layout[plate, row-1, num_cols-1] != plate_layout[plate, row, num_cols] /\
     plate_layout[plate, row-1, num_cols]   != plate_layout[plate, row, num_cols] /\
     plate_layout[plate, row, num_cols-1]   != plate_layout[plate, row, num_cols] /\
     plate_layout[plate, row+1, num_cols-1] != plate_layout[plate, row, num_cols] /\
     plate_layout[plate, row+1, num_cols]   != plate_layout[plate, row, num_cols]);

% Corner cell predicates (3 neighbors each)

% Upper-left corner (row=1, col=1)
predicate no_adjacent_same_control_upper_left(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: control_value
) = 
  plate_layout[plate, 1, 1] == control_value ->
    (plate_layout[plate, 1, 2]   != control_value /\
     plate_layout[plate, 2, 1]   != control_value /\
     plate_layout[plate, 2, 2]   != control_value);

% Upper-right corner (row=1, col=num_cols_line)
predicate no_adjacent_same_control_upper_right(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_cols,
  int: control_value
) = 
  plate_layout[plate, 1, num_cols] == control_value ->
    (plate_layout[plate, 1, num_cols-1] != control_value /\
     plate_layout[plate, 2, num_cols-1] != control_value /\
     plate_layout[plate, 2, num_cols]   != control_value);

% Lower-left corner (row=num_rows_line, col=1)
predicate no_adjacent_same_control_lower_left(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_rows,
  int: control_value
) = 
  plate_layout[plate, num_rows, 1] == control_value ->
    (plate_layout[plate, num_rows-1, 1] != control_value /\
     plate_layout[plate, num_rows-1, 2] != control_value /\
     plate_layout[plate, num_rows, 2]   != control_value);

% Lower-right corner (row=num_rows_line, col=num_cols_line)
predicate no_adjacent_same_control_lower_right(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_rows,
  int: num_cols,
  int: control_value
) = 
  plate_layout[plate, num_rows, num_cols] == control_value ->
    (plate_layout[plate, num_rows-1, num_cols-1] != control_value /\
     plate_layout[plate, num_rows-1, num_cols]   != control_value /\
     plate_layout[plate, num_rows, num_cols-1]   != control_value);


% ==============================================================================
% NO CONTROLS ADJACENT PREDICATES
% ==============================================================================
%
% These predicates enforce that if a cell contains a control, its neighbors
% must NOT be controls (i.e., neighbors must be experiments or empty).
%
% Pattern: plates[i,j,k] > experiments -> neighbors <= experiments
%
% This is different from the "no_adjacent_same_control" predicates which
% enforce that neighbors are different control types.
% ==============================================================================

% Center cells: 4 orthogonal neighbors must not be controls
predicate no_controls_adjacent_center(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row,
  int: col
) = 
  is_control_cell(plate_layout[plate, row, col]) ->
    (plate_layout[plate, row-1, col] <= experiments /\
     plate_layout[plate, row, col-1] <= experiments /\
     plate_layout[plate, row, col+1] <= experiments /\
     plate_layout[plate, row+1, col] <= experiments);

% Top edge: 3 neighbors must not be controls
predicate no_controls_adjacent_top_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: col
) = 
  plate_layout[plate, 1, col] > experiments ->
    (plate_layout[plate, 1, col-1] <= experiments /\
     plate_layout[plate, 1, col+1] <= experiments /\
     plate_layout[plate, 2, col]   <= experiments);

% Bottom edge: 3 neighbors must not be controls
predicate no_controls_adjacent_bottom_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_rows,
  int: col
) = 
  plate_layout[plate, num_rows, col] > experiments ->
    (plate_layout[plate, num_rows-1, col] <= experiments /\
     plate_layout[plate, num_rows, col-1] <= experiments /\
     plate_layout[plate, num_rows, col+1] <= experiments);

% Left edge: 3 neighbors must not be controls
predicate no_controls_adjacent_left_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row
) = 
  plate_layout[plate, row, 1] > experiments ->
    (plate_layout[plate, row-1, 1] <= experiments /\
     plate_layout[plate, row, 2]   <= experiments /\
     plate_layout[plate, row+1, 1] <= experiments);

% Right edge: 3 neighbors must not be controls
predicate no_controls_adjacent_right_edge(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: row,
  int: num_cols
) = 
  plate_layout[plate, row, num_cols] > experiments ->
    (plate_layout[plate, row-1, num_cols]   <= experiments /\
     plate_layout[plate, row, num_cols-1]   <= experiments /\
     plate_layout[plate, row+1, num_cols]   <= experiments);

% Upper-left corner: 2 neighbors must not be controls
predicate no_controls_adjacent_upper_left(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate
) = 
  plate_layout[plate, 1, 1] > experiments ->
    (plate_layout[plate, 1, 2] <= experiments /\
     plate_layout[plate, 2, 1] <= experiments);

% Upper-right corner: 2 neighbors must not be controls
predicate no_controls_adjacent_upper_right(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_cols
) = 
  plate_layout[plate, 1, num_cols] > experiments ->
    (plate_layout[plate, 1, num_cols-1] <= experiments /\
     plate_layout[plate, 2, num_cols]   <= experiments);

% Lower-left corner: 2 neighbors must not be controls
predicate no_controls_adjacent_lower_left(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_rows
) = 
  plate_layout[plate, num_rows, 1] > experiments ->
    (plate_layout[plate, num_rows-1, 1] <= experiments /\
     plate_layout[plate, num_rows, 2]   <= experiments);

% Lower-right corner: 2 neighbors must not be controls
predicate no_controls_adjacent_lower_right(
  array[Plates, Rows, Columns] of var int: plate_layout,
  int: plate,
  int: num_rows,
  int: num_cols
) = 
  plate_layout[plate, num_rows, num_cols] > experiments ->
    (plate_layout[plate, num_rows-1, num_cols] <= experiments /\
     plate_layout[plate, num_rows, num_cols-1] <= experiments);
