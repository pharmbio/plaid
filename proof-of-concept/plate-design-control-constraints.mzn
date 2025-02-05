%%%% AUXILIARY VARIABLES %%%%

% Number of controls in the upper and lower halves of each plate
array [Plates,Vertical] of var int: sum_controls_ul_plates;

% Number of controls in the left and halves of each plate
array [Plates,Horizontal] of var int: sum_controls_lr_plates;


array [Plates,Vertical,1..num_controls*max_control_concentrations] of var int: controls_ul_plates;

array [Plates,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_lr_plates;


array [Plates,Vertical,Horizontal] of var int: sum_controls_quarter_plates;

array [Plates,Vertical,Horizontal,1..num_controls*max_control_concentrations] of var int: controls_quarter_plates;

%array [Plates,Rows,{0} union 1..num_controls*max_control_concentrations] of var int: controls_row_plates;

%array [Plates,Columns,{0} union 1..num_controls*max_control_concentrations] of var int: controls_column_plates;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% VARIABLE DEFINITION %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constraint forall(i in Plates) (sum_controls_ul_plates[i,upper] == sum(controls_layout[i,1..(numrows div 2),..]));

constraint forall(i in Plates) (sum_controls_ul_plates[i,lower] == sum(controls_layout[i, (numrows - (numrows div 2) +1..numrows) ,..]));


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% MODEL CONSTRAINTS %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 50% of the total number of controls should be on the upper half of all plates and 50% at the bottom half of all plates
% The upper limit in the number of controls works for any plate. 
% However, it might be weak for plates with an odd number of rows.
constraint if even_rows then forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) <= ((total_controls + 1) div 2))
           else forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) <= ((total_controls - sum(controls_layout[..,(numrows div 2) +1,..]) + 1) div 2)) endif;



% The lower limit in the number of controls per half must take into account if the plate has an odd number of rows.
constraint if even_rows then forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) >= (total_controls div 2))
           else forall(v in Vertical) (sum(sum_controls_ul_plates[..,v]) >= ( (total_controls - sum(controls_layout[..,(numrows div 2) +1,..])) div 2)) endif;


% 50% of the total number of controls should be on the upper half of all plates and 50% at the bottom half of all plates
% If the controls are the same on every plate, then we know the exact number of total controls per plate
constraint if equal_controls /\ even_rows then forall(i in Plates, v in Vertical) ( sum_controls_ul_plates[i,v] >= ((total_controls div numplates) div 2)) 
           elseif equal_controls then forall(i in Plates, v in Vertical) ( sum_controls_ul_plates[i,v] >= (((total_controls div numplates) - sum(controls_layout[i,(numrows div 2) +1,..]) ) div 2)) 
           else true
           endif;

%% The upper limit can still be half of the controls, but it is not a tight limit if there are many controls in the middle row.
constraint if equal_controls then forall(i in Plates, v in Vertical) ( sum_controls_ul_plates[i,v] <= (((total_controls div numplates) + 1) div 2)) endif;

constraint if equal_controls /\ even_rows then forall(i in Plates) ( sum(sum_controls_ul_plates[i,..]) == (total_controls div numplates) ) 
           elseif equal_controls then forall(i in Plates) ( sum(sum_controls_ul_plates[i,..]) <= (total_controls div numplates) ) 
           else true
           endif;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% IMPLIED CONSTRAINTS %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The total sum of the controls, across all plates, of the upper and lower halves in the total number of controls
constraint if even_rows then sum(sum_controls_ul_plates) == total_controls 
           else sum(sum_controls_ul_plates) + sum(controls_layout[..,(numrows div 2) +1,..]) == total_controls endif;

% The total sum of the controls, across all plates, of the left and right halves in the total number of controls
%constraint sum(sum_controls_lr_plates) == total_controls;