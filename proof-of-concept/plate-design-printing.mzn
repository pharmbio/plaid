

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Pretty printing %%%

output [if testing \/ debugging then "\(numplates) plates\n" else "" endif];

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
        if j== numrows /\ k == numcols /\ i<numplates then "\n\n Plate \(i+1):\n" 
        elseif k == numcols then "\n" else " " endif else "" endif|
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
%%% CSV Ouput %%%

array[int] of string: letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
         
output ["plateID,well,cmpdname,CONCuM,cmpdnum,VOLuL\n"];

%%% TODO: FIX ME!!!!!! Problem when there are vertical/horizontal cell lines!
output [if fix(plates[i,j,k]) > 0 then "plate_\(i)," ++   letters[size_empty_edge+j+(numrows+(1+inner_empty_edge)*size_empty_edge)*(h-1)] ++ if (size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1)) < 10 then "0" else "" endif ++ "\(size_empty_edge+k+((1+inner_empty_edge)*size_empty_edge+numcols)*(v-1))," ++ 

%%%% Compounds %%%%

if fix(plates[i,j,k]) <= experiments then compound_names[(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) div max_compound_concentrations)+1] 
++ "," ++ compound_concentration_names[(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) div max_compound_concentrations)+1,(((fix(plates[i,j,k])-1) mod (compounds*max_compound_concentrations)) mod max_compound_concentrations)+1] ++ "," ++

compound_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1] ++ "_" ++ compound_concentration_names[(floor(((fix(plates[i,j,k])-1)/max_compound_concentrations)) mod compounds)+1,((floor((fix(plates[i,j,k])-1))) mod max_compound_concentrations)+1]


%%%% Controls %%%%

else
%control_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1] ++ "," ++  
control_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1] ++ "," ++  
%% control concentration

%control_concentration_names[((fix(plates[i,j,k])-experiments-1) mod num_controls)+1,floor((fix(plates[i,j,k])-experiments-1)/num_controls)+1] 
control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1] 

++ "," ++ 
%% Latex name of the control. Right now it's the same as the name
control_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1] ++ "_" ++ control_concentration_names[floor((fix(plates[i,j,k])-experiments-1)/max_control_concentrations)+1,((fix(plates[i,j,k])-experiments-1) mod max_control_concentrations)+1]  

endif 

++ "\n" else "" endif|
        i in Plates, j in Rows, k in Columns, v in 1..vertical_cell_lines, h in 1..horizontal_cell_lines];       
         

