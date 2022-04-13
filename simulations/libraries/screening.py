## Library used for the screening simulation
# https://academic.oup.com/bioinformatics/article/31/23/3815/208794

import numpy as np

def control_stats(plate_array, layout, neg_control_id, pos_control_id):
    num_rows, num_columns = layout.shape
    
    #neg_control_id = np.max(layout)
    #pos_control_id = neg_control_id -1 
    
    pos_values = np.empty(0,np.float64)
    neg_values = np.empty(0,np.float64)
    
    ## Collect controls
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (layout[row_index][col_index] == neg_control_id):
                 neg_values = np.append(neg_values,plate_array[row_index][col_index])
            elif (layout[row_index][col_index] == pos_control_id):
                 pos_values = np.append(pos_values,plate_array[row_index][col_index])

    # neg_control_mean, pos_control_mean, neg_stdev, pos_stdev
    return np.mean(neg_values), np.mean(pos_values), np.std(neg_values), np.std(pos_values)


# Strictly Standarized Mean Difference
def ssmd(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev):
    return (pos_control_mean - neg_control_mean)/np.sqrt(pos_stdev**2+neg_stdev**2)


def zfactor(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev):  
    return 1 - (3*(pos_stdev+neg_stdev)/abs(pos_control_mean - neg_control_mean))
                
                
                
def fill_plate(layout, neg_control_id, pos_control_id, neg_control_mean = 95, pos_control_mean = 5, neg_stdev = 5, pos_stdev = 5):
    num_rows, num_columns = layout.shape
    
    plate = np.full((num_rows, num_columns), 0.0)
    
    #neg_control_id = np.max(layout)
    #pos_control_id = neg_control_id -1 

    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (layout[row_index][col_index] == neg_control_id):
                 plate[row_index][col_index] = np.random.normal(neg_control_mean, neg_stdev)
            elif (layout[row_index][col_index] > 2*(neg_control_id-2)//3):
                 plate[row_index][col_index] = np.random.normal(pos_control_mean, pos_stdev)
            else:
                 plate[row_index][col_index] = np.random.normal(neg_control_mean, neg_stdev)
                    
    return np.abs(plate)



def screen(layout_dir,layout_file,neg_control_mean, pos_control_mean, neg_stdev, pos_stdev,error_function,error,normalization_function,min_dist,lose_from_row=0,lose_to_row=0):
    
    layout = np.load(layout_dir+layout_file)    

    # Fill plate
    plate = fill_plate(layout, neg_control_mean, pos_control_mean, neg_stdev, pos_stdev)
    
    # Add errors
    plate = error_function(plate, error)
    
    plate = dt.lose_rows(plate, lose_from_row, lose_to_row)
    
    # Fix errors
    neg_control_locations = util.get_controls_layout(layout.astype(np.float32))
    neg_control_locations = dt.lose_rows(neg_control_locations, lose_from_row, lose_to_row)
    layout = dt.lose_rows(layout, lose_from_row, lose_to_row)
    
    plate = normalization_function(plate,neg_control_locations,min_dist=min_dist)
    
    neg_control_mean, pos_control_mean, neg_stdev, pos_stdev = control_stats(plate, layout)
        
    ssmd = ssmd(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev)
    zfactor = zfactor(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev)
    
    return ssmd, zfactor
