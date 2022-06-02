import numpy as np
import pandas as pd
from moepy import lowess, eda
import libraries.utilities as util
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from loess.loess_2d import loess_2d
import warnings

def normalize_plate_nearest_control(plate_array, layout, neg_control_id, min_dist=0):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    normalized_plate = np.full((16, 24), 0.0)

    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if control_locations[row_index][col_index] > 0:
                normalized_plate[row_index][col_index] = 100
            else:
                dist = min_dist-1
                total_controls = 0
            
                while(total_controls == 0):
                    dist+=1
                    relevant_locations = control_locations[max(0,row_index-dist):min(num_rows,row_index+dist+1),max(0,col_index-dist):min(num_columns,col_index+dist+1)]
                    total_controls = relevant_locations.sum()
                
                
                relevant_plate = plate_array[max(0,row_index-dist):min(num_rows,row_index+dist+1),max(0,col_index-dist):min(num_columns,col_index+dist+1)]
            
                control_values = (np.multiply(relevant_plate,relevant_locations)).sum()
                normalized_plate[row_index][col_index] = plate_array[row_index][col_index]*100*total_controls/control_values
            
                
    return normalized_plate



def normalize_plate_mean(plate_array,layout, neg_control_id):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape

    mean = np.mean([plate_array[i,j] for i in range(num_rows) for j in range(num_columns) if control_locations[i,j]==1 ])
    
    normalized_plate = (100/mean)*plate_array
    
    return normalized_plate



def normalize_plate_median(plate_array,layout, neg_control_id):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape

    median = np.median([plate_array[i,j] for i in range(num_rows) for j in range(num_columns) if control_locations[i,j]==1 ])
    
    normalized_plate = (100/median)*plate_array
    
    return normalized_plate



def normalize_plate_clean_mean(plate_array,control_locations):
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape

    controls = [plate_array[i,j] for i in range(num_rows) for j in range(num_columns) if control_locations[i,j]==1 ]
    controls.sort()
    mean = np.mean(controls[7:18])
    
    normalized_plate = (100/mean)*plate_array
    
    return normalized_plate



## Normalization 
def normalize_plate_column_effect(plate_array, layout, neg_control_id, min_dist=1):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    normalized_plate = np.full((num_rows, num_columns), 0.0)

    for col_index in range(num_columns):
        dist = min_dist-1
        total_controls = 0
            
        while(total_controls == 0):
            dist+=1
            relevant_locations = control_locations[:,max(0,col_index-dist):min(num_columns,col_index+dist+1)]
            total_controls = relevant_locations.sum()
        
        relevant_plate = plate_array[:,max(0,col_index-dist):min(num_columns,col_index+dist+1)]
        control_values = (np.multiply(relevant_plate,relevant_locations)).sum()
        
        normalization_value = 100*total_controls/control_values
        
        for row_index in range(num_rows):
            normalized_plate[row_index][col_index] = plate_array[row_index][col_index]*normalization_value
            
    return normalized_plate





## Normalization 
def normalize_plate_row_effect(plate_array, layout, neg_control_id, min_dist=1):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    normalized_plate = np.full((16, 24), 0.0)

    for row_index in range(num_rows):
        dist = min_dist-1
        total_controls = 0
            
        while(total_controls == 0):
            dist+=1
            relevant_locations = control_locations[max(0,row_index-dist):min(num_rows,row_index+dist+1),:]
            total_controls = relevant_locations.sum()
        
        relevant_plate = plate_array[max(0,row_index-dist):min(num_rows,row_index+dist+1),:]
        control_values = (np.multiply(relevant_plate,relevant_locations)).sum()
        
        normalization_value = 100*total_controls/control_values
        
        for col_index in range(num_columns):
            normalized_plate[row_index][col_index] = plate_array[row_index][col_index]*normalization_value
            
    return normalized_plate


## Normalization 
def normalize_plate_striped_even_rows_column_effect(plate_array, layout, neg_control_id, min_dist=1):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    normalized_plate = np.full((16, 24), 0.0)

    for col_index in range(num_columns):
        dist = min_dist-1
        total_controls = 0
            
        while(total_controls == 0):
            dist+=1
            relevant_locations = control_locations[:,max(0,col_index-dist):min(num_columns,col_index+dist+1)]
            x,y = relevant_locations.shape
            relevant_locations = np.array([[relevant_locations[i,j] if i%2==0 else 0 for j in range(y)] for i in range(x)])
            total_controls = relevant_locations.sum()
        
        relevant_plate = plate_array[:,max(0,col_index-dist):min(num_columns,col_index+dist+1)]
        control_values = (np.multiply(relevant_plate,relevant_locations)).sum()
        
        normalization_value = 100*total_controls/control_values
        
        for row_index in range(num_rows):
            if row_index%2==0:
                normalized_plate[row_index][col_index] = plate_array[row_index][col_index]*normalization_value
            else:
                normalized_plate[row_index][col_index] = plate_array[row_index][col_index]
            
    return normalized_plate


## Normalization 
def normalize_plate_striped_odd_rows_column_effect(plate_array, layout, neg_control_id,min_dist=1):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)
    if control_locations.sum() == 0:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    normalized_plate = np.full((16, 24), 0.0)

    for col_index in range(num_columns):
        dist = min_dist-1
        total_controls = 0
            
        while(total_controls == 0):
            dist+=1
            relevant_locations = control_locations[:,max(0,col_index-dist):min(num_columns,col_index+dist+1)]
            x,y = relevant_locations.shape
            relevant_locations = np.array([[relevant_locations[i,j] if i%2==1 else 0 for j in range(y)] for i in range(x)])
            total_controls = relevant_locations.sum()
        
        relevant_plate = plate_array[:,max(0,col_index-dist):min(num_columns,col_index+dist+1)]
        control_values = (np.multiply(relevant_plate,relevant_locations)).sum()
        
        normalization_value = 100*total_controls/control_values
        
        for row_index in range(num_rows):
            if row_index%2==1:
                normalized_plate[row_index][col_index] = plate_array[row_index][col_index]*normalization_value
            else:
                normalized_plate[row_index][col_index] = plate_array[row_index][col_index]
            
    return normalized_plate



def normalize_plate_lowess_deprecated(plate_array, layout, neg_control_id=None, min_dist=None, my_frac = 1.0):
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)

    if control_locations.sum() < 2:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    
    plate_df = pd.DataFrame(plate_array)
    intensity_df = plate_df.stack().reset_index()
    intensity_df.columns = ["Rows","Columns","Intensity"]

    controls_df = pd.DataFrame(layout).stack().reset_index()
    controls_df.columns = ["Rows","Columns","Type"]
    
    y_adjusted = pd.merge(intensity_df, controls_df,  how='left', on=['Rows','Columns'])

    y_adjusted.reset_index()
    
    #neg_control_id = np.max(layout)
    
    ### Adjust rows ###
    lowess_rows_model = lowess.Lowess()
        
    lowess_rows_model.fit(y_adjusted.loc[y_adjusted['Type']==neg_control_id,['Rows']].to_numpy(),y_adjusted[y_adjusted.Type==neg_control_id].Intensity.to_numpy(),frac=my_frac,num_fits=5000)

    xnew_rows = np.array([i for i in range(0,num_rows)])
    y_pred_rows = lowess_rows_model.predict(xnew_rows)
    
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] -= y_pred_rows[y_adjusted.loc[y_adjusted['Type']>0,['Rows']]] 
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] += np.nanmean(y_pred_rows)
    
    
    ### Adjustc columns ###
    lowess_model = lowess.Lowess()
    lowess_model.fit(y_adjusted[y_adjusted.Type==neg_control_id].Columns.to_numpy(),y_adjusted[y_adjusted.Type==neg_control_id].Intensity.to_numpy(),frac=my_frac,num_fits=5000)

    xnew = np.array([i for i in range(0,num_columns)])
    y_pred = lowess_model.predict(xnew)
    
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] -= y_pred[y_adjusted.loc[y_adjusted['Type']>0,['Columns']]] 
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] += np.nanmean(y_pred)
    
    ### Back to plate format ###
    unstack_adjusted_df = y_adjusted[["Rows","Columns","Intensity"]].copy()
    
    unstacked_adjusted_df = pd.pivot_table(unstack_adjusted_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)
    

    #return unstacked_adjusted_df.to_numpy()
    
    return normalize_plate_mean(unstacked_adjusted_df.to_numpy(), layout, neg_control_id)




# This is the preferred method of smoothing!
def normalize_plate_lowess_2d(plate_array_in, layout, neg_control_id, min_dist=None, frac = 1.0):
    """ Smoothing using loess_2d """

    #warnings.filterwarnings("error")
    
    #plate_array = plate_array_in
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
        plate_array = np.log10(plate_array_in)

    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)

    if control_locations.sum() < 2:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    
    plate_df = pd.DataFrame(plate_array)

    intensity_df = plate_df.stack().reset_index()
    intensity_df.columns = ["Rows","Columns","Intensity"]

    
    controls_df = pd.DataFrame(layout).stack().reset_index()
    controls_df.columns = ["Rows","Columns","Type"]
    
    y_adjusted = pd.merge(intensity_df, controls_df,  how='left', on=['Rows','Columns'])

    y_adjusted.reset_index()
    
    ## LOESS
    
    x = y_adjusted.loc[y_adjusted['Type']==neg_control_id,['Columns']].to_numpy().reshape((-1,))
    y = y_adjusted.loc[y_adjusted['Type']==neg_control_id,['Rows']].to_numpy().reshape((-1,))
    z = y_adjusted.loc[y_adjusted['Type']==neg_control_id,['Intensity']].to_numpy().reshape((-1,))
    
    xnew = y_adjusted['Columns'].to_numpy().reshape((-1,))
    ynew = y_adjusted['Rows'].to_numpy().reshape((-1,))
    znew = y_adjusted['Intensity'].to_numpy().reshape((-1,))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        zout, _ = loess_2d(x, y, z, xnew=xnew, ynew=ynew, degree=1, frac=frac, npoints=None, rescale=False, sigz=None)
        zout_controls, _ = loess_2d(x, y, z, degree=1, frac=frac, npoints=None, rescale=False, sigz=None)
    
    z_norm = znew - zout + np.nanmean(zout_controls)
    
        
    new_plate = z_norm.reshape((num_rows, num_columns))
    
    #return new_plate
    #return normalize_plate_nearest_control(new_plate, layout, neg_control_id)
    return normalize_plate_mean(np.power(10,new_plate), layout, neg_control_id)
    #return normalize_plate_mean(new_plate, layout, neg_control_id)
    
    
    
def normalize_plate_linear(plate_array_in, layout, neg_control_id, min_dist=None):
    #neg_control_id = np.max(layout)
    
    #plate_array = plate_array_in
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
        plate_array = np.log10(plate_array_in)
    
    control_locations = util.get_controls_layout(layout,neg_control=neg_control_id)

    if control_locations.sum() < 2:
        return plate_array.copy()
    
    num_rows, num_columns = plate_array.shape
    
    plate_df = pd.DataFrame(plate_array)
    intensity_df = plate_df.stack().reset_index()
    intensity_df.columns = ["Rows","Columns","Intensity"]
    
    types_df = pd.DataFrame(layout).stack().reset_index()
    types_df.columns = ["Rows","Columns","Type"]
    
    y_adjusted = pd.merge(intensity_df, types_df,  how='left', on=['Rows','Columns'])
    
    y_adjusted.reset_index()
    
    ###################
    
    ### Adjust rows ###
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        linear_model_rows = LinearRegression()
        linear_model_rows.fit(y_adjusted[y_adjusted.Type==neg_control_id].Rows.to_numpy().reshape(-1,1), y_adjusted[y_adjusted.Type==neg_control_id].Intensity.to_numpy().reshape(-1, 1))

    xnew_rows = np.array([i for i in range(0,num_rows)])
    y_pred_rows = linear_model_rows.predict(xnew_rows.reshape(-1, 1))
    mean_y = np.nanmean(y_pred_rows)
    
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] -= linear_model_rows.predict(y_adjusted.loc[y_adjusted['Type']>0,['Rows']].values)
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] += mean_y
    
    ### Adjust columns ###
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        linear_model_columns = LinearRegression()
        linear_model_columns.fit(y_adjusted[y_adjusted.Type==neg_control_id].Columns.to_numpy().reshape(-1,1), y_adjusted[y_adjusted.Type==neg_control_id].Intensity.to_numpy().reshape(-1, 1))

    xnew_columns = np.array([i for i in range(0,num_columns)])
    y_pred_columns = linear_model_columns.predict(xnew_columns.reshape(-1, 1))
    mean_y = np.nanmean(y_pred_columns)    
    
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] -= linear_model_columns.predict(y_adjusted.loc[y_adjusted['Type']>0,['Columns']].values)
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] += mean_y

    
    unstack_adjusted_df = y_adjusted[["Rows","Columns","Intensity"]].copy()
    unstacked_adjusted_df = pd.pivot_table(unstack_adjusted_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)
    
    
    #return unstacked_adjusted_df.to_numpy()
    
    #return normalize_plate_nearest_control(unstacked_adjusted_df.to_numpy(), layout, neg_control_id)
    return normalize_plate_mean(np.power(10,unstacked_adjusted_df.to_numpy()), layout, neg_control_id)
    #return normalize_plate_mean(unstacked_adjusted_df.to_numpy(), layout, neg_control_id)




############


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    
    num_rows, num_columns = df.shape
        
    # apply the z-score method
    for col_index in range(num_columns):
        df_std[:,col_index] = (df_std[:,col_index] - df_std[:,col_index].mean()) / df_std[:,col_index].std()
        
    return df_std