import numpy as np
import math

def add_bowlshaped_errors(plate, error):
    
    plate_array = __add_bowlshaped_errors_to_columns(plate, error)
    plate_array = __add_bowlshaped_errors_to_rows(plate_array, error)
    
    return plate_array



def __add_bowlshaped_errors_to_columns(plate, error):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    translation_const = (num_columns + 1)/2
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index] > 0):
                plate_array[row_index][col_index] += error*abs(col_index - translation_const)
            
    return plate_array



def __add_bowlshaped_errors_to_rows(plate, error):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()

    translation_const = (num_rows + 1)/2
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index] > 0):
                plate_array[row_index][col_index] += error*abs(row_index - translation_const)
                
    return plate_array


def add_bowlshaped_errors_nl(plate, error=0.125):
    
    plate_array = __add_bowlshaped_errors_to_columns_nl(plate, error)
    plate_array = __add_bowlshaped_errors_to_rows_nl(plate_array, error)
    
    return plate_array



def __add_bowlshaped_errors_to_columns_nl(plate, error):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)
    
    translation_const = (num_columns + 1)/2
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*abs(col_index - translation_const))
#            plate_array[row_index][col_index] += (np.random.random()-0.5)*error*abs(col_index - translation_const)
            
    return plate_array



def __add_bowlshaped_errors_to_rows_nl(plate, error):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)

    translation_const = (num_rows + 1)/2
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*abs(row_index - translation_const))
#            plate_array[row_index][col_index] += (np.random.random()-0.5)*error*abs(col_index - translation_const)
            
    return plate_array


def add_errors_to_upper_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_rows-row_index))
            
    return plate_array


def add_linear_errors_to_upper_rows(plate, error=8):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] += error*(num_rows-row_index)
            
    return plate_array


def add_linear_errors_to_upper_rows_neg(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] -= error*(num_rows-row_index)
                plate_array[row_index][col_index] = min(plate_array[row_index][col_index],0)
            
    return plate_array



def add_linear_errors_to_upper_rows_half(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows//2):
        for col_index in range(num_columns):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] += 2*error*(num_rows//2-row_index)
            
    return plate_array


def add_errors_to_lower_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*row_index)
            
    return plate_array


def add_errors_to_lower_rows_half(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows//2,num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_rows//2 -row_index))
            
    return plate_array



def add_linear_errors_to_lower_rows_neg(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] -= error*row_index
                plate_array[row_index][col_index] = min(plate_array[row_index][col_index],0)
            
    return plate_array



def add_linear_errors_to_lower_rows_half(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows//2,num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] += 2*error*(row_index-num_rows//2)
            
    return plate_array


def add_errors_to_left_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns-col_index))
    return plate_array


def add_linear_errors_to_left_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] += 2*error*(num_columns-col_index)
    return plate_array



def add_errors_to_right_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*col_index)
            #if (plate[row_index][col_index]>0):
             #   plate_array[row_index][col_index] += error*col_index
            
    return plate_array


def add_errors_to_left_columns_half(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns//2):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] += error*(1-col_index+num_columns//2)
            
    return plate_array


def add_errors_to_right_columns_half(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns//3,num_columns):
            if (plate[row_index][col_index]>0):
                plate_array[row_index][col_index] = plate[row_index][col_index]*(1 + error*abs(col_index-num_columns//3))
    return plate_array



def add_striped_errors_even_rows_left(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 0):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns-col_index))
                
    return plate_array



def add_striped_errors_odd_rows_left(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 1):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns-col_index))
                
    return plate_array



def add_striped_errors_even_rows_right(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 0):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(col_index))
                
    return plate_array




def add_striped_errors_odd_rows_right(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 1):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(col_index))
                
    return plate_array




def lose_columns(plate, from_col, to_col, empty_value=0):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    # 0 is used as 'empty' in layout arrays, but NaN is preferable for plate/results information
    # It can be an important difference if the value in a well is (close to) 0 or if it's empty.
    
    from_col = min(max(from_col,0),num_columns)
    to_col = min(to_col,num_columns)
    
    for col_index in range(from_col,to_col):
        for row_index in range(num_rows):
            plate_array[row_index][col_index] = empty_value
            
    return plate_array



def lose_rows(plate, from_row, to_row, empty_value=0):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    # 0 is used as 'empty' in layout arrays, but NaN is preferable for plate/results information
    # It can be an important difference if the value in a well is (close to) 0 or if it's empty.
        
    from_row = min(max(from_row,0),num_rows)
    to_row = min(to_row,num_rows)
    
    for row_index in range(from_row,to_row):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = empty_value
            
    return plate_array


def add_exponential_errors_to_upper_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**(num_rows-row_index))
            
    return plate_array


def add_exponential_errors_to_lower_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**row_index)
            
    return plate_array



def add_exponential_errors_to_left_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**(num_columns-col_index))
            
    return plate_array



def add_exponential_errors_to_right_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = np.full((num_rows, num_columns), 0.0)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**col_index)
            
    return plate_array


def add_linear_errors_to_bottom_left_half(plate, error=0.125):
    plate_array = add_linear_errors_to_lower_rows(plate, error)
    plate_array = add_linear_errors_to_left_columns_half(plate_array, error)
    return plate_array


def add_linear_errors_to_bottom_right_half(plate, error=0.125):
    plate_array = add_linear_errors_to_lower_rows(plate, error)
    plate_array = add_linear_errors_to_right_columns_half(plate_array, error)
    return plate_array


def add_linear_errors_to_upper_left(plate, error=0.125):
    plate_array = add_linear_errors_to_left_columns(plate, error)
    plate_array = add_linear_errors_to_upper_rows(plate_array, error)
    return plate_array



def add_diagonal_errors_x(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows//6,num_rows):
        for col_index in range(num_columns//6,num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*((num_rows//6)-row_index)*((num_columns//6)-col_index))
            
    return plate_array


def add_diagonal_errors(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if math.sqrt((num_rows-row_index)**2 + (num_columns - col_index)**2) < num_columns:
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns - math.sqrt((num_rows-row_index)**2 + (num_columns - col_index)**2)))
#            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(col_index - num_columns//2 + 1))
            
    return plate_array
