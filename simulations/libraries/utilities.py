import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import os
import re
from statannotations.Annotator import Annotator # to add p-values to plots
from scipy import stats
from random import randrange


box_pairs_erb = [("Effective", "Random"), ("Random", "Border"), ("Effective", "Border")]
box_pairs_bre = [("Random", "Effective"), ("Border", "Random"), ("Border", "Effective")]

order_erb = ["Effective","Random","Border"]
order_bre = ["Border","Random","Effective"]

def plot_plate(plate_array, title="", mask=None, filename=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.xaxis.tick_top()
    plt.title(title, fontsize = 15) 
    sns.heatmap(plate_array,linewidth=0.3,square=True,mask=mask,vmin=vmin,vmax=vmax)
    plt.show()
    
    if filename:
        fig.savefig(filename,bbox_inches='tight')
    
    
def create_random_layout_controls(plate_id, num_controls=20, num_rows=16, num_columns=24, directory="layouts/controls_manual_layouts"):
    '''Create a random layout of (negative) controls'''
    
    layout = np.full(num_rows*num_columns, 0)

    while layout.sum()<num_controls:
        idx = randrange(0,num_rows*num_columns)
        layout[idx] = 1

    B = np.reshape(layout, (-1, num_columns))

    print(B)
    np.save(directory+'/plate_layout_rand_'+plate_id+'.npy',B)

    

def create_random_layout_compounds(plate_id, num_controls=20, num_rows=16, num_columns=24, directory="layouts/compounds_manual_layouts", size_empty_edge=1):
    '''Create a random layout of negative controls and compounds'''
    
    total_compounds = (num_rows-2*size_empty_edge)*(num_columns-2*size_empty_edge)-num_controls
    
    layout = [(total_compounds+1) for i in range(num_controls)]+[i for i in range(1,total_compounds+1)]
    random.shuffle(layout)
    
    layout = np.reshape(layout,(-1, num_columns-2*size_empty_edge))
    
    vertical_edge = np.reshape(np.full(size_empty_edge*(num_rows-2*size_empty_edge),0), (-1,size_empty_edge))
                         
    layout = np.hstack((vertical_edge,layout))
    
    layout = np.hstack((layout,vertical_edge))
    
    horizontal_edge = np.reshape(np.full(size_empty_edge*num_columns,0), (-1,num_columns))
    
    layout = np.vstack((horizontal_edge,layout))
    
    layout = np.vstack((layout,horizontal_edge))
    
    print(layout)
    np.save(directory+'/plate_layout_rand_'+str(num_controls)+"_"+plate_id+'.npy',layout)
    
    
def save_plaid_layout(plate_id, layout_array, num_rows=16, num_columns=24, compounds=36, concentrations=4, replicates=2, size_empty_edge=1, neg_controls=20,directory="layouts/compounds_PLAID_layouts"):

    assert size_empty_edge >= 0
    assert num_rows >= 0
    assert num_columns >= 0
    
    layout = np.reshape(layout_array, (-1, num_columns-2*size_empty_edge))
    
    if size_empty_edge > 0:
        vertical_edge = np.reshape(np.full(size_empty_edge*(num_rows-2*size_empty_edge),0), (-1,size_empty_edge))
                         
        layout = np.hstack((vertical_edge,layout))
    
        layout = np.hstack((layout,vertical_edge))
    
        horizontal_edge = np.reshape(np.full(size_empty_edge*num_columns,0), (-1,num_columns))
    
        layout = np.vstack((horizontal_edge,layout))
    
        layout = np.vstack((layout,horizontal_edge))
    
    print(layout)
    np.save(directory+'/plate_layout_'+str(neg_controls)+"-"+str(compounds)+"-"+str(concentrations)+"-"+str(replicates)+"_"+plate_id+'.npy',layout)

    
def save_plaid_controls_layout(plate_id, layout_array, num_rows=16, num_columns=24, size_empty_edge=1, directory="layouts/controls_PLAID_layouts"):
    layout = __shape_layout(layout_array, num_rows, num_columns, size_empty_edge)
    
    layout = get_controls_layout(layout)
    
    print(layout)
    np.save(directory+'/plate_layout_'+plate_id+'.npy',layout)
    
    
def get_controls_layout(layout, neg_control = None):
    num_rows, num_columns = layout.shape
    
    if neg_control is None:
        control = np.max(layout)
    else:
        control = neg_control

    control_layout = np.full((num_rows, num_columns), 0)

    for row_i in range(num_rows):
        for col_i in range(num_columns):
            if layout[row_i][col_i] == control:
                control_layout[row_i][col_i] = 1

    return(control_layout)



def save_plaid_screening_layout(plate_id, layout_array, num_rows=16, num_columns=24, size_empty_edge=1, directory="layouts/screening_PLAID_layouts"):
    neg_id = np.max(layout_array)
    pos_id = neg_id - 1
    
    neg_controls = layout_array.count(neg_id)
    pos_controls = layout_array.count(pos_id)
    
    layout = __shape_layout(layout_array, num_rows, num_columns, size_empty_edge)
    
    print(layout)
    np.save(directory+'/plate_layout_'+str(neg_controls)+"-"+str(pos_controls)+"_"+plate_id+'.npy',layout)
    
    
def create_random_layout_screening(plate_id, neg_controls=10, pos_controls = 10, num_rows=16, num_columns=24, directory="layouts/screening_manual_layouts", size_empty_edge=1):
    '''Create a random layout for a screening experiment'''
    
    total_compounds = (num_rows-2*size_empty_edge)*(num_columns-2*size_empty_edge)-neg_controls-pos_controls
    
    layout = [(total_compounds+1) for i in range(pos_controls)]+[(total_compounds+2) for i in range(neg_controls)]+[(i+1) for i in range(total_compounds)]
    random.shuffle(layout)
    
    layout = __shape_layout(layout, num_rows, num_columns, size_empty_edge)
    
    print(layout)
    np.save(directory+'/plate_layout_rand_'+str(neg_controls)+"-"+str(pos_controls)+"_"+plate_id+'.npy',layout)
    
    
def __shape_layout(layout, num_rows, num_columns, size_empty_edge):
    layout = np.reshape(layout,(-1, num_columns-2*size_empty_edge))
    
    if size_empty_edge > 0:
        vertical_edge = np.reshape(np.full(size_empty_edge*(num_rows-2*size_empty_edge),0), (-1,size_empty_edge))
                         
        layout = np.hstack((vertical_edge,layout))
    
        layout = np.hstack((layout,vertical_edge))
    
        horizontal_edge = np.reshape(np.full(size_empty_edge*num_columns,0), (-1,num_columns))
    
        layout = np.vstack((horizontal_edge,layout))
    
        layout = np.vstack((layout,horizontal_edge))
    
    return layout


def fill_in_border_layout_screening(plate_id, control_layout, directory="layouts/screening_manual_layouts", size_empty_edge=1):
    '''Fill in a border layout for a screening experiment'''
    num_rows, num_columns = control_layout.shape
    
    neg_id = np.max(control_layout) #Should be >1
    pos_id = neg_id - 1 #Should be >0
    
    neg_controls = np.count_nonzero(control_layout==neg_id)
    pos_controls = np.count_nonzero(control_layout==pos_id)
    
    total_compounds = (num_rows-2*size_empty_edge)*(num_columns-2*size_empty_edge)-neg_controls-pos_controls
    
    pos_ctr = total_compounds + 1
    neg_ctr = pos_ctr + 1
    
    layout = np.full((num_rows, num_columns), 0)

    compound = 1
    
    for row_i in range(size_empty_edge,num_rows-size_empty_edge):
        for col_i in range(size_empty_edge,num_columns-size_empty_edge):
            if control_layout[row_i][col_i] == neg_id:
                layout[row_i][col_i] = neg_ctr
            elif control_layout[row_i][col_i] == pos_id:
                layout[row_i][col_i] = pos_ctr
            else:
                layout[row_i][col_i] = compound
                compound = compound + 1
    
    print(layout)
    np.save(directory+'/plate_layout_border_'+str(neg_controls)+"-"+str(pos_controls)+"_"+plate_id+'.npy',layout)
    

    
def fill_in_border_layout_screening_vertically(plate_id, control_layout, directory="layouts/screening_manual_layouts", size_empty_edge=1):
    '''Fill in a border layout for a screening experiment'''
    num_rows, num_columns = control_layout.shape
    
    neg_id = np.max(control_layout) #Should be >1
    pos_id = neg_id - 1 #Should be >0
    
    neg_controls = np.count_nonzero(control_layout==neg_id)
    pos_controls = np.count_nonzero(control_layout==pos_id)
    
    total_compounds = (num_rows-2*size_empty_edge)*(num_columns-2*size_empty_edge)-neg_controls-pos_controls
    
    pos_ctr = total_compounds + 1
    neg_ctr = pos_ctr + 1
    
    layout = np.full((num_rows, num_columns), 0)

    compound = 1
    
    
    for col_i in range(size_empty_edge,num_columns-size_empty_edge):
        for row_i in range(size_empty_edge,num_rows-size_empty_edge):
            if control_layout[row_i][col_i] == neg_id:
                layout[row_i][col_i] = neg_ctr
            elif control_layout[row_i][col_i] == pos_id:
                layout[row_i][col_i] = pos_ctr
            else:
                layout[row_i][col_i] = compound
                compound = compound + 1
    
    print(layout)
    np.save(directory+'/plate_layout_border_'+str(neg_controls)+"-"+str(pos_controls)+"_"+plate_id+'.npy',layout)
    

    
    
def fill_in_border_layout(plate_id, control_layout, directory="layouts/screening_manual_layouts", size_empty_edge=1):
    '''Fill in a border layout for a screening experiment'''
    num_rows, num_columns = control_layout.shape
    
    neg_id = np.max(control_layout)
    
    neg_controls = np.count_nonzero(control_layout==neg_id)
    
    total_compounds = (num_rows-2*size_empty_edge)*(num_columns-2*size_empty_edge)-neg_controls
    
    neg_ctr = total_compounds + 1
    
    layout = np.full((num_rows, num_columns), 0)

    compound = 1
    
    for row_i in range(size_empty_edge,num_rows-size_empty_edge):
        for col_i in range(size_empty_edge,num_columns-size_empty_edge):
            if control_layout[row_i][col_i] == neg_id:
                layout[row_i][col_i] = neg_ctr
            else:
                layout[row_i][col_i] = compound
                compound += 1
    
    print(layout)
    np.save(directory+'/plate_layout_border_'+str(neg_controls)+"_"+plate_id+'.npy',layout)
    

    
    
    
# Returns True if there are duplicated layouts
def check_duplicated_layouts(layout_dir = 'screening_manual_layouts/'):

    layouts = os.listdir(layout_dir)
    duplicates = False
    
    for layout_file in layouts:
        match = re.search('plate_layout_*',layout_file)

        if match == None:
            continue

        layout_1 = np.load(layout_dir+layout_file)  
            
        for layout_file_2 in layouts:
            match = re.search('plate_layout_*',layout_file_2)

            if (match == None) or (layout_file == layout_file_2):
                continue
                
            layout_2 = np.load(layout_dir+layout_file_2)  

            if (layout_1 == layout_2).all():
                print("Layouts "+layout_file+" "+layout_file_2+" are the same")
                duplicates = True
    
    if not duplicates:
        print('There are no duplicates')
        
    return duplicates





def plot_well_series(plate_array, norm_plate, layout, neg_control_id, pos_control_id,order=0, vmin=None, vmax=None, filename=None):    
    ''' Creates the well series plots used for the PLAID bioseminar presentation
    
    Args:
        plate_array: np array containing the raw data from the experiments
        norm_plate: np array containing the corrected/normalized data from the experiments
        layout: an np array containing the layout used in plate_array
        neg_control_id: id (number) of the negative controls such that if layout[i][j] == neg_control_id then the i,j well 
            contains a negative control
        norm_function: function used to normalize the plate, for example nrm.normalize_plate_lowess_2d
    '''
    
    num_rows, num_columns = layout.shape
    
    ### Reformat original input data
    plate_df = pd.DataFrame(plate_array)
    
    intensity_df = plate_df.stack().reset_index()
    intensity_df.columns = ["Rows","Columns","Intensity"]
    
    types_df = pd.DataFrame(layout).stack().reset_index()
    types_df.columns = ["Rows","Columns","Type"]
    
    combined_df = pd.merge(intensity_df, types_df,  how='left', on=['Rows','Columns'])
    combined_df['Rows'] += 1
    combined_df['Columns'] += 1
    
    
    ### Reformat normalized/corrected plate
    n_plate_df = pd.DataFrame(norm_plate)
    
    n_intensity_df = n_plate_df.stack().reset_index()
    n_intensity_df.columns = ["Rows","Columns","Intensity"]
    
    n_combined_df = pd.merge(n_intensity_df, types_df,  how='left', on=['Rows','Columns'])
    n_combined_df['Rows'] += 1
    n_combined_df['Columns'] += 1
    
    
    ### Plot heatmap before normalization
    unstack_df = combined_df[["Rows","Columns","Intensity"]].copy()
    unstacked_df = pd.pivot_table(unstack_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)

    #if filename:
     #   plot_plate(unstacked_df, title="Input",filename=filename+'heatmap-before')
    #else:
     #   plot_plate(unstacked_df, title="Input")
    
    
    ### Plot heatmap after normalization
    unstack_adjusted_df = n_combined_df[["Rows","Columns","Intensity"]].copy()
    unstacked_adjusted_df = pd.pivot_table(unstack_adjusted_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)
    
  #  if filename:
   #     plot_plate(unstacked_adjusted_df, title="Normalized",vmin=vmin,vmax=vmax,filename=filename+'heatmap-after')
    #else:
     #   plot_plate(unstacked_adjusted_df, title="Normalized",vmin=vmin,vmax=vmax)
    
    
    ### Plotting well series with original and normalized data
    fig, ax = plt.subplots(figsize=(11,7))
    
    ax.set(xlim=(0,num_columns+1))
    
    ## Add all the samples in the original data (except controls)
    ax = sns.regplot(data=combined_df[(combined_df.Type!=pos_control_id) & (combined_df.Type!=neg_control_id)], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, scatter_kws={"color":"orange","alpha":0.3})
    
    ## Positive controls from the raw/original data
    ax = sns.regplot(data=combined_df[combined_df.Type==pos_control_id], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, scatter_kws={"color":"orange","alpha":0.3})

    # Add negative controls from the raw/original data
    ax = sns.regplot(data=combined_df[combined_df.Type==neg_control_id], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, marker='*',scatter_kws={"color":"purple"}, truncate=False, order=order)
    
    # Add normalized data
    ax = sns.regplot(data=n_combined_df, x="Columns", y="Intensity", x_jitter=0.3, fit_reg=True, marker='x',scatter_kws={"color":"blue"}, truncate=False, order=order)
    
    ax.set_xticks(range(1,num_columns+1))
    
    if filename:
        fig.savefig(filename)
    
    plt.show()
    
    

    
    
def plot_barplot_residuals_data(residuals_1rep, residuals_2rep, residuals_3rep, fig_name, y_max=None, leg_loc="lower center", leg_ncol=3, leg_fontsize=8, pvalue_thresholds = [[1e-43, "***"], [1e-12, "**"], [1e-4, "*"], [1, "ns"]]):
    """ Plots residual plots for dose response experiments as in the manuscript.
    
    Args:
        residuals_1rep:
        residuals_2rep:
        residuals_3rep:
        fig_name:
        y_max: maximum value for the y (vertical) axis
        leg_loc: location of the legend, for example, "upper left", "lower right"
        leg_ncol: number of columns in the legend
        leg_fontsize: font size for the legend
    """
        
    residuals_df = pd.DataFrame(residuals_1rep, columns=["layout", "error_type", "Error", "E", "rows lost", "residuals", "true_residuals"])
    residuals_df_2rep = pd.DataFrame(residuals_2rep, columns=["layout", "error_type", "Error", "E", "rows lost", "residuals", "true_residuals"])
    residuals_df_3rep = pd.DataFrame(residuals_3rep, columns=["layout", "error_type", "Error", "E", "rows lost", "residuals", "true_residuals"])

    residuals_df.insert(0, 'replicates', 1)
    residuals_df_2rep.insert(0, 'replicates', 2)
    residuals_df_3rep.insert(0, 'replicates', 3)

    residuals_df = residuals_df.append(residuals_df_2rep)
    residuals_df = residuals_df.append(residuals_df_3rep)

    residuals_df.residuals = pd.to_numeric(residuals_df.residuals, errors='coerce')
    residuals_df.true_residuals = pd.to_numeric(residuals_df.true_residuals, errors='coerce')
    residuals_df['rows lost'] = pd.to_numeric(residuals_df['rows lost'], errors='coerce')

    residuals_df = residuals_df[(residuals_df['rows lost']<=1)]

    ## Rename the layouts so they can be grouped as PLAID, RANDOM or Border
    residuals_df.loc[(residuals_df['layout'] >= "plate_layout_rand"), 'layout'] = "Random"
    residuals_df.loc[(residuals_df['layout'] >= "plate_layout_border") & (residuals_df['layout'] != "Random"), 'layout'] = "Border"
    residuals_df.loc[(residuals_df['layout'] >= "plate_layout") & (residuals_df['layout'] != "Random"), 'layout'] = "Effective"

    fig, ax = plt.subplots(figsize=(4, 3))
    
    if y_max:
        ax.set(ylim=(0,y_max))

    palette = ["#b7a2d8", "#765591", "#37185d"] #Dark purple
    ax = sns.barplot(x='replicates', y="true_residuals", data=residuals_df, hue="layout", palette=sns.color_palette(palette, 3))#'rocket_r')
    plt.ylabel("Mean residuals", fontsize = 10)
    plt.legend(ncol=leg_ncol, loc=leg_loc, fontsize = leg_fontsize)
        
    #palette = ["#fac8eb", "#b75591", "#76185d"] #Pink
    #ax = sns.barplot(x='replicates', y="residuals", data=residuals_df, hue="layout", palette=sns.color_palette(palette, 3))#'rocket_r')
    #plt.ylabel("Fit Residuals")

    box_pairs = [((3,"Effective"),(3,"Random")),((1,"Effective"),(1,"Random")),((1,"Random"),(1,"Border")),((1,"Effective"),(1,"Border")),((2,"Random"),(2,"Border")),((2,"Effective"),(2,"Border")),((3,"Random"),(3,"Border")),((3,"Effective"),(3,"Border"))]
    
    ## Add annotations to the plot
    annotator = Annotator(ax, pairs=[((2,"Effective"),(2,"Random"))], data=residuals_df, x='replicates', y="true_residuals",hue='layout', order=[1,2,3],hue_order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside',pvalue_thresholds=pvalue_thresholds, text_offset=-1)
    annotator.apply_and_annotate()

    annotator = Annotator(ax, pairs=box_pairs, data=residuals_df, x='replicates', y="true_residuals",hue='layout', order=[1,2,3],hue_order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside',pvalue_thresholds=pvalue_thresholds, text_offset=-1)
    annotator.apply_and_annotate()
    

    plt.show()
    fig.savefig("residuals"+fig_name+".png",bbox_inches='tight',dpi=800)

  

    
def plot_barplot_replicate_data(data_1rep, data_2rep, data_3rep, fig_name='', fig_type='', plot_mse = True, y_max=None, leg_ncol=None, leg_loc=None, leg_fontsize=8, pvalue_thresholds=None):
    """ Plots barplots for absolute and relative EC50/IC50 for dose response experiments as in the manuscript. 
        It also plots d_diff, that is, the average difference between the expected and obtained maximum (d) of the
        dose-response 4PL sigmoid curve.
    
    Args:
        data_1rep:
        data_2rep:
        data_3rep:
        fig_name: string added to the image file name.
        fig_type: string. "relic50" for relative IC50, "absic50" for absolute IC50, 
                  "diff_d" for difference in the maximum (d) value
        y_max: maximum value for the y (vertical) axis
        leg_loc: location of the legend, for example, "upper left", "lower right"
        leg_ncol: number of columns in the legend
        leg_fontsize: font size for the legend
        pvalue_thresholds:
    """    
    
    results_df = pd.DataFrame(data_1rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df_2rep = pd.DataFrame(data_2rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df_3rep = pd.DataFrame(data_3rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df.insert(0, 'replicates', 1)
    results_df_2rep.insert(0, 'replicates', 2)
    results_df_3rep.insert(0, 'replicates', 3)

    results_df = results_df.append(results_df_2rep)
    results_df = results_df.append(results_df_3rep)

    results_df.MSE = pd.to_numeric(results_df.MSE, errors='coerce')
    results_df.E = pd.to_numeric(results_df.E, errors='coerce')
    results_df.r2_score = pd.to_numeric(results_df.r2_score, errors='coerce')
    results_df.d = pd.to_numeric(results_df.d, errors='coerce')
    results_df.fit_d = pd.to_numeric(results_df.fit_d, errors='coerce')

    results_df.insert(0, 'diff_d', 0)
    results_df.diff_d = abs(results_df.d - results_df.fit_d)

    results_df = results_df[np.logical_not(np.isnan(results_df['MSE']))]

    #results_df = results_df[(results_df["error type"]!="bowl") & (results_df["error type"]!="left")]

    #results_df = results_df[(results_df["error type"]=="right-half")]
    #results_df = results_df[(results_df["E"]<90)]
    #results_df = results_df[(results_df["E"]>10)]

    results_df.loc[(results_df['layout'] >= "plate_layout_rand"), 'layout'] = "Random"
    results_df.loc[(results_df['layout'] >= "plate_layout_border") & (results_df['layout'] != "Random"), 'layout'] = "Border"
    results_df.loc[(results_df['layout'] >= "plate_layout") & (results_df['layout'] != "Random") & (results_df['layout'] != "Border"), 'layout'] = "Effective"

    fig, ax = plt.subplots(figsize=(4, 3))

    if pvalue_thresholds is None:
        # * indicates p < 10−4, ** indicates p < 10−12, *** indicates p < 10−43.
        pvalue_thresholds = [[1e-43, "***"], [1e-12, "**"], [1e-4, "*"], [1, "ns"]] #[1e-64, "****"], 

    box_pairs = [((3,"Effective"),(3,"Random")),((1,"Effective"),(1,"Random")),((1,"Random"),(1,"Border")),((1,"Effective"),(1,"Border")),((2,"Random"),(2,"Border")),((2,"Effective"),(2,"Border")),((3,"Random"),(3,"Border")),((3,"Effective"),(3,"Border"))]

    if y_max:
        ax.set_ylim(top = y_max)
    
    ## Plotting
    if fig_type == "relic50":
        relic50_palette = ["#91d1c2", "#00A087", "#236e56"] #"#3bccaa", 
        ax = sns.barplot(x='replicates', y="MSE", data=results_df[results_df['MSE']!=np.inf], hue="layout", palette=relic50_palette)
        plt.ylabel("Mean absolute log10 difference", fontsize = 10)
               
    elif fig_type == "absic50":
        ax = sns.barplot(x='replicates', y="MSE", data=results_df[results_df['MSE']!=np.inf], hue="layout", palette='YlOrBr')
        plt.ylabel("Mean absolute log10 difference", fontsize = 10)
        
    else:
        ax = sns.barplot(x='replicates', y="diff_d", data=results_df, hue="layout", palette = "GnBu")#, palette='YlOrBr')
        plt.ylabel("Mean absolute d difference", fontsize = 10)
        fig_type = "d_diff"
        

    plt.legend(fontsize = leg_fontsize)
    
    if leg_ncol:
        plt.legend(ncol=leg_ncol)
    if leg_loc:
        plt.legend(loc=leg_loc)

    annotator = Annotator(ax, pairs=[((2,"Effective"),(2,"Random"))], data=results_df[results_df['MSE']!=np.inf], x='replicates', y="MSE",hue='layout', order=[1,2,3],hue_order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside',pvalue_thresholds=pvalue_thresholds, text_offset=-1)
    annotator.apply_and_annotate()


    annotator = Annotator(ax, pairs=box_pairs, data=results_df[results_df['MSE']!=np.inf], x='replicates', y="MSE",hue='layout', order=[1,2,3],hue_order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside',pvalue_thresholds=pvalue_thresholds, text_offset=-1)
    annotator.apply_and_annotate()


    #plt.legend().set_title(None)
    plt.show()

    fig.savefig("dose-response-"+fig_type+fig_name+".png",bbox_inches='tight',dpi=800)
    
    
    
    
    
    
    
def plot_r2_percentage(data_1rep, data_2rep, data_3rep, fig_name='', y_max=None, leg_loc="upper left", leg_ncol=1, leg_fontsize=8):
    """
    Plotting the percentage of low-quality curves for dose-response simulations as in the manuscript.
    
    """
    
    results_df = pd.DataFrame(data_1rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_2rep = pd.DataFrame(data_2rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_3rep = pd.DataFrame(data_3rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df.insert(0, 'replicates', 1)
    results_df_2rep.insert(0, 'replicates', 2)
    results_df_3rep.insert(0, 'replicates', 3)

    results_df = results_df.append(results_df_2rep)
    results_df = results_df.append(results_df_3rep)

    results_df.r2_score = pd.to_numeric(results_df.r2_score, errors='coerce')

    results_df.loc[(results_df['layout'] >= "plate_layout_rand"), 'layout'] = "RANDOM"
    results_df.loc[(results_df['layout'] >= "plate_layout_border") & (results_df['layout'] != "RANDOM"), 'layout'] = "BORDER"
    results_df.loc[(results_df['layout'] >= "plate_layout") & (results_df['layout'] != "RANDOM") & (results_df['layout'] != "BORDER"), 'layout'] = "PLAID"

    all_plaid_1rep = results_df.loc[(results_df['layout']=="PLAID") & (results_df['replicates']==1)].shape[0]
    all_plaid_2rep = results_df.loc[(results_df['layout']=="PLAID") & (results_df['replicates']==2)].shape[0]
    all_plaid_3rep = results_df.loc[(results_df['layout']=="PLAID") & (results_df['replicates']==3)].shape[0]

    all_random_1rep = results_df.loc[(results_df['layout']=="RANDOM") & (results_df['replicates']==1)].shape[0]
    all_random_2rep = results_df.loc[(results_df['layout']=="RANDOM") & (results_df['replicates']==2)].shape[0]
    all_random_3rep = results_df.loc[(results_df['layout']=="RANDOM") & (results_df['replicates']==3)].shape[0]

    all_border_1rep = results_df.loc[(results_df['layout']=="BORDER") & (results_df['replicates']==1)].shape[0]
    all_border_2rep = results_df.loc[(results_df['layout']=="BORDER") & (results_df['replicates']==2)].shape[0]
    all_border_3rep = results_df.loc[(results_df['layout']=="BORDER") & (results_df['replicates']==3)].shape[0]

    #results_df = results_df[np.logical_not(np.isinf(results_df['MSE']))]
    results_df = results_df[results_df.r2_score<0.8]

    plaid_1rep = results_df.loc[(results_df['layout']=="PLAID") & (results_df['replicates']==1)].shape[0]
    plaid_2rep = results_df.loc[(results_df['layout']=="PLAID") & (results_df['replicates']==2)].shape[0]
    plaid_3rep = results_df.loc[(results_df['layout']=="PLAID") & (results_df['replicates']==3)].shape[0]

    random_1rep = results_df.loc[(results_df['layout']=="RANDOM") & (results_df['replicates']==1)].shape[0]
    random_2rep = results_df.loc[(results_df['layout']=="RANDOM") & (results_df['replicates']==2)].shape[0]
    random_3rep = results_df.loc[(results_df['layout']=="RANDOM") & (results_df['replicates']==3)].shape[0]

    border_1rep = results_df.loc[(results_df['layout']=="BORDER") & (results_df['replicates']==1)].shape[0]
    border_2rep = results_df.loc[(results_df['layout']=="BORDER") & (results_df['replicates']==2)].shape[0]
    border_3rep = results_df.loc[(results_df['layout']=="BORDER") & (results_df['replicates']==3)].shape[0]

    percentage_data = [
        ["Effective",  1, 100*plaid_1rep/all_plaid_1rep],
        ["Random", 1, 100*random_1rep/all_random_1rep],
        ["Border", 1, 100*border_1rep/all_border_1rep],
        ["Effective", 2, 100*plaid_2rep/all_plaid_2rep],
        ["Random", 2, 100*random_2rep/all_random_2rep],
        ["Border", 2, 100*border_2rep/all_border_2rep],
        ["Effective",  3, 100*plaid_3rep/all_plaid_3rep],
        ["Random", 3, 100*random_3rep/all_random_3rep],
        ["Border", 3, 100*border_3rep/all_border_3rep]
    ]

    print(percentage_data)

    percentage_data_df = pd.DataFrame(percentage_data, columns=["layout", "replicates", "percent"])

    fig, ax = plt.subplots(figsize=(4, 3))
    
    if y_max:
        ax.set(ylim=(0,y_max))
        
    palette = ["#bc5090", "#ff6361", "#ffa600"]
    ax = sns.barplot(x='replicates', y="percent", data=percentage_data_df, hue="layout", palette=sns.color_palette(palette, 3))    
    
    plt.ylabel("Percentage", fontsize = 10)
    plt.legend(ncol=leg_ncol, loc=leg_loc, fontsize = leg_fontsize)

    
    # Show and save figure!
    plt.show()
    fig.savefig("percentage-low-r2-curves"+fig_name+".png",bbox_inches='tight')

    
    
def create_latex_table(data, tex_filename, column_name="MSE"):
    # Open file
    latex_f=open(tex_filename,'w')
    
    results_df = pd.DataFrame(data, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df.MSE = pd.to_numeric(results_df.MSE, errors='coerce')
    results_df = results_df.sort_values("MSE")
    results_df = results_df[np.logical_not(np.isnan(results_df['MSE']))]

    results_df.loc[(results_df['layout'] >= "plate_layout_rand"), 'layout'] = "Random"
    results_df.loc[(results_df['layout'] >= "plate_layout_border") & (results_df['layout'] != "Random"), 'layout'] = "Border"
    results_df.loc[(results_df['layout'] >= "plate_layout") & (results_df['layout'] != "Random") & (results_df['layout'] != "Border"), 'layout'] = "Effective"

    results_df.d = pd.to_numeric(results_df.d, errors='coerce')
    results_df.fit_d = pd.to_numeric(results_df.fit_d, errors='coerce')

    results_df.insert(0, 'diff_d', 0)
    results_df.diff_d = abs(results_df.d - results_df.fit_d)    
    
    plaid_description = results_df[results_df['layout']=='Effective'].describe()
    random_description = results_df[results_df['layout']=='Random'].describe()
    border_description = results_df[results_df['layout']=='Border'].describe()

    latex_f.write(" & Effective & Random & Border \\\\ ")
    latex_f.write("\n\\hline\n")
    
    rows = [{'row_id':'count', 'row_name':'\\tabCount{}'},
            {'row_id':'mean',  'row_name':'\\tabMean{}'},
            {'row_id':'std',   'row_name':'\\tabSTD{}'},
            {'row_id':'min',   'row_name':'\\tabMin{}'},
            {'row_id':'25%',   'row_name':'\\tabQone{}'},
            {'row_id':'50%',   'row_name':'\\tabMedian{}'},
            {'row_id':'75%',   'row_name':'\\tabQthree{}'},
            {'row_id':'max',   'row_name':'\\tabMax{}'}]
    
    for row in rows:
        latex_f.write(row['row_name']+" & "+str(round(plaid_description.loc[row['row_id'],column_name],2))+" & "+str(round(random_description.loc[row['row_id'],column_name],2))+" & "+str(round(border_description.loc[row['row_id'],column_name],2))+"\\\\ \n")
    
    latex_f.write("\\hline")
        
    # Close file
    latex_f.close()
    
    
def create_latex_table_wide(data_1rep, data_2rep, data_3rep, tex_filename, table_text = "Relative \\ECIC", column_name="MSE"):
    # Open file
    latex_f=open(tex_filename,'w')
    
    results_df = pd.DataFrame(data_1rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_2rep = pd.DataFrame(data_2rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_3rep = pd.DataFrame(data_3rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df.insert(0, 'replicates', 1)
    results_df_2rep.insert(0, 'replicates', 2)
    results_df_3rep.insert(0, 'replicates', 3)

    results_df = results_df.append(results_df_2rep)
    results_df = results_df.append(results_df_3rep)

    results_df.MSE = pd.to_numeric(results_df[column_name], errors='coerce')
    results_df = results_df.sort_values(column_name)
    results_df = results_df[np.logical_not(np.isnan(results_df[column_name]))]
    
    results_df.loc[(results_df['layout'] >= "plate_layout_rand"), 'layout'] = "Random"
    results_df.loc[(results_df['layout'] >= "plate_layout_border") & (results_df['layout'] != "Random"), 'layout'] = "Border"
    results_df.loc[(results_df['layout'] >= "plate_layout") & (results_df['layout'] != "Random") & (results_df['layout'] != "Border"), 'layout'] = "Effective"
    
#    plaid_description = results_df[results_df['layout']=='Effective'].describe()
 #   random_description = results_df[results_df['layout']=='Random'].describe()
  #  border_description = results_df[results_df['layout']=='Border'].describe()

#    latex_f.write(" & Effective & Random & Border \\\\ ")
#    latex_f.write("\n\\hline\n")

    layouts = ['Effective','Random','Border']
    
    latex_f.write("\\multirow{4}{*}{"+table_text+"}")
    
    for layout in layouts:
        latex_f.write(" & "+layout)
        
        for rep in range(1,4):
            description = results_df[(results_df['layout']==layout) & (results_df['replicates']==rep)].describe()
            latex_f.write(" & "+str(round(description.loc['mean',column_name],2))+" $\\pm$ ("+str(round(description.loc['std',column_name],2))+")")
            
        latex_f.write("\\\\ \n")
    
 #   for row in rows:
  #      latex_f.write(row['row_name']+" & "+str(round(plaid_description.loc[row['row_id'],column_name],2))+" & "+str(round(random_description.loc[row['row_id'],column_name],2))+" & "+str(round(border_description.loc[row['row_id'],column_name],2))+"\\\\ \n")
    
    latex_f.write("\\hline \n")
        
    # Close file
    latex_f.close()

    
    
    
    
    
def create_latex_table_pvalues_wide(data_1rep, data_2rep, data_3rep, tex_filename, table_text = "Relative \\ECIC", column_name="MSE"):
    # Open file
    latex_f=open(tex_filename,'w')
    
    results_df = pd.DataFrame(data_1rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_2rep = pd.DataFrame(data_2rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_3rep = pd.DataFrame(data_3rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df.insert(0, 'replicates', 1)
    results_df_2rep.insert(0, 'replicates', 2)
    results_df_3rep.insert(0, 'replicates', 3)

    results_df = results_df.append(results_df_2rep)
    results_df = results_df.append(results_df_3rep)

    results_df[column_name] = pd.to_numeric(results_df[column_name], errors='coerce')
    results_df = results_df.sort_values(column_name)
    results_df = results_df[np.logical_not(np.isnan(results_df[column_name]))]
    results_df = results_df[np.logical_not(np.isinf(results_df[column_name]))]
    
    results_df.loc[(results_df['layout'] >= "plate_layout_rand"), 'layout'] = "Random"
    results_df.loc[(results_df['layout'] >= "plate_layout_border") & (results_df['layout'] != "Random"), 'layout'] = "Border"
    results_df.loc[(results_df['layout'] >= "plate_layout") & (results_df['layout'] != "Random") & (results_df['layout'] != "Border"), 'layout'] = "Effective"

    layouts = ['Effective','Random','Border']
    
    latex_f.write("\\multirow{4}{*}{"+table_text+"}")
    
    for layout_1 in range(3):
        for layout_2 in range(layout_1+1,3):
            latex_f.write(" & "+layouts[layout_1]+" -- "+layouts[layout_2])

            for rep in range(1,4):
                results_array_1 = results_df.loc[(results_df.layout==layouts[layout_1]) & (results_df.replicates==rep),column_name]
                results_array_2 = results_df.loc[(results_df.layout==layouts[layout_2]) & (results_df.replicates==rep),column_name]

                _, pvalue = stats.ttest_ind(results_array_1,results_array_2,equal_var = False)
                latex_f.write(" & "+'{:.2e}'.format(pvalue))

            latex_f.write("\\\\ \n")
    
    latex_f.write("\\hline \n")
        
    # Close file
    latex_f.close()

    
    
def plotting_ssmd_scores(screening_scores_data_filename, fig_name, y_min=None, y_max=None):
    screening_scores_df = pd.read_csv(screening_scores_data_filename)

    ## No rows lost!
    screening_scores_df = screening_scores_df[screening_scores_df['lost_rows']<1]

    screening_scores_df['Zfactor_expected'] = pd.to_numeric(screening_scores_df['Zfactor_expected'], errors='coerce')
    screening_scores_df['Zfactor_norm'] = pd.to_numeric(screening_scores_df['Zfactor_norm'], errors='coerce')
    screening_scores_df['Zfactor_raw'] = pd.to_numeric(screening_scores_df['Zfactor_raw'], errors='coerce')

    screening_scores_df['SSMD_expected'] = pd.to_numeric(screening_scores_df['SSMD_expected'], errors='coerce')
    screening_scores_df['SSMD_norm'] = pd.to_numeric(screening_scores_df['SSMD_norm'], errors='coerce')
    screening_scores_df['SSMD_raw'] = pd.to_numeric(screening_scores_df['SSMD_raw'], errors='coerce')

    screening_scores_df['SSMD_abs'] = np.abs(screening_scores_df['SSMD_raw'])
    screening_scores_df['SSMD_norm_abs'] = np.abs(screening_scores_df['SSMD_norm'])
    
    ### This is only needed when the specific layout name is stored in 'layout' ###
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_rand"), 'layout'] = "RANDOM"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_border") & (screening_scores_df['layout'] != "RANDOM"), 'layout'] = "BORDER"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout") & (screening_scores_df['layout'] != "RANDOM") & (screening_scores_df['layout'] != "BORDER"), 'layout'] = "PLAID"

    # Plotting
    plotting_ssmd_scores_df = screening_scores_df[['layout','SSMD_norm_abs','lost_rows']]
    plotting_ssmd_scores_df = plotting_ssmd_scores_df.rename(columns={'SSMD_norm_abs': 'SSMD'})

    plotting_ssmd_scores_df.insert(0, 'type', 'Normalised')


    plotting_ssmd_scores_temp_df = screening_scores_df[['layout','SSMD_abs','lost_rows']]
    plotting_ssmd_scores_temp_df = plotting_ssmd_scores_temp_df.rename(columns={'SSMD_abs': 'SSMD'})

    plotting_ssmd_scores_temp_df.insert(0, 'type', 'Raw')

    plotting_ssmd_scores_df = plotting_ssmd_scores_df.append(plotting_ssmd_scores_temp_df)

    plotting_ssmd_scores_df.loc[(plotting_ssmd_scores_df['layout'] == "RANDOM"), 'layout'] = "Random"
    plotting_ssmd_scores_df.loc[(plotting_ssmd_scores_df['layout'] == "BORDER"), 'layout'] = "Border"
    plotting_ssmd_scores_df.loc[(plotting_ssmd_scores_df['layout'] == "PLAID"), 'layout'] = "Effective"

    sns.set_style("whitegrid", {'axes.grid' : False})

    #palette = sns.color_palette("BuPu",3)
    palette = sns.color_palette("BuPu",4)

    fig, ax = plt.subplots(figsize=(4, 3))

    if y_min:
        ax.set_ylim(bottom = y_min)
    if y_max:
        ax.set_ylim(top = y_max)
        
    ax = sns.barplot(x='type', y="SSMD", data=plotting_ssmd_scores_df[plotting_ssmd_scores_df.lost_rows<1], palette=palette,hue='layout', order = ['Raw','Normalised'])#,showfliers = False)
    ax.set(xlabel='', ylabel='SSMD')
    plt.legend(ncol=3, loc="lower center", fontsize = 8)
    plt.ylabel("Mean SSMD", fontsize = 10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    #plt.yticks([i for i in range(1,8)])
    
    pvalue_thresholds = [[1e-4, "***"], [1e-2, "**"], [0.05, "*"],[1, "ns"]]
    box_pairs = [(("Raw","Effective"),("Raw","Random")),(("Normalised","Effective"),("Normalised","Random")),(("Normalised","Random"),("Normalised","Border")),(("Normalised","Effective"),("Normalised","Border")),(("Raw","Random"),("Raw","Border")),(("Raw","Effective"),("Raw","Border"))]

    annotator = Annotator(ax, pairs=box_pairs, data=plotting_ssmd_scores_df[plotting_ssmd_scores_df.lost_rows<1], x='type', y="SSMD",hue='layout', order=['Raw','Normalised'],hue_order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', pvalue_thresholds=pvalue_thresholds,text_offset=-1)
    annotator.apply_and_annotate()

    plt.show()
    fig.savefig("screening-ssmd-"+fig_name+".png",bbox_inches='tight',dpi=800)


    
    
def plotting_z_scores(screening_scores_data_filename, fig_name, y_min=None, y_max=None):
    screening_scores_df = pd.read_csv(screening_scores_data_filename)

    ## No rows lost!
    screening_scores_df = screening_scores_df[screening_scores_df['lost_rows']<1]

    screening_scores_df['Zfactor_expected'] = pd.to_numeric(screening_scores_df['Zfactor_expected'], errors='coerce')
    screening_scores_df['Zfactor_norm'] = pd.to_numeric(screening_scores_df['Zfactor_norm'], errors='coerce')
    screening_scores_df['Zfactor_raw'] = pd.to_numeric(screening_scores_df['Zfactor_raw'], errors='coerce')

    screening_scores_df['SSMD_expected'] = pd.to_numeric(screening_scores_df['SSMD_expected'], errors='coerce')
    screening_scores_df['SSMD_norm'] = pd.to_numeric(screening_scores_df['SSMD_norm'], errors='coerce')
    screening_scores_df['SSMD_raw'] = pd.to_numeric(screening_scores_df['SSMD_raw'], errors='coerce')

    ### This is only needed when the specific layout name is stored in 'layout' ###
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_rand"), 'layout'] = "RANDOM"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_border") & (screening_scores_df['layout'] != "RANDOM"), 'layout'] = "BORDER"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout") & (screening_scores_df['layout'] != "RANDOM") & (screening_scores_df['layout'] != "BORDER"), 'layout'] = "PLAID"

    ## Plotting
    plotting_z_scores_df = screening_scores_df[['layout','Zfactor_norm']]
    plotting_z_scores_df = plotting_z_scores_df.rename(columns={'Zfactor_norm': 'Zfactor'})

    plotting_z_scores_df.insert(0, 'type', 'Normalised')

    plotting_z_scores_temp_df = screening_scores_df[['layout','Zfactor_raw']]
    plotting_z_scores_temp_df = plotting_z_scores_temp_df.rename(columns={'Zfactor_raw': 'Zfactor'})

    plotting_z_scores_temp_df.insert(0, 'type', 'Raw')

    plotting_z_scores_df = plotting_z_scores_df.append(plotting_z_scores_temp_df)

    plotting_z_scores_df.loc[(plotting_z_scores_df['layout'] == "RANDOM"), 'layout'] = "Random"
    plotting_z_scores_df.loc[(plotting_z_scores_df['layout'] == "BORDER"), 'layout'] = "Border"
    plotting_z_scores_df.loc[(plotting_z_scores_df['layout'] == "PLAID"), 'layout'] = "Effective"

    sns.set_style("whitegrid", {'axes.grid' : False})

    palette = sns.color_palette("Greens",5)

    fig, ax = plt.subplots(figsize=(4, 3))
    
    if y_min:
        ax.set_ylim(bottom = y_min)
    if y_max:
        ax.set_ylim(top = y_max)

    ax = sns.barplot(x='type', y="Zfactor", data=plotting_z_scores_df, palette=palette,hue='layout', order = ['Raw','Normalised'])
    ax.set(xlabel="", ylabel="Z' factor")
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(ncol=3, loc="upper center", fontsize = 8)
    plt.ylabel("Mean Z' factor", fontsize = 10)

    pvalue_thresholds = [[1e-4, "***"], [1e-2, "**"], [0.05, "*"],[1, "ns"]]
    box_pairs = [(("Raw","Effective"),("Raw","Random")),(("Normalised","Effective"),("Normalised","Random")),(("Normalised","Random"),("Normalised","Border")),(("Normalised","Effective"),("Normalised","Border")),(("Raw","Random"),("Raw","Border")),(("Raw","Effective"),("Raw","Border"))]

    annotator = Annotator(ax, pairs=box_pairs, data=plotting_z_scores_df, x='type', y="Zfactor",hue='layout', order=['Raw','Normalised'],hue_order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', pvalue_thresholds=pvalue_thresholds,text_offset=-1)
    annotator.apply_and_annotate()

    plt.show()
    fig.savefig("screening-zpfactor-"+fig_name+".png",bbox_inches='tight',dpi=800)
    
    
def plotting_ssmd_scores_norm(screening_scores_data_filename, fig_name, y_min=None, y_max=None):
    screening_scores_df = pd.read_csv(screening_scores_data_filename)

    ## No rows lost!
    screening_scores_df = screening_scores_df[screening_scores_df['lost_rows']<1]

    screening_scores_df['SSMD_expected'] = pd.to_numeric(screening_scores_df['SSMD_expected'], errors='coerce')
    screening_scores_df['SSMD_norm'] = pd.to_numeric(screening_scores_df['SSMD_norm'], errors='coerce')
    screening_scores_df['SSMD_raw'] = pd.to_numeric(screening_scores_df['SSMD_raw'], errors='coerce')

    screening_scores_df['SSMD_abs'] = np.abs(screening_scores_df['SSMD_raw'])
    screening_scores_df['SSMD_norm_abs'] = np.abs(screening_scores_df['SSMD_norm'])
    
    ### This is only needed when the specific layout name is stored in 'layout' ###
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_rand"), 'layout'] = "RANDOM"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_border") & (screening_scores_df['layout'] != "RANDOM"), 'layout'] = "BORDER"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout") & (screening_scores_df['layout'] != "RANDOM") & (screening_scores_df['layout'] != "BORDER"), 'layout'] = "PLAID"

    screening_scores_df.loc[(screening_scores_df['layout'] == "RANDOM"), 'layout'] = "Random"
    screening_scores_df.loc[(screening_scores_df['layout'] == "BORDER"), 'layout'] = "Border"
    screening_scores_df.loc[(screening_scores_df['layout'] == "PLAID"), 'layout'] = "Effective"

    sns.set_style("whitegrid", {'axes.grid' : True})

    palette = sns.color_palette("BuPu",4)

    fig, ax = plt.subplots(figsize=(3, 3))

    if y_min:
        ax.set_ylim(bottom = y_min)
    if y_max:
        ax.set_ylim(top = y_max)

        
    ax = sns.barplot(x='layout', y="SSMD_norm_abs", data=screening_scores_df, palette=palette)#,showfliers = False)
    ax.set(xlabel='', ylabel='SSMD')
    #plt.legend(ncol=3, loc="lower center", fontsize = 8)
    plt.ylabel("Mean SSMD", fontsize = 10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    #plt.yticks([i for i in range(1,8)])
    
    pvalue_thresholds = [[1e-4, "***"], [1e-2, "**"], [0.05, "*"],[1, "ns"]]
    box_pairs = [("Effective", "Random"), ("Random", "Border"), ("Effective", "Border")]

    annotator = Annotator(ax, pairs=box_pairs, data=screening_scores_df, x='layout', y="SSMD_norm_abs", order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', pvalue_thresholds=pvalue_thresholds,text_offset=-1)
    annotator.apply_and_annotate()

    plt.show()
    fig.savefig("screening-ssmd-"+fig_name+".png",bbox_inches='tight',dpi=800)


    
def plotting_z_scores_norm(screening_scores_data_filename, fig_name, y_min=None, y_max=None):
    screening_scores_df = pd.read_csv(screening_scores_data_filename)

    ## No rows lost!
    screening_scores_df = screening_scores_df[screening_scores_df['lost_rows']<1]

    screening_scores_df['Zfactor_expected'] = pd.to_numeric(screening_scores_df['Zfactor_expected'], errors='coerce')
    screening_scores_df['Zfactor_norm'] = pd.to_numeric(screening_scores_df['Zfactor_norm'], errors='coerce')
    screening_scores_df['Zfactor_raw'] = pd.to_numeric(screening_scores_df['Zfactor_raw'], errors='coerce')

    ### This is only needed when the specific layout name is stored in 'layout' ###
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_rand"), 'layout'] = "RANDOM"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout_border") & (screening_scores_df['layout'] != "RANDOM"), 'layout'] = "BORDER"
    screening_scores_df.loc[(screening_scores_df['layout'] >= "plate_layout") & (screening_scores_df['layout'] != "RANDOM") & (screening_scores_df['layout'] != "BORDER"), 'layout'] = "PLAID"

    screening_scores_df.loc[(screening_scores_df['layout'] == "RANDOM"), 'layout'] = "Random"
    screening_scores_df.loc[(screening_scores_df['layout'] == "BORDER"), 'layout'] = "Border"
    screening_scores_df.loc[(screening_scores_df['layout'] == "PLAID"), 'layout'] = "Effective"

    
    sns.set_style("whitegrid", {'axes.grid' : True})

    palette = sns.color_palette("Greens",5)

    fig, ax = plt.subplots(figsize=(3, 3))
    
    if y_min:
        ax.set_ylim(bottom = y_min)
    if y_max:
        ax.set_ylim(top = y_max)
    else:
        ax.set_ylim(top = 1)


    ax = sns.barplot(x='layout', y="Zfactor_norm", data=screening_scores_df, palette=palette)
    ax.set(xlabel="", ylabel="Z' factor")
    plt.tick_params(axis='both', which='major', labelsize=10)
    #plt.legend(ncol=3, loc="upper center", fontsize = 8)
    plt.ylabel("Mean Z' factor", fontsize = 10)

    pvalue_thresholds = [[1e-4, "***"], [1e-2, "**"], [0.05, "*"],[1, "ns"]]
    box_pairs = [("Effective", "Random"), ("Random", "Border"), ("Effective", "Border")]

    annotator = Annotator(ax, pairs=box_pairs, data=screening_scores_df, x='layout', y="Zfactor_norm", order=["Effective","Random","Border"])
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', pvalue_thresholds=pvalue_thresholds,text_offset=-1)
    annotator.apply_and_annotate()

    plt.show()
    fig.savefig("screening-zpfactor-"+fig_name+".png",bbox_inches='tight',dpi=800)
    
    
    
## Functions used to create the screening/control layouts for the quality assessment metrics experiments

def full_controls_layout(layout, activity_layout, neg_control_id, pos_control_id):
    extended_controls_layout = np.copy(layout)
    num_rows, num_columns = layout.shape
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (layout[row_index][col_index] > 0 and activity_layout[row_index][col_index] == 0):
                extended_controls_layout[row_index][col_index] = neg_control_id
                            
            elif (layout[row_index][col_index] > 0 and activity_layout[row_index][col_index] == 1):
                 extended_controls_layout[row_index][col_index] = pos_control_id
                    
    return extended_controls_layout


def plate_to_random_layout(layout, activity_layout, num_neg_controls, num_pos_controls, neg_control_id, pos_control_id):
    num_rows, num_columns = layout.shape
    random_layout = np.full((num_rows, num_columns), 0)
    
    while num_neg_controls>0:
        rand_row = randrange(num_rows)
        rand_col = randrange(num_columns)
        
        if (layout[rand_row][rand_col] > 0) and (random_layout[rand_row][rand_col] == 0) and (activity_layout[rand_row][rand_col] == 0):
            random_layout[rand_row][rand_col] = neg_control_id
            num_neg_controls = num_neg_controls - 1

    while num_pos_controls>0:
        rand_row = randrange(num_rows)
        rand_col = randrange(num_columns)
        
        if (layout[rand_row][rand_col] > 0) and (random_layout[rand_row][rand_col] == 0) and (activity_layout[rand_row][rand_col] == 1):
            random_layout[rand_row][rand_col] = pos_control_id
            num_pos_controls = num_pos_controls - 1
            
    
    return random_layout


def plate_to_border_layout(layout, activity_layout, num_neg_controls, num_pos_controls, neg_control_id, pos_control_id):
    num_rows, num_columns = layout.shape
    border_layout = np.full((num_rows, num_columns), 0)

    for col_i in range(num_columns):
        for row_i in range(num_rows):
            for j in [col_i,num_columns-col_i-1]:
                if (layout[row_i][j] > 0) and (border_layout[row_i][j] == 0):
                    if num_neg_controls > 0 and (activity_layout[row_i][j] == 0):
                        border_layout[row_i][j] = neg_control_id
                        num_neg_controls = num_neg_controls - 1

                    elif num_pos_controls > 0 and (activity_layout[row_i][j] == 1):
                        border_layout[row_i][j] = pos_control_id
                        num_pos_controls = num_pos_controls - 1
        
        if num_neg_controls == 0 and num_pos_controls == 0: break
    
    return border_layout



def plotting_residual_metrics(screening_scores_data_filename, metric='Zfactor', fig_name=None, y_min=None, y_max=None, palette=None, plots_directory = '',box_pairs=box_pairs_bre, order=order_bre ):
    print(screening_scores_data_filename)
    screening_scores_df = pd.read_csv(screening_scores_data_filename)

    screening_scores_df[metric+'_expected'] = pd.to_numeric(screening_scores_df[metric+'_expected'], errors='coerce')
    screening_scores_df[metric+'_plaid'] = pd.to_numeric(screening_scores_df[metric+'_plaid'], errors='coerce')
    screening_scores_df[metric+'_rand'] = pd.to_numeric(screening_scores_df[metric+'_rand'], errors='coerce')
    screening_scores_df[metric+'_border'] = pd.to_numeric(screening_scores_df[metric+'_border'], errors='coerce')
    
    results_df = pd.DataFrame(np.square(screening_scores_df[metric+'_expected'] - screening_scores_df[metric+'_plaid']), columns = ['MSE'])
    rand_df = pd.DataFrame(np.square(screening_scores_df[metric+'_expected'] - screening_scores_df[metric+'_rand']), columns = ['MSE'])
    border_df = pd.DataFrame(np.square(screening_scores_df[metric+'_expected'] - screening_scores_df[metric+'_border']), columns = ['MSE'])
    
    results_df.insert(0, 'layout', 'Effective')
    rand_df.insert(0, 'layout', 'Random')
    border_df.insert(0, 'layout', 'Border')

    results_df = results_df.append(rand_df)
    results_df = results_df.append(border_df)
    
    sns.set_style("whitegrid", {'axes.grid' : True})

    if palette is None:
        palette = sns.color_palette("Greens",5)

    fig, ax = plt.subplots(figsize=(4,3))
    
    if y_min:
        ax.set_ylim(bottom = y_min)
    if y_max:
        ax.set_ylim(top = y_max)


    ax = sns.barplot(x='layout', y="MSE", data=results_df, palette=palette, order=order)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel("MSE", fontsize = 10)

    #pvalue_thresholds = [[1e-4, "***"], [1e-2, "**"], [0.05, "*"],[1, "ns"]]
    

    annotator = Annotator(ax, pairs=box_pairs, data=results_df, x='layout', y="MSE", order=order)
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', text_offset=-1)
    annotator.apply_and_annotate()

    plt.show()
    
    if fig_name:
        fig.savefig(plots_directory+"screening-"+metric+"-mse-"+fig_name+".png",bbox_inches='tight',dpi=800)

