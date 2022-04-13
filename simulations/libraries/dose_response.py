import numpy as np
import pandas as pd
import scipy.optimize as opt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score

import utilities as util
import disturbances as dt

# Function taken from https://gist.github.com/yannabraham/5f210fed773785d8b638
def ll4(x,b,c,d,e):
    '''Same as the LL.4 function from the R drc package with:
     - b: hill slope
     - c: min response
     - d: max response
     - e: EC50'''
    result = c + (d-c)/(1+np.exp(b*(np.log(x)-np.log(e))))
    
#    result = ( (d-c)/(1+ (x/e)**b  ) ) + c 
#    Y= c + (d-c)/(1+10^((LogAbsoluteIC50-X)*b + log((d-c)/(Fifty-c)-1))) 
    
    #with np.errstate(over='print'): ## raise for debugging
        #try:
            #result = c + (d-c)/(1+np.exp(b*(np.log(x)-np.log(e))))
            
            #if np.isnan(result).any():
            #    print("NaN in ll4 with input:","x:",x,"b:",b,"c:",c,"d:",d,"e:",e,"end")
            #    print("result:",result)
                
        #except FloatingPointError:
            #print('FloatingPointError: Very large exp in ll4! b:',b,"x:",x,"e:",e)
        
    return(result)


# Function taken from https://gist.github.com/yannabraham/5f210fed773785d8b638
def pDose(x):
    '''Helper function to easily compute log transformed concentrations used in drug discovery'''
    result = -np.log10(1e-5*x)
    
    #Used for debugging and validation
    #if np.isnan(result).any():
    #    print("NaN in pDose with param:",x)
    return(result)


def IC50(b,c,d,e):
#    with np.errstate(over='print'):
 #       try:
            #ic50 = np.exp(np.log(((d-c)/(50-c)) -1)*(e**b)/b)
            #ic50 =  np.exp(np.log((((d-c)/(50-c)) -1)*(e**b))/b)
            
            #if np.isnan(ic50).any():
            #    print("NaN in IC50 with params: b:",b.loc[np.isnan(ic50)],"c:",c.loc[np.isnan(ic50)],"d:",d.loc[np.isnan(ic50)],"e:",e.loc[np.isnan(ic50)])
                
        #except FloatingPointError:
        #    print('oh no! Very large exp in IC50! b:',b,"c:",c,"d:",d,"e:",e)
    ic50 =  np.exp(np.log((((d-c)/(50-c)) -1)*(e**b))/b)        
    return(ic50)


def fill_plate(layout, plate_content, neg_control_value = 100, num_compounds=36, concentrations=4, replicates=2, expected_noise = 0.25):
    num_rows, num_columns = layout.shape
    
    plate = np.full((num_rows, num_columns), 0.0)
    
    #plate_content = generate_plate_content(dose_response_params, replicates, expected_noise)
    
    neg_control = np.max(layout)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if layout[row_index][col_index] == neg_control:
                #plate[row_index][col_index] = (1 + expected_noise*(np.random.random()-0.5))*neg_control_value
                plate[row_index][col_index] = neg_control_value + expected_noise*(np.random.random()-0.5)
            elif 0 < layout[row_index][col_index]:
                #plate[row_index][col_index] = (1 + expected_noise*(np.random.random()-0.5))*plate_content.iloc[layout[row_index][col_index]-1].response
                plate[row_index][col_index] = plate_content.iloc[neg_control-layout[row_index][col_index]-1].response + abs(expected_noise*(np.random.random()-0.5))
                
    return(plate)


def generate_plate_content(dose_response_params, replicates):
    drData=[]

    for _ in range(replicates):
        for curve in dose_response_params:
            # generate base curve
            curData = pd.DataFrame(data={'compound':curve['compound'],
                                         'dose':curve['startDose']/np.power(curve['dilution'],range(curve['nDose']))})
            curData['logDose'] = pDose(curData.dose)
            curData['response'] = curData.dose.apply(lambda x: ll4(x,*[curve[i] for i in ['b','c','d','e']]))

            # generate replicates
            repData = []

            rep = curData
            
            repData.append(rep.copy())

            repData = pd.concat(repData)
            drData.append(repData)

    # assemble data
    drData = pd.concat(drData)

    
    return(drData)


def collect_plate_results(layout, plate):
    
    num_rows, num_columns = layout.shape
    
    neg_control = np.max(layout)
    
    results = np.full(neg_control-1, np.NaN)
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (0 < layout[row_index][col_index] & layout[row_index][col_index] < neg_control):
                results[neg_control-layout[row_index][col_index]-1] = plate[row_index][col_index]
                
    return(results)



def fit_data(result_data, response_column, result_column, df_params=None, layout_type="", neg_control_values = None):
    compoundData = result_data.groupby(['compound'])
    fitData = []
    
    #print(result_data)
    ## Values are aprox 0.5 (slope),  0,  100,  IC50
    ## #'b','c','d','e'
    #low_b = [0,min(min(result_data[result_column]),0),min(result_data[result_column]),np.min(result_data.dose)] 
    #low_b = [0,-np.inf,min(result_data[result_column]),0] 
    
    #up_b = [np.inf,np.inf,np.inf,np.inf]        
    #up_b = [np.inf,max(result_data[result_column]),np.inf,np.inf]
            
#    print("bounds\n")    
 #   print(low_b)
  #  print(up_b)

    for name, group_t in compoundData:
        group = group_t[np.logical_not(np.isnan(group_t[result_column]))]
        
        if len(group)>0:
            p0 = [0.5,min(group[result_column]),max(group[result_column]),np.median(group.dose)]
            #p0 = [0.5,0,100,np.median(group.dose)]
            #low_b = [-np.inf,0,-np.inf,0] #'b','c','d','e'
            #low_b = [-np.inf,min(group[result_column]+[0]),min(group[result_column]+[0]),0] #'b','c','d','e'
            #low_b = [0,-np.inf,min(group[result_column]),0] 
            
            ## Fixing the top (negative control) plateau
            if neg_control_values is not None:
                low_b = [0,-np.inf,0.95*np.mean(neg_control_values),0] 
            else:
                low_b = [0,-np.inf,min(group[result_column]),0] 
            
            ## Use this one when NOT fixing the top plateau
            low_b = [0,-np.inf,min(group[result_column]),0] 
            
            #low_b = [0,min(min(result_data[result_column]),0),min(group[result_column]),0] 
            
            #up_b = [np.inf,np.inf,np.inf,np.inf]
            up_b = [np.inf,max(group[result_column]),np.inf,np.inf]
            #up_b = [np.inf,max(group[result_column]),np.inf,np.inf]
            
            #maxfev
            #fitCoefs, covMatrix = opt.curve_fit(ll4, group.dose, group[result_column],p0,max_nfev=10000000,bounds=(low_b,up_b))
        
            #fitCoefs, covMatrix = opt.curve_fit(ll4, group.dose, group[result_column],p0,maxfev=10000000)
            
            
            
            try:
                if neg_control_values is None:
                    fitCoefs, covMatrix = opt.curve_fit(ll4, group.dose, group[result_column], p0,max_nfev=10000000,bounds=(low_b,up_b))
                    
                else:
                    neg_dose = np.min(group.dose)/100
                    ## FIX ME!
                    mean_neg_ctrl = [np.mean([neg_control_values[4*i],neg_control_values[4*i+1],neg_control_values[4*i+2],neg_control_values[4*i+3]]) for i in range(4)]
                    neg_dose_array = np.full_like(mean_neg_ctrl, neg_dose, dtype=np.double)
                    
                    fitCoefs, covMatrix = opt.curve_fit(ll4, np.concatenate([group.dose,neg_dose_array]), np.concatenate([group[result_column],mean_neg_ctrl]),p0,max_nfev=10000000,bounds=(low_b,up_b))
            
                resids = group[result_column]-group.dose.apply(lambda x: ll4(x,*fitCoefs))
        
                ## Ground-truth values minus experimental results
                true_resids = group[response_column] - group[result_column]
        
                r2s = r2_score(group[result_column], group.dose.apply(lambda x: ll4(x,*fitCoefs)))
            
                #curFit = dict(list(zip(['b','c','d','e'],fitCoefs))+[('residuals',resids**2),('true_residuals',true_resids**2)])
        
                curFit = dict(list(zip(['b','c','d','e'],fitCoefs))+[('r2_score',r2s)]+[('residuals',resids**2),('true_residuals',true_resids**2)])
            
            except:    
                print('too many iteration (probably)')
                curFit = dict([('b',np.NaN),('c',np.NaN),('d',np.NaN),('e',np.NaN),('r2_score',-np.inf)]+[('residuals',[np.inf]),('true_residuals',[np.inf])])

        else:
            curFit = dict([('b',np.NaN),('c',np.NaN),('d',np.NaN),('e',np.NaN),('r2_score',-np.inf)]+[('residuals',[np.inf]),('true_residuals',[np.inf])])

        
        
        curFit['compound']=name
        #curFit['residuals']=sum(resids**2)
        #curFit['true_residuals']=sum(true_resids**2)
        
        fitData.append(curFit)
        
        #### Code used for debugging
        ## Useful when the absolute IC50/EC50 generates an error (because it doesn't exist)
        #if curFit["c"]>50:
        #    print("Houston, we have a problem!",curFit)
        if df_params is not None and curFit['b'] is not np.NaN:
            true_curve = df_params.iloc[[name]]
            
            print("\n"+"Plotting curve for compound "+str(name))
            print("The score of this fit is: "+str(r2s)+"\n")
            print("The original IC50 is: "+str(true_curve['e'])+"\n")
            print("The estimated IC50 is: "+str(curFit['e'])+"\n")
            print("The absolute log error is: "+str(abs(np.log10(true_curve['e'])-np.log10(curFit['e'])))+"\n")
            
            print(group)
            
            refDose = np.linspace(min(result_data.logDose)*0.9,max(result_data.logDose)*1.1,256)
            refDose = (10**-refDose)*1e5
            
            sns.set_style("ticks")
            
            
            
            sns.lmplot(x='logDose',y='results',data=result_data[result_data['compound']==name],fit_reg=False,height=2.75,scatter_kws={'s':6},palette=['#4DBBD599'])
                        
            #plt.subplots(figsize=(3, 3))
            plt.plot([pDose(i) for i in refDose],[ll4(i,*[true_curve[i] for i in ['b','c','d','e']]) for i in refDose],color='#DC000099', label="Original", linewidth=2)
            plt.plot([pDose(i) for i in refDose],[ll4(i,*[curFit[i] for i in ['b','c','d','e']]) for i in refDose],color='#3C5488FF', label="Estimated", linewidth=2)
            plt.ylim(-5, 135)
            
            
            if neg_control_values is not None:
                
                sns.regplot(x=pDose(neg_dose_array), y=mean_neg_ctrl, scatter=True, fit_reg=False, color='orange', marker="*",scatter_kws={'s':6}, label="Controls") 

            
            plt.ylabel("Response (%)", fontsize = 10)
            plt.xlabel("Log(Concentration)", fontsize = 10)
            lgnd = plt.legend(loc='lower right', fontsize = 10)
            lgnd.legendHandles[2].set_sizes([30])
            
            #plt.savefig(layout_type+"_compound_"+str(name)+"-right-half.png",bbox_inches='tight',dpi=1200)
            plt.show()
        
    fitCompound = [ item['compound'] for item in fitData]
    
    fitTable = pd.DataFrame(fitData).set_index('compound')
    
    return fitTable



def fit_data_min_req(result_data, response_column, result_column):
    compoundData = result_data.groupby(['compound'])
    fitData = []

    for name, group_t in compoundData:
        group = group_t[np.logical_not(np.isnan(group_t[result_column]))]
        
        if len(group)>0:
            p0 = [0.5,min(group[result_column]),max(group[result_column]),np.median(group.dose)]
            low_b = [-np.inf,0,-np.inf,0] #'b','c','d','e'
            up_b = [np.inf,np.inf,np.inf,np.inf]
        
            fitCoefs, covMatrix = opt.curve_fit(ll4, group.dose, group[result_column],p0,maxfev=10000000,bounds=(low_b,up_b))
        
            resids = group[result_column]-group.dose.apply(lambda x: ll4(x,*fitCoefs))
        
            ## Ground-truth values minus experimental results
            true_resids = group[response_column] - group[result_column]
        
            curFit = dict(list(zip(['b','c','d','e'],fitCoefs))+[('residuals',resids**2),('true_residuals',true_resids**2)])
        
        else:
            curFit = dict([('b',np.NaN),('c',np.NaN),('d',np.NaN),('e',np.NaN)]+[('residuals',[np.inf]),('true_residuals',[np.inf])])
        
        
        curFit['compound']=name
        #curFit['residuals']=sum(resids**2)
        #curFit['true_residuals']=sum(true_resids**2)
        
        fitData.append(curFit)
    
        
    fitCompound = [ item['compound'] for item in fitData]
    
    fitTable = pd.DataFrame(fitData).set_index('compound')
    
    return fitTable




def plate_curves_after_error(layout_dir,layout_file,plate_content,expected_noise,error_function,error,normalization_function,min_dist,lose_from_row=0,lose_to_row=0, df_params=None):

    plate_content, neg_control_values = __run_experiment(layout_dir,layout_file,plate_content,expected_noise,error_function,error,normalization_function,min_dist,lose_from_row,lose_to_row)
    
    
    fitTable_new = fit_data(plate_content, 'response', 'results', neg_control_values = neg_control_values, df_params=df_params, layout_type=layout_file)
    
    return(fitTable_new)


def __run_experiment(layout_dir,layout_file,plate_content,expected_noise,error_function,error,normalization_function,min_dist,lose_from_row,lose_to_row):

    layout = np.load(layout_dir+layout_file)    
    neg_control_id = np.max(layout)
    
    # Fill plate
    plate = fill_plate(layout, plate_content, neg_control_value = 100, expected_noise = expected_noise)
    
    # Add errors
    plate = error_function(plate, error)
    
    plate = dt.lose_rows(plate, lose_from_row, lose_to_row)
    
    # Fix errors
    #control_locations = util.get_controls_layout(layout.astype(np.float32))
    #control_locations = dt.lose_rows(control_locations, lose_from_row, lose_to_row)
    layout = dt.lose_rows(layout, lose_from_row, lose_to_row)
    
    plate = normalization_function(plate,layout,neg_control_id, min_dist=min_dist)
    
    # Collect negative controls
    mean_neg_ctrl = mean_controls(plate,layout,neg_control_id)
    
    # Collect results
    results = collect_plate_results(layout, plate)
    plate_content['results'] = results

    return plate_content, mean_neg_ctrl


def plate_min_curves_after_error(layout_dir,layout_file,plate_content,expected_noise,error_function,error,normalization_function,min_dist,lose_from_row=0,lose_to_row=0):
    
    plate_content = __run_experiment(layout_dir,layout_file,plate_content,expected_noise,error_function,error,normalization_function,min_dist,lose_from_row,lose_to_row)
    
    fitTable_new = fit_data_min_req(plate_content, 'response', 'results')
    
    return(fitTable_new)


def mean_controls(plate_array,layout,control_id):
    
    control_locations = util.get_controls_layout(layout,neg_control=control_id)

    if control_locations.sum() < 1:
        return np.NaN
    
    num_rows, num_columns = plate_array.shape
    
    plate_df = pd.DataFrame(plate_array)
    intensity_df = plate_df.stack().reset_index()
    intensity_df.columns = ["Rows","Columns","Intensity"]

    controls_df = pd.DataFrame(layout).stack().reset_index()
    controls_df.columns = ["Rows","Columns","Type"]
    
    data = pd.merge(intensity_df, controls_df,  how='left', on=['Rows','Columns'])

    data.reset_index()
    
    z = data.loc[data['Type']==control_id,['Intensity']].to_numpy().reshape((-1,))
    
    return z
    #return np.mean(z)