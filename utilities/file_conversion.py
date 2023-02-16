import pandas as pd
import numpy as np
import csv
import re
from datetime import datetime

def plaid_to_echo(plaid_filename, source_plate_filename,total_volume, echo_filename,cmpdname_src_column='Compound', backfill_compound='DMSO'):

    # Open PLAID file
    plaid_df = pd.read_csv(plaid_filename)
    
    # Open source plate file
    source_df = pd.read_csv(source_plate_filename)

    # Clean units
    plaid_df[['CONCuM','unit']] = plaid_df['CONCuM'].str.split(' ',expand=True)
    plaid_df['CONCuM'] = pd.to_numeric(plaid_df['CONCuM'], errors='coerce')
    
    source_df[['CONCuM','unit']] = source_df['CONCuM'].str.split(' ',expand=True)
    source_df['CONCuM'] = pd.to_numeric(source_df['CONCuM'], errors='coerce')
    
    echo_df = plaid_df.merge(source_df, left_on='cmpdname', right_on=cmpdname_src_column, suffixes=('_plaid','_source'))
        
    # Calculate transfer volume
    echo_df['Transfer Volume'] = echo_df['CONCuM_plaid']*total_volume/echo_df['CONCuM_source']
    echo_df['Backfill Volume'] = total_volume - echo_df['Transfer Volume']
    echo_df['Backfill Compound'] = backfill_compound
    
    # Open/create ECHO file
    echo_output_f=open(echo_filename,'a')

    # Write headers
    echo_writer = csv.writer(echo_output_f)
    echo_writer.writerow(['Compound', 'Source plate', 'Source well', 'Destination plate', 'Destination well', 'Transfer Volume', 'Source Plate Type'])

    # Write compounds
    echo_df.to_csv(echo_output_f,index=False, header=False, columns=['cmpdname','plateID_source','well_source','plateID_plaid','well_plaid','Transfer Volume','type'],mode='a')
    
    # Write DMSO Backfill
    echo_df.to_csv(echo_output_f,index=False, header=False, columns=['Backfill Compound','plateID_source','well_source','plateID_plaid','well_plaid','Backfill Volume','type'],mode='a')
    
    ## Close file before the end
    echo_output_f.close()


    

def plaid_to_idot(plaid_filename, source_plate_filename, total_volume, dmso_max_perc, idot_filename, cmpdname_src_column='Compound', backfill_compound='DMSO', protocol_name = "My_First_Experiment", software = "1.7.2021.1019", user_name = "pharmb_io", sourceplate_type = "S.100 Plate", sourceplate_name = "source_name", max_volume = 8e-5, target_plate_type = "MWP 384", targetplate_name = "target_name", waste_well = "Waste Tube", dispense_to_waste = True, dispense_to_waste_cycles = 3, dispense_to_waste_volume = 1e-7, use_deionisation = True, optimization_level = "ReorderAndParallel", waste_error_handling_level = "Ask", save_liquids = "Ask"):
    
    # Open PLAID file
    plaid_df = pd.read_csv(plaid_filename)
    
    # Open source plate file
    source_df = pd.read_csv(source_plate_filename)

    # Clean units
    plaid_df[['CONCuM','unit']] = plaid_df['CONCuM'].str.split(' ',expand=True)
    plaid_df['CONCuM'] = pd.to_numeric(plaid_df['CONCuM'], errors='coerce')
    plaid_df['well'] = plaid_df['well'].map(strip_zeros)
    
    source_df[['CONCuM','unit']] = source_df['CONCuM'].str.split(' ',expand=True)
    source_df['CONCuM'] = pd.to_numeric(source_df['CONCuM'], errors='coerce')
    source_df['well'] = source_df['well'].map(strip_zeros)
    
    idot_df = plaid_df.merge(source_df, left_on='cmpdname', right_on=cmpdname_src_column, suffixes=('_plaid','_source'))

    # Calculate dmso volume per well in µL
    dmso_vol = (dmso_max_perc / 100) * (total_volume)

    # Calculate transfer volume
    idot_df['Transfer Volume'] = idot_df['CONCuM_plaid']*total_volume/idot_df['CONCuM_source']
    idot_df['Backfill Volume'] = dmso_vol - idot_df['Transfer Volume']
    idot_df['Backfill Compound'] = backfill_compound
    wells_backfill = source_df.loc[source_df[cmpdname_src_column] == backfill_compound, 'well']
    idot_df['well_backfill'] = np.resize(wells_backfill, len(idot_df))
    idot_df['X'] = ''
    idot_df['Y'] = ''
    idot_df['New Array'] = ''
    
    # Open/create iDOT file
    idot_output_f=open(idot_filename,'a')
    idot_writer = csv.writer(idot_output_f)
    
    # Write fist line
    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y")
    time_str = now.strftime("%H:%M")
    
    idot_writer.writerow([protocol_name, software, "<"+user_name+">", date_str, time_str,"","",""])
    
    # Write second line
    idot_writer.writerow([sourceplate_type, sourceplate_name, "",max_volume, target_plate_type, targetplate_name, "",waste_well])

    # Write parameters
    idot_writer.writerow(['DispenseToWaste='+str(dispense_to_waste), 'DispenseToWasteCycles='+ str(dispense_to_waste_cycles), 'DispenseToWasteVolume='+str(dispense_to_waste_volume), 'UseDeionisation='+ str(use_deionisation), 'OptimizationLevel='+optimization_level, 'WasteErrorHandlingLevel='+ waste_error_handling_level, 'SaveLiquids='+ save_liquids, ""])
    
    # Write headers
    idot_writer.writerow(['Source Well', 'Target Well', 'Volume [uL]', 'Liquid Name',"","","",""])

    # Write compounds
    idot_df.to_csv(idot_output_f,index=False, header=False, columns=['well_source','well_plaid','Transfer Volume','cmpdname','X','Y','New Array','New Array'],mode='a')
    
    # Write DMSO Backfill
    idot_df.to_csv(idot_output_f,index=False, header=False, columns=['well_backfill','well_plaid','Backfill Volume','Backfill Compound','X','Y','New Array','New Array'],mode='a')
    
    ## Close file before the end
    idot_output_f.close()
    


def strip_zeros(well):

    number = re.split('(\d+)', well)[1].lstrip('0')
    letter = re.split('(\d+)', well)[0]
    new_well = letter + number
    return new_well