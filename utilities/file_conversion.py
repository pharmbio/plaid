import pandas as pd
import csv 

def plaid_to_echo_(plaid_filename,source_plate_filename,total_volume,echo_filename,cmpdname_src_column='Compound',backfill_compound='DMSO'):

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
    print("All done! :-)")
