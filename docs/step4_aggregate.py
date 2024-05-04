import sys

# Check if at least one argument (apart from the script name) is provided
if len(sys.argv) > 1:
    # The first command line argument after the script name
    wpdir = sys.argv[1]
    print(f"The argument passed is: {wpdir}")
else:
    print("No argument was passed.")
    wpdir='processedCSVSRFIRST'
    # sys.exit(0)
import antspymm
import glob as glob
import re
import pandas as pd
import os
df = pd.read_csv( "matched_mm_data5.csv" )
df.columns.values[0] = 'filename'
# df = df.iloc[0:23, :]
print( df )
print( df.keys() )
pdir='./' + wpdir + '/'
outfn = "ppmi_matched_qc_mm_"+wpdir+"_v"+str( antspymm.version()['antspymm'] ) + ".csv"
print( outfn )
df['projectID']='PPMI'
df['perfid']="NA"
merged=antspymm.aggregate_antspymm_results_sdf( df, 
    subject_col='subjectID', date_col='date', image_col='imageID',  base_path=pdir, 
    splitsep='-', idsep='-', wild_card_modality_id=True, verbose=False )
print(merged.shape)
merged.to_csv( outfn )
