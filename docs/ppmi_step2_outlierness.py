import sklearn.impute
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values="NaN")
from sklearn.impute import KNNImputer
imputer = KNNImputer()
from os.path import exists

import pandas as pd
alldf=pd.DataFrame()
import antspymm
mymods = antspymm.get_valid_modalities( )
for n in range(len(mymods)):
    m=mymods[n]
    print( m )
    if m != 'perf':
        if not exists( "ppmi_mm_outlierness_"+m+"_x2.csv" ):
            jj=antspymm.collect_blind_qc_by_modality("vizx_2024/*"+m+"*csv")
            print( m + " shape " + str(jj.shape[0]), flush=True )
            selit = []
            for k in range(jj.shape[0]):
                if jj['slice'].iloc[k] in [ 0, 30, 60, 90, 120 ]:
                    selit.append(True)
                else:
                    selit.append(False)
            jjj = jj[ selit ]
            print( m + " new shape " + str( jjj.shape[0] ), flush=True )
            jjj=antspymm.average_blind_qc_by_modality(jjj,verbose=False) ## reduce the time series qc
            print( m + " avg shape " + str( jjj.shape[0] ), flush=True )
            print("calculate outlierness "+m)
            jjj=antspymm.outlierness_by_modality( jjj, verbose=False)
            jjj.to_csv( "ppmi_mm_outlierness_"+m+"_x2.csv"  )
        else:
            jjj=pd.read_csv( "ppmi_mm_outlierness_"+m+"_x2.csv" )
        alldf = pd.concat( [alldf, jjj ], axis=0 )
        jjj.to_csv( "ppmi_mm_outlierness_"+m+"_x2.csv")
        print(m+" done")

# write the joined data out
if alldf.shape[0] > 0:
    alldf.to_csv( "ppmi_mm_outlierness_x2.csv", index=False )
# find the best mm collection
import pandas as pd
import antspymm
alldf=pd.read_csv( "ppmi_mm_outlierness_x2.csv" )
# Add a new column named 'failed' initialized to False
alldf['dti_failed'] = False
alldf['rsf_failed'] = False
# Extract unique 'fn' values where 'modality' is either 'DTIdwi' or 'DTIb0'
uids = alldf.loc[alldf['modality'].isin(['DTIdwi', 'DTIb0','DTI']), 'filename'].unique()
print(len(uids))
# Loop through each unique 'fn' value
for u in uids:
    # Select rows where 'fn' matches the current unique value
    selu = alldf['filename'] == u
    # Update the 'failed' column for the selected rows
    # Set to True if the number of unique 'slice' values is less than 2
    tester = (alldf.loc[selu, 'dimt'] < 8) | (alldf.loc[selu, 'dimensionality'] != 4) | alldf.loc[selu, 'dti_bvalueMax'].isna()
    alldf.loc[selu, 'dti_failed'] = tester


filtered_failed = alldf[alldf['modality'].isin(['DTIdwi', 'DTIb0','DTI'])]['dti_failed']
# Use value_counts() to get the count of each unique value in the 'failed' column
count_table = filtered_failed.value_counts()
print(count_table)
alldf=alldf[~alldf['dti_failed']]
# Extract unique 'fn' values where 'modality' is either 'DTIdwi' or 'DTIb0'
uids = alldf.loc[alldf['modality'].isin(['rsfMRI']), 'filename'].unique()
print(len(uids))
# Loop through each unique 'fn' value
for u in uids:
    # Select rows where 'fn' matches the current unique value
    selu = alldf['filename'] == u
    # Update the 'failed' column for the selected rows
    # Set to True if the number of unique 'slice' values is less than 2
    tester = (alldf.loc[selu, 'dimt'] < 8) | (alldf.loc[selu, 'dimensionality'] != 4)
    alldf.loc[selu, 'rsf_failed'] = tester


filtered_failed = alldf[alldf['modality'].isin(['rsfMRI'])]['rsf_failed']
count_table = filtered_failed.value_counts()
print(count_table)
alldf=alldf[~alldf['rsf_failed']]

if alldf.shape[0] > 0:
    alldf.to_csv( "ppmi_mm_outlierness_x2_filt.csv", index=False )

import antspymm
import pandas as pd
alldf=pd.read_csv("ppmi_mm_outlierness_x2_filt.csv")
matched_mm_data2=antspymm.mm_match_by_qc_scoring_all( alldf, verbose=True )
matched_mm_data2.to_csv( "matched_mm_data5.csv", index=False )
