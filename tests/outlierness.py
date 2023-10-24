import sklearn.impute
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values="NaN")
from sklearn.impute import KNNImputer
imputer = KNNImputer()

import pandas as pd
# pd.set_option('mode.use_inf_as_na', True)
alldf=pd.DataFrame()
import antspymm
mymods = antspymm.get_valid_modalities( )
for n in range(len(mymods)):
    m=mymods[n]
    jj=antspymm.collect_blind_qc_by_modality("vizx/*"+m+"*csv")
    print( m + " shape " + str(jj.shape[0]), flush=True )
    selit = []
    for k in range(jj.shape[0]):
        if jj['slice'].iloc[k] in [0  ,30,  60,  90, 120, 150, 180, 210]:
            selit.append(True)
        else:
            selit.append(False)
    jjj = jj[ selit ]
    print( m + " new shape " + str( jjj.shape[0] ), flush=True )
    jjj=antspymm.outlierness_by_modality( jjj, verbose=False)
    alldf = pd.concat( [alldf, jjj ], axis=0 )
    jjj.to_csv( "mystudy_mm_outlierness_"+m+"_x.csv")
    print(m+" done")
# write the joined data out
alldf.to_csv( "mystudy_mm_outlierness_x.csv", index=False )
# find the best mm collection

import pandas as pd
import antspymm
alldf=pd.read_csv( "mystudy_mm_outlierness_x.csv" )
matched_mm_data=antspymm.match_modalities( alldf, verbose=True )
matched_mm_data.to_csv( "matched_mm_data.csv", index=False )
matched_mm_data['negative_outlier_factor'] = 1.0 - matched_mm_data['ol_loop'].astype("float")
matched_mm_data2 = antspymm.highest_quality_repeat( matched_mm_data, 'subjectID', 'date', qualityvar='negative_outlier_factor')
matched_mm_data2.to_csv( "matched_mm_data2.csv", index=False )
