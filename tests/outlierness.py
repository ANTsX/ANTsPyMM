import pandas as pd
alldf=pd.DataFrame()
import antspymm
mymods = antspymm.get_valid_modalities( )
for n in range(len(mymods)):
    m=mymods[n]
    jj=antspymm.collect_blind_qc_by_modality("viz/*"+m+"*csv")
    print( m + " shape " + str(jj.shape[0]) )
    jjj=antspymm.average_blind_qc_by_modality(jj,verbose=False) ## reduce the time series qc
    jjj=antspymm.outlierness_by_modality( jjj, verbose=False)
    alldf = pd.concat( [alldf, jjj ], axis=0 )
    jjj.to_csv( "ppmi_mm_outlierness_"+m+".csv")
    print(m+" done")
# write the joined data out
alldf.to_csv( "ppmi_mm_outlierness.csv", index=False )
# find the best mm collection
matched_mm_data=antspymm.match_modalities( alldf, verbose=True )
matched_mm_data.to_csv( "matched_mm_data.csv", index=False )
matched_mm_data['negative_outlier_factor'] = 1.0 - matched_mm_data['ol_loop'].astype("float")
matched_mm_data2 = antspymm.highest_quality_repeat( matched_mm_data, 'subjectID', 'date', qualityvar='negative_outlier_factor')
matched_mm_data2.to_csv( "matched_mm_data2.csv", index=False )

