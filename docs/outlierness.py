import subprocess
import antspymm
import pandas as pd
from tabulate import tabulate

ii=antspymm.collect_blind_qc_by_modality("viz/inspect*T1w*brain.csv")
jj=antspymm.collect_blind_qc_by_modality("viz/viz*T1w*csv")
jj=pd.concat( [jj,ii],axis=1)
jj = jj.loc[:,~jj.columns.duplicated()].copy()
goodsel = (jj['resnetGrade'] >= 1.5 )
##########
# generic outlierness wrt all data
jjj=antspymm.outlierness_by_modality( jj, verbose=False)
jjj.to_csv( "abb_outlierness.csv", index=False )
##########
basecols = [ 'noise', 'snr', 'cnr' ]
olcols = [ 'noise',  'cnr', 'psnr', 'ssim',  'mi', 
    'reflection_err',  'EVR', 'msk_vol'  ]
olcols2 = [ 'RandBasisProj01', 'RandBasisProj02', 'RandBasisProj03',
    'RandBasisProj04', 'RandBasisProj05', 'RandBasisProj06',
    'RandBasisProj07', 'RandBasisProj08', 'RandBasisProj09',
    'RandBasisProj10' ]
olcols = olcols + olcols2
# wrt good data
jjgood=jj[goodsel]
jjbad=jj[~goodsel]

for fn in jj['fn']:
    jjbadsel=jj[jj['fn']==fn]
    dfG = ( jjgood[olcols] )
    dfT = ( jjbadsel[olcols] )
    univar = antspymm.novelty_detection_quantile( dfG, dfT )
    temp0=antspymm.novelty_detection_lof( dfG, dfT, n_neighbors=8 )
    temp1=antspymm.novelty_detection_svm( dfG, dfT )
    temp2=antspymm.novelty_detection_ee( dfG, dfT )
    temp3=antspymm.novelty_detection_loop( dfG, dfT, n_neighbors=12 )
    if jjbadsel['resnetGrade'][0] < 1.5:
#        if ( ( temp0[0] + temp1[0] + temp2[0] + temp3[0] ) > 2 ) or ( temp0[0] ) > 0 or temp3[0] > 0.90:
#        if (univar > 0.99 ).to_numpy().any() or temp3[0] > 0.5:
        if temp3[0] > 0.5:
            print( fn )
            print(tabulate(univar, headers='keys', tablefmt='psql',showindex=False))
            #print( fn + ' lof: ' + str(temp0[0]) + ' svm: ' + str(temp1[0]) +' ee: ' + str(temp2[0])+ ' loop: ' + str(temp3[0]) )
            cmd_str = "open viz/viz_abb_"+fn+"*slice*png"
            # subprocess.run(cmd_str, shell=True)
            cmd_str = "open viz/inspect*"+fn+"*brain.png"
            subprocess.run(cmd_str, shell=True)
            wait = input("Press Enter to continue.")

