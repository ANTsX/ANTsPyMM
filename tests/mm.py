# convert to pynb via p2j mm.py -o
# convert the ipynb to html via:
#   jupyter nbconvert ANTsPyMM/tests/mm.ipynb --execute --to html
#
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import re
import nibabel as nib
import numpy as np
import ants
import antspymm
import antspyt1w
import pandas as pd
import glob
import antspynet
from os.path import exists
ex_path = os.path.expanduser( "~/.antspyt1w/" )
templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
if not exists( templatefn ):
    print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
    stophere
template = ants.image_read( templatefn ) # Read in template

subjectrootpath = os.path.expanduser( "~/data/PPMI/MV/PPMI/nifti/40543/20210819/" )
identifier = '40543_20210819'
myod = os.path.expanduser(subjectrootpath + 'processed/') # FIXME -- better choice here
os.makedirs(myod,  exist_ok = True)
t1fn = os.path.expanduser( subjectrootpath + "3D_T1-weighted/14_22_13.0/40543-20210819-3D_T1-weighted-14_22_13.0_repeat_1.nii.gz")
ddir = os.path.expanduser( subjectrootpath + "DTI_LR/14_51_41.0/" )
pfx = ddir + "40543-20210819-DTI_LR-14_51_41.0_repeat_1"
dwi_fname = pfx + ".nii.gz"
bvec_fname = pfx + ".bvec"
bval_fname = pfx + ".bval"
rsfdir = os.path.expanduser( subjectrootpath + "rsfMRI_RL/14_22_13.0/" )
rsf_fname = rsfdir + "40543-20210819-rsfMRI_RL-14_22_13.0_repeat_1.nii.gz"
flair_fname = os.path.expanduser( subjectrootpath + "3D_T2_FLAIR/14_22_13.0/40543-20210819-3D_T2_FLAIR-14_22_13.0_repeat_1.nii.gz" )
t1 = ants.image_read( t1fn )
rsf = ants.image_read( rsf_fname )
nmfn = glob.glob( os.path.expanduser(subjectrootpath + "2D_GRE-MT/14_22_13.0/*nii.gz" ))
dwi = ants.image_read( dwi_fname )
flair = ants.image_read( flair_fname )
import tensorflow as tf
import tensorflow.keras as keras
# srmdl = tf.keras.models.load_model( '~/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0.h5', compile=False )
## >> the end of input functions << ##
#####################
#  T1 hierarchical  #
#####################
myop = myod + identifier
t1widefn = myop + "_t1mergewide.csv"
mmwidefn = myop + "_mmmergewide.csv"
print("begin: "  + myop )
hier = antspyt1w.read_hierarchical( myop )
t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
    hier['dataframes'], identifier=None )
t1imgbrn = hier['brain_n4_dnz']
t1atropos = hier['dkt_parc']['tissue_segmentation']
ants.plot( t1imgbrn,  axis=2, nslices=21, ncol=7, crop=True, title='brain extraction' )
ants.plot( t1imgbrn, t1atropos, axis=2, nslices=21, ncol=7, crop=True, title='segmentation'  )
ants.plot( t1imgbrn, hier['dkt_parc']['dkt_cortex'], axis=2, nslices=21, ncol=7, crop=True, title='cortex'   )

mynm = list()
for k in range( len( nmfn ) ):
    mynm.append( ants.image_read( nmfn[k] ) )

if not 'tabPro' in locals():
    tabPro, normPro = antspymm.mm( t1, hier, 
        nm_image_list = mynm,
        rsf_image = rsf,
        dw_image = dwi,
        bvals = bval_fname,
        bvecs = bvec_fname,
        flair_image = flair,
        srmodel=None,
        do_tractography=False, 
        do_kk=False, 
        do_normalization=True, 
        verbose=True )

antspymm.write_mm( output_prefix=myop, mm=tabPro, mm_norm=normPro, t1wide=t1wide, separator='_'  )

if tabPro['kk'] is not None:
    ants.plot( hier['brain_n4_dnz'], tabpro['kk']['thickness_image'], axis=2, nslices=21, ncol=7, crop=True, title='kk' )

################################## do the rsf .....
if tabPro['rsf'] is not None:
    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['DefaultMode'],
        axis=2, nslices=21, ncol=7, crop=True, title='DefaultMode' )
    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['FrontoparietalTaskControl'],
        axis=2, nslices=21, ncol=7, crop=True, title='FrontoparietalTaskControl' )

################################## do the nm .....
if tabPro['NM'] is not None:
    nmpro = tabPro['NM']
    mysl = range( nmpro['NM_avg'].shape[2] )
    ants.plot( nmpro['NM_avg'],  nmpro['t1_to_NM'], slices=mysl, axis=2, title='nm + t1' )
    mysl = range( nmpro['NM_avg_cropped'].shape[2] )
    ants.plot( nmpro['NM_avg_cropped'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop' )
    ants.plot( nmpro['NM_avg_cropped'], nmpro['t1_to_NM'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop + t1' )
    ants.plot( nmpro['NM_avg_cropped'], nmpro['NM_labels'], axis=2, slices=mysl, title='nm crop + labels' )
################################## do the dti .....
if tabPro['DTI'] is not None:
    mydti = tabPro['DTI']
    antspymm.write_bvals_bvecs( mydti['bval_LR'], mydti['bvec_LR'], myop + '_reoriented' )
    ants.plot( mydti['dtrecon_LR']['FA'],  axis=2, nslices=21, ncol=7, crop=True, title='FA pre correction' )
    ants.plot( mydti['recon_fa'],  axis=2, nslices=21, ncol=7, crop=True, title='FA (supposed to be better)' )
    ants.plot( mydti['recon_fa'], mydti['jhu_labels'], axis=2, nslices=21, ncol=7, crop=True, title='FA + JHU' )
    ants.plot( mydti['recon_md'],  axis=2, nslices=21, ncol=7, crop=True, title='MD' )

################################## do the flair .....
if tabPro['flair'] is not None:
    ants.plot( flair,   axis=2, nslices=21, ncol=7, crop=True, title='Flair' )
    ants.plot( flair, tabPro['flair']['WMH_probability_map'],  axis=2, nslices=21, ncol=7, crop=True, title='Flair + WMH' )

# now normalized visualization
for mykey in normPro.keys():
    if normPro[mykey] is not None:
        ants.plot( template, normPro[mykey], axis=2, nslices=21, ncol=7, crop=True, title=mykey  )
