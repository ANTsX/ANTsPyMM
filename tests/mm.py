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
ex_path_mm = os.path.expanduser( "~/.antspymm/" )
mycsv = pd.read_csv(  os.path.expanduser( ex_path + "FA_JHU_labels_edited.csv" ) )
citcsv = pd.read_csv(  os.path.expanduser(  ex_path + "CIT168_Reinf_Learn_v1_label_descriptions_pad.csv" ) )
dktcsv = pd.read_csv(  os.path.expanduser( ex_path + "dkt.csv" ) )
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz') # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz') # Read in JHU labels
t1fn = os.path.expanduser( "~/data/PPMI/MV/PPMI/nifti/40543/20210819/3D_T1-weighted/14_22_13.0/40543-20210819-3D_T1-weighted-14_22_13.0_repeat_1.nii.gz")
ddir = os.path.expanduser( "~/data/PPMI/MV/PPMI/nifti/40543/20210819/DTI_LR/14_51_41.0/" )
pfx = ddir + "40543-20210819-DTI_LR-14_51_41.0_repeat_1"
dwi_fname = pfx + ".nii.gz"
bvec_fname = pfx + ".bvec"
bval_fname = pfx + ".bval"
rsfdir = os.path.expanduser( "~/data/PPMI/MV/PPMI/nifti/40543/20210819/rsfMRI_RL/14_22_13.0/" )
rsf_fname = rsfdir + "40543-20210819-rsfMRI_RL-14_22_13.0_repeat_1.nii.gz"
flair_fname = os.path.expanduser( "~/data/PPMI/MV/PPMI/nifti/40543/20210819/3D_T2_FLAIR/14_22_13.0/40543-20210819-3D_T2_FLAIR-14_22_13.0_repeat_1.nii.gz" )
t1 = ants.image_read( t1fn )
dwi = ants.image_read( dwi_fname )
rsf = ants.image_read( rsf_fname )
flair = ants.image_read( flair_fname )
nmfn = glob.glob( os.path.expanduser("~/data/PPMI/MV/PPMI/nifti/40543/20210819/2D_GRE-MT/14_22_13.0/*nii.gz" ))
fnt=os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni.nii.gz" )
fntbst=os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni_brainstem.nii.gz")
fnslab=os.path.expanduser("~/.antspyt1w/CIT168_MT_Slab_adni.nii.gz")
fntseg=os.path.expanduser("~/.antspyt1w/det_atlas_25_pad_LR_adni.nii.gz")
#####################
#  T1 hierarchical  #
#####################
myod = os.path.expanduser('~/data/PPMI/MV/processed/40543/20210819/')
os.makedirs(myod,  exist_ok = True)
identifier = '40543_20210819'
myop = myod + identifier
t1widefn = myop + "_mergewide.csv"
print("begin: "  + myop )
if not exists( t1widefn ):
    hier = antspyt1w.hierarchical( t1, output_prefix=myop )
    antspyt1w.write_hierarchical(  hier , myop )
else:
    hier = antspyt1w.read_hierarchical( myop )
t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
    hier['dataframes'], identifier=identifier )
t1wide.to_csv( t1widefn )
t1imgbrn = hier['brain_n4_dnz']
t1atropos = hier['dkt_parc']['tissue_segmentation']
ants.plot( t1imgbrn,  axis=2, nslices=21, ncol=7, crop=True, title='brain extraction' )
ants.plot( t1imgbrn, t1atropos, axis=2, nslices=21, ncol=7, crop=True, title='segmentation'  )
ants.plot( t1imgbrn, hier['dkt_parc']['dkt_cortex'], axis=2, nslices=21, ncol=7, crop=True, title='cortex'   )
################################## do the rsf .....
rsfpro = antspymm.resting_state_fmri_networks( rsf, hier['brain_n4_dnz'], t1atropos,
    f=[0.03,0.08],   spa = 1.5, spt = 0.5, nc = 6 )
ants.plot( rsfpro['meanBold'], rsfpro['DefaultMode'],
    title='DefaultMode',
    axis=2, nslices=21, ncol=7, crop=True )
ants.plot( rsfpro['meanBold'], rsfpro['FrontoparietalTaskControl'],
    title='FrontoparietalTaskControl',
    axis=2, nslices=21, ncol=7, crop=True )
# dataframe output is called rsfpro['corr_wide']
################################## do the nm .....
import tensorflow as tf
import tensorflow.keras as keras
srmdl = tf.keras.models.load_model( '/Users/stnava/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0.h5', compile=False )
mynm = list()
for k in range( len( nmfn ) ):
    mynm.append( ants.image_read( nmfn[k] ) )
nmpro = antspymm.neuromelanin( mynm, t1imgbrn, t1, hier['deep_cit168lab'] )
nmprosr = antspymm.neuromelanin( mynm, t1imgbrn, t1, hier['deep_cit168lab'], srmodel=srmdl )
# this is for checking the processing
mysl = [8,10,12]
ants.plot( nmpro['NM_avg'],  nmpro['t1_to_NM'], slices=mysl, axis=2 )
ants.plot( nmpro['NM_cropped'], axis=2, slices=mysl, overlay_alpha=0.3 )
ants.plot( nmpro['NM_cropped'], nmpro['t1_to_NM'], axis=2, slices=mysl, overlay_alpha=0.3 )
ants.plot( nmpro['NM_cropped'], nmpro['NM_labels'], axis=2, slices=mysl )
################################## do the dti .....
dtibxt_data = antspymm.t1_based_dwi_brain_extraction( hier['brain_n4_dnz'], dwi, transform='Rigid' )
mydti = antspymm.joint_dti_recon(
        dwi,
        bval_fname,
        bvec_fname,
        jhu_atlas=JHU_atlas,
        jhu_labels=JHU_labels,
        t1w = hier['brain_n4_dnz'],
        brain_mask = dtibxt_data['b0_mask'],
        reference_image = dtibxt_data['b0_avg'],
        srmodel=None,
        motion_correct=True, # set to False if using input from qsiprep
        verbose = True)
ants.plot( mydti['recon_fa'],  axis=2, nslices=21, ncol=7, crop=True, title='FA' )
ants.plot( mydti['recon_fa'], mydti['jhu_labels'], axis=2, nslices=21, ncol=7, crop=True, title='FA + JHU' )
ants.plot( mydti['recon_md'],  axis=2, nslices=21, ncol=7, crop=True, title='MD' )
# summarize dwi with T1 outputs
# first - register ....
reg = ants.registration( mydti['recon_fa'], hier['brain_n4_dnz'], 'Rigid' )
##################################################
fat1summ = antspymm.hierarchical_modality_summary(
    mydti['recon_fa'],
    hier=hier,
    modality_name='fa',
    transformlist=reg['fwdtransforms'],
    verbose = True )
##################################################
mdt1summ = antspymm.hierarchical_modality_summary(
    mydti['recon_md'],
    hier=hier,
    modality_name='md',
    transformlist=reg['fwdtransforms'],
    verbose = True )
# these inputs should come from nicely processed data
dktmapped = ants.apply_transforms(
    mydti['recon_fa'],
    hier['dkt_parc']['dkt_cortex'],
    reg['fwdtransforms'], interpolator='nearestNeighbor' )
mask = ants.threshold_image( mydti['recon_fa'], 0.05, 2.0 ).iMath("GetLargestComponent")
mystr = antspymm.dwi_deterministic_tracking(
    mydti['dwi_LR_dewarped'],
    mydti['recon_fa'],
    mydti['bval_LR'],
    mydti['bvec_LR'],
    seed_density = 1,
    mask=mask,
    verbose=True )
##########################################
cnxmat = antspymm.dwi_streamline_connectivity( mystr['streamlines'], dktmapped, dktcsv, verbose=True )
# FIXME --- put all these output together in wide format
################################## do the flair .....
flairpro = antspymm.wmh( flair, t1, t1atropos, mmfromconvexhull=12 )
ants.plot( flair,   axis=2, nslices=21, ncol=7, crop=True, title='Flair' )
ants.plot( flair, flairpro['WMH_probability_map'],  axis=2, nslices=21, ncol=7, crop=True, title='Flair + WMH' )
# FIXME --- put all these output together in wide format
