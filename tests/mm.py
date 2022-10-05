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
mycsvfn = ex_path + "FA_JHU_labels_edited.csv"
citcsvfn = ex_path + "CIT168_Reinf_Learn_v1_label_descriptions_pad.csv"
dktcsvfn = ex_path + "dkt.csv"
JHU_atlasfn = ex_path + 'JHU-ICBM-FA-1mm.nii.gz' # Read in JHU atlas
JHU_labelsfn = ex_path + 'JHU-ICBM-labels-1mm.nii.gz' # Read in JHU labels
templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
if not exists( mycsvfn ) or not exists( citcsvfn ) or not exists( dktcsvfn ) or not exists( JHU_atlasfn ) or not exists( JHU_labelsfn ) or not exists( templatefn ):
    print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
    stophere
mycsv = pd.read_csv(  mycsvfn )
citcsv = pd.read_csv(  os.path.expanduser(  citcsvfn ) )
dktcsv = pd.read_csv(  os.path.expanduser( dktcsvfn ) )
JHU_atlas = ants.image_read( JHU_atlasfn ) # Read in JHU atlas
JHU_labels = ants.image_read( JHU_labelsfn ) # Read in JHU labels
template = ants.image_read( templatefn ) # Read in template
subjectrootpath = os.path.expanduser( "~/data/PPMI/MV/PPMI/nifti/40543/20210819/" )
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
## >> the end of input functions << ##
#####################
#  T1 hierarchical  #
#####################
myod = os.path.expanduser('~/data/PPMI/MV/processed/40543/20210819/')
os.makedirs(myod,  exist_ok = True)
identifier = '40543_20210819'
myop = myod + identifier
t1widefn = myop + "_t1mergewide.csv"
mmwidefn = myop + "_mmmergewide.csv"
print("begin: "  + myop )
if not exists( t1widefn ):
    hier = antspyt1w.hierarchical( t1, output_prefix=myop )
    antspyt1w.write_hierarchical(  hier , myop )
else:
    hier = antspyt1w.read_hierarchical( myop )
t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
    hier['dataframes'], identifier=None )
t1wide.to_csv( t1widefn )
t1imgbrn = hier['brain_n4_dnz']
t1atropos = hier['dkt_parc']['tissue_segmentation']
ants.plot( t1imgbrn,  axis=2, nslices=21, ncol=7, crop=True, title='brain extraction' )
ants.plot( t1imgbrn, t1atropos, axis=2, nslices=21, ncol=7, crop=True, title='segmentation'  )
ants.plot( t1imgbrn, hier['dkt_parc']['dkt_cortex'], axis=2, nslices=21, ncol=7, crop=True, title='cortex'   )
kkthk = antspyt1w.kelly_kapowski_thickness( hier['brain_n4_dnz'],
    labels=hier['dkt_parc']['dkt_cortex'], iterations=45 ) # FIXME 3=testing, >=45=real
ants.plot( hier['brain_n4_dnz'], kkthk['thickness_image'], axis=2, nslices=21, ncol=7, crop=True, title='kk' )
ants.image_write( kkthk['thickness_image'],  myop + '_kkthickness.nii.gz' )
################################## do the rsf .....
rsfpro = antspymm.resting_state_fmri_networks( rsf, hier['brain_n4_dnz'], t1atropos,
    f=[0.03,0.08],   spa = 1.5, spt = 0.5, nc = 6 )
ants.plot( rsfpro['meanBold'], rsfpro['DefaultMode'],
    axis=2, nslices=21, ncol=7, crop=True, title='DefaultMode' )
ants.plot( rsfpro['meanBold'], rsfpro['FrontoparietalTaskControl'],
    axis=2, nslices=21, ncol=7, crop=True, title='FrontoparietalTaskControl' )
# dataframe output is called rsfpro['corr_wide']
################################## do the nm .....
import tensorflow as tf
import tensorflow.keras as keras
# srmdl = tf.keras.models.load_model( '~/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0.h5', compile=False )
mynm = list()
for k in range( len( nmfn ) ):
    mynm.append( ants.image_read( nmfn[k] ) )
nmpro = antspymm.neuromelanin( mynm, t1imgbrn, t1, hier['deep_cit168lab'] )
# nmprosr = antspymm.neuromelanin( mynm, t1imgbrn, t1, hier['deep_cit168lab'], srmodel=srmdl )
# this is for checking the processing
mysl = [8,10,12]
ants.plot( nmpro['NM_avg'],  nmpro['t1_to_NM'], slices=mysl, axis=2, title='nm + t1' )
ants.plot( nmpro['NM_cropped'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop' )
ants.plot( nmpro['NM_cropped'], nmpro['t1_to_NM'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop + t1' )
ants.plot( nmpro['NM_cropped'], nmpro['NM_labels'], axis=2, slices=mysl, title='nm crop + labels' )
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
# write the bvals, bvecs, (large file) DWI, the DTI, the labels and the streamlines
antspymm.write_bvals_bvecs( mydti['bval_LR'], mydti['bvec_LR'], myop + '_reoriented' )
ants.image_write( mydti['dwi_LR_dewarped'],  myop + '_dwi.nii.gz' )
ants.image_write( mydti['dtrecon_LR_dewarp']['RGB'] ,  myop + '_DTIRGB.nii.gz' )
ants.image_write( mydti['jhu_labels'],  myop+'_dtijhulabels.nii.gz' )
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
from dipy.io.streamline import save_tractogram
save_tractogram(mystr['tractogram'], myop+'_dtitracts.trk')
##########################################
cnxmat = antspymm.dwi_streamline_connectivity( mystr['streamlines'], dktmapped, dktcsv, verbose=True )
# FIXME --- put all these output together in wide format
################################## do the flair .....
flairpro = antspymm.wmh( flair, t1, t1atropos, mmfromconvexhull=12 )
ants.plot( flair,   axis=2, nslices=21, ncol=7, crop=True, title='Flair' )
ants.plot( flair, flairpro['WMH_probability_map'],  axis=2, nslices=21, ncol=7, crop=True, title='Flair + WMH' )
################################################
# put all these output together in wide format #
# joint [ t1, rsf, dti, nm, flair ]
mm_wide = pd.concat( [
  t1wide.iloc[: , 1:],
  kkthk['thickness_dataframe'].iloc[: , 1:],
  nmpro['NM_dataframe_wide'].iloc[: , 1:],
  mydti['recon_fa_summary'].iloc[: , 1:],
  mydti['recon_md_summary'].iloc[: , 1:],
  fat1summ.iloc[: , 1:],
  mdt1summ.iloc[: , 1:],
  cnxmat['connectivity_wide'].iloc[: , 1:] # NOTE: connectivity_wide is not much tested
  ], axis=1 )
mm_wide = mm_wide.copy()
mm_wide['flair_wmh'] = flairpro['wmh_mass']
mm_wide['rsf_FD_mean'] = rsfpro['FD_mean']
mm_wide['rsf_FD_max'] = rsfpro['FD_max']
if mydti['dtrecon_LR']['framewise_displacement'] is not None:
    mm_wide['dti_FD_mean'] = mydti['dtrecon_LR']['framewise_displacement'].mean()
    mm_wide['dti_FD_max'] = mydti['dtrecon_LR']['framewise_displacement'].max()
else:
    mm_wide['dti_FD_mean'] = mm_wide['dti_FD_max'] = 'NA'
# mm_wide.shape
mm_wide.to_csv( mmwidefn )
# write out csvs
rsfpro['corr'].to_csv( myop+'_rsfcorr.csv' )
pd.DataFrame(cnxmat['connectivity_matrix']).to_csv( myop+'_dtistreamlinecorr.csv' )


#################################################################
### NOTES: deforming to a common space and writing out images ###
### images we want come from: DTI, NM, rsf, thickness ###########
#################################################################
if True:
    # might reconsider this template space - cropped and/or higher res?
    template = ants.resample_image( template, [1,1,1], use_voxels=False )
    t1reg = ants.registration( template, hier['brain_n4_dnz'], "antsRegistrationSyNQuickRepro[s]")
    thk2template = ants.apply_transforms( template, kkthk['thickness_image'], t1reg['fwdtransforms'])
    ants.image_write( thk2template, myop+'_thickness2template.nii.gz' )
    dtirig = ants.registration( hier['brain_n4_dnz'], mydti['recon_fa'], 'Rigid' )
    rsfrig = ants.registration( hier['brain_n4_dnz'], rsfpro['meanBold'], 'Rigid' )
    md2template = ants.apply_transforms( template, mydti['recon_md'],t1reg['fwdtransforms']+dtirig['fwdtransforms'] )
    ants.image_write( md2template, myop+'_md2template.nii.gz' )
    ants.plot(template, md2template, crop=True, axis=2, ncol=7, nslices=21, title='md 2 template' )
    fa2template = ants.apply_transforms( template, mydti['recon_fa'],t1reg['fwdtransforms']+dtirig['fwdtransforms'] )
    ants.image_write( fa2template, myop+'_fa2template.nii.gz' )
    ants.plot(template, fa2template, crop=True, axis=2, ncol=7, nslices=21, title='fa 2 template' )
    mynets = list([ 'CinguloopercularTaskControl', 'DefaultMode', 'MemoryRetrieval', 'VentralAttention', 'Visual', 'FrontoparietalTaskControl', 'Salience', 'Subcortical', 'DorsalAttention'])
    for netid in mynets:
        dfn2template = ants.apply_transforms( template, rsfpro[netid],t1reg['fwdtransforms']+rsfrig['fwdtransforms'] )
        ants.image_write( dfn2template, myop+'_'+netid+'2template.nii.gz' )
        ants.plot(template, dfn2template, crop=True, axis=2, ncol=7, nslices=21, title=netid + ' 2 template' )
    nmrig = nmpro['t1_to_NM_transform'] # this is an inverse tx
    nm2template = ants.apply_transforms( template, nmpro['NM_avg'],t1reg['fwdtransforms']+nmrig,
        whichtoinvert=[False,False,True])
    ants.image_write( nm2template, myop+'_nm2template.nii.gz' )
    ants.plot(template, nm2template, crop=True, axis=2, ncol=7, nslices=21, title='nm 2 template' )
