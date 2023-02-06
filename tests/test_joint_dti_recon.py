import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
from os.path import exists
from dipy.io.image import save_nifti, load_nifti
import antspymm
import antspyt1w
import antspynet
import ants
import pandas as pd
import tensorflow as tf
print(" Load in JHU atlas and labels ")
ex_path = os.path.expanduser( "~/.antspyt1w/" )
ex_path_mm = os.path.expanduser( "~/.antspymm/" )
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz' ) # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz' ) # Read in JHU labels
#### Load in data ####
print("Load in subject data ...")
lrid = ex_path_mm + "I1499279_Anon_20210819142214_5"
rlid = ex_path_mm + "I1499337_Anon_20210819142214_6"
t1id = ex_path_mm + "t1_rand.nii.gz"
# Load in image L-R
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'
# Load in image R-L
img_RL_in = ants.image_read( rlid + '.nii.gz' ) # RL dwi image
img_RL_bval = lrid + '.bval' # bval
img_RL_bvec = lrid + '.bvec'
t1wh = ants.iMath( ants.image_read( t1id ) , 'Normalize' )
t1w = t1wh * antspyt1w.brain_extraction( t1wh )
ants.plot( t1w , axis=2 )
bxtdwi = antspymm.t1_based_dwi_brain_extraction( t1wh, t1w, img_LR_in,
    transform='Rigid', deform=True, verbose=True )
derka
myoutx = antspymm.joint_dti_recon(
    img_LR_in,
    img_LR_bval,
    img_LR_bvec,
    jhu_atlas = JHU_atlas,
    jhu_labels = JHU_labels,
    srmodel = None,
    img_RL = img_RL_in,
    bval_RL = img_RL_bval,
    bvec_RL = img_RL_bvec,
    motion_correct = 'Rigid',
    dewarp_modality = 'FA',
    brain_mask = bxtdwi['b0_mask' ],
    t1w=t1w,
    verbose = True)
derka
######
myoutx = antspymm.joint_dti_recon(
    img_LR_in,
    img_LR_bval,
    img_LR_bvec,
    jhu_atlas = JHU_atlas,
    jhu_labels = JHU_labels,
    srmodel = None,
    img_RL = img_RL_in,
    bval_RL = img_RL_bval,
    bvec_RL = img_RL_bvec,
    motion_correct = 'Rigid',
    dewarp_modality = 'FA',
    verbose = True)

if True:
    ants.image_write( myoutx['dtrecon_LR']['FA'], '/tmp/temp1fa1.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR']['motion_corrected'], '/tmp/temp1moco.nii.gz'  )
    ants.image_write( myoutx['dtrecon_RL']['FA'], '/tmp/temp2fa1.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['FA'], '/tmp/temp1fa2.nii.gz'  )
    ants.image_write( myoutx['dtrecon_RL_dewarp']['FA'], '/tmp/temp2fa2.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['RGB'], '/tmp/temp1rgb.nii.gz'  )
    ants.image_write( myoutx['dtrecon_RL_dewarp']['RGB'], '/tmp/temp2rgb.nii.gz'  )
    ants.image_write( myoutx['recon_fa'], '/tmp/temp1fa.nii.gz'  )
    ants.image_write( myoutx['recon_md'], '/tmp/temp1md.nii.gz'  )
