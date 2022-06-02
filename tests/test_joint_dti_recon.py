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
ex_path = "~/.antspyt1w/"
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz') # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz') # Read in JHU labels

#### Load in data ####
print("Load in subject data ...")
lrid = ex_path + "I1499279_Anon_20210819142214_5"
rlid = ex_path + "I1499337_Anon_20210819142214_6"
# Load in image L-R
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'
# Load in image R-L
img_RL_in = ants.image_read( rlid + '.nii.gz' ) # RL dwi image
img_RL_bval = lrid + '.bval' # bval
img_RL_bvec = lrid + '.bvec'

myoutx = antspymm.joint_dti_recon(
    img_LR_in, img_RL_in,
    img_LR_bval, img_RL_bval,
    img_LR_bvec, img_RL_bvec,
    jhu_atlas=JHU_atlas, jhu_labels=JHU_labels,
    srmodel=None, verbose = True)

if False:
    ants.image_write( myoutx['recon_RL']['RGB'], '/tmp/temp1.nii.gz'  )
    ants.image_write( myoutx['recon_LR']['RGB'], '/tmp/temp2.nii.gz'  )
    ants.image_write( myoutx['mean_fa'], '/tmp/temp1fa.nii.gz'  )
    ants.image_write( myoutx['mean_md'], '/tmp/temp1md.nii.gz'  )
