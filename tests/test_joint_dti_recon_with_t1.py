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

print("THIS TEST USES DATA FROM DIFFERENT SUBJECTS'T1 and DWI - as such, not a great example of performance but demonstrates utility nonetheless")
ex_path = os.path.expanduser( "~/.antspyt1w/" )
ex_path_mm = os.path.expanduser( "~/.antspymm/" )
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz') # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz') # Read in JHU labels

#### Load in data ####
print("Load in subject data ...")
t1id = ex_path_mm + "t1.nii.gz"
lrid = ex_path_mm + "I1499279_Anon_20210819142214_5"
# Load in image L-R
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'
t1w = ants.image_read( t1id )
t1w = t1w * ants.threshold_image( antspynet.brain_extraction( t1w, 't1') , 0.5, 1)
t1w = ants.resample_image( t1w, [3,3,3] )

myoutx = antspymm.joint_dti_recon(
        img_LR_in,
        img_LR_bval,
        img_LR_bvec,
        jhu_atlas=JHU_atlas,
        jhu_labels=JHU_labels,
        t1w = t1w,
        srmodel=None,
        motion_correct=True,
        verbose = True)

doit=True
if doit:
    ants.image_write( myoutx['t1w_rigid'], '/tmp/tempt1w.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['RGB'], '/tmp/temp2.nii.gz'  )
    ants.image_write( myoutx['recon_fa'], '/tmp/temp1fa.nii.gz'  )
    ants.image_write( myoutx['recon_md'], '/tmp/temp1md.nii.gz'  )
