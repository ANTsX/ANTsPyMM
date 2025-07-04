# based on https://github.com/cookpa/antsDTOrientationTests
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
from tempfile import mktemp
import numpy as np
import antspymm
import sys
# sys.exit(0) # this is not an actual test that we want to run regularly
print(" Load in JHU atlas and labels ")
ex_path = os.path.expanduser( "~/.antspyt1w/" )
ex_path_mm = os.path.expanduser( "~/.antspymm/" )
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz' ) # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz' ) # Read in JHU labels
#### Load in data ####
print("Load in subject data ...")
rotdir='yaw'
lrid=os.path.expanduser('~/code/extern/antsDTOrientationTests/nii/'+rotdir+'/'+rotdir)
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'
#
rotdir2='roll'
rlid=os.path.expanduser('~/code/extern/antsDTOrientationTests/'+'nii/'+rotdir2+'/'+rotdir2)
img_RL_in = ants.image_read( rlid + '.nii.gz') # LR dwi image
img_RL_bval = rlid + '.bval' # bval
img_RL_bvec = rlid + '.bvec'
#
print("build the DTI templates")
a1b,a1w=antspymm.get_average_dwi_b0(img_LR_in)
a2b,a2w=antspymm.get_average_dwi_b0(img_RL_in)
ants.copy_image_info( a1b, a2b )
ants.copy_image_info( a1b, a2w )
bxt1=antspynet.brain_extraction( a1b, 'bold' ).threshold_image(0.5,1)
bxt2=antspynet.brain_extraction( a2b, 'bold' ).threshold_image(0.5,1)
rig = ants.registration( a2b * bxt2, a1b * bxt1, 'Rigid'  )
# the 1st test looks at reorienting the tensor after reconstruction
myoutx = antspymm.joint_dti_recon(
        img_LR_in,
        img_LR_bval,
        img_LR_bvec,
        jhu_atlas = JHU_atlas,
        jhu_labels = JHU_labels,
        reference_B0=a1b,
        reference_DWI=a1w,
        srmodel = None,
        motion_correct = 'Rigid',
        brain_mask = bxt1,
        denoise = True,
        verbose = True )
#
# dti0 = myoutx['dtrecon_LR_dewarp']['tensormodel']# 
dti0 = antspymm.get_dti( a1b, myoutx['dtrecon_LR_dewarp']['tensormodel'], return_image=True )
#
comptx = ants.apply_transforms( a2b, a2b, rig['fwdtransforms'], compose='/tmp/XXX' )
# transform_and_reorient_dti(fixed, moving_dti, composite_transform, py_based=True, verbose=False, **kwargs)
dti0reo = antspymm.transform_and_reorient_dti( a2b, dti0, comptx, py_based=True )
ants.image_write( dti0, '/tmp/temp_dti.nii.gz' )
ants.image_write( dti0reo, '/tmp/temp_dtireo.nii.gz' )
# check this visually ....
# ImageMath 3 /tmp/temp_dti_rgb.nii.gz TensorColor /tmp/temp_dti.nii.gz 
# ImageMath 3 /tmp/temp_dtireo_rgb.nii.gz TensorColor /tmp/temp_dtireo.nii.gz 
#
# just do something simple:  map to template FA then run the same reo as above
# from antspymm data: https://figshare.com/articles/dataset/antspymm_helper_data/16912366
# ImageMath 3 ~/.antspymm/PPMI_template0_tensor_fa.nii.gz TensorFA ~/.antspymm/PPMI_template0_tensor.nii.gz 
tfa=ants.image_read("~/.antspymm/PPMI_template0_tensor_fa.nii.gz").resample_image([2,2,2])
sfa=myoutx['recon_fa']
fareg=ants.registration( tfa, sfa, 'SyN' )
comptx = ants.apply_transforms( tfa, sfa, fareg['fwdtransforms'], compose='/tmp/YYY' )
dti0reo = antspymm.transform_and_reorient_dti( tfa, dti0, comptx, py_based=True )
ants.image_write( dti0, '/tmp/temp_dti.nii.gz' )
ants.image_write( dti0reo, '/tmp/temp_dtireo.nii.gz' )
# ImageMath 3 /tmp/temp_dtireo_rgb.nii.gz TensorColor /tmp/temp_dtireo.nii.gz 
