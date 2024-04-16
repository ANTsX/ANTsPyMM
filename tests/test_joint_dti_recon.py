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
print("Load in image L-R")
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'
print("Load in image R-L")
img_RL_in = ants.image_read( rlid + '.nii.gz' ) # RL dwi image
img_RL_bval = lrid + '.bval' # bval
img_RL_bvec = lrid + '.bvec'

print("brain extract the T1")
t1wh = ants.iMath( ants.image_read( t1id ) , 'Normalize' )
mybxt = antspyt1w.brain_extraction( t1wh )
t1w = t1wh * mybxt

print("build the DTI templates")
a1b,a1w=antspymm.get_average_dwi_b0(img_LR_in)
a2b,a2w=antspymm.get_average_dwi_b0(img_RL_in)
btpB0, btpDW = antspymm.dti_template(
        b_image_list=[a1b,a2b],
        w_image_list=[a1w,a2w],
        iterations=2, verbose=True )

print("bxt the DTI template space")
reg = ants.registration( btpDW, t1w, 'SyN', verbose=False)
mask = ants.apply_transforms( btpDW, mybxt, reg['fwdtransforms'], interpolator='nearestNeighbor')

print("Begin joint DTI recon")
myoutx = antspymm.joint_dti_recon(
        img_LR_in,
        img_LR_bval,
        img_LR_bvec,
        jhu_atlas = JHU_atlas,
        jhu_labels = JHU_labels,
        reference_B0=btpB0,
        reference_DWI=btpDW,
        srmodel = None,
        img_RL = img_RL_in,
        bval_RL = img_RL_bval,
        bvec_RL = img_RL_bvec,
        motion_correct = 'Rigid',
        brain_mask = mask,
        denoise = False,
        verbose = True )

if True:
    ants.image_write( myoutx['recon_fa'], '/tmp/temp1fa.nii.gz'  )
    ants.image_write( myoutx['recon_md'], '/tmp/temp1md.nii.gz'  )
    ants.image_write( myoutx['dwi_LR_dewarped'], '/tmp/temp1moco.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['RGB'], '/tmp/temp1rgb.nii.gz'  )


myoutx = antspymm.joint_dti_recon(
        img_LR_in,
        img_LR_bval,
        img_LR_bvec,
        jhu_atlas = JHU_atlas,
        jhu_labels = JHU_labels,
        reference_B0=btpB0,
        reference_DWI=btpDW,
        srmodel = None,
        img_RL = img_RL_in,
        bval_RL = img_RL_bval,
        bvec_RL = img_RL_bvec,
        motion_correct = 'SyN',
        brain_mask = mask,
        denoise = False,
        verbose = True )

if True:
    ants.image_write( myoutx['recon_fa'], '/tmp/temp1fas.nii.gz'  )
    ants.image_write( myoutx['recon_md'], '/tmp/temp1mds.nii.gz'  )
    ants.image_write( myoutx['dwi_LR_dewarped'], '/tmp/temp1mocos.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['RGB'], '/tmp/temp1rgbs.nii.gz'  )
