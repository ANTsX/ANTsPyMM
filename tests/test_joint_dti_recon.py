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
# Load in image L-R
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'
# Load in image R-L
img_RL_in = ants.image_read( rlid + '.nii.gz' ) # RL dwi image
img_RL_bval = lrid + '.bval' # bval
img_RL_bvec = lrid + '.bvec'

btpB0=ants.image_read('/tmp/tempbtpB.nii.gz')
btpDW=ants.image_read('/tmp/tempbtpD.nii.gz')
btpB0=ants.n4_bias_field_correction(btpB0)
btpDW=ants.n4_bias_field_correction(btpDW)

t1wh = ants.iMath( ants.image_read( t1id ) , 'Normalize' )
mybxt = antspyt1w.brain_extraction( t1wh )
t1w = t1wh * mybxt
reg = ants.registration( btpDW, t1w, 'SyN', verbose=False)
mask = ants.apply_transforms( btpDW, mybxt, reg['fwdtransforms'], interpolator='nearestNeighbor')
print("Begin Join")
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
        denoise=True,
        verbose = True)

if True:
    ants.image_write( myoutx['dtrecon_LR']['FA'], '/tmp/temp1fa1.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['motion_corrected'], '/tmp/temp1moco.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['FA'], '/tmp/temp1fa2.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['RGB'], '/tmp/temp1rgb.nii.gz'  )
    ants.image_write( myoutx['recon_fa'], '/tmp/temp1fa.nii.gz'  )
    ants.image_write( myoutx['recon_md'], '/tmp/temp1md.nii.gz'  )

derka

if False:
    print("dipy dti recon")
    dd = antspymm.dipy_dti_recon(
        img_LR_in, img_LR_bval, img_LR_bvec,
        average_b0=bxtdwi['b0_avg'],
        mask=bxtdwi['b0_mask'],
        motion_correct='Rigid', 
        trim_the_mask=4,
        verbose=True )
    ants.image_write( dd['RGB'], '/tmp/temprgb.nii.gz' )
    ants.image_write( dd['MD'], '/tmp/tempmd.nii.gz' ) 
    ants.image_write( dd['FA'], '/tmp/tempfa.nii.gz' )
    ants.image_write( dd['dwi_mask'], '/tmp/tempmask.nii.gz' )
    derka



if False:    
    a1b,a1w=antspymm.get_average_dwi_b0(img_LR_in)
    a2b,a2w=antspymm.get_average_dwi_b0(img_RL_in)
    btpB0, btpDW = antspymm.dti_template(
        b_image_list=[a1b,a2b],
        w_image_list=[a1w,a2w],
        iterations=7, verbose=True )
    ants.image_write( btpB0, '/tmp/tempbtpB.nii.gz' )
    ants.image_write( btpDW, '/tmp/tempbtpD.nii.gz' )
    ants.image_write( a1w, '/tmp/temp.nii.gz' )
    ants.image_write( a2w, '/tmp/temp2.nii.gz' )
    moco = antspymm.dti_reg( img_RL_in,
                avg_b0=t_B0,
                avg_dwi=t_DWI,
                type_of_transform='SyN',
                verbose=True)


print("dipy dti recon")
dd = antspymm.dipy_dti_recon(
        img_LR_in, img_LR_bval, img_LR_bvec,
        average_b0=btpB0,
        average_dwi=btpDW,
        motion_correct='SyN',
        mask=mask,
        trim_the_mask=4,
        verbose=True )
ants.image_write( dd['motion_corrected'], '/tmp/temp.nii.gz')
ants.image_write( dd['RGB'], '/tmp/temprgb.nii.gz')
ants.image_write( dd['FA'], '/tmp/tempfa.nii.gz')

print("dipy dti recon-B")
ee = antspymm.dipy_dti_recon(
        img_RL_in, img_RL_bval, img_RL_bvec,
        average_b0=btpB0,
        average_dwi=btpDW,
        motion_correct='SyN',
        mask=mask,
        trim_the_mask=4,
        verbose=True )
ants.image_write( ee['motion_corrected'], '/tmp/tempB.nii.gz')
ants.image_write( ee['RGB'], '/tmp/temprgbB.nii.gz')
ants.image_write( ee['FA'], '/tmp/tempfaB.nii.gz')

derka
##########################################
if False:
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


