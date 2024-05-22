import sys, os
import unittest

os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import tempfile
import shutil
import tensorflow as tf
import antspymm
import antspyt1w
import antspynet
import ants
from dipy.io.gradients import read_bvals_bvecs                                 

testingClass = unittest.TestCase( )
islocal = False
downloaddir = os.path.expanduser( "~/data/PTBP/images/" )
if not "prefix" in globals():
    prefix = downloaddir + "PEDS097/20130215/DWI/PEDS097_20130215_0021_DTI_1_1x0_30x1000." # has problems
    # prefix = downloaddir + "PEDS022/20131101/DWI/PEDS022_20131101_0015_DTI_1_1x0_30x1000." # "normal" quality
    imgfn = prefix + "nii.gz"
    dwi = ants.image_read( imgfn )
    dwi = ants.motion_correction( dwi )['motion_corrected']
    dd = antspymm.dipy_dti_recon( dwi, prefix + "bval", prefix + "bvec", verbose=True )

bvalr, bvecr = read_bvals_bvecs( prefix + "bval", prefix + "bvec" )
mask=None
mask=dd['dwi_mask']
th=0.20
for mask in [None,dd['dwi_mask']]:
    print("Mask is" + str( mask ) )
    # scrub example
    dwi_sc, bval_sc, bvec_sc = antspymm.scrub_dwi( dwi, bvalr, bvecr, threshold=th, mask=mask, verbose=True )
antspymm.write_bvals_bvecs( bvals=bval_sc, bvecs=bvec_sc, prefix='/tmp/temp' )
ee = antspymm.dipy_dti_recon( dwi_sc, '/tmp/temp.bval', '/tmp/temp.bvec', verbose=True )
# impute example
dwi_imp = antspymm.impute_dwi( dwi, threshold=th, mask=mask, verbose=True )
ff = antspymm.dipy_dti_recon( dwi_imp, prefix + "bval", prefix + "bvec", verbose=True )
ants.image_write( dd['FA'], '/tmp/temp0.nii.gz' )
ants.image_write( ee['FA'], '/tmp/temp1.nii.gz' )
ants.image_write( ff['FA'], '/tmp/temp2.nii.gz' )
