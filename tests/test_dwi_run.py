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
id1 = "I1499279_Anon_20210819142214_5"
id2 = "I1499337_Anon_20210819142214_6"
img1 = ants.image_read( antspymm.get_data( id1, target_extension=".nii.gz") )
img2 = ants.image_read( antspymm.get_data( id2, target_extension=".nii.gz") )

bvec = antspymm.get_data( id1, target_extension=".bvec")
bval = antspymm.get_data( id1, target_extension=".bval")
dd = antspymm.dipy_dti_recon( img1, bval, bvec, verbose=True )
# scrub example
bvalr, bvecr = read_bvals_bvecs( bval, bvec )
img1_sc, bval_sc, bvec_sc = antspymm.scrub_dwi( img1, bvalr, bvecr, threshold=0.01, verbose=True )
antspymm.write_bvals_bvecs( bvals=bval_sc, bvecs=bvec_sc, prefix='/tmp/temp' )
ee = antspymm.dipy_dti_recon( img1_sc, '/tmp/temp.bval', '/tmp/temp.bvec', verbose=True )
ants.image_write( dd['FA'], '/tmp/temp0.nii.gz' )
ants.image_write( ee['FA'], '/tmp/temp1.nii.gz' )
exit(0)
