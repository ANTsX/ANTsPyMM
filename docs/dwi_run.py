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

# Create a temp directory using tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # Define file prefixes and paths
    prefix = os.path.join(tmpdir, 'temp')
    bval_path = prefix + '.bval'
    bvec_path = prefix + '.bvec'
    fa_path_0 = os.path.join(tmpdir, 'temp0.nii.gz')
    fa_path_1 = os.path.join(tmpdir, 'temp1.nii.gz')

    # Write bvals and bvecs
    antspymm.write_bvals_bvecs(bvals=bval_sc, bvecs=bvec_sc, prefix=prefix)

    # DTI reconstruction
    ee = antspymm.dipy_dti_recon(img1_sc, bval_path, bvec_path, verbose=True)

    # Write output images for comparison
    # ants.image_write(dd['FA'], fa_path_0)
    # ants.image_write(ee['FA'], fa_path_1)

    # Temporary files will be cleaned up automatically when the block ends