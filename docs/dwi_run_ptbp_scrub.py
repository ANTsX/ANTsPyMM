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

import glob
from typing import List

downloaddir = os.path.expanduser( "~/data/PTBP/images/" )

import glob
import os
from typing import List

def extract_dti_prefix_paths(base_pattern: str) -> List[str]:
    """
    Find DTI-related files matching a pattern and extract the full path
    with the filename minus its extension.

    Args:
        base_pattern (str): Glob pattern to find files (e.g., *.bval, *.nii.gz)

    Returns:
        List[str]: List of full prefix paths (without file extension)
    """
    files = glob.glob(base_pattern)
    prefix_paths = []

    for filepath in files:
        root, _ = os.path.splitext(filepath)  # removes .bval, .nii.gz, etc.
        prefix_paths.append(root)

    return sorted(prefix_paths)

files = extract_dti_prefix_paths(downloaddir + 'PEDS*/20*/DWI/*.bval')


testingClass = unittest.TestCase( )
islocal = False
if not "prefix" in globals():
    prefix = files[0]
    print( "process: " + prefix )
    # prefix = downloaddir + "PEDS097/20130215/DWI/PEDS097_20130215_0021_DTI_1_1x0_30x1000." # has problems
    # prefix = downloaddir + "PEDS022/20131101/DWI/PEDS022_20131101_0015_DTI_1_1x0_30x1000." # "normal" quality
    imgfn = prefix + ".nii.gz"
    dwi = ants.image_read( imgfn )
    dwi = ants.motion_correction( dwi )['motion_corrected']
    dd = antspymm.dipy_dti_recon( dwi, prefix + ".bval", prefix + ".bvec", verbose=True )

bvalr, bvecr = read_bvals_bvecs( prefix + ".bval", prefix + ".bvec" )
mask=None
mask=dd['dwi_mask']
th=0.20
# th=0.05
for mask in [None,dd['dwi_mask']]:
    print("Mask is" + str( mask ) )
    # scrub example
    # def censor_dwi( dwi, bval, bvec, threshold = 0.20, imputeb0=False, mask=None, verbose=False ):
    dwi_sc, bval_sc, bvec_sc = antspymm.censor_dwi( dwi, bvalr, bvecr, threshold=th, mask=mask, verbose=True )
    ## this is the right answer for PEDS097_20130215_0021_DTI_1_1x0_30x1000
    # censoring: [4, 5, 6, 28]
antspymm.write_bvals_bvecs( bvals=bval_sc, bvecs=bvec_sc, prefix='/tmp/temp' )
ee = antspymm.dipy_dti_recon( dwi_sc, '/tmp/temp.bval', '/tmp/temp.bvec', verbose=True )
# impute example
dwi_imp = antspymm.impute_dwi( dwi, threshold=th, mask=mask, verbose=True )
ff = antspymm.dipy_dti_recon( dwi_imp, prefix + ".bval", prefix + ".bvec", verbose=True )
ants.image_write( dd['FA'], '/tmp/temp0.nii.gz' )
ants.image_write( ee['FA'], '/tmp/temp1.nii.gz' )
ants.image_write( ff['FA'], '/tmp/temp2.nii.gz' )
