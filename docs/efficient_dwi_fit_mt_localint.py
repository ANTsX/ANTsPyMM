import os
mynt="2" # should be number of cores ... or thereabouts
os.environ["TF_NUM_INTEROP_THREADS"] = mynt
os.environ["TF_NUM_INTRAOP_THREADS"] = mynt
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = mynt
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
import time

from dipy.core.histeq import histeq
import dipy.reconst.dti as dti
from dipy.core.gradients import (gradient_table, gradient_table_from_gradient_strength_bvecs)
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa

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
if not "mask" in globals():
        print("build the DTI templates")
        # btpB0, btpDW=antspymm.get_average_dwi_b0(img_LR_in)
        btpDW=ants.get_average_of_timeseries(img_LR_in)
        print("brain extract the T1")
        t1wh = ants.iMath( ants.image_read( t1id ) , 'Normalize' )
        mybxt = antspyt1w.brain_extraction( t1wh )
        t1w = t1wh * mybxt
        print("bxt the DTI template space")
        reg = ants.registration( btpDW, t1w, 'antsRegistrationSyNQuickRepro[s]', verbose=False)
        mask = ants.apply_transforms( btpDW, mybxt, 
                reg['fwdtransforms'], interpolator='nearestNeighbor')


from dipy.io.gradients import read_bvals_bvecs
bvals, bvecs = read_bvals_bvecs( img_LR_bval, img_LR_bvec )
gtab = gradient_table(bvals, bvecs=bvecs, atol=2.0 )

start_time = time.time()
xxx = antspymm.efficient_dwi_fit(
        gtab, 
        diffusion_model_name='DTI', 
        imagein=img_LR_in, 
        maskin=mask, 
        chunk_size=1024, 
        num_threads=int(mynt), verbose=True)
end_time = time.time()
elapsed = end_time - start_time

print(f"Execution time for efficient_dwi_fit: {elapsed:.2f} seconds with {mynt} threads ")
print("Done with efficient_dwi_fit")
# Execution time for efficient_dwi_fit: 24.31 seconds with 1 threads 
# Execution time for efficient_dwi_fit: 8.59 seconds with 8 threads 
