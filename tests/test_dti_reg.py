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
t1id = ex_path_mm + "t1_rand.nii.gz"
print("Load in image L-R")
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'

print("brain extract the T1")
t1wh = ants.iMath( ants.image_read( t1id ) , 'Normalize' )
mybxt = antspyt1w.brain_extraction( t1wh )
t1w = t1wh * mybxt

print("build the DTI templates")
a1b,a1w=antspymm.get_average_dwi_b0(img_LR_in)

dtireg = antspymm.dti_reg(
    img_LR_in,
    avg_b0=a1b,
    avg_dwi=a1w,
    bvals=img_LR_bval,
    bvecs=img_LR_bvec,
    type_of_transform='SyN',
    verbose=True )

if False:
    ants.image_write( dtireg['motion_corrected'], '/tmp/temp.nii.gz' )
