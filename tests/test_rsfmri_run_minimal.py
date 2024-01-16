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
import numpy as np

testingClass = unittest.TestCase( )
islocal = False
id1 = "LS2001_3T_rfMRI_REST1_LR_gdc"
img1 = ants.image_read( antspymm.get_data( id1, target_extension=".nii.gz") )

import pandas as pd
und = ants.get_average_of_timeseries( img1 )

t1fn = antspymm.get_data( 'LS2001_3T_T1w_MPR1_gdc' , target_extension='.nii.gz' )
print("do t1")
if not "t1" in globals():
    t1 = ants.image_read( t1fn ).n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
    t1bxt = antspynet.brain_extraction( t1, 't1' ).threshold_image( 0.3, 1.0 )
    t1seg = antspynet.deep_atropos( t1 )
    t1segmentation = t1seg['segmentation_image']

print("do rsf")
imgrm=antspymm.remove_volumes_from_timeseries( img1, list(range(60,420)) )

rsf = antspymm.resting_state_fmri_networks(
  imgrm, und, t1 * t1bxt, t1segmentation, spa=0, spt=0.5, nc=6, 
  outlier_threshold=0.2, verbose=True )
