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




# FIXME - need to return FD and other motion parameters from dewarp function
# then incorporate those parameters in this example

testingClass = unittest.TestCase( )
islocal = False
id1 = "LS2001_3T_rfMRI_REST1_LR_gdc"
id2 = "LS2001_3T_rfMRI_REST1_RL_gdc"
img1 = ants.image_read( antspymm.get_data( id1, target_extension=".nii.gz") )
img2 = ants.image_read( antspymm.get_data( id2, target_extension=".nii.gz") )

if 'dwp' not in globals() and False:
    dwp = antspymm.dewarp_imageset( [img1,img2], iterations=3, padding=0,
        target_idx = [10,11,12],
        syn_sampling = 20, syn_metric='mattes',
        type_of_transform = 'SyN',
        total_sigma = 0.0, random_seed=1,
        reg_iterations = [200,50,20] )
    if islocal:
      print('rsfmri dewarp done')
      ants.image_write( dwp['dewarped'][0], './rsfmridewarped0.nii.gz' )
      ants.image_write( dwp['dewarped'][1], './rsfmridewarped1.nii.gz' )

# now process fMRI as usual --- do we concatenate the two dewarped images?
# for now, just processing dwp0
import pandas as pd
und = ants.get_average_of_timeseries( img1 ) # dwp['dewarpedmean']
bmask = antspynet.brain_extraction( und, 'bold' ).threshold_image( 0.3, 1.0 )
powers_areal_mni_itk = pd.read_csv(antspymm.get_data('powers_mni_itk', target_extension=".csv")) # power coordinates
t1fn = antspymm.get_data( 'LS2001_3T_T1w_MPR1_gdc' , target_extension='.nii.gz' )
print("do t1")
t1 = ants.image_read( t1fn ).n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
t1bxt = antspynet.brain_extraction( t1, 't1' ).threshold_image( 0.3, 1.0 )
t1seg = antspynet.deep_atropos( t1 )
t1segmentation = t1seg['segmentation_image']
print("do rsf")
rsf = antspymm.resting_state_fmri_networks(
  img1, t1 * t1bxt, t1segmentation, f=[0.03, 0.08], spa=1.5, spt=0.5, nc=6)
