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

testingClass = unittest.TestCase( )
islocal = True
id1 = "LS2001_3T_rfMRI_REST1_LR_gdc"
id2 = "LS2001_3T_rfMRI_REST1_RL_gdc"
img1 = ants.image_read( antspymm.get_data( id1, target_extension=".nii.gz") )
img2 = ants.image_read( antspymm.get_data( id2, target_extension=".nii.gz") )
# FIXME: - test that these are the same values
# NOTE: could run SR at this point - will take a long time - example here:
# mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
# mdl = tf.keras.models.load_model( mdlfn )
# srimg = antspymm.super_res_mcimage( img, mdl, verbose=False )
dwp = antspymm.dewarp_imageset( [img1,img2], iterations=2, padding=0,
    target_idx = [10,11,12],
    syn_sampling = 20, syn_metric='mattes',
    type_of_transform = 'SyN',
    total_sigma = 0.0, random_seed=1,
    reg_iterations = [200,50,20] )

if islocal:
    print('rsfmri dewarp done')
    ants.image_write( dwp['dewarped'][0], '~/Downloads/PPMI_DTI_EX/processed/rsfmridewarped0.nii.gz' )
    ants.image_write( dwp['dewarped'][1], '~/Downloads/PPMI_DTI_EX/processed/rsfmridewarped1.nii.gz' )

# now process fMRI as usual --- do we concatenate the two dewarped images?
