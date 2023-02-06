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
islocal = False
id1 = "I1499279_Anon_20210819142214_5"
id2 = "I1499337_Anon_20210819142214_6"
img1 = ants.image_read( antspymm.get_data( id1, target_extension=".nii.gz") )
img2 = ants.image_read( antspymm.get_data( id2, target_extension=".nii.gz") )

bvec = antspymm.get_data( id1, target_extension=".bvec")
bval = antspymm.get_data( id1, target_extension=".bval")
dd = antspymm.dipy_dti_recon( img1, bval, bvec, motion_correct='Rigid', verbose=True )
derka
exit(0)
# img1 = ants.image_read( "processed/dwp0sr.nii.gz" )
# img2 = ants.image_read( "processed/dwp1sr.nii.gz" )
b0indices = antspymm.segment_timeseries_by_meanvalue( img1 )['highermeans']
b0indices2 = antspymm.segment_timeseries_by_meanvalue( img1 )['highermeans']
# FIXME: - test that these are the same values
# NOTE: could run SR at this point - will take a long time - example here:
# mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
# mdl = tf.keras.models.load_model( mdlfn )
# srimg = antspymm.super_res_mcimage( img, mdl, verbose=False )

dwp = antspymm.dewarp_imageset( [img1,img2], iterations=2, padding=6,
    target_idx = b0indices,
    syn_sampling = 20, syn_metric='mattes',
    type_of_transform = 'SyN',
    total_sigma = 0.0, random_seed=1,
    reg_iterations = [200,50,20] )

if islocal:
    print('dewarp done')
    ants.image_write( dwp['dewarped'][0], './dewarped0.nii.gz' )
    ants.image_write( dwp['dewarped'][1], './dewarped1.nii.gz' )

# FIXME: - add test
# testingClass.assertAlmostEqual(
#    float( dwp['dewarpedmean'].mean() ),
#    float( 108.2 ), 0, "template mean not close enough")

# now reconstruct DTI
bvec = antspymm.get_data( id1, target_extension=".bvec")
bval = antspymm.get_data( id1, target_extension=".bval")
dd = antspymm.dipy_dti_recon( dwp['dewarped'][0], bval, bvec, b0_idx=b0indices )
# ants.image_write( dd['RGB'], '/tmp/tempsr_rgb.nii.gz' )
bvec = antspymm.get_data( id2, target_extension=".bvec")
bval = antspymm.get_data( id2, target_extension=".bval")
ee = antspymm.dipy_dti_recon( dwp['dewarped'][1], bval, bvec, b0_idx=b0indices )
# ants.image_write( ee['RGB'], '/tmp/temp_rgb2.nii.gz' )
# FIXME: - add test

# sys.exit(os.EX_OK) # code 0, all ok
