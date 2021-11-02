import sys, os
import unittest
import tensorflow as tf
import ants
import antspynet
import antspymm

import tempfile
import shutil

mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
mdl = tf.keras.models.load_model( mdlfn )
img = ants.image_read( antspymm.get_data( "I1499279_Anon_20210819142214_5", target_extension=".nii.gz") )
lod = 16
imgd = ants.resample_image( img, (lod,lod,lod,lod), use_voxels=True)
srimg = antspymm.super_res_mcimage( imgd, mdl, verbose=False )

testingClass = unittest.TestCase( )

testingClass.assertEqual(
    srimg.shape[0], lod*2, "dimension incorrect")
