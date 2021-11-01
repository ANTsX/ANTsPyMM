import sys, os
import unittest

os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import tempfile
import shutil

import antspymm
import antspyt1w
import antspynet
import ants

testingClass = unittest.TestCase( )

img1 = ants.image_read( antspymm.get_data( "I1499279_Anon_20210819142214_5", target_extension=".nii.gz") )
img2 = ants.image_read( antspymm.get_data( "I1499337_Anon_20210819142214_6", target_extension=".nii.gz") )
dwp = antspymm.dewarp_imageset( [img1,img2] )

# testingClass.assertAlmostEqual(
#    float( rbp['RandBasisProj01'] ),
#    float( -0.4285278603876295 ), 5, "RBP result not close enough")


temp_dir.cleanup()

##### specialized labeling for hypothalamus
# FIXME hypothalamus
# sys.exit(os.EX_OK) # code 0, all ok
