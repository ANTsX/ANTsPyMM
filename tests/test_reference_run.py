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
dwp = antspymm.dewarp_imageset( [img1,img2], iterations=4,
    syn_sampling=16, syn_metric='mattes', type_of_transform='SyN',
    aff_metric='GC', random_seed=1,
    reg_iterations = [200,200,20] )
# moco = ants.motion_correction( image=dwp['dewarped'][0],  fixed=dwp['dewarpedmean'], type_of_transform='Rigid' )
# ants.image_write( dwp['dewarpedmean'], '/Users/stnava/Downloads/PPMI_DTI_EX/processed/mean.nii.gz' )
# ants.image_write( dwp['dewarped'][0], '/Users/stnava/Downloads/PPMI_DTI_EX/processed/dwp0.nii.gz' )
# ants.image_write( dwp['dewarped'][1], '/Users/stnava/Downloads/PPMI_DTI_EX/processed/dwp1.nii.gz' )
testingClass.assertAlmostEqual(
    float( dwp['dewarpedmean'].mean() ),
    float( 108.2 ), 1, "template mean not close enough")


# temp_dir.cleanup()

##### specialized labeling for hypothalamus
# FIXME hypothalamus
# sys.exit(os.EX_OK) # code 0, all ok
