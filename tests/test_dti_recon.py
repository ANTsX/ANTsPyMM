import ants
import antspymm
img1 = ants.image_read( antspymm.get_data( "I1499279_Anon_20210819142214_5", target_extension=".nii.gz") )
bvec = antspymm.get_data( "I1499279_Anon_20210819142214_5", target_extension=".bvec")
bval = antspymm.get_data( "I1499279_Anon_20210819142214_5", target_extension=".bval")
dd = antspymm.dipy_dti_recon( img1, bval, bvec, motion_correct=True, mask_dilation=0 )
