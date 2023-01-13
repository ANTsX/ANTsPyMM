import ants
import antspynet
import antspymm
flairfn = antspymm.get_data( "flair", target_extension=".nii.gz")
t1fn = antspymm.get_data( "t1", target_extension=".nii.gz")
t1segfn = antspymm.get_data( "t1seg", target_extension=".nii.gz")
flair = ants.image_read( flairfn )
t1 = ants.image_read( t1fn )
t1seg = ants.image_read( t1segfn )
output = antspymm.wmh( flair, t1, t1seg, model='hyper' )
# ants.plot( flair, output['WMH_probability_map'] )
