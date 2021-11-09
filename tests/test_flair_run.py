import ants
import antspynet
import antspymm

flair = ants.image_read('flair.nii.gz')
t1 = ants.image_read('t1.nii.gz') 
t1seg = antspynet.deep_atropos( t1, do_preprocessing = True )  
output = antspymm.wmh( flair, t1, t1seg )

# Plot WMH probability map over white matter segmentation 
t1_2_flair_reg = ants.registration(flair, t1, type_of_transform = 'Rigid') 
wmseg_mask = ants.threshold_image(t1seg['segmentation_image'], low_thresh = 3, high_thresh = 3).iMath("FillHoles")
wmseg_2_flair = ants.apply_transforms(flair, wmseg_mask, transformlist = t1_2_flair_reg['fwdtransforms'])
ants.plot(wmseg_2_flair, output['WMH_probability_map'], axis=2, overlay_alpha = 0.6, cbar=False, nslices = 24, ncol=8, cbar_length=0.3, cbar_vertical=True, filename='wmh.png' )

