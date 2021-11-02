import tensorflow as tf
import ants
import antspynet
import antspymm
mdlfn = antspynet.get_pretrained_network( "mriSuperResolution" )
mdl = tf.keras.models.load_model( mdlfn )
img = ants.image_read( antspymm.get_data( "I1499279_Anon_20210819142214_5", target_extension=".nii.gz") )
imgd = ants.resample_image( img, (16,16,16,16), use_voxels=True)
srimg = antspymm.super_res_mcimage( imgd, mdl, verbose=True )
