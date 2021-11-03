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
    ants.image_write( dwp['dewarped'][0], './rsfmridewarped0.nii.gz' )
    ants.image_write( dwp['dewarped'][1], './rsfmridewarped1.nii.gz' )

# now process fMRI as usual --- do we concatenate the two dewarped images?
# for now, just processing dwp0
import pandas as pd
und = dwp['dewarpedmean']
bmask = antspynet.brain_extraction( und, 'bold' ).threshold_image( 0.3, 1.0 )
powers_areal_mni_itk = pd.read_csv(antspymm.get_data('powers_mni_itk', target_extension=".csv")) # power coordinates
t1fn = antspymm.get_data( 'LS2001_3T_T1w_MPR1_gdc' , target_extension='.nii.gz' )
t1 = ants.image_read( t1fn ).n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
t1bxt = antspynet.brain_extraction( t1, 't1' ).threshold_image( 0.3, 1.0 )
t1seg = antspynet.deep_atropos( t1 )
t1reg = ants.registration( und * bmask, t1 * t1bxt, "SyN" ) # in practice use something different
# ants.plot( t1*t1bxt, t1reg['warpedfixout'] , axis=2, overlay_alpha=0.25, ncol=8, nslices=24 )
# ants.plot( und, t1reg['warpedmovout'], overlay_alpha = 0.25, axis=2, nslices=24, ncol=6 )
boldseg = ants.apply_transforms( und, t1seg['segmentation_image'],
  t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' )
ants.plot( und, boldseg, overlay_alpha = 0.25, axis=2, nslices=24, ncol=6 )
csfAndWM = ( ants.threshold_image( boldseg, 1, 1 ) +
             ants.threshold_image( boldseg, 3, 3 ) ).morphology("erode",1)
mycompcor = ants.compcor( dwp['dewarped'][0],
  ncompcor=4, quantile=0.95, mask = csfAndWM,
  filter_type='polynomial', degree=2 )


nt = dwp['dewarped'][0].shape[3]
import matplotlib.pyplot as plt
plt.plot(  range( nt ), mycompcor['components'][:,0] )
plt.show()
plt.plot(  range( nt ), mycompcor['components'][:,1] )
plt.show()

myvoxes = range(powers_areal_mni_itk.shape[0])
anat = powers_areal_mni_itk['Anatomy']
syst = powers_areal_mni_itk['SystemName']
Brod = powers_areal_mni_itk['Brodmann']
xAAL  = powers_areal_mni_itk['AAL']
ch2 = ants.image_read( ants.get_ants_data( "ch2" ) )
if 'treg' not in globals():
    treg = ants.registration( t1 * t1bxt, ch2, 'SyN' )
concatx2 = treg['invtransforms'] + t1reg['invtransforms']
pts2bold = ants.apply_transforms_to_points( 3, powers_areal_mni_itk, concatx2,whichtoinvert = ( True, False, True, False ) )
locations = pts2bold.iloc[:,:3].values
ptImg = ants.make_points_image( locations, bmask, radius = 2 )
ants.plot( und, ptImg, axis=2, nslices=24, ncol=8 )

bold = dwp['dewarped'][0]
tr = ants.get_spacing( bold )[3]
gmseg = ants.threshold_image( boldseg, 2, 2 )
spa, spt = 1.5, 0.0 # spatial, temporal - which we ignore b/c of frequency filtering
smth = ( spa, spa, spa, spt ) # this is for sigmaInPhysicalCoordinates = F
simg = ants.smooth_image(simg, smth, sigma_in_physical_coordinates = False )
nuisance = mycompcor['components']
gmmat = ants.timeseries_to_matrix( simg, gmseg )
gmmat = ants.bandpass_filter_matrix( gmmat, tr = tr ) # some would argue against this
gmmat = ants.regress_components( gmmat, nuisance )

import numpy as np
postCing = powers_areal_mni_itk['AAL'].unique()[9]
networks = powers_areal_mni_itk['SystemName'].unique()
ww = np.where( powers_areal_mni_itk['SystemName'] == networks[5] )[0]
dfnImg = ants.make_points_image(pts2bold.iloc[ww,:3].values, bmask, radius=1).threshold_image( 1, 400 )
ants.plot( und, dfnImg, axis=2, nslices=24, ncol=8 )


dfnmat = ants.timeseries_to_matrix( simg, ants.threshold_image( dfnImg * gmseg, 1, dfnImg.max() ) )
dfnmat = ants.bandpass_filter_matrix( dfnmat, tr = tr )
dfnmat = ants.regress_components( dfnmat, nuisance )
dfnsignal = dfnmat.mean( axis = 1 )

from scipy.stats.stats import pearsonr
gmmatDFNCorr = np.zeros( gmmat.shape[1] )
for k in range( gmmat.shape[1] ):
    gmmatDFNCorr[ k ] = pearsonr( dfnsignal, gmmat[:,k] )[0]

corrImg = ants.make_image( gmseg, gmmatDFNCorr  )

corrImgPos = corrImg * ants.threshold_image( corrImg, 0.5, 1 )
ants.plot( und, corrImgPos, axis=2, overlay_alpha = 0.6, cbar=False, nslices = 24, ncol=8, cbar_length=0.3, cbar_vertical=True )
