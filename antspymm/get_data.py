
__all__ = ['get_data','dewarp_imageset','super_res_mcimage','dipy_dti_recon',
    'segment_timeseries_by_meanvalue', 'wmh', 'neuromelanin',
    'resting_state_fmri_networks']

from pathlib import Path
from pathlib import PurePath
import os
import pandas as pd
import math
import os.path
from os import path
import pickle
import sys
import numpy as np
import random
import functools
from operator import mul
from scipy.sparse.linalg import svds
from scipy.stats.stats import pearsonr
import re

from dipy.core.histeq import histeq
import dipy.reconst.dti as dti
from dipy.core.gradients import (gradient_table, gradient_table_from_gradient_strength_bvecs)
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa
import nibabel as nib

import ants
import antspynet
import tensorflow as tf

from multiprocessing import Pool

DATA_PATH = os.path.expanduser('~/.antspymm/')

def get_data( name=None, force_download=False, version=6, target_extension='.csv' ):
    """
    Get ANTsPyMM data filename

    The first time this is called, it will download data to ~/.antspymm.
    After, it will just read data from disk.  The ~/.antspymm may need to
    be periodically deleted in order to ensure data is current.

    Arguments
    ---------
    name : string
        name of data tag to retrieve
        Options:
            - 'all'

    force_download: boolean

    version: version of data to download (integer)

    Returns
    -------
    string
        filepath of selected data

    Example
    -------
    >>> import antspymm
    >>> antspymm.get_data()
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    def download_data( version ):
        url = "https://figshare.com/ndownloader/articles/16912366/versions/" + str(version)
        target_file_name = "16912366.zip"
        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
            cache_subdir=DATA_PATH, extract = True )
        os.remove( DATA_PATH + target_file_name )

    if force_download:
        download_data( version = version )


    files = []
    for fname in os.listdir(DATA_PATH):
        if ( fname.endswith(target_extension) ) :
            fname = os.path.join(DATA_PATH, fname)
            files.append(fname)

    if len( files ) == 0 :
        download_data( version = version )
        for fname in os.listdir(DATA_PATH):
            if ( fname.endswith(target_extension) ) :
                fname = os.path.join(DATA_PATH, fname)
                files.append(fname)

    if name == 'all':
        return files

    datapath = None

    for fname in os.listdir(DATA_PATH):
        mystem = (Path(fname).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        if ( name == mystem and fname.endswith(target_extension) ) :
            datapath = os.path.join(DATA_PATH, fname)

    return datapath




def dewarp_imageset( image_list, iterations=None, padding=0, target_idx=[0], **kwargs ):
    """
    Dewarp a set of images

    Makes simplifying heuristic decisions about how to transform an image set
    into an unbiased reference space.  Will handle plenty of decisions
    automatically so beware.  Computes an average shape space for the images
    and transforms them to that space.

    Arguments
    ---------
    image_list : list containing antsImages 2D, 3D or 4D

    iterations : number of template building iterations

    padding:  will pad the images by an integer amount to limit edge effects

    target_idx : the target indices for the time series over which we should average;
        a list of integer indices into the last axis of the input images.

    kwargs : keyword args
        arguments passed to ants registration - these must be set explicitly

    Returns
    -------
    a dictionary with the mean image and the list of the transformed images as
    well as motion correction parameters for each image in the input list

    Example
    -------
    >>> import antspymm
    """
    outlist = []
    avglist = []
    if len(image_list[0].shape) > 3:
        imagetype = 3
        for k in range(len(image_list)):
            for j in range(len(target_idx)):
                avglist.append( ants.slice_image( image_list[k], axis=3, idx=target_idx[j] ) )
    else:
        imagetype = 0
        avglist=image_list

    if iterations is None:
        iterations = 2

    pw=[]
    for k in range(len(avglist[0].shape)):
        pw.append( padding )
    for k in range(len(avglist)):
        avglist[k] = ants.pad_image( avglist[k], pad_width=pw  )

    btp = ants.build_template( image_list=avglist,
        gradient_step=0.5, blending_weight=0.8,
        iterations=iterations, **kwargs )

    # last - warp all images to this frame
    mocoplist = []
    mocofdlist = []
    reglist = []
    for k in range(len(image_list)):
        moco0 = ants.motion_correction( image=image_list[k], fixed=btp, type_of_transform='BOLDRigid' )
        mocoplist.append( moco0['motion_parameters'] )
        mocofdlist.append( moco0['FD'] )
        locavg = ants.slice_image( moco0['motion_corrected'], axis=3, idx=0 ) * 0.0
        for j in range(len(target_idx)):
            locavg = locavg + ants.slice_image( moco0['motion_corrected'], axis=3, idx=target_idx[j] )
        locavg = locavg * 1.0 / len(target_idx)
        reg = ants.registration( btp, locavg, **kwargs )
        reglist.append( reg )
        if imagetype == 3:
            myishape = image_list[k].shape
            mytslength = myishape[ len(myishape) - 1 ]
            mywarpedlist = []
            for j in range(mytslength):
                locimg = ants.slice_image( image_list[k], axis=3, idx = j )
                mywarped = ants.apply_transforms( btp, locimg,
                    reg['fwdtransforms'] + moco0['motion_parameters'][j], imagetype=0 )
                mywarpedlist.append( mywarped )
            mywarped = ants.list_to_ndimage( image_list[k], mywarpedlist )
        else:
            mywarped = ants.apply_transforms( btp, image_list[k], reg['fwdtransforms'], imagetype=imagetype )
        outlist.append( mywarped )

    return {
        'dewarpedmean':btp,
        'dewarped':outlist,
        'deformable_registrations': reglist,
        'FD': mocofdlist,
        'motionparameters': mocoplist }


def super_res_mcimage( image, srmodel, truncation=[0.0001,0.995],
    poly_order=1,
    target_range=[-127.5,127.5],
    verbose=False ):
    """
    Super resolution on a timeseries or multi-channel image

    Arguments
    ---------
    image : an antsImage

    srmodel : a tensorflow fully convolutional model

    truncation :  quantiles at which we truncate intensities to limit impact of outliers e.g. [0.005,0.995]

    poly_order : if not None, will fit a global regression model to map
        intensity back to original histogram space

    target_range : 2-element tuple
        a tuple or array defining the (min, max) of the input image
        (e.g., -127.5, 127.5).  Output images will be scaled back to original
        intensity. This range should match the mapping used in the training
        of the network.

    verbose : boolean

    Returns
    -------
    super resolution version of the image

    Example
    -------
    >>> import antspymm
    """
    idim = image.dimension
    ishape = image.shape
    nTimePoints = ishape[idim - 1]
    mcsr = list()
    counter = 0
    for k in range(nTimePoints):
        mycount = round(k / nTimePoints * 100)
        if verbose and mycount == counter:
            counter = counter + 10
            print(mycount, end="%.", flush=True)
        temp = ants.slice_image( image, axis=idim - 1, idx=k )
        temp = ants.iMath( temp, "TruncateIntensity", truncation[0], truncation[1] )
        mysr = antspynet.apply_super_resolution_model_to_image( temp, srmodel,
            target_range = target_range )
        if k == 0:
            upshape = list()
            for j in range(len(ishape)-1):
                upshape.append( mysr.shape[j] )
            upshape.append( ishape[ idim-1 ] )
            if verbose:
                print("SR will be of voxel size:" + str(upshape) )
        if poly_order is not None:
            bilin = ants.resample_image_to_target( temp, mysr )
            mysr = antspynet.regression_match_image( mysr, bilin, poly_order = poly_order )
        mcsr.append( mysr )

    imageup = ants.resample_image( image, upshape, use_voxels = True )
    if verbose:
        print("Done")

    return ants.list_to_ndimage( imageup, mcsr )



def segment_timeseries_by_meanvalue( image, quantile = 0.995 ):
    """
    Identify indices of a time series where we assume there is a different mean
    intensity over the volumes.  The indices of volumes with higher and lower
    intensities is returned.  Can be used to automatically identify B0 volumes
    in DWI timeseries.

    Arguments
    ---------
    image : an antsImage holding B0 and DWI

    quantile : a quantile for splitting the indices of the volume - should be greater than 0.5

    Returns
    -------
    dictionary holding the two sets of indices

    Example
    -------
    >>> import antspymm
    """
    ishape = image.shape
    lastdim = len(ishape)-1
    meanvalues = list()
    for x in range(ishape[lastdim]):
        meanvalues.append(  ants.slice_image( image, axis=lastdim, idx=x ).mean() )
    myhiq = np.quantile( meanvalues, quantile )
    myloq = np.quantile( meanvalues, 1.0 - quantile )
    lowerindices = list()
    higherindices = list()
    for x in range(len(meanvalues)):
        hiabs = abs( meanvalues[x] - myhiq )
        loabs = abs( meanvalues[x] - myloq )
        if hiabs < loabs:
            higherindices.append(x)
        else:
            lowerindices.append(x)

    return {
    'lowermeans':lowerindices,
    'highermeans':higherindices }

def dipy_dti_recon( image, bvalsfn, bvecsfn, median_radius = 4, numpass = 4, dilate = 2, vol_idx=None ):
    """
    Super resolution on a timeseries or multi-channel image

    Arguments
    ---------
    image : an antsImage holding B0 and DWI

    bvalsfn : bvalue filename

    bvecsfn : bvector filename

    median_radius : median_radius from dipy median_otsu function

    numpass : numpass from dipy median_otsu function

    dilate : dilate from dipy median_otsu function

    vol_idx : the indices of the B0; if None, use segment_timeseries_by_meanvalue to guess

    Returns
    -------
    dictionary holding the tensorfit, MD, FA and RGB images

    Example
    -------
    >>> import antspymm
    """
    if vol_idx is None:
        vol_idx = segment_timeseries_by_meanvalue( image )['highermeans']
    bvals, bvecs = read_bvals_bvecs( bvalsfn , bvecsfn   )
    gtab = gradient_table(bvals, bvecs)
    img = image.to_nibabel()
    data = img.get_fdata()
    data3d = data[:,:,:,0] * 1/6
    for x in range(1, 5):
      data3d = data3d + data[:,:,:,0] * 1/6

    maskdata, mask = median_otsu(
        data,
        vol_idx=vol_idx,
        median_radius = median_radius,
        numpass = numpass,
        autocrop = True,
        dilate = dilate )

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0

    MD1 = dti.mean_diffusivity(tenfit.evals)
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, tenfit.evecs)
    return {
        'tensormodel':tenfit,
        'MD':ants.from_nibabel(nib.Nifti1Image(MD1.astype(np.float32), img.affine)),
        'FA':ants.from_nibabel(nib.Nifti1Image(FA.astype(np.float32), img.affine)),
        'RGB': ants.from_nibabel(nib.Nifti1Image(RGB.astype(np.float32), img.affine)) }

def wmh( flair, t1, t1seg, mmfromconvexhull = 12 ) :

  """
  Outputs the WMH probability mask and a summary single measurement

  Arguments
  ---------
  flair : ANTsImage
    input 3-D FLAIR brain image (not skull-stripped).

  t1 : ANTsImage
    input 3-D T1 brain image (not skull-stripped).

  t1seg : ANTsImage
    T1 segmentation image

  mmfromconvexhull : float
    restrict WMH to regions that are WM or mmfromconvexhull mm away from the
    convex hull of the cerebrum

  Returns
  ---------
  WMH probability map and a summary single measurement which is the sum of the WMH map

  """
  import numpy as np
  import math
  probability_mask = antspynet.sysu_media_wmh_segmentation(flair)
  t1_2_flair_reg = ants.registration(flair, t1, type_of_transform = 'Rigid') # Register T1 to Flair
  wmseg_mask = ants.threshold_image( t1seg,
    low_thresh = 3, high_thresh = 3).iMath("FillHoles")
  if mmfromconvexhull > 0:
        convexhull = ants.threshold_image( t1seg, 1, 4 )
        spc2vox = np.prod( ants.get_spacing( t1seg ) )
        voxdist = 0.0
        myspc = ants.get_spacing( t1seg )
        for k in range( t1seg.dimension ):
            voxdist = voxdist + myspc[k] * myspc[k]
        voxdist = math.sqrt( voxdist )
        nmorph = round( 2.0 / voxdist )
        convexhull = ants.morphology( convexhull, "close", nmorph )
        dist = ants.iMath( convexhull, "MaurerDistance" ) * -1.0
        wmseg_mask = wmseg_mask + ants.threshold_image( dist, mmfromconvexhull, 1.e80 )
        wmseg_mask = ants.threshold_image( wmseg_mask, 1, 2 )

  wmseg_2_flair = ants.apply_transforms(flair, wmseg_mask,
    transformlist = t1_2_flair_reg['fwdtransforms'],
    interpolator = 'nearestNeighbor' )
  probability_mask_WM = wmseg_2_flair * probability_mask # Remove WMH signal outside of WM
  label_stats = ants.label_stats(probability_mask_WM, wmseg_2_flair)
  label1 = label_stats[label_stats["LabelValue"]==1.0]
  wmh_sum = label1['Mass'].values[0]

  return{
      'WMH_probability_map_raw': probability_mask,
      'WMH_probability_map' : probability_mask_WM,
      'wmh_mass': wmh_sum }



def neuromelanin( list_nm_images, t1, t1slab, t1lab ) :

  """
  Outputs the averaged and registered neuromelanin image, and neuromelanin labels

  Arguments
  ---------
  list_nm_image : list of ANTsImages
    list of neuromenlanin repeat images

  t1 : ANTsImage
    input 3-D T1 brain image

  t1slab : ANTsImage
    t1 slab - a label image that roughly locates the likely position of the NM slab acquisition

  t1lab : ANTsImage
    t1 labels that will be propagated to the NM


  Returns
  ---------
  Averaged and registered neuromelanin image and neuromelanin labels

  """

  # Average images in image_list
  avg = list_nm_images[0]*0.0
  for k in range(len( list_nm_images )):
    avg = avg + ants.resample_image_to_target( list_nm_images[k], avg ) / len( list_nm_images )

  # Register each nm image in list_nm_images to the averaged nm image (avg)
  reglist = []
  for k in range(len( list_nm_images )):
    current_image = ants.registration( avg, list_nm_images[k], type_of_transform = 'Rigid' )
    current_image = current_image['warpedmovout']
    reglist.append( current_image )

  # Average the reglist
  new_ilist = []
  nm_avg = reglist[0]*0
  for k in range(len( reglist )):
    nm_avg = nm_avg + reglist[k] / len( reglist )

  t1c = ants.crop_image( t1, t1slab )
  slabreg = ants.registration( nm_avg, t1c, 'Rigid' )
  nmlab = ants.apply_transforms( nm_avg, t1lab, slabreg['fwdtransforms'],
    interpolator = 'genericLabel' )

  return{
      'NM_avg' : nm_avg,
      'NM_labels': nmlab }




def resting_state_fmri_networks( fmri, t1, t1segmentation,
    f=[0.03,0.08],   spa = 1.5, spt = 0.5, nc = 6 ):

  """
  Compute resting state network correlation maps based on the J Power labels.
  This will output a map for each of the major network systems.

  Arguments
  ---------
  fmri : BOLD fmri antsImage

  t1 : ANTsImage
    input 3-D T1 brain image (brain extracted)

  t1segmentation : ANTsImage
    t1 segmentation - a six tissue segmentation image in T1 space

  f : band pass limits for frequency filtering

  spa : gaussian smoothing for spatial component

  spt : gaussian smoothing for temporal component

  nc  : number of components for compcor filtering

  Returns
  ---------
  a dictionary containing the derived network maps

  """
  import numpy as np
  import pandas as pd
  A = np.zeros((1,1))
  powers_areal_mni_itk = pd.read_csv( get_data('powers_mni_itk', target_extension=".csv")) # power coordinates

  dwp = dewarp_imageset( [fmri], iterations=1, padding=8,
          target_idx = [7,8,9],
          syn_sampling = 20, syn_metric='mattes',
          type_of_transform = 'SyN',
          total_sigma = 0.0, random_seed=1,
          reg_iterations = [50,20] )
  und = dwp['dewarpedmean']
  bmask = antspynet.brain_extraction( und, 'bold' ).threshold_image( 0.3, 1.0 )
  bmask = ants.iMath( bmask, "MD", 4 ).iMath( "ME", 4 ).iMath( "FillHoles" )

  t1reg = ants.registration( und * bmask, t1, "SyNBold" )
  boldseg = ants.apply_transforms( und, t1segmentation,
    t1reg['fwdtransforms'], interpolator = 'genericLabel' ) * bmask
  gmseg = ants.threshold_image( t1segmentation, 2, 2 ).iMath("MD",1)
  gmseg = gmseg + ants.threshold_image( t1segmentation, 4, 4 )
  gmseg = ants.threshold_image( gmseg, 1, 4 )
  gmseg = ants.apply_transforms( und, gmseg,
    t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' )  * bmask
  csfAndWM = ( ants.threshold_image( t1segmentation, 1, 1 ) +
               ants.threshold_image( t1segmentation, 3, 3 ) ).morphology("erode",2)
  csfAndWM = ants.apply_transforms( und, csfAndWM,
    t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' )  * bmask

  dwpind = 0
  mycompcor = ants.compcor( dwp['dewarped'][dwpind],
    ncompcor=nc, quantile=0.90, mask = csfAndWM,
    filter_type='polynomial', degree=2 )

  nt = dwp['dewarped'][dwpind].shape[3]

  myvoxes = range(powers_areal_mni_itk.shape[0])
  anat = powers_areal_mni_itk['Anatomy']
  syst = powers_areal_mni_itk['SystemName']
  Brod = powers_areal_mni_itk['Brodmann']
  xAAL  = powers_areal_mni_itk['AAL']
  ch2 = ants.image_read( ants.get_ants_data( "ch2" ) )
  treg = ants.registration( t1, ch2, 'SyN' )
  concatx2 = treg['invtransforms'] + t1reg['invtransforms']
  pts2bold = ants.apply_transforms_to_points( 3, powers_areal_mni_itk, concatx2,
    whichtoinvert = ( True, False, True, False ) )
  locations = pts2bold.iloc[:,:3].values
  ptImg = ants.make_points_image( locations, bmask, radius = 2 )

  tr = ants.get_spacing( dwp['dewarped'][dwpind] )[3]
  highMotionTimes = np.where( dwp['FD'][dwpind] >= 1.0 )
  goodtimes = np.where( dwp['FD'][dwpind] < 0.5 )
  smth = ( spa, spa, spa, spt ) # this is for sigmaInPhysicalCoordinates = F
  simg = ants.smooth_image(dwp['dewarped'][dwpind], smth, sigma_in_physical_coordinates = False )

  nuisance = mycompcor[ 'components' ]
  nuisance = np.c_[ nuisance, mycompcor['basis'] ]
  nuisance = np.c_[ nuisance, dwp['FD'][dwpind] ]

  gmmat = ants.timeseries_to_matrix( simg, gmseg )
  gmmat = ants.bandpass_filter_matrix( gmmat, tr = tr, lowf=f[0], highf=f[1] ) # some would argue against this
  gmsignal = gmmat.mean( axis = 1 )
  nuisance = np.c_[ nuisance, gmsignal ]
  gmmat = ants.regress_components( gmmat, nuisance )

  networks = powers_areal_mni_itk['SystemName'].unique()

  outdict = {}
  outdict['meanBold'] = und
  outdict['pts2bold'] = pts2bold
  outdict['nuisance'] = nuisance
  outdict['FD'] = dwp['FD'][dwpind]

  # this is just for human readability - reminds us of which we choose by default
  netnames = ['Cingulo-opercular Task Control', 'Default Mode',
                'Memory Retrieval', 'Ventral Attention', 'Visual',
                'Fronto-parietal Task Control', 'Salience', 'Subcortical',
                'Dorsal Attention']
  # cerebellar is 12
  ct = 0
  numofnets = [3,5,6,7,8,9,10,11,13]
  for mynet in numofnets:
    netname = re.sub( " ", "", networks[mynet] )
    netname = re.sub( "-", "", netname )
    ww = np.where( powers_areal_mni_itk['SystemName'] == networks[mynet] )[0]
    dfnImg = ants.make_points_image(pts2bold.iloc[ww,:3].values, bmask, radius=1).threshold_image( 1, 1e9 )
    dfnmat = ants.timeseries_to_matrix( simg, ants.threshold_image( dfnImg, 1, dfnImg.max() ) )
    dfnmat = ants.bandpass_filter_matrix( dfnmat, tr = tr, lowf=f[0], highf=f[1]  )
    dfnmat = ants.regress_components( dfnmat, nuisance )
    dfnsignal = dfnmat.mean( axis = 1 )
    gmmatDFNCorr = np.zeros( gmmat.shape[1] )
    for k in range( gmmat.shape[1] ):
      gmmatDFNCorr[ k ] = pearsonr( dfnsignal, gmmat[:,k] )[0]
    corrImg = ants.make_image( gmseg, gmmatDFNCorr  )
    outdict[ netname ] = corrImg
    ct = ct + 1

  A = np.zeros( ( len( numofnets ), len( numofnets ) ) )
  newnames=[]

  for i in range( len( numofnets ) ):
      netnamei = re.sub( " ", "", networks[numofnets[i]] )
      netnamei = re.sub( "-", "", netnamei )
      newnames.append( netnamei )
      binmask = ants.threshold_image( outdict[ netnamei ], 0.2, 1.0 )
      ww = np.where( powers_areal_mni_itk['SystemName'] == networks[numofnets[i]] )[0]
      dfnImg = ants.make_points_image(pts2bold.iloc[ww,:3].values, bmask, radius=1).threshold_image( 1, 1e9 )
      for j in range( len( numofnets ) ):
          netnamej = re.sub( " ", "", networks[numofnets[j]] )
          netnamej = re.sub( "-", "", netnamej )
          A[i,j] = outdict[ netnamej ][ dfnImg == 1].mean()

  A = pd.DataFrame( A )
  A.columns = newnames
  A['networks']=newnames
  outdict['corr'] = A
  outdict['brainmask'] = bmask

  return outdict
