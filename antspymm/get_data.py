
__all__ = ['get_data','dewarp_imageset','super_res_mcimage','dipy_dti_recon',
    'segment_timeseries_by_meanvalue', 'wmh', 'neuromelanin',
    'resting_state_fmri_networks', 'dwi_deterministic_tracking']

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
from scipy.stats import pearsonr
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
import antspyt1w
import tensorflow as tf

from multiprocessing import Pool

DATA_PATH = os.path.expanduser('~/.antspymm/')

def get_data( name=None, force_download=False, version=8, target_extension='.csv' ):
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




def dewarp_imageset( image_list, initial_template=None,
    iterations=None, padding=0, target_idx=[0], **kwargs ):
    """
    Dewarp a set of images

    Makes simplifying heuristic decisions about how to transform an image set
    into an unbiased reference space.  Will handle plenty of decisions
    automatically so beware.  Computes an average shape space for the images
    and transforms them to that space.

    Arguments
    ---------
    image_list : list containing antsImages 2D, 3D or 4D

    initial_template : optional

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

    pw=[]
    for k in range(len(avglist[0].shape)):
        pw.append( padding )
    for k in range(len(avglist)):
        avglist[k] = ants.pad_image( avglist[k], pad_width=pw  )

    if initial_template is None:
        initial_template = avglist[0] * 0
        for k in range(len(avglist)):
            initial_template = initial_template + avglist[k]/len(avglist)

    if iterations is None:
        iterations = 2

    btp = ants.build_template(
        initial_template=initial_template,
        image_list=avglist,
        gradient_step=0.5, blending_weight=0.8,
        iterations=iterations, **kwargs )

    # last - warp all images to this frame
    mocoplist = []
    mocofdlist = []
    reglist = []
    for k in range(len(image_list)):
        if imagetype == 3:
            moco0 = ants.motion_correction( image=image_list[k], fixed=btp, type_of_transform='BOLDRigid' )
            mocoplist.append( moco0['motion_parameters'] )
            mocofdlist.append( moco0['FD'] )
            locavg = ants.slice_image( moco0['motion_corrected'], axis=3, idx=0 ) * 0.0
            for j in range(len(target_idx)):
                locavg = locavg + ants.slice_image( moco0['motion_corrected'], axis=3, idx=target_idx[j] )
            locavg = locavg * 1.0 / len(target_idx)
        else:
            locavg = image_list[k]
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
    poly_order=None,
    target_range=[0,1],
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

def t1_based_dwi_brain_extraction(
    t1w,
    dwi,
    b0_idx = None,
    transform='Rigid'
):
    """
    Map a t1-based brain extraction to b0 and return a mask and average b0

    Arguments
    ---------
    t1w : an antsImage probably but not necessarily T1-weighted

    dwi : an antsImage holding B0 and DWI

    b0_idx : the indices of the B0; if None, use segment_timeseries_by_meanvalue to guess

    transform : string Rigid or SyNBold

    Returns
    -------
    dictionary holding the avg_b0 and its mask

    Example
    -------
    >>> import antspymm
    """
    t1w_use = ants.iMath( t1w, "Normalize" )
    t1bxt = ants.threshold_image( t1w_use, 0.05, 1 ).iMath("FillHoles")
    if b0_idx is None:
        b0_idx = segment_timeseries_by_meanvalue( dwi )['highermeans']
    # first get the average b0
    if len( b0_idx ) > 1:
        b0_avg = ants.slice_image( dwi, axis=3, idx=b0_idx[0] ).iMath("Normalize")
        for n in range(1,len(b0_idx)):
            temp = ants.slice_image( dwi, axis=3, idx=b0_idx[n] )
            reg = ants.registration( b0_avg, temp, 'Rigid' )
            b0_avg = b0_avg + ants.iMath( reg['warpedmovout'], "Normalize")
    else:
        b0_avg = ants.slice_image( dwi, axis=3, idx=b0_idx[0] )
    b0_avg = ants.iMath(b0_avg,"Normalize")
    reg = ants.registration( b0_avg, t1w_use, transform, syn_metric='mattes', total_sigma=3.0, verbose=False )
    outmsk = ants.apply_transforms( b0_avg, t1bxt, reg['fwdtransforms'], interpolator='linear').threshold_image( 0.5, 1.0 )
    return  {
    'b0_avg':b0_avg,
    'b0_mask':outmsk }


def dipy_dti_recon(
    image,
    bvalsfn,
    bvecsfn,
    mask = None,
    b0_idx = None,
    motion_correct = False,
    mask_dilation = 0,
    average_b0 = None, 
    verbose=False ):
    """
    DiPy DTI reconstruction - following their own basic example

    Arguments
    ---------
    image : an antsImage holding B0 and DWI

    bvalsfn : bvalues  obtained by dipy read_bvals_bvecs or the values themselves

    bvecsfn : bvectors obtained by dipy read_bvals_bvecs or the values themselves

    mask : brain mask for the DWI/DTI reconstruction; if it is not in the same
        space as the image, we will resample directly to the image space.  This
        could lead to problems if the inputs are really incorrect.

    b0_idx : the indices of the B0; if None, use segment_timeseries_by_meanvalue to guess

    motion_correct : boolean

    mask_dilation : integer zero or more dilates the brain mask

    average_b0 : optional reference average b0; if it is not in the same
        space as the image, we will resample directly to the image space.  This
        could lead to problems if the inputs are really incorrect.

    verbose : boolean

    Returns
    -------
    dictionary holding the tensorfit, MD, FA and RGB images and motion parameters (optional)

    NOTE -- see dipy reorient_bvecs(gtab, affines, atol=1e-2):

    Example
    -------
    >>> import antspymm
    """

    from scipy.linalg import inv, polar
    from dipy.core.gradients import reorient_bvecs
    if b0_idx is None:
        b0_idx = segment_timeseries_by_meanvalue( image )['highermeans']

    if isinstance(bvecsfn, str):
        bvals, bvecs = read_bvals_bvecs( bvalsfn , bvecsfn   )
    else: # assume we already read them
        bvals = bvalsfn.copy()
        bvecs = bvecsfn.copy()
    gtab = gradient_table(bvals, bvecs)
    b0 = ants.slice_image( image, axis=3, idx=b0_idx[0] )
    FD = None
    motion_corrected = None
    maskedimage = None
    motion_parameters = None

    # first get the average images
    haveB0 = True
    if average_b0 is None:
        haveB0 = False
        average_b0 = ants.slice_image( image, axis=3, idx=0 ) * 0
        for myidx in b0_idx:
            b0 = ants.slice_image( image, axis=3, idx=myidx)
            average_b0 = average_b0 + b0
    else:
        average_b0 = ants.resample_image_to_target( average_b0, b0, interp_type='linear' )

    if mask is not None:
        mask = ants.resample_image_to_target( mask, b0, interp_type='nearestNeighbor' )

    average_dwi = average_b0.clone() * 0.0
    for myidx in range(image.shape[3]):
        b0 = ants.slice_image( image, axis=3, idx=myidx)
        average_dwi = average_dwi + ants.iMath( b0,'Normalize' )

    average_b0 = ants.iMath( average_b0, 'Normalize' )
    average_dwi = ants.iMath( average_dwi, 'Normalize' )

    bxtmod='bold'
    bxtmod='t2'
    get_mask = False
    if mask is None:
        get_mask = True
#        mask = antspynet.brain_extraction( average_dwi, 'flair' ).threshold_image(0.5,1).iMath("FillHoles").iMath("GetLargestComponent")
        mask = antspynet.brain_extraction( average_b0, bxtmod ).threshold_image(0.5,1).iMath("GetLargestComponent").morphology("close",2).iMath("FillHoles")

    maskdil = ants.iMath( mask, "MD", mask_dilation )

    if verbose:
        print("recon A",flush=True)

    # now extract the masked image data with or without motion correct
    if not motion_correct:
        maskedimage = []
        for myidx in range(image.shape[3]):
            b0 = ants.slice_image( image, axis=3, idx=myidx)
            maskedimage.append( b0 * maskdil )
        maskedimage = ants.list_to_ndimage( image, maskedimage )
        maskdata = maskedimage.numpy()
    if motion_correct:
        maskedimage = []
        for myidx in range(image.shape[3]):
                b0 = ants.slice_image( image, axis=3, idx=myidx)
                maskedimage.append( ants.iMath( b0,'Normalize' ) )
        maskedimage = ants.list_to_ndimage( image, maskedimage )
        moco0 = ants.motion_correction(
                image=maskedimage,
                fixed=average_b0,
                type_of_transform='Rigid',
                aff_metric='Mattes',
                aff_sampling=20,
                aff_smoothing_sigmas=(2,1,0),
                aff_shrink_factors=(3,2,1),
                aff_random_sampling_rate=0.5,
                grad_step=0.025,
                aff_iterations=(200,200,20) )
        motion_parameters = moco0['motion_parameters']
        FD = moco0['FD']
        maskedimage = []
        mocoimage = []
        dipymoco = np.zeros( [image.shape[3],3,3] )
        for myidx in range(image.shape[3]):
            dipymoco[myidx,:,:] = np.eye( 3 )
            if moco0['motion_parameters'][myidx] != 'NA':
                txparam = ants.read_transform(moco0['motion_parameters'][myidx][0] )
                txparam = ants.get_ants_transform_parameters(txparam)[0:9].reshape( [3,3])
                Rinv = inv( txparam )
                bvecs[myidx,:] = np.dot( Rinv, bvecs[myidx,:] )
            b0 = ants.slice_image( image, axis=3, idx=myidx)
            b0 = ants.apply_transforms( average_b0, b0, moco0['motion_parameters'][myidx] )
            mocoimage.append( b0 )
            maskedimage.append( b0 * maskdil )
        gtab = gradient_table(bvals, bvecs)
        if verbose:
            print("recon B",flush=True)
        motion_corrected = ants.list_to_ndimage( image, mocoimage )
        maskedimage = ants.list_to_ndimage( image, maskedimage )
        maskdata = maskedimage.numpy()
        if get_mask:
            if not haveB0:
                average_b0 = ants.slice_image( image, axis=3, idx=0 ) * 0
            average_dwi = average_b0.clone()
            for myidx in range(image.shape[3]):
                    b0 = ants.slice_image( image, axis=3, idx=myidx)
                    average_dwi = average_dwi + ants.iMath( b0,'Normalize' )
            for myidx in b0_idx:
                    b0 = ants.slice_image( image, axis=3, idx=myidx)
                    if not haveB0:
                        average_b0 = average_b0 + b0
            average_b0 = ants.iMath( average_b0, 'Normalize' )
            average_dwi = ants.iMath( average_dwi, 'Normalize' )
            # mask = antspynet.brain_extraction( average_dwi, 'flair' ).threshold_image(0.5,1).iMath("FillHoles").iMath("GetLargestComponent")
            mask = antspynet.brain_extraction( average_b0, bxtmod ).threshold_image(0.5,1).iMath("GetLargestComponent").morphology("close",2).iMath("FillHoles")

    if verbose:
        print("recon dti.TensorModel",flush=True)

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    if verbose:
        print("recon dti.TensorModel done",flush=True)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0

    MD1 = dti.mean_diffusivity(tenfit.evals)
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, tenfit.evecs)
    MD1 = ants.copy_image_info( b0, ants.from_numpy( MD1.astype(np.float32) ) )
    FA = ants.copy_image_info(  b0, ants.from_numpy( FA.astype(np.float32) ) )
    RGB = ants.from_numpy( RGB.astype(np.float32) )
    RGB0 = ants.copy_image_info( b0, ants.slice_image( RGB, axis=3, idx=0 ) )
    RGB1 = ants.copy_image_info( b0, ants.slice_image( RGB, axis=3, idx=1 ) )
    RGB2 = ants.copy_image_info( b0, ants.slice_image( RGB, axis=3, idx=2 ) )
    famask = antspynet.brain_extraction( FA, 'fa' ).threshold_image(0.5,1).iMath("GetLargestComponent").morphology("close",2).iMath("FillHoles")
    return {
        'tensormodel' : tenfit,
        'MD' : MD1,
        'FA' : FA,
        'RGB' : ants.merge_channels( [RGB0,RGB1,RGB2] ),
        'motion_corrected' : motion_corrected,
        'motion_corrected_masked' : maskedimage,
        'framewise_displacement' : FD,
        'motion_parameters':motion_parameters,
        'average_b0':average_b0,
        'average_dwi':average_dwi,
        'dwi_mask':mask,
        'famask':famask,
        'bvals':bvals,
        'bvecs':bvecs
        }

def joint_dti_recon(
    img_LR,
    bval_LR,
    bvec_LR,
    jhu_atlas,
    jhu_labels,
    srmodel = None,
    img_RL = None,
    bval_RL = None,
    bvec_RL = None,
    t1w = None,
    brain_mask = None,
    reference_image = None,
    motion_correct = False,
    dewarp_modality = 'FA',
    verbose = False ):
    """
    1. pass in subject data and 1mm JHU atlas/labels
    2. perform initial LR, RL reconstruction (2nd is optional) and motion correction (optional)
    3. dewarp the images using dewarp_modality or T1w
    4. apply dewarping to the original data
        ===> may want to apply SR at this step
    5. reconstruct DTI again
    6. label images and do registration
    7. return relevant outputs

    NOTE: RL images are optional; should pass t1w in this case.

    NOTE: the user may want to perform motion correction externally as this
    function does not rotate bvectors.

    Arguments
    ---------

    img_LR : an antsImage holding B0 and DWI LR acquisition

    bval_LR : bvalue filename LR

    bvec_LR : bvector filename LR

    jhu_atlas : atlas FA image

    jhu_labels : atlas labels

    srmodel : optional h5 (tensorflow) model

    img_RL : an antsImage holding B0 and DWI RL acquisition

    bval_RL : bvalue filename RL

    bvec_RL : bvector filename RL

    t1w : antsimage t1w neuroimage (brain-extracted)

    brain_mask : mask for the DWI - just 3D

    reference_image : the "target" image for the DWI (e.g. average B0)

    motion_correct : boolean

    dewarp_modality : string average_dwi, average_b0, MD or FA

    verbose : boolean

    Returns
    -------
    dictionary holding the mean_fa, its summary statistics via JHU labels,
        the JHU registration, the JHU labels, the dewarping dictionary and the
        dti reconstruction dictionaries.

    Example
    -------
    >>> import antspymm
    """

    if verbose:
        print("Recon DTI on OR images ...")


    # RL image
    mymd = 1
    recon_RL = None
    bval_RL = None
    bvec_RL = None

    temp = ants.get_average_of_timeseries( img_LR )
    maskInRightSpace = True
    if not brain_mask is None:
        maskInRightSpace = ants.image_physical_space_consistency( brain_mask, temp )
        if not maskInRightSpace :
            raise ValueError('not maskInRightSpace ... provided brain mask should be in DWI space;see  ants.get_average_of_timeseries(dwi) to find the right space')

    if img_RL is not None :
        recon_RL = dipy_dti_recon( img_RL, bval_RL, bvec_RL,
            mask = brain_mask, average_b0 = reference_image,
            motion_correct=motion_correct, mask_dilation=mymd )
        bval_RL = recon_RL['bvals']
        bvec_RL = recon_RL['bvecs']
        OR_RLFA = recon_RL['FA']

    recon_LR = dipy_dti_recon( img_LR, bval_LR, bvec_LR,
            mask = brain_mask, average_b0 = reference_image,
            motion_correct=motion_correct,
            mask_dilation=mymd )
    bval_LR = recon_LR['bvals']
    bvec_LR = recon_LR['bvecs']

    OR_LRFA = recon_LR['FA']

    if verbose:
        print("JHU initialization ...")

    JHU_atlas_aff = ants.registration(
        OR_LRFA * recon_LR['famask'],
        jhu_atlas, 'Affine' )['warpedmovout']
    JHU_atlas_aff_mask = ants.threshold_image( JHU_atlas_aff, 0.1, 2.0 ).iMath("GetLargestComponent").iMath("FillHoles").iMath("MD",20)
    JHU_atlas_aff = ants.crop_image( JHU_atlas_aff, JHU_atlas_aff_mask )

    synreg = None
    ts_LR_avg = recon_LR[dewarp_modality] * recon_LR['dwi_mask']
    ts_RL_avg = None

    t1wrig = None
    if t1w is not None:
        t1wtarget = recon_LR[ 'dwi_mask' ] * recon_LR[dewarp_modality]
        t1wrig = ants.registration( t1wtarget, t1w, 'Rigid' )['warpedmovout']

    if img_RL is not None:
        if dewarp_modality == 'FA':
            targeter = JHU_atlas_aff
        else:
            targeter = ts_LR_avg * recon_RL['dwi_mask']
        ts_RL_avg = recon_RL[dewarp_modality] * recon_RL['dwi_mask'] # ants.get_average_of_timeseries( recon_RL['motion_corrected'] )
        dwp_OR = dewarp_imageset(
            [ts_LR_avg, ts_RL_avg],
            initial_template=targeter,
            iterations = 5,
            syn_metric='CC', syn_sampling=2, reg_iterations=[20,100,100,20] )
    else:
        synreg = ants.registration(
            t1wrig,
            t1wtarget,
            'SyNOnly',
            total_sigma=0.0,
            # syn_metric='CC', syn_sampling=2,
            reg_iterations=[100,100,20],
            gradient_step=0.1 )
        dwp_OR ={
            'deformable_registrations':[synreg],
            'dewarpedmean':synreg['warpedmovout']
            }

    def concat_dewarp(
            refimg,
            originalDWI,
            physSpaceDWI,
            dwpTx,
            motion_parameters,
            motion_correct=True ):
        # apply the dewarping tx to the original dwi and reconstruct again
        # NOTE: refimg must be in the same space for this to work correctly
        # due to the use of ants.list_to_ndimage( originalDWI, dwpimage )
        dwpimage = []
        for myidx in range(originalDWI.shape[3]):
            b0 = ants.slice_image( originalDWI, axis=3, idx=myidx)
            concatx = dwpTx.copy()
            if motion_correct:
                concatx.append( motion_parameters[myidx][0] )
            warpedb0 = ants.apply_transforms( refimg, b0, concatx )
            dwpimage.append( warpedb0 )
        return ants.list_to_ndimage( physSpaceDWI, dwpimage )
 
    img_RLdwp = None
    img_LRdwp = concat_dewarp( dwp_OR['dewarpedmean'],
            img_LR,
            img_LR, # phys-space == original space
            dwp_OR['deformable_registrations'][0]['fwdtransforms'],
            recon_LR['motion_parameters'],
            motion_correct=motion_correct
            )
    if img_RL is not None:
        img_RLdwp = concat_dewarp( dwp_OR['dewarpedmean'],
            img_RL,
            img_LR, # phys-space != original space
            dwp_OR['deformable_registrations'][1]['fwdtransforms'],
            recon_RL['motion_parameters'],
            motion_correct=motion_correct
            )

    reg_its = [100,50,10]
    if srmodel is not None:
        reg_its = [100] + reg_its
        if img_RL is not None:
            if verbose:
                print("convert img_RL_dwp to img_RL_dwp_SR")
                img_RLdwp = super_res_mcimage( img_RLdwp, srmodel, verbose=verbose )
        if verbose:
            print("convert img_LR_dwp to img_LR_dwp_SR")
        img_LRdwp = super_res_mcimage( img_LRdwp, srmodel, verbose=verbose )

    if verbose:
        print("recon after distortion correction", flush=True)

    recon_RL_dewarp = None
    if img_RL is not None:
        recon_RL_dewarp = dipy_dti_recon( img_RLdwp, bval_RL, bvec_RL,
            mask = brain_mask, average_b0 = reference_image,
                motion_correct=False,
                mask_dilation=0 )

    recon_LR_dewarp = dipy_dti_recon( img_LRdwp, bval_LR, bvec_LR,
            mask = brain_mask, average_b0 = reference_image,
                motion_correct=False,
                mask_dilation=0, verbose=True )

    if verbose:
        print("recon done", flush=True)

    if img_RL is not None:
        reconFA = recon_RL_dewarp['FA'] * 0.5 + recon_LR_dewarp['FA'] * 0.5
        reconMD = recon_RL_dewarp['MD'] * 0.5 + recon_LR_dewarp['MD'] * 0.5
    else:
        reconFA = recon_LR_dewarp['FA']
        reconMD = recon_LR_dewarp['MD']

    if verbose:
        print("JHU reg",flush=True)

    OR_FA2JHUreg = ants.registration( reconFA, jhu_atlas,
        type_of_transform = 'SyN', syn_metric='CC', syn_sampling=2,
        reg_iterations=reg_its, verbose=False )
    OR_FA_jhulabels = ants.apply_transforms( reconFA, jhu_labels,
        OR_FA2JHUreg['fwdtransforms'], interpolator='genericLabel')

    df_FA_JHU_ORRL = antspyt1w.map_intensity_to_dataframe(
        'FA_JHU_labels_edited',
        reconFA,
        OR_FA_jhulabels)
    df_FA_JHU_ORRL_bfwide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
            {'df_FA_JHU_ORRL' : df_FA_JHU_ORRL},
            col_names = ['Mean'] )

    df_MD_JHU_ORRL = antspyt1w.map_intensity_to_dataframe(
        'FA_JHU_labels_edited',
        reconMD,
        OR_FA_jhulabels)
    df_MD_JHU_ORRL_bfwide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
            {'df_MD_JHU_ORRL' : df_MD_JHU_ORRL},
            col_names = ['Mean'] )

    return {
        'recon_fa':reconFA,
        'recon_fa_summary':df_FA_JHU_ORRL_bfwide,
        'recon_md':reconMD,
        'recon_md_summary':df_MD_JHU_ORRL_bfwide,
        'jhu_labels':OR_FA_jhulabels,
        'jhu_registration':OR_FA2JHUreg,
        'dtrecon_LR':recon_LR,
        'dtrecon_LR_dewarp':recon_LR_dewarp,
        'dtrecon_RL':recon_RL,
        'dtrecon_RL_dewarp':recon_RL_dewarp,
        'dewarping_object':dwp_OR,
        'dwi_LR_dewarped':img_LRdwp,
        'dwi_RL_dewarped':img_RLdwp,
        't1w_rigid':t1wrig,
        'bval_LR':bval_LR,
        'bvec_LR':bvec_LR,
        'bval_RL':bval_RL,
        'bvec_RL':bvec_RL,
    }



def middle_slice_snr( x, background_dilation=5 ):
    """

    Estimate signal to noise ratio (SNR) in 2D mid image from a 3D image.  
    Estimates noise from a background mask which is a 
    dilation of the foreground mask minus the foreground mask.
    Actually estimates the reciprocal of the coefficient of variation.

    Arguments
    ---------

    x : an antsImage

    background_dilation : integer - amount to dilate foreground mask

    """
    xshp = x.shape
    xmidslice = ants.slice_image( x, 2, int( xshp[2]/2 )  )
    xmidslice = ants.iMath( xmidslice - xmidslice.min(), "Normalize" )
    xmidslice = ants.n3_bias_field_correction( xmidslice )
    xmidslice = ants.n3_bias_field_correction( xmidslice )
    xmidslicemask = ants.threshold_image( xmidslice, "Otsu", 1 ).morphology("close",2).iMath("FillHoles")
    xbkgmask = ants.iMath( xmidslicemask, "MD", background_dilation ) - xmidslicemask
    signal = (xmidslice[ xmidslicemask == 1] ).mean()
    noise = (xmidslice[ xbkgmask == 1] ).std()
    return signal / noise

def foreground_background_snr( x, background_dilation=10, 
        erode_foreground=False):
    """

    Estimate signal to noise ratio (SNR) in an image.  
    Estimates noise from a background mask which is a 
    dilation of the foreground mask minus the foreground mask.
    Actually estimates the reciprocal of the coefficient of variation.

    Arguments
    ---------

    x : an antsImage

    background_dilation : integer - amount to dilate foreground mask

    erode_foreground : boolean - 2nd option which erodes the initial 
    foregound mask  to create a new foreground mask.  the background 
    mask is the initial mask minus the eroded mask.

    """
    xshp = x.shape
    xbc = ants.iMath( x - x.min(), "Normalize" )
    xbc = ants.n3_bias_field_correction( xbc )
    xmask = ants.threshold_image( xbc, "Otsu", 1 ).morphology("close",2).iMath("FillHoles")
    xbkgmask = ants.iMath( xmask, "MD", background_dilation ) - xmask
    fgmask = xmask
    if erode_foreground:
        fgmask = ants.iMath( xmask, "ME", background_dilation )
        xbkgmask = xmask - fgmask
    signal = (xbc[ fgmask == 1] ).mean()
    noise = (xbc[ xbkgmask == 1] ).std()
    return signal / noise

def quantile_snr( x, 
    lowest_quantile=0.01, 
    low_quantile=0.1, 
    high_quantile=0.5,
    highest_quantile=0.95 ):
    """

    Estimate signal to noise ratio (SNR) in an image.  
    Estimates noise from a background mask which is a 
    dilation of the foreground mask minus the foreground mask.
    Actually estimates the reciprocal of the coefficient of variation.

    Arguments
    ---------

    x : an antsImage

    lowest_quantile : float value < 1 and > 0

    low_quantile : float value < 1 and > 0

    high_quantile : float value < 1 and > 0

    highest_quantile : float value < 1 and > 0

    """
    import numpy as np
    xshp = x.shape
    xbc = ants.iMath( x - x.min(), "Normalize" )
    xbc = ants.n3_bias_field_correction( xbc )
    xbc = ants.iMath( xbc - xbc.min(), "Normalize" )
    y = xbc.numpy()
    ylowest = np.quantile( y[y>0], lowest_quantile )
    ylo = np.quantile( y[y>0], low_quantile )
    yhi = np.quantile( y[y>0], high_quantile )
    yhiest = np.quantile( y[y>0], highest_quantile )
    xbkgmask = ants.threshold_image( xbc, ylowest, ylo )
    fgmask = ants.threshold_image( xbc, yhi, yhiest )
    signal = (xbc[ fgmask == 1] ).mean()
    noise = (xbc[ xbkgmask == 1] ).std()
    return signal / noise

def mask_snr( x, background_mask, foreground_mask, bias_correct=True ):
    """

    Estimate signal to noise ratio (SNR) in an image using 
    a user-defined foreground and background mask.  
    Actually estimates the reciprocal of the coefficient of variation.

    Arguments
    ---------

    x : an antsImage

    background_mask : binary antsImage

    foreground_mask : binary antsImage

    bias_correct : boolean

    """
    import numpy as np
    xbc = ants.iMath( x - x.min(), "Normalize" )
    if bias_correct:
        xbc = ants.n3_bias_field_correction( xbc )
    xbc = ants.iMath( xbc - xbc.min(), "Normalize" )
    signal = (xbc[ foreground_mask == 1] ).mean()
    noise = (xbc[ background_mask == 1] ).std()
    return signal / noise

#    print( x + " mid-snr: " + str(middle_slice_snr( ants.image_read(x))))
# for x in fns:
#    print( x + " fg-snr: " + str(foreground_background_snr( ants.image_read(x),20)))
#    print( x + " mid-snr: " + str(middle_slice_snr( ants.image_read(x))))
# for x in fns:
#    qsnr = quantile_snr( ants.image_read(x),0.1,0.2,0.6,0.7)
#    print( x + " mid-snr: " + str( qsnr ) )


def dwi_deterministic_tracking(
    dwi,
    fa,
    bvals,
    bvecs,
    num_processes=1,
    mask=None,
    label_image = None,
    seed_labels = None,
    fa_thresh = 0.05,
    seed_density = 1,
    step_size = 0.15,
    peak_indices = None,
    verbose = False ):
    """

    Performs deterministic tractography from the DWI and returns a tractogram
    and path length data frame.

    Arguments
    ---------

    dwi : an antsImage holding DWI acquisition

    fa : an antsImage holding FA values

    bvals : bvalues

    bvecs : bvectors

    num_processes : number of subprocesses

    mask : mask within which to do tracking - if None, we will make a mask using the fa_thresh
        and the code ants.threshold_image( fa, fa_thresh, 2.0 ).iMath("GetLargestComponent")

    label_image : atlas labels

    seed_labels : list of label numbers from the atlas labels

    fa_thresh : 0.25 defaults

    seed_density : 1 default number of seeds per voxel

    step_size : for tracking

    peak_indices : pass these in, if they are previously estimated.  otherwise, will
        compute on the fly (slow)

    verbose : boolean

    Returns
    -------
    dictionary holding tracts and stateful object.

    Example
    -------
    >>> import antspymm
    """
    import os
    import re
    import nibabel as nib
    import numpy as np
    import ants
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.tracking import utils
    import dipy.reconst.dti as dti
    from dipy.segment.clustering import QuickBundles
    from dipy.tracking.utils import path_length
    if verbose:
        print("begin tracking",flush=True)
    dwi_img = dwi.to_nibabel()
    affine = dwi_img.affine
    if isinstance( bvals, str ) or isinstance( bvecs, str ):
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    gtab = gradient_table(bvals, bvecs)
    if mask is None:
        mask = ants.threshold_image( fa, fa_thresh, 2.0 ).iMath("GetLargestComponent")
    dwi_data = dwi_img.get_fdata()
    dwi_mask = mask.numpy() == 1
    dti_model = dti.TensorModel(gtab)
    if verbose:
        print("begin tracking fit",flush=True)
    dti_fit = dti_model.fit(dwi_data, mask=dwi_mask)  # This step may take a while
    evecs_img = dti_fit.evecs
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    stopping_criterion = ThresholdStoppingCriterion(fa.numpy(), fa_thresh)
    from dipy.data import get_sphere
    sphere = get_sphere('symmetric362')
    from dipy.direction import peaks_from_model
    if peak_indices is None:
        # problems with multi-threading ...
        # see https://github.com/dipy/dipy/issues/2519
        if verbose:
            print("begin peaks",flush=True)
        mynump=1
        if os.getenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
            mynump = os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']
        current_openblas = os.environ.get('OPENBLAS_NUM_THREADS', '')
        current_mkl = os.environ.get('MKL_NUM_THREADS', '')
        os.environ['DIPY_OPENBLAS_NUM_THREADS'] = current_openblas
        os.environ['DIPY_MKL_NUM_THREADS'] = current_mkl
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        peak_indices = peaks_from_model(
            model=dti_model,
            data=dwi_data,
            sphere=sphere,
            relative_peak_threshold=.5,
            min_separation_angle=25,
            mask=dwi_mask,
            npeaks=3, return_odf=False,
            return_sh=False, 
            parallel=int(mynump) > 1,
            num_processes=int(mynump)
            )
        if 'DIPY_OPENBLAS_NUM_THREADS' in os.environ:
            os.environ['OPENBLAS_NUM_THREADS'] = \
                os.environ.pop('DIPY_OPENBLAS_NUM_THREADS', '')
            if os.environ['OPENBLAS_NUM_THREADS'] in ['', None]:
                os.environ.pop('OPENBLAS_NUM_THREADS', '')
        if 'DIPY_MKL_NUM_THREADS' in os.environ:
            os.environ['MKL_NUM_THREADS'] = \
                os.environ.pop('DIPY_MKL_NUM_THREADS', '')
            if os.environ['MKL_NUM_THREADS'] in ['', None]:
                os.environ.pop('MKL_NUM_THREADS', '')

    if label_image is None or seed_labels is None:
        seed_mask = fa.numpy().copy()
        seed_mask[seed_mask >= fa_thresh] = 1
        seed_mask[seed_mask < fa_thresh] = 0
    else:
        labels = label_image.numpy()
        seed_mask = labels * 0
        for u in seed_labels:
            seed_mask[ labels == u ] = 1
    seeds = utils.seeds_from_mask(seed_mask, affine=affine, density=seed_density)
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines
    if verbose:
        print("streamlines begin ...", flush=True)
    streamlines_generator = LocalTracking(
        peak_indices, stopping_criterion, seeds, affine=affine, step_size=step_size)
    streamlines = Streamlines(streamlines_generator)
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram
    sft = StatefulTractogram(streamlines, dwi_img, Space.RASMM)
    if verbose:
        print("streamlines done", flush=True)
    return {
          'tractogram': sft,
          'streamlines': streamlines,
          'peak_indices': peak_indices
          }



def dwi_closest_peak_tracking(
    dwi,
    fa,
    bvals,
    bvecs,
    num_processes=1,
    mask=None,
    label_image = None,
    seed_labels = None,
    fa_thresh = 0.05,
    seed_density = 1,
    step_size = 0.15,
    peak_indices = None,
    verbose = False ):
    """

    Performs deterministic tractography from the DWI and returns a tractogram
    and path length data frame.

    Arguments
    ---------

    dwi : an antsImage holding DWI acquisition

    fa : an antsImage holding FA values

    bvals : bvalues

    bvecs : bvectors

    num_processes : number of subprocesses

    mask : mask within which to do tracking - if None, we will make a mask using the fa_thresh
        and the code ants.threshold_image( fa, fa_thresh, 2.0 ).iMath("GetLargestComponent")

    label_image : atlas labels

    seed_labels : list of label numbers from the atlas labels

    fa_thresh : 0.25 defaults

    seed_density : 1 default number of seeds per voxel

    step_size : for tracking

    peak_indices : pass these in, if they are previously estimated.  otherwise, will
        compute on the fly (slow)

    verbose : boolean

    Returns
    -------
    dictionary holding tracts and stateful object.

    Example
    -------
    >>> import antspymm
    """
    import os
    import re
    import nibabel as nib
    import numpy as np
    import ants
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.tracking import utils
    import dipy.reconst.dti as dti
    from dipy.segment.clustering import QuickBundles
    from dipy.tracking.utils import path_length
    from dipy.core.gradients import gradient_table
    from dipy.data import small_sphere
    from dipy.direction import BootDirectionGetter, ClosestPeakDirectionGetter
    from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                    auto_response_ssst)
    from dipy.reconst.shm import CsaOdfModel
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

    if verbose:
        print("begin tracking",flush=True)
    dwi_img = dwi.to_nibabel()
    affine = dwi_img.affine
    if isinstance( bvals, str ) or isinstance( bvecs, str ):
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    gtab = gradient_table(bvals, bvecs)
    if mask is None:
        mask = ants.threshold_image( fa, fa_thresh, 2.0 ).iMath("GetLargestComponent")
    dwi_data = dwi_img.get_fdata()
    dwi_mask = mask.numpy() == 1


    response, ratio = auto_response_ssst(gtab, dwi_data, roi_radii=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_fit = csd_model.fit(dwi_data, mask=dwi_mask)
    csa_model = CsaOdfModel(gtab, sh_order=6)
    gfa = csa_model.fit(dwi_data, mask=dwi_mask).gfa
    stopping_criterion = ThresholdStoppingCriterion(gfa, .25)


    if label_image is None or seed_labels is None:
        seed_mask = fa.numpy().copy()
        seed_mask[seed_mask >= fa_thresh] = 1
        seed_mask[seed_mask < fa_thresh] = 0
    else:
        labels = label_image.numpy()
        seed_mask = labels * 0
        for u in seed_labels:
            seed_mask[ labels == u ] = 1
    seeds = utils.seeds_from_mask(seed_mask, affine=affine, density=seed_density)
    if verbose:
        print("streamlines begin ...", flush=True)

    pmf = csd_fit.odf(small_sphere).clip(min=0)
    if verbose:
        print("ClosestPeakDirectionGetter begin ...", flush=True)
    peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                sphere=small_sphere)
    if verbose:
        print("local tracking begin ...", flush=True)
    streamlines_generator = LocalTracking(peak_dg, stopping_criterion, seeds,
                                            affine, step_size=.5)
    streamlines = Streamlines(streamlines_generator)
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram
    sft = StatefulTractogram(streamlines, dwi_img, Space.RASMM)
    if verbose:
        print("streamlines done", flush=True)
    return {
          'tractogram': sft,
          'streamlines': streamlines
          }

def dwi_streamline_pairwise_connectivity( streamlines, label_image, labels_to_connect=[1,None], verbose=False ):
    """

    Return streamlines connecting all of the regions in the label set. Ideal
    for just 2 regions.

    Arguments
    ---------

    streamlines : streamline object from dipy

    label_image : atlas labels

    labels_to_connect : list of 2 labels or [label,None]

    verbose : boolean

    Returns
    -------
    the subset of streamlines and a streamline count

    Example
    -------
    >>> import antspymm
    """
    from dipy.tracking.streamline import Streamlines
    keep_streamlines = Streamlines()
    affine = label_image.to_nibabel().affine
    lin_T, offset = utils._mapping_to_voxel(affine)
    label_image_np = label_image.numpy()
    def check_it( sl, target_label, label_image, index, full=False ):
        if full:
            maxind=sl.shape[0]
            for index in range(maxind):
                pt = utils._to_voxel_coordinates(sl[index,:], lin_T, offset)
                mylab = (label_image[ pt[0], pt[1], pt[2] ]).astype(int)
                if mylab == target_label[0] or mylab == target_label[1]:
                    return { 'ok': True, 'label':mylab }
        else:
            pt = utils._to_voxel_coordinates(sl[index,:], lin_T, offset)
            mylab = (label_image[ pt[0], pt[1], pt[2] ]).astype(int)
            if mylab == target_label[0] or mylab == target_label[1]:
                return { 'ok': True, 'label':mylab }
        return { 'ok': False, 'label':None }
    ct=0
    for k in range( len( streamlines ) ):
        sl = streamlines[k]
        mycheck = check_it( sl, labels_to_connect, label_image_np, index=0, full=True )
        if mycheck['ok']:
            otherind=1
            if mycheck['label'] == labels_to_connect[1]:
                otherind=0
            lsl = len( sl )-1
            pt = utils._to_voxel_coordinates(sl[lsl,:], lin_T, offset)
            mylab_end = (label_image_np[ pt[0], pt[1], pt[2] ]).astype(int)
            accept_point = mylab_end == labels_to_connect[otherind]
            if verbose and accept_point:
                print( mylab_end )
            if labels_to_connect[1] is None:
                accept_point = mylab_end != 0
            if accept_point:
                keep_streamlines.append(sl)
                ct=ct+1
    return { 'streamlines': keep_streamlines, 'count': ct }

def dwi_streamline_pairwise_connectivity_old(
    streamlines,
    label_image,
    exclusion_label = None,
    verbose = False ):
    import os
    import re
    import nibabel as nib
    import numpy as np
    import ants
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.tracking import utils
    import dipy.reconst.dti as dti
    from dipy.segment.clustering import QuickBundles
    from dipy.tracking.utils import path_length
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines
    volUnit = np.prod( ants.get_spacing( label_image ) )
    labels = label_image.numpy()
    affine = label_image.to_nibabel().affine
    import numpy as np
    from dipy.io.image import load_nifti_data, load_nifti, save_nifti
    import pandas as pd
    ulabs = np.unique( labels[ labels > 0 ] )
    if exclusion_label is not None:
        ulabs = ulabs[ ulabs != exclusion_label ]
        exc_slice = labels == exclusion_label
    if verbose:
        print("Begin connectivity")
    tracts = []
    for k in range(len(ulabs)):
        cc_slice = labels == ulabs[k]
        cc_streamlines = utils.target(streamlines, affine, cc_slice)
        cc_streamlines = Streamlines(cc_streamlines)
        if exclusion_label is not None:
            cc_streamlines = utils.target(cc_streamlines, affine, exc_slice, include=False)
            cc_streamlines = Streamlines(cc_streamlines)
        for j in range(len(ulabs)):
            cc_slice2 = labels == ulabs[j]
            cc_streamlines2 = utils.target(cc_streamlines, affine, cc_slice2)
            cc_streamlines2 = Streamlines(cc_streamlines2)
            if exclusion_label is not None:
                cc_streamlines2 = utils.target(cc_streamlines2, affine, exc_slice, include=False)
                cc_streamlines2 = Streamlines(cc_streamlines2)
            tracts.append( cc_streamlines2 )
        if verbose:
            print("end connectivity")
    return {
          'pairwise_tracts': tracts
          }


def dwi_streamline_connectivity(
    streamlines,
    label_image,
    label_dataframe,
    verbose = False ):
    """

    Summarize network connetivity of the input streamlines between all of the
    regions in the label set.

    Arguments
    ---------

    streamlines : streamline object from dipy

    label_image : atlas labels

    label_dataframe : pandas dataframe containing descriptions for the labels in antspy style (Label,Description columns)

    verbose : boolean

    Returns
    -------
    dictionary holding summary connection statistics in wide format and matrix format.

    Example
    -------
    >>> import antspymm
    """
    import os
    import re
    import nibabel as nib
    import numpy as np
    import ants
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.tracking import utils
    import dipy.reconst.dti as dti
    from dipy.segment.clustering import QuickBundles
    from dipy.tracking.utils import path_length
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines
    import os
    import re
    import nibabel as nib
    import numpy as np
    import ants
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.tracking import utils
    import dipy.reconst.dti as dti
    from dipy.segment.clustering import QuickBundles
    from dipy.tracking.utils import path_length
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines
    volUnit = np.prod( ants.get_spacing( label_image ) )
    labels = label_image.numpy()
    affine = label_image.to_nibabel().affine
    import numpy as np
    from dipy.io.image import load_nifti_data, load_nifti, save_nifti
    import pandas as pd
    ulabs = label_dataframe['Label']
    labels_to_connect = ulabs[ulabs > 0]
    Ctdf = None
    lin_T, offset = utils._mapping_to_voxel(affine)
    label_image_np = label_image.numpy()
    def check_it( sl, target_label, label_image, index, not_label = None ):
        pt = utils._to_voxel_coordinates(sl[index,:], lin_T, offset)
        mylab = (label_image[ pt[0], pt[1], pt[2] ]).astype(int)
        if not_label is None:
            if ( mylab == target_label ).sum() > 0 :
                return { 'ok': True, 'label':mylab }
        else:
            if ( mylab == target_label ).sum() > 0 and ( mylab == not_label ).sum() == 0:
                return { 'ok': True, 'label':mylab }
        return { 'ok': False, 'label':None }
    ct=0
    which = lambda lst:list(np.where(lst)[0])
    myCount = np.zeros( [len(ulabs),len(ulabs)])
    for k in range( len( streamlines ) ):
            sl = streamlines[k]
            mycheck = check_it( sl, labels_to_connect, label_image_np, index=0 )
            if mycheck['ok']:
                exclabel=mycheck['label']
                lsl = len( sl )-1
                mycheck2 = check_it( sl, labels_to_connect, label_image_np, index=lsl, not_label=exclabel )
                if mycheck2['ok']:
                    myCount[ulabs == mycheck['label'],ulabs == mycheck2['label']]+=1
                    ct=ct+1
    Ctdf = label_dataframe.copy()
    for k in range(len(ulabs)):
            nn3 = "CnxCount"+str(k).zfill(3)
            Ctdf.insert(Ctdf.shape[1], nn3, myCount[k,:] )
    Ctdfw = antspyt1w.merge_hierarchical_csvs_to_wide_format( { 'networkc': Ctdf },  Ctdf.keys()[2:Ctdf.shape[1]] )
    return { 'connectivity_matrix' :  myCount, 'connectivity_wide' : Ctdfw }

def dwi_streamline_connectivity_old(
    streamlines,
    label_image,
    label_dataframe,
    verbose = False ):
    """

    Summarize network connetivity of the input streamlines between all of the
    regions in the label set.

    Arguments
    ---------

    streamlines : streamline object from dipy

    label_image : atlas labels

    label_dataframe : pandas dataframe containing descriptions for the labels in antspy style (Label,Description columns)

    verbose : boolean

    Returns
    -------
    dictionary holding summary connection statistics in wide format and matrix format.

    Example
    -------
    >>> import antspymm
    """

    if verbose:
        print("streamline connections ...")

    import os
    import re
    import nibabel as nib
    import numpy as np
    import ants
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.tracking import utils
    import dipy.reconst.dti as dti
    from dipy.segment.clustering import QuickBundles
    from dipy.tracking.utils import path_length
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines

    volUnit = np.prod( ants.get_spacing( label_image ) )
    labels = label_image.numpy()
    affine = label_image.to_nibabel().affine

    if verbose:
        print("path length begin ... volUnit = " + str( volUnit ) )
    import numpy as np
    from dipy.io.image import load_nifti_data, load_nifti, save_nifti
    import pandas as pd
    ulabs = label_dataframe['Label']
    pathLmean = np.zeros( [len(ulabs)])
    pathLtot = np.zeros( [len(ulabs)])
    pathCt = np.zeros( [len(ulabs)])
    for k in range(len(ulabs)):
        cc_slice = labels == ulabs[k]
        cc_streamlines = utils.target(streamlines, affine, cc_slice)
        cc_streamlines = Streamlines(cc_streamlines)
        if len(cc_streamlines) > 0:
            wmpl = path_length(cc_streamlines, affine, cc_slice)
            mean_path_length = wmpl[wmpl>0].mean()
            total_path_length = wmpl[wmpl>0].sum()
            pathLmean[int(k)] = mean_path_length
            pathLtot[int(k)] = total_path_length
            pathCt[int(k)] = len(cc_streamlines) * volUnit

    # convert paths to data frames
    pathdf = label_dataframe.copy()
    pathdf.insert(pathdf.shape[1], "mean_path_length", pathLmean )
    pathdf.insert(pathdf.shape[1], "total_path_length", pathLtot )
    pathdf.insert(pathdf.shape[1], "streamline_count", pathCt )
    pathdfw =antspyt1w.merge_hierarchical_csvs_to_wide_format(
        {path_length:pathdf }, ['mean_path_length', 'total_path_length', 'streamline_count'] )
    allconnexwide = pathdfw

    if verbose:
        print("path length done ...")

    Mdfw = None
    Tdfw = None
    Mdf = None
    Tdf = None
    Ctdf = None
    Ctdfw = None
    if True:
        if verbose:
            print("Begin connectivity")
        M = np.zeros( [len(ulabs),len(ulabs)])
        T = np.zeros( [len(ulabs),len(ulabs)])
        myCount = np.zeros( [len(ulabs),len(ulabs)])
        for k in range(len(ulabs)):
            cc_slice = labels == ulabs[k]
            cc_streamlines = utils.target(streamlines, affine, cc_slice)
            cc_streamlines = Streamlines(cc_streamlines)
            for j in range(len(ulabs)):
                cc_slice2 = labels == ulabs[j]
                cc_streamlines2 = utils.target(cc_streamlines, affine, cc_slice2)
                cc_streamlines2 = Streamlines(cc_streamlines2)
                if len(cc_streamlines2) > 0 :
                    wmpl = path_length(cc_streamlines2, affine, cc_slice2)
                    mean_path_length = wmpl[wmpl>0].mean()
                    total_path_length = wmpl[wmpl>0].sum()
                    M[int(j),int(k)] = mean_path_length
                    T[int(j),int(k)] = total_path_length
                    myCount[int(j),int(k)] = len( cc_streamlines2 ) * volUnit
        if verbose:
            print("end connectivity")
        Mdf = label_dataframe.copy()
        Tdf = label_dataframe.copy()
        Ctdf = label_dataframe.copy()
        for k in range(len(ulabs)):
            nn1 = "CnxMeanPL"+str(k).zfill(3)
            nn2 = "CnxTotPL"+str(k).zfill(3)
            nn3 = "CnxCount"+str(k).zfill(3)
            Mdf.insert(Mdf.shape[1], nn1, M[k,:] )
            Tdf.insert(Tdf.shape[1], nn2, T[k,:] )
            Ctdf.insert(Ctdf.shape[1], nn3, myCount[k,:] )
        Mdfw = antspyt1w.merge_hierarchical_csvs_to_wide_format( { 'networkm' : Mdf },  Mdf.keys()[2:Mdf.shape[1]] )
        Tdfw = antspyt1w.merge_hierarchical_csvs_to_wide_format( { 'networkt' : Tdf },  Tdf.keys()[2:Tdf.shape[1]] )
        Ctdfw = antspyt1w.merge_hierarchical_csvs_to_wide_format( { 'networkc': Ctdf },  Ctdf.keys()[2:Ctdf.shape[1]] )
        allconnexwide = pd.concat( [
            pathdfw,
            Mdfw,
            Tdfw,
            Ctdfw ], axis=1 )

    return {
          'connectivity': allconnexwide,
          'connectivity_matrix_mean': Mdf,
          'connectivity_matrix_total': Tdf,
          'connectivity_matrix_count': Ctdf
          }


def hierarchical_modality_summary(
    target_image,
    hier,
    transformlist,
    modality_name,
    return_keys = ["Mean","Volume"],
    verbose = False ):
    """

    Use output of antspyt1w.hierarchical to summarize a modality

    Arguments
    ---------

    target_image : the image to summarize - should be brain extracted

    hier : dictionary holding antspyt1w.hierarchical output

    transformlist : spatial transformations mapping from T1 to this modality (e.g. from ants.registration)

    modality_name : adds the modality name to the data frame columns

    return_keys = ["Mean","Volume"] keys to return

    verbose : boolean

    Returns
    -------
    data frame holding summary statistics in wide format

    Example
    -------
    >>> import antspymm
    """
    dfout = pd.DataFrame()
    def myhelper( target_image, seg, mytx, mapname, modname, mydf, extra='', verbose=False ):
        if verbose:
            print( mapname )
        cortmapped = ants.apply_transforms(
            target_image,
            seg,
            mytx, interpolator='nearestNeighbor' )
        mapped = antspyt1w.map_intensity_to_dataframe(
            mapname,
            target_image,
            cortmapped)
        mapped.iloc[:,1] = modname + '_' + extra + mapped.iloc[:,1]
        mappedw = antspyt1w.merge_hierarchical_csvs_to_wide_format(
            { 'x' : mapped},
            col_names = return_keys )
        if verbose:
            print( mappedw.keys() )
        if mydf.shape[0] > 0:
            mydf = pd.concat( [ mydf, mappedw], axis=1 )
        else:
            mydf = mappedw
        return mydf
    if hier['dkt_parc']['dkt_cortex'] is not None:
        dfout = myhelper( target_image, hier['dkt_parc']['dkt_cortex'], transformlist,
            "dkt", modality_name, dfout, extra='', verbose=verbose )
    if hier['deep_cit168lab'] is not None:
        dfout = myhelper( target_image, hier['deep_cit168lab'], transformlist,
            "CIT168_Reinf_Learn_v1_label_descriptions_pad", modality_name, dfout, extra='deep_', verbose=verbose )
    if hier['cit168lab'] is not None:
        dfout = myhelper( target_image, hier['cit168lab'], transformlist,
            "CIT168_Reinf_Learn_v1_label_descriptions_pad", modality_name, dfout, extra='', verbose=verbose  )
    if hier['bf'] is not None:
        dfout = myhelper( target_image, hier['bf'], transformlist,
            "nbm3CH13", modality_name, dfout, extra='', verbose=verbose  )
    # if hier['mtl'] is not None:
    #    dfout = myhelper( target_image, hier['mtl'], reg,
    #        "mtl_description", modality_name, dfout, extra='', verbose=verbose  )
    return dfout


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



def neuromelanin( list_nm_images, t1, t1_head, t1lab, brain_stem_dilation=8,
    bias_correct=True,
    denoise=1,
    srmodel=None, verbose=False ) :

  """
  Outputs the averaged and registered neuromelanin image, and neuromelanin labels

  Arguments
  ---------
  list_nm_image : list of ANTsImages
    list of neuromenlanin repeat images

  t1 : ANTsImage
    input 3-D T1 brain image

  t1_head : ANTsImage
    input 3-D T1 head image

  t1lab : ANTsImage
    t1 labels that will be propagated to the NM

  brain_stem_dilation : integer default 8
    dilates the brain stem mask to better match coverage of NM

  bias_correct : boolean

  denoise : None or integer

  srmodel : None -- this is a work in progress feature, probably not optimal

  verbose : boolean

  Returns
  ---------
  Averaged and registered neuromelanin image and neuromelanin labels and wide csv

  """

  fnt=os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni.nii.gz" )
  fntbst=os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni_brainstem.nii.gz")
  fnslab=os.path.expanduser("~/.antspyt1w/CIT168_MT_Slab_adni.nii.gz")
  fntseg=os.path.expanduser("~/.antspyt1w/det_atlas_25_pad_LR_adni.nii.gz")

  template = ants.image_read( fnt )
  templatebstem = ants.image_read( fntbst ).threshold_image( 1, 1000 )
  # reg = ants.registration( t1, template, 'antsRegistrationSyNQuickRepro[s]' )
  reg = ants.registration( t1, template, 'SyN' )
  # map brain stem and slab to t1 for neuromelanin processing
  bstem2t1 = ants.apply_transforms( t1, templatebstem,
    reg['fwdtransforms'],
    interpolator='nearestNeighbor' ).iMath("MD",1)
  slab2t1 = ants.apply_transforms( t1, ants.image_read( fnslab ),
    reg['fwdtransforms'], interpolator = 'nearestNeighbor')
  bstem2t1 = ants.crop_image( bstem2t1, slab2t1 )
  cropper = ants.decrop_image( bstem2t1, slab2t1 ).iMath("MD",brain_stem_dilation)

  # Average images in image_list
  nm_avg = list_nm_images[0]*0.0
  for k in range(len( list_nm_images )):
    if denoise is not None:
        list_nm_images[k] = ants.denoise_image( list_nm_images[k],
            shrink_factor=1,
            p=denoise,
            r=denoise+1,
            noise_model='Gaussian' )
    if bias_correct :
        n4mask = ants.threshold_image( ants.iMath(list_nm_images[k], "Normalize" ), 0.05, 1 )
        list_nm_images[k] = ants.n4_bias_field_correction( list_nm_images[k], mask=n4mask )
    nm_avg = nm_avg + ants.resample_image_to_target( list_nm_images[k], nm_avg ) / len( list_nm_images )

  if verbose:
      print("Register each nm image in list_nm_images to the averaged nm image (avg)")
  nm_avg_new = nm_avg * 0.0
  txlist = []
  for k in range(len( list_nm_images )):
    current_image = ants.registration( list_nm_images[k], nm_avg, type_of_transform = 'Rigid' )
    txlist.append( current_image['fwdtransforms'][0] )
    current_image = current_image['warpedfixout']
    nm_avg_new = nm_avg_new + current_image / len( list_nm_images )
  nm_avg = nm_avg_new

  t1c = ants.crop_image( t1_head, slab2t1 ).iMath("Normalize")
  slabreg = ants.registration( nm_avg, t1c, 'Rigid' )
  labels2nm = ants.apply_transforms( nm_avg, t1lab, slabreg['fwdtransforms'],
    interpolator = 'genericLabel' )
  cropper2nm = ants.apply_transforms( nm_avg, cropper, slabreg['fwdtransforms'], interpolator='nearestNeighbor' )
  nm_avg_cropped = ants.crop_image( nm_avg, cropper2nm )

  if verbose:
      print("now map these labels to each individual nm")
  crop_mask_list = []
  crop_nm_list = []
  for k in range(len( list_nm_images )):
      concattx = []
      concattx.append( txlist[k] )
      concattx.append( slabreg['fwdtransforms'][0] )
      cropmask = ants.apply_transforms( list_nm_images[k], cropper,
        concattx, interpolator = 'nearestNeighbor' )
      crop_mask_list.append( cropmask )
      temp = ants.crop_image( list_nm_images[k], cropmask )
      crop_nm_list.append( temp )

  if srmodel is not None:
      if verbose:
          print( " start sr " + str(len( crop_nm_list )) )
      for k in range(len( crop_nm_list )):
          if verbose:
              print( " do sr " + str(k) )
              print( crop_nm_list[k] )
          crop_nm_list[k] = antspynet.apply_super_resolution_model_to_image(
                crop_nm_list[k], srmodel, target_range=[0,1], regression_order=None )

  nm_avg_cropped = crop_nm_list[0]*0.0
  if verbose:
      print( "cropped average" )
      print( nm_avg_cropped )
  for k in range(len( crop_nm_list )):
      nm_avg_cropped = nm_avg_cropped + ants.apply_transforms( nm_avg_cropped,
        crop_nm_list[k], txlist[k] ) / len( crop_nm_list )
  for loop in range( 3 ):
      nm_avg_cropped_new = nm_avg_cropped * 0.0
      for k in range(len( crop_nm_list )):
            myreg = ants.registration(
                ants.iMath(nm_avg_cropped,"Normalize"),
                ants.iMath(crop_nm_list[k],"Normalize"),
                'antsRegistrationSyNRepro[r]' )
            warpednext = ants.apply_transforms(
                nm_avg_cropped_new,
                crop_nm_list[k],
                myreg['fwdtransforms'] )
            nm_avg_cropped_new = nm_avg_cropped_new + warpednext
      nm_avg_cropped = nm_avg_cropped_new / len( crop_nm_list )

  slabreg = ants.registration( nm_avg_cropped, t1c, 'Rigid',
    initial_transform=slabreg['fwdtransforms'][0], verbose=verbose )

  labels2nm = ants.apply_transforms( nm_avg_cropped, t1lab,
        slabreg['fwdtransforms'], interpolator='nearestNeighbor' )

  if verbose:
      print( "map summary measurements to wide format" )
  nmdf = antspyt1w.map_intensity_to_dataframe(
          'CIT168_Reinf_Learn_v1_label_descriptions_pad',
          nm_avg_cropped,
          labels2nm)
  if verbose:
      print( "merge to wide format" )
  nmdf_wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
              {'NM' : nmdf},
              col_names = ['Mean'] )
  if verbose:
      print( "nm done" )

  rr_mask = ants.mask_image( labels2nm, labels2nm, [33,34] , binarize=True )
  sn_mask = ants.mask_image( labels2nm, labels2nm, [7,9,23,25] , binarize=True )
  nmavgsnr = mask_snr( nm_avg_cropped, rr_mask, sn_mask, bias_correct = False )

  snavg = nm_avg_cropped[ sn_mask == 1].mean()
  rravg = nm_avg_cropped[ rr_mask == 1].mean()
  snstd = nm_avg_cropped[ sn_mask == 1].std()
  rrstd = nm_avg_cropped[ rr_mask == 1].std()

  return{
      'NM_avg' : nm_avg,
      'NM_avg_cropped' : nm_avg_cropped,
      'NM_labels': labels2nm,
      'NM_cropped': crop_nm_list,
      'NM_midbrainROI': cropper2nm,
      'NM_dataframe': nmdf,
      'NM_dataframe_wide': nmdf_wide,
      't1_to_NM': slabreg['warpedmovout'],
      't1_to_NM_transform' : slabreg['fwdtransforms'],
      'NM_avg_signaltonoise' : nmavgsnr,
      'NM_avg_substantianigra' : snavg,
      'NM_std_substantianigra' : snstd,
      'NM_avg_refregion' : rravg,
      'NM_std_refregion' : rrstd
       }

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

  A = np.zeros( ( len( numofnets ) , len( numofnets ) ) )
  A_wide = np.zeros( ( 1, len( numofnets ) * len( numofnets ) ) )
  newnames=[]
  newnames_wide=[]
  ct = 0
  for i in range( len( numofnets ) ):
      netnamei = re.sub( " ", "", networks[numofnets[i]] )
      netnamei = re.sub( "-", "", netnamei )
      newnames.append( netnamei  )
      binmask = ants.threshold_image( outdict[ netnamei ], 0.2, 1.0 )
      ww = np.where( powers_areal_mni_itk['SystemName'] == networks[numofnets[i]] )[0]
      dfnImg = ants.make_points_image(pts2bold.iloc[ww,:3].values, bmask, radius=1).threshold_image( 1, 1e9 )
      for j in range( len( numofnets ) ):
          netnamej = re.sub( " ", "", networks[numofnets[j]] )
          netnamej = re.sub( "-", "", netnamej )
          newnames_wide.append( netnamei + "_2_" + netnamej )
          A[i,j] = outdict[ netnamej ][ dfnImg == 1].mean()
          A_wide[0,ct] = A[i,j]
          ct=ct+1

  A = pd.DataFrame( A )
  A.columns = newnames
  A['networks']=newnames
  A_wide = pd.DataFrame( A_wide )
  A_wide.columns = newnames_wide
  outdict['corr'] = A
  outdict['corr_wide'] = A_wide
  outdict['brainmask'] = bmask

  rsfNuisance = pd.DataFrame( nuisance )
  rsfNuisance['FD']=dwp['FD'][dwpind]

  outdict['nuisance'] = rsfNuisance
  outdict['FD_max'] = rsfNuisance['FD'].max()
  outdict['FD_mean'] = rsfNuisance['FD'].mean()

  return outdict



def write_bvals_bvecs(bvals, bvecs, prefix ):
    ''' Write FSL FDT bvals and bvecs files

    adapted from dipy.external code

    Parameters
    -------------
    bvals : (N,) sequence
       Vector with diffusion gradient strength (one per diffusion
       acquisition, N=no of acquisitions)
    bvecs : (N, 3) array-like
       diffusion gradient directions
    prefix : string
       path to write FDT bvals, bvecs text files
       None results in current working directory.
    '''
    _VAL_FMT = '   %e'
    bvals = tuple(bvals)
    bvecs = np.asarray(bvecs)
    bvecs[np.isnan(bvecs)] = 0
    N = len(bvals)
    fname = prefix + '.bval'
    fmt = _VAL_FMT * N + '\n'
    open(fname, 'wt').write(fmt % bvals)
    fname = prefix + '.bvec'
    bvf = open(fname, 'wt')
    for dim_vals in bvecs.T:
        bvf.write(fmt % tuple(dim_vals))

def crop_mcimage( x, mask ):
    """ 
    crop a time series (4D) image by a 3D mask

    Parameters
    -------------

    x : raw image

    mask  : mask for cropping

    """ 
    cropmask = ants.crop_image( mask, mask )
    myorig = list( ants.get_origin(cropmask) )
    myorig.append( ants.get_origin( x )[3] )
    croplist = []
    if len(x.shape) > 3:
        for k in range(x.shape[3]):
            temp = ants.slice_image( x, axis=3, idx=k )
            croplist.append( ants.crop_image( temp, mask ) )
        temp = ants.list_to_ndimage( x, croplist )
        temp.set_origin( myorig )
        return temp
    else:
        return( ants.crop_image( x, mask ) )


def mm( 
    t1_image, 
    hier,
    rsf_image=None, 
    flair_image=None, 
    nm_image_list=None, 
    dw_image=None, bvals=None, bvecs=None, 
    srmodel=None,
    do_tractography = False,
    do_kk = False,
    do_normalization = True, 
    verbose = False ):
    """ 
    Multiple modality processing and normalization

    aggregates modality-specific processing under one roof.  see individual 
    modality specific functions for details.

    Parameters
    -------------

    t1_image : raw t1 image

    hier  : output of antspyt1w.hierarchical ( see read hierarchical )

    rsf_image : resting state fmri 

    flair_image : flair 

    nm_image_list : list of neuromelanin images

    dw_image : diffusion weighted image

    bvals : bvals file name

    bvecs : bvecs file name 

    srmodel : optional srmodel

    do_tractography : boolean

    do_kk : boolean to control whether we compute kelly kapowski thickness image (slow)

    do_normalization : boolean

    verbose : boolean

    """
    from os.path import exists
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    ex_path_mm = os.path.expanduser( "~/.antspymm/" )
    mycsvfn = ex_path + "FA_JHU_labels_edited.csv"
    citcsvfn = ex_path + "CIT168_Reinf_Learn_v1_label_descriptions_pad.csv"
    dktcsvfn = ex_path + "dkt.csv"
    JHU_atlasfn = ex_path + 'JHU-ICBM-FA-1mm.nii.gz' # Read in JHU atlas
    JHU_labelsfn = ex_path + 'JHU-ICBM-labels-1mm.nii.gz' # Read in JHU labels
    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
    if not exists( mycsvfn ) or not exists( citcsvfn ) or not exists( dktcsvfn ) or not exists( JHU_atlasfn ) or not exists( JHU_labelsfn ) or not exists( templatefn ):
        print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
        stophere
    mycsv = pd.read_csv(  mycsvfn )
    citcsv = pd.read_csv(  os.path.expanduser(  citcsvfn ) )
    dktcsv = pd.read_csv(  os.path.expanduser( dktcsvfn ) )
    JHU_atlas = ants.image_read( JHU_atlasfn ) # Read in JHU atlas
    JHU_labels = ants.image_read( JHU_labelsfn ) # Read in JHU labels
    template = ants.image_read( templatefn ) # Read in template
    #####################
    #  T1 hierarchical  #
    #####################
    t1imgbrn = hier['brain_n4_dnz']
    t1atropos = hier['dkt_parc']['tissue_segmentation']
    mynets = list([ 'CinguloopercularTaskControl', 'DefaultMode', 
        'MemoryRetrieval', 'VentralAttention', 'Visual', 
        'FrontoparietalTaskControl', 'Salience', 'Subcortical', 
        'DorsalAttention'])
    output_dict = { 
        'kk': None,
        'rsf': None,
        'flair' : None,
        'NM' : None,
        'DTI' : None,
        'FA_summ' : None,
        'MD_summ' : None,
        'tractography' : None,
        'tractography_connectivity' : None
    }
    normalization_dict = {
        'kk_norm': None,
        'NM_norm' : None,
        'FA_norm' : None,
        'MD_norm' : None,
        'CinguloopercularTaskControl_norm' : None, 
        'DefaultMode_norm' : None,
        'MemoryRetrieval_norm' : None,
        'VentralAttention_norm' : None, 
        'Visual_norm' : None,
        'FrontoparietalTaskControl_norm' : None,
        'Salience_norm' : None,
        'Subcortical_norm' : None,
        'DorsalAttention_norm' : None
    }
    if do_kk:
        if verbose:
            print('kk')
        output_dict['kk'] = antspyt1w.kelly_kapowski_thickness( hier['brain_n4_dnz'],
            labels=hier['dkt_parc']['dkt_cortex'], iterations=45 )
    ################################## do the rsf .....
    if rsf_image is not None:
        if verbose:
            print('rsf')
        if rsf_image.shape[3] > 40: # FIXME - better heuristic?
            output_dict['rsf'] = resting_state_fmri_networks( rsf_image, hier['brain_n4_dnz'], t1atropos,
                f=[0.03,0.08],   spa = 1.5, spt = 0.5, nc = 6 )
    if nm_image_list is not None:
        if verbose:
            print('nm')
        if srmodel is None:
            output_dict['NM'] = neuromelanin( nm_image_list, t1imgbrn, t1_image, hier['deep_cit168lab'] )
        else:
            output_dict['NM'] = neuromelanin( nm_image_list, t1imgbrn, t1_image, hier['deep_cit168lab'], srmodel=srmodel )
################################## do the dti .....
    if dw_image is not None:
        if verbose:
            print('dti')
        dtibxt_data = t1_based_dwi_brain_extraction( hier['brain_n4_dnz'], dw_image, transform='Rigid' )
        cropmask = ants.iMath( dtibxt_data['b0_mask'], 'MD', 6 )
        dw_image = crop_mcimage( dw_image, cropmask  )
        dtibxt_data['b0_mask'] = ants.crop_image( dtibxt_data['b0_mask'], cropmask )
        dtibxt_data['b0_avg'] = ants.crop_image( dtibxt_data['b0_avg'], cropmask )
        output_dict['DTI'] = joint_dti_recon(
            dw_image,
            bvals,
            bvecs,
            jhu_atlas=JHU_atlas,
            jhu_labels=JHU_labels,
            t1w = hier['brain_n4_dnz'],
            brain_mask = dtibxt_data['b0_mask'],
            reference_image = dtibxt_data['b0_avg'],
            srmodel=srmodel,
            motion_correct=True, # set to False if using input from qsiprep
            verbose = verbose)
        mydti = output_dict['DTI']
        # summarize dwi with T1 outputs
        # first - register ....
        reg = ants.registration( mydti['recon_fa'], hier['brain_n4_dnz'], 'Rigid' )
        ##################################################
        output_dict['FA_summ'] = hierarchical_modality_summary(
            mydti['recon_fa'],
            hier=hier,
            modality_name='fa',
            transformlist=reg['fwdtransforms'],
            verbose = False )
        ##################################################
        output_dict['MD_summ'] = hierarchical_modality_summary(
            mydti['recon_md'],
            hier=hier,
            modality_name='md',
            transformlist=reg['fwdtransforms'],
            verbose = False )
        # these inputs should come from nicely processed data
        dktmapped = ants.apply_transforms(
            mydti['recon_fa'],
            hier['dkt_parc']['dkt_cortex'],
            reg['fwdtransforms'], interpolator='nearestNeighbor' )
        mask = ants.threshold_image( mydti['recon_fa'], 0.05, 2.0 ).iMath("GetLargestComponent")
        if do_tractography: # dwi_deterministic_tracking dwi_closest_peak_tracking
            output_dict['tractography'] = dwi_deterministic_tracking(
                mydti['dwi_LR_dewarped'],
                mydti['recon_fa'],
                mydti['bval_LR'],
                mydti['bvec_LR'],
                seed_density = 1,
                mask=mask,
                verbose = verbose )
            mystr = output_dict['tractography']
            output_dict['tractography_connectivity'] = dwi_streamline_connectivity( mystr['streamlines'], dktmapped, dktcsv, verbose=verbose )
    ################################## do the flair .....
    if flair_image is not None:
        if verbose:
            print('flair')
        output_dict['flair'] = wmh( flair_image, t1_image, t1atropos, mmfromconvexhull=12 )
    #################################################################
    ### NOTES: deforming to a common space and writing out images ###
    ### images we want come from: DTI, NM, rsf, thickness ###########
    #################################################################
    if do_normalization:
        if verbose:
            print('normalization')
        # might reconsider this template space - cropped and/or higher res?
        template = ants.resample_image( template, [1,1,1], use_voxels=False )
        t1reg = ants.registration( template, hier['brain_n4_dnz'], "antsRegistrationSyNQuickRepro[s]")
        if do_kk:
            normalization_dict['kk_norm'] = ants.apply_transforms( template, output_dict['kk']['thickness_image'], t1reg['fwdtransforms'])
        if dw_image is not None:
            mydti = output_dict['DTI']
            dtirig = ants.registration( hier['brain_n4_dnz'], mydti['recon_fa'], 'Rigid' )
            normalization_dict['MD_norm'] = ants.apply_transforms( template, mydti['recon_md'],t1reg['fwdtransforms']+dtirig['fwdtransforms'] )
            normalization_dict['FA_norm'] = ants.apply_transforms( template, mydti['recon_fa'],t1reg['fwdtransforms']+dtirig['fwdtransforms'] )
        if output_dict['rsf'] is not None:
            rsfpro = output_dict['rsf']
            rsfrig = ants.registration( hier['brain_n4_dnz'], rsfpro['meanBold'], 'Rigid' )
            for netid in mynets:
                rsfkey = netid + "_norm"
                normalization_dict[rsfkey] = ants.apply_transforms( 
                    template, rsfpro[netid],
                    t1reg['fwdtransforms']+rsfrig['fwdtransforms'] )
        if nm_image_list is not None:
            nmpro = output_dict['NM']
            nmrig = nmpro['t1_to_NM_transform'] # this is an inverse tx
            normalization_dict['NM_norm'] = ants.apply_transforms( template, nmpro['NM_avg'],t1reg['fwdtransforms']+nmrig,
                whichtoinvert=[False,False,True])

    if verbose:
        print('mm done')
    return output_dict, normalization_dict


def write_mm( output_prefix, mm, mm_norm=None, t1wide=None, separator='_' ):
    """
    write the tabular and normalization output of the mm function

    Parameters
    -------------

    output_prefix : prefix for file outputs - modality specific postfix will be added

    mm  : output of mm function for modality-space processing

    mm_norm : output of mm function for normalized processing

    t1wide : wide output data frame from t1 hierarchical

    separator : string or character separator for filenames

    Returns
    ---------
  
    both csv and image files written to disk.  the primary outputs will be 
    output_prefix + separator + 'mmwide.csv' and *norm.nii.gz images

    """
    from dipy.io.streamline import save_tractogram
    if mm_norm is not None:
        for mykey in mm_norm.keys():
            tempfn = output_prefix + separator + mykey + '.nii.gz'
            if mm_norm[mykey] is not None:
                ants.image_write( mm_norm[mykey], tempfn )
    thkderk = None
    if t1wide is not None:
        thkderk = t1wide.iloc[: , 1:]
    kkderk = None
    if mm['kk'] is not None:
        kkderk = mm['kk']['thickness_dataframe'].iloc[: , 1:]
        mykey='thickness_image'
        tempfn = output_prefix + separator + mykey + '.nii.gz'
        ants.image_write( mm['kk'][mykey], tempfn )
    nmderk = None
    if mm['NM'] is not None:
        nmderk = mm['NM']['NM_dataframe_wide'].iloc[: , 1:]
        for mykey in ['NM_avg_cropped', 'NM_avg', 'NM_labels' ]:
            tempfn = output_prefix + separator + mykey + '.nii.gz'
            ants.image_write( mm['NM'][mykey], tempfn )

    faderk = mdderk = fat1derk = mdt1derk = None
    if mm['DTI'] is not None:
        mydti = mm['DTI']
        myop = output_prefix + separator
        write_bvals_bvecs( mydti['bval_LR'], mydti['bvec_LR'], myop + 'reoriented' )
        ants.image_write( mydti['dwi_LR_dewarped'],  myop + 'dwi.nii.gz' )
        ants.image_write( mydti['dtrecon_LR_dewarp']['RGB'] ,  myop + 'DTIRGB.nii.gz' )
        ants.image_write( mydti['jhu_labels'],  myop+'dtijhulabels.nii.gz' )
        ants.image_write( mydti['recon_fa'],  myop+'dtifa.nii.gz' )
        ants.image_write( mydti['recon_md'],  myop+'dtimd.nii.gz' )
        faderk = mm['DTI']['recon_fa_summary'].iloc[: , 1:]
        mdderk = mm['DTI']['recon_md_summary'].iloc[: , 1:]
        fat1derk = mm['FA_summ'].iloc[: , 1:]
        mdt1derk = mm['MD_summ'].iloc[: , 1:]
    if mm['tractography'] is not None:
        ofn = output_prefix + separator + 'tractogram.trk'
        save_tractogram( mm['tractography']['tractogram'], ofn )
    cnxderk = None
    if mm['tractography_connectivity'] is not None:
        cnxderk = mm['tractography_connectivity']['connectivity_wide'].iloc[: , 1:] # NOTE: connectivity_wide is not much tested
        ofn = output_prefix + separator + 'dtistreamlinecorr.csv'
        pd.DataFrame(mm['tractography_connectivity']['connectivity_matrix']).to_csv( ofn )
    mm_wide = pd.concat( [
        thkderk,
        kkderk,
        nmderk,
        faderk,
        mdderk,
        fat1derk,
        mdt1derk,
        cnxderk
        ], axis=1 )
    mm_wide = mm_wide.copy()
    if mm['NM'] is not None:
        mm_wide['NM_avg_signaltonoise'] = mm['NM']['NM_avg_signaltonoise']
        mm_wide['NM_avg_substantianigra'] = mm['NM']['NM_avg_substantianigra']
        mm_wide['NM_std_substantianigra'] = mm['NM']['NM_std_substantianigra']
        mm_wide['NM_avg_refregion'] = mm['NM']['NM_avg_refregion']
        mm_wide['NM_std_refregion'] = mm['NM']['NM_std_refregion']
    if mm['flair'] is not None:
        myop = output_prefix + separator + 'wmh.nii.gz'
        ants.image_write( mm['flair']['WMH_probability_map'], myop )
        mm_wide['flair_wmh'] = mm['flair']['wmh_mass']
    if mm['rsf'] is not None:
        mynets = list([ 'meanBold', 'CinguloopercularTaskControl', 'DefaultMode', 'MemoryRetrieval', 'VentralAttention', 'Visual', 'FrontoparietalTaskControl', 'Salience', 'Subcortical', 'DorsalAttention'])
        rsfpro = mm['rsf']
        for mykey in mynets:
            myop = output_prefix + separator + mykey + '.nii.gz'
            ants.image_write( rsfpro[mykey], myop )
        rsfpro['corr_wide'].set_index( mm_wide.index, inplace=True )
        mm_wide = pd.concat( [ mm_wide, rsfpro['corr_wide'] ], axis=1 )
        mm_wide['rsf_FD_mean'] = rsfpro['FD_mean']
        mm_wide['rsf_FD_max'] = rsfpro['FD_max']
        ofn = output_prefix + separator + 'rsfcorr.csv'
        rsfpro['corr'].to_csv( ofn )
    if mm['DTI'] is not None:
        mydti = mm['DTI']
        if mydti['dtrecon_LR']['framewise_displacement'] is not None:
            mm_wide['dti_FD_mean'] = mydti['dtrecon_LR']['framewise_displacement'].mean()
            mm_wide['dti_FD_max'] = mydti['dtrecon_LR']['framewise_displacement'].max()
        else:
            mm_wide['dti_FD_mean'] = mm_wide['dti_FD_max'] = 'NA'
    mmwidefn = output_prefix + separator + 'mmwide.csv'
    mm_wide.to_csv( mmwidefn )
    return



def mm_nrg( 
    sourcedir = os.path.expanduser( "~/data/PPMI/MV/example_s3_b/images/PPMI/" ), # study folder
    sid  = "100898",   # subject unique id
    dtid = "20210331", # date
    iid  = "1496183",  # image unique id for t1 - should have highest grade if repeats exist
    sourcedatafoldername = 'images', # root for source data
    processDir = "processed", # where output will go - parallel to sourcedatafoldername
    mysep = '-', # define a separator for filename components
    srmodel_NM = None, # optional - will add a great deal of time
    srmodel_DTI = None, # optional - will add a great deal of time
    visualize = True,
    verbose = True
):
    """
    too dangerous to document ... use with care.

    processes multiple modality MRI specifically:
    
    * T1w
    * T2Flair
    * DTI, DTI_LR, DTI_RL
    * rsfMRI, rsfMRI_LR, rsfMRI_RL
    * NM2DMT (neuromelanin)

    other modalities may be added later ...

    "trust me, i know what i'm doing" - sledgehammer

    convert to pynb via:
        p2j mm.py -o

    convert the ipynb to html via:
        jupyter nbconvert ANTsPyMM/tests/mm.ipynb --execute --to html

    this function assumes NRG format for the input data .... 
    we also assume that t1w hierarchical (if already done) was written 
    via its standardized write function.
    NRG = https://github.com/stnava/biomedicalDataOrganization

    this function is verbose

    Parameters
    -------------

    sourcedir : a study specific folder containing individual subject folders

    sid  : subject unique id e.g. S001

    dtid : date eg "20210331"

    iid  : image unique id for t1 e.g. "1496183"  - this image should have the 
        highest grade if repeats exist

    sourcedatafoldername : root for source data e.g. "images"

    processDir : where output will go - parallel to sourcedatafoldername e.g. 
        "processed"

    mysep : define a character separator for filename components

    srmodel_NM : None (optional) - will add a great deal of time

    srmodel_DTI : None (optional) - will add a great deal of time

    visualize : True - will plot some results to screen / notebook

    verbose : boolean

    Returns
    ---------

    writes output to disk and potentially produces figures that may be 
    captured in a ipynb / html file.

    """
    import glob as glob
    from os.path import exists
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
    if not exists( templatefn ):
        print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
        antspyt1w.get_data( force_download=True )
        get_data( force_download=True )
    temp = sourcedir.split( "/" )
    splitCount = len( temp )
    template = ants.image_read( templatefn ) # Read in template
    realrun = True
    subjectrootpath = sourcedir +sid+"/"+ dtid+ "/"
    myimgs = glob.glob( subjectrootpath+"*" )
    myimgs.sort( )
    if verbose:
        print( myimgs )
    # hierarchical
    # NOTE: if there are multiple T1s for this time point, should take 
    # the one with the highest resnetGrade
    t1fn = glob.glob( subjectrootpath + "/T1w/"+iid+"/*nii.gz")[0]
    t1 = ants.image_read( t1fn )
    hierfn = re.sub( sourcedatafoldername, processDir, t1fn)
    hierfn = re.sub( "T1w", "T1wHierarchical", hierfn)
    hierfn = re.sub( ".nii.gz", "", hierfn)
    hierfntest = hierfn + 'snseg.csv'
    if verbose:
        print( hierfntest )
    hierexists = exists( hierfntest ) # FIXME should test this explicitly but we assume it here
    hier = None
    if not hierexists:
        subjectpropath = os.path.dirname( hierfn )
        if verbose:
            print( subjectpropath )
        os.makedirs( subjectpropath, exist_ok=True  )
        hier = antspyt1w.hierarchical( t1, hierfn, labels_to_register=None )
        antspyt1w.write_hierarchical( hier, hierfn )
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
        t1wide.to_csv( hierfn + 'wide.csv' )
    ################################################
    hier = antspyt1w.read_hierarchical( hierfn )
    if exists( hierfn + 'wide.csv' ) :
        t1wide = pd.read_csv( hierfn + 'wide.csv' )
    else:
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
    t1imgbrn = hier['brain_n4_dnz']
    t1atropos = hier['dkt_parc']['tissue_segmentation']
    testloop = False
    if visualize and not testloop:
        ants.plot( t1imgbrn,  axis=2, nslices=21, ncol=7, crop=True, title='brain extraction' )
        ants.plot( t1imgbrn, t1atropos, axis=2, nslices=21, ncol=7, crop=True, title='segmentation'  )
        ants.plot( t1imgbrn, hier['dkt_parc']['dkt_cortex'], axis=2, nslices=21, ncol=7, crop=True, title='cortex'   )
    # loop over modalities and then unique image IDs
    # we treat NM in a "special" way -- aggregating repeats 
    # other modalities (beyond T1) are treated individually
    for xnum in range( len( myimgs ) ):
        if verbose:
            print( "we have : " + str(len(myimgs)) + " modalities.")
        dowrite=False
        x = myimgs[xnum]
        myimgsr = glob.glob( x+"/*" )
        overmod = x.split( "/" )
        overmod = overmod[ len(overmod)-1 ]
        if verbose:
            print( 'overmod is : ' + overmod )
            print( 'x is : ' + x )
        if overmod == 'NM2DMT':
            myimgsr = glob.glob( x+"/*/*nii.gz" )
            myimgsr.sort()
            subjectpropath = os.path.dirname( x )
            subjectpropath = re.sub( sourcedatafoldername, processDir, x )
            if verbose:
                print( myimgsr )
                print( "subjectpropath " + subjectpropath )
            mysplit = subjectpropath.split( "/" )
            os.makedirs( subjectpropath, exist_ok=True  )
            mysplitCount = len( mysplit )
            identifier = mysplit[mysplitCount-3] + mysep + mysplit[mysplitCount-2] + mysep + 'NM2DMT'
            mymm = subjectpropath + "/" + identifier
            if verbose:
                print( "NM " + mymm )
            nmlist = []
            for zz in myimgsr:
                nmlist.append( ants.image_read( zz ) )
            if not testloop:
                tabPro, normPro = mm( t1, hier, 
                        nm_image_list = nmlist,
                        srmodel=srmodel_NM,
                        do_tractography=False, 
                        do_kk=False, 
                        do_normalization=True, 
                        verbose=True )
                write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=None, separator=mysep )
                nmpro = tabPro['NM']
                mysl = range( nmpro['NM_avg'].shape[2] )
                if visualize:
                    ants.plot( nmpro['NM_avg'],  nmpro['t1_to_NM'], slices=mysl, axis=2, title='nm + t1' )
                    mysl = range( nmpro['NM_avg_cropped'].shape[2] )
                    ants.plot( nmpro['NM_avg_cropped'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop' )
                    ants.plot( nmpro['NM_avg_cropped'], nmpro['t1_to_NM'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop + t1' )
                    ants.plot( nmpro['NM_avg_cropped'], nmpro['NM_labels'], axis=2, slices=mysl, title='nm crop + labels' )
        else :
            for y in myimgsr:
                dowrite=False
                myimg = glob.glob( y+"/*nii.gz" )
                subjectpropath = os.path.dirname( myimg[0] )
                subjectpropath = re.sub( sourcedatafoldername, processDir, subjectpropath )
                mysplit = subjectpropath.split("/")
                mysplitCount = len( mysplit )
                mymod = mysplit[mysplitCount-2] # FIXME system dependent
                uid = mysplit[mysplitCount-1] # unique image id
                os.makedirs( subjectpropath, exist_ok=True  )
                identifier = mysplit[mysplitCount-4] + mysep + mysplit[mysplitCount-3] + mysep + mymod + mysep + uid
                mymm = subjectpropath + "/" + identifier
                if verbose:
                    print("Modality specific processing: " + mymod )
                    print( mymm )
                if verbose:
                    print(subjectpropath)
                    print(identifier)
                    print( myimg[0] )
                if not testloop:
                    img = ants.image_read( myimg[0] )
                    if mymod == 'T1w' : # for a real run, set to True
                        dowrite=True
                        if verbose:
                            print('start kk')
                        tabPro, normPro = mm( t1, hier, 
                            srmodel=None,
                            do_tractography=False, 
                            do_kk=True, 
                            do_normalization=True, 
                            verbose=True )
                        if visualize:
                            ants.plot( hier['brain_n4_dnz'], tabPro['kk']['thickness_image'], axis=2, nslices=21, ncol=7, crop=True, title='kk' )
                    if mymod == 'T2Flair':
                        dowrite=True
                        tabPro, normPro = mm( t1, hier, 
                            flair_image = img,
                            srmodel=None,
                            do_tractography=False, 
                            do_kk=False, 
                            do_normalization=True, 
                            verbose=True )
                        if visualize:
                            ants.plot( img,   axis=2, nslices=21, ncol=7, crop=True, title='Flair' )
                            ants.plot( img, tabPro['flair']['WMH_probability_map'],  axis=2, nslices=21, ncol=7, crop=True, title='Flair + WMH' )
                    if mymod == 'rsfMRI_LR' or mymod == 'rsfMRI_RL' or mymod == 'rsfMRI' :
                        dowrite=True
                        tabPro, normPro = mm( t1, hier, 
                            rsf_image=img,
                            srmodel=None,
                            do_tractography=False, 
                            do_kk=False, 
                            do_normalization=True, 
                            verbose=True )
                        if tabPro['rsf'] is not None and visualize:
                            ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['DefaultMode'],
                                axis=2, nslices=21, ncol=7, crop=True, title='DefaultMode' )
                            ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['FrontoparietalTaskControl'],
                                axis=2, nslices=21, ncol=7, crop=True, title='FrontoparietalTaskControl' )
                    if mymod == 'DTI_LR' or mymod == 'DTI_RL' or mymod == 'DTI':
                        dowrite=True
                        bvalfn = re.sub( '.nii.gz', '.bval' , myimg[0] )
                        bvecfn = re.sub( '.nii.gz', '.bvec' , myimg[0] )
                        tabPro, normPro = mm( t1, hier, 
                            dw_image=img,
                            bvals = bvalfn,
                            bvecs = bvecfn,
                            srmodel=srmodel_DTI,
                            do_tractography=realrun, 
                            do_kk=False, 
                            do_normalization=True, 
                            verbose=True )
                        mydti = tabPro['DTI']
                        if visualize:
                            ants.plot( mydti['dtrecon_LR']['FA'],  axis=2, nslices=21, ncol=7, crop=True, title='FA pre correction' )
                            ants.plot( mydti['recon_fa'],  axis=2, nslices=21, ncol=7, crop=True, title='FA (supposed to be better)' )
                            ants.plot( mydti['recon_fa'], mydti['jhu_labels'], axis=2, nslices=21, ncol=7, crop=True, title='FA + JHU' )
                            ants.plot( mydti['recon_md'],  axis=2, nslices=21, ncol=7, crop=True, title='MD' )
                    if dowrite:
                        write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=t1wide, separator=mysep )
                        for mykey in normPro.keys():
                            if normPro[mykey] is not None:
                                if visualize:
                                    ants.plot( template, normPro[mykey], axis=2, nslices=21, ncol=7, crop=True, title=mykey  )
