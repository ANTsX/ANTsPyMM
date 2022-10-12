
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
    average_b0 = None ):
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

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

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
        print("recon after distortion correction")

    recon_RL_dewarp = None
    if img_RL is not None:
        recon_RL_dewarp = dipy_dti_recon( img_RLdwp, bval_RL, bvec_RL,
            mask = brain_mask, average_b0 = reference_image,
                motion_correct=False,
                mask_dilation=0 )

    recon_LR_dewarp = dipy_dti_recon( img_LRdwp, bval_LR, bvec_LR,
            mask = brain_mask, average_b0 = reference_image,
                motion_correct=False,
                mask_dilation=0  )

    if img_RL is not None:
        reconFA = recon_RL_dewarp['FA'] * 0.5 + recon_LR_dewarp['FA'] * 0.5
        reconMD = recon_RL_dewarp['MD'] * 0.5 + recon_LR_dewarp['MD'] * 0.5
    else:
        reconFA = recon_LR_dewarp['FA']
        reconMD = recon_LR_dewarp['MD']

    if verbose:
        print("JHU reg")

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
    dti_fit = dti_model.fit(dwi_data, mask=dwi_mask)  # This step may take a while
    evecs_img = dti_fit.evecs
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    stopping_criterion = ThresholdStoppingCriterion(fa.numpy(), fa_thresh)
    from dipy.data import get_sphere
    sphere = get_sphere('symmetric362')
    from dipy.direction import peaks_from_model
    if peak_indices is None:
        if verbose:
            print("begin peaks")
        peak_indices = peaks_from_model(
            model=dti_model, data=dwi_data, sphere=sphere, relative_peak_threshold=.2,
            min_separation_angle=25, mask=dwi_mask, npeaks=5, return_odf=False,
            return_sh=False, parallel=True, num_processes=num_processes
            )

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
        print("streamlines begin ...")
    streamlines_generator = LocalTracking(
        peak_indices, stopping_criterion, seeds, affine=affine, step_size=step_size)
    streamlines = Streamlines(streamlines_generator)
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram
    sft = StatefulTractogram(streamlines, dwi_img, Space.RASMM)
    if verbose:
        print("done")
    return {
          'tractogram': sft,
          'streamlines': streamlines,
          'peak_indices': peak_indices
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



def neuromelanin( list_nm_images, t1, t1_head, t1lab, brain_stem_dilation=8, srmodel=None ) :

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

  srmodel : None -- this is a work in progress feature, probably not optimal

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
  reg = ants.registration( t1, template, 'antsRegistrationSyNQuickRepro[s]' )
  # map brain stem and slab to t1 for neuromelanin processing
  bstem2t1 = ants.apply_transforms( t1, templatebstem,
    reg['fwdtransforms'],
    interpolator='nearestNeighbor' ).iMath("MD",1)
  slab2t1 = ants.apply_transforms( t1, ants.image_read( fnslab ),
    reg['fwdtransforms'], interpolator = 'nearestNeighbor')
  cropper = ants.iMath( bstem2t1 , "MD", brain_stem_dilation ) * slab2t1

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

  t1c = ants.crop_image( t1_head, cropper ).iMath("Normalize")
  slabreg = ants.registration( nm_avg, t1c, 'Rigid' )
  labels2nm = ants.apply_transforms( nm_avg, t1lab, slabreg['fwdtransforms'],
    interpolator = 'genericLabel' )
  cropper2nm = ants.apply_transforms( nm_avg, cropper, slabreg['fwdtransforms'], interpolator='nearestNeighbor' )
  # ants.plot( nm_avg,  slabreg['warpedmovout'], slices=[8,10,12], axis=2 )
  # ants.plot( nm_avg, nmpro['NM_labels'], axis=2, slices=[8,10,12] )
  # now do SR processing
  nmcropped = ants.crop_image( nm_avg, cropper2nm )
  if srmodel is not None: # over-writes prior results with SR results
      nm_avg = antspynet.apply_super_resolution_model_to_image(
        nmcropped, srmodel, target_range=[0,1], regression_order=None )
      slabreg = ants.registration( nm_avg, t1c, 'Rigid' )
      labels2nm = ants.apply_transforms( nm_avg, t1lab,
        slabreg['fwdtransforms'], interpolator='nearestNeighbor' )

  #  map summary measurements to wide format
  nmdf = antspyt1w.map_intensity_to_dataframe(
          'CIT168_Reinf_Learn_v1_label_descriptions_pad',
          nm_avg,
          labels2nm)
  nmdf_wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
              {'NM' : nmdf},
              col_names = ['Mean'] )

  return{
      'NM_avg' : nm_avg,
      'NM_labels': labels2nm,
      'NM_cropped': nmcropped,
      'NM_dataframe': nmdf,
      'NM_dataframe_wide': nmdf_wide,
      't1_to_NM': slabreg['warpedmovout'],
      't1_to_NM_transform' : slabreg['fwdtransforms'] }





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
