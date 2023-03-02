Module antspymm.get_data
========================

Functions
---------

    
`dewarp_imageset(image_list, initial_template=None, iterations=None, padding=0, target_idx=[0], **kwargs)`
:   Dewarp a set of images
    
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

    
`dipy_dti_recon(image, bvalsfn, bvecsfn, mask=None, b0_idx=None, mask_dilation=2, mask_closing=5, fit_method='WLS', trim_the_mask=2, verbose=False)`
:   DiPy DTI reconstruction - building on the DiPy basic DTI example
    
    Arguments
    ---------
    image : an antsImage holding B0 and DWI
    
    bvalsfn : bvalues  obtained by dipy read_bvals_bvecs or the values themselves
    
    bvecsfn : bvectors obtained by dipy read_bvals_bvecs or the values themselves
    
    mask : brain mask for the DWI/DTI reconstruction; if it is not in the same
        space as the image, we will resample directly to the image space.  This
        could lead to problems if the inputs are really incorrect.
    
    b0_idx : the indices of the B0; if None, use segment_timeseries_by_meanvalue to guess
    
    mask_dilation : integer zero or more dilates the brain mask
    
    mask_closing : integer zero or more closes the brain mask
    
    fit_method : string one of WLS LS NLLS or restore - see import dipy.reconst.dti as dti and help(dti.TensorModel) ... if None, will not reconstruct DTI.
    
    trim_the_mask : boolean post-hoc method for trimming the mask
    
    verbose : boolean
    
    Returns
    -------
    dictionary holding the tensorfit, MD, FA and RGB images and motion parameters (optional)
    
    NOTE -- see dipy reorient_bvecs(gtab, affines, atol=1e-2)
    
    NOTE -- if the bvec.shape[0] is smaller than the image.shape[3], we neglect
        the tailing image volumes.
    
    Example
    -------
    >>> import antspymm

    
`dwi_deterministic_tracking(dwi, fa, bvals, bvecs, num_processes=1, mask=None, label_image=None, seed_labels=None, fa_thresh=0.05, seed_density=1, step_size=0.15, peak_indices=None, fit_method='WLS', verbose=False)`
:   Performs deterministic tractography from the DWI and returns a tractogram
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
    
    fit_method : string one of WLS LS NLLS or restore - see import dipy.reconst.dti as dti and help(dti.TensorModel)
    
    verbose : boolean
    
    Returns
    -------
    dictionary holding tracts and stateful object.
    
    Example
    -------
    >>> import antspymm

    
`get_data(name=None, force_download=False, version=11, target_extension='.csv')`
:   Get ANTsPyMM data filename
    
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

    
`neuromelanin(list_nm_images, t1, t1_head, t1lab, brain_stem_dilation=8, bias_correct=True, denoise=None, srmodel=None, target_range=[0, 1], poly_order='hist', normalize_nm=False, verbose=False)`
:   Outputs the averaged and registered neuromelanin image, and neuromelanin labels
    
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
    
    target_range : 2-element tuple
          a tuple or array defining the (min, max) of the input image
          (e.g., [-127.5, 127.5] or [0,1]).  Output images will be scaled back to original
          intensity. This range should match the mapping used in the training
          of the network.
    
    poly_order : if not None, will fit a global regression model to map
        intensity back to original histogram space; if 'hist' will match
        by histogram matching - ants.histogram_match_image
    
    normalize_nm : boolean - WIP not validated
    
    verbose : boolean
    
    Returns
    ---------
    Averaged and registered neuromelanin image and neuromelanin labels and wide csv

    
`resting_state_fmri_networks(fmri, fmri_template, t1, t1segmentation, f=[0.03, 0.08], spa=1.5, spt=0.5, nc=6, type_of_transform='SyN', verbose=False)`
:   Compute resting state network correlation maps based on the J Power labels.
    This will output a map for each of the major network systems.
    
    Arguments
    ---------
    fmri : BOLD fmri antsImage
    
    fmri_template : reference space for BOLD
    
    t1 : ANTsImage
      input 3-D T1 brain image (brain extracted)
    
    t1segmentation : ANTsImage
      t1 segmentation - a six tissue segmentation image in T1 space
    
    f : band pass limits for frequency filtering
    
    spa : gaussian smoothing for spatial component
    
    spt : gaussian smoothing for temporal component
    
    nc  : number of components for compcor filtering
    
    type_of_transform : SyN or Rigid
    
    verbose : boolean
    
    Returns
    ---------
    a dictionary containing the derived network maps

    
`segment_timeseries_by_meanvalue(image, quantile=0.995)`
:   Identify indices of a time series where we assume there is a different mean
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

    
`super_res_mcimage(image, srmodel, truncation=[0.0001, 0.995], poly_order='hist', target_range=[0, 1], isotropic=False, verbose=False)`
:   Super resolution on a timeseries or multi-channel image
    
    Arguments
    ---------
    image : an antsImage
    
    srmodel : a tensorflow fully convolutional model
    
    truncation :  quantiles at which we truncate intensities to limit impact of outliers e.g. [0.005,0.995]
    
    poly_order : if not None, will fit a global regression model to map
        intensity back to original histogram space; if 'hist' will match
        by histogram matching - ants.histogram_match_image
    
    target_range : 2-element tuple
        a tuple or array defining the (min, max) of the input image
        (e.g., [-127.5, 127.5] or [0,1]).  Output images will be scaled back to original
        intensity. This range should match the mapping used in the training
        of the network.
    
    isotropic : boolean
    
    verbose : boolean
    
    Returns
    -------
    super resolution version of the image
    
    Example
    -------
    >>> import antspymm

    
`wmh(flair, t1, t1seg, mmfromconvexhull=3.0, strict=True, probability_mask=None, prior_probability=None, model='sysu', verbose=False)`
:   Outputs the WMH probability mask and a summary single measurement
    
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
      convex hull of the cerebrum.   we choose a default value based on
      Figure 4 from:
      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6240579/pdf/fnagi-10-00339.pdf
    
    strict: boolean - if True, only use convex hull distance
    
    probability_mask : None - use to compute wmh just once - then this function
          just does refinement and summary
    
    prior_probability : optional prior probability image in space of the input t1
    
    model : either sysu or hyper
    
    verbose : boolean
    
    Returns
    ---------
    WMH probability map and a summary single measurement which is the sum of the WMH map