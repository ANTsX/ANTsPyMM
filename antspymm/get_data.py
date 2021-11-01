
__all__ = ['get_data','dewarp_dti']

from pathlib import Path
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

import ants
import antspynet
import tensorflow as tf

from multiprocessing import Pool

DATA_PATH = os.path.expanduser('~/.antspymm/')

def get_data( name=None, force_download=False, version=1, target_extension='.csv' ):
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

    if datapath is None:
        raise ValueError('File doesnt exist. Options: ' , os.listdir(DATA_PATH))
    return datapath




def dewarp_imageset( image_list, iterations=None, **kwargs ):
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

    kwargs : keyword args
        arguments passed to ants registration - these must be set explicitly

    Returns
    -------
    a dictionary with the mean image and the list of the transformed images

    Example
    -------
    >>> import antspymm
    """
    outlist = []
    avglist = []
    if len(image_list[0].shape) > 3:
        imagetype = 3
        for k in range(len(image_list)):
            avglist.append( ants.slice_image( image_list[k], axis=3, idx=0 ) )
    else:
        imagetype = 0
        avglist=image_list

    if iterations is None:
        iterations = 2

    btp = ants.build_template( image_list=avglist,
        gradient_step=0.5, blending_weight=0.8,
        iterations=iterations, **kwargs )

    # last - warp all images to this frame
    for k in range(len(image_list)):
        reg=ants.registration( btp, avglist[k], **kwargs )
        # now apply the transformation parameters to all in the time series, if needed
        mywarped = ants.apply_transforms( btp, image_list[k], reg['fwdtransforms'], imagetype=imagetype )
        outlist.append( mywarped )


    return {'dewarpedmean':btp, 'dewarped':outlist }
