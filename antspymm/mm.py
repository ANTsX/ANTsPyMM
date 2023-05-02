
__all__ = ['version',
    'mm_read',
    'mm_read_to_3d',
    'image_write_with_thumbnail',
    'nrg_format_path',
    'highest_quality_repeat',
    'match_modalities',
    'mc_resample_image_to_target',
    'nrg_filelist_to_dataframe',
    'merge_timeseries_data',
    'timeseries_reg',
    'merge_dwi_data',
    'outlierness_by_modality',
    'bvec_reorientation',
    'dti_reg',
    'mc_reg',
    'get_data',
    'get_models',
    'get_valid_modalities',
    'dewarp_imageset',
    'super_res_mcimage',
    'segment_timeseries_by_meanvalue',
    'get_average_rsf',
    'get_average_dwi_b0',
    'dti_template',
    't1_based_dwi_brain_extraction',
    'mc_denoise',
    'tsnr',
    'dvars',
    'slice_snr',
    'impute_fa',
    'trim_dti_mask',
    'dipy_dti_recon',
    'concat_dewarp',
    'joint_dti_recon',
    'middle_slice_snr',
    'foreground_background_snr',
    'quantile_snr',
    'mask_snr',
    'dwi_deterministic_tracking',
    'dwi_closest_peak_tracking',
    'dwi_streamline_pairwise_connectivity',
    'dwi_streamline_connectivity',
    'hierarchical_modality_summary',
    'tra_initializer',
    'neuromelanin',
    'resting_state_fmri_networks',
    'write_bvals_bvecs',
    'crop_mcimage',
    'mm',
    'write_mm',
    'mm_nrg',
    'mm_csv',
    'collect_blind_qc_by_modality',
    'alffmap',
    'alff_image',
    'down2iso',
    'read_mm_csv',
    'assemble_modality_specific_dataframes',
    'bind_wide_mm_csvs',
    'merge_mm_dataframe',
    'augment_image',
    'boot_wmh',
    'threaded_bind_wide_mm_csvs',
    'get_names_from_data_frame',
    'average_mm_df',
    'quick_viz_mm_nrg',
    'blind_image_assessment',
    'average_blind_qc_by_modality',
    'best_mmm',
    'nrg_2_bids',
    'bids_2_nrg',
    'parse_nrg_filename',
    'novelty_detection_svm',
    'novelty_detection_ee',
    'novelty_detection_lof',
    'novelty_detection_loop',
    'novelty_detection_quantile',
    'generate_mm_dataframe',
    'wmh']

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
import datetime as dt
from collections import Counter
import tempfile


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
import siq
import tensorflow as tf

from multiprocessing import Pool
import glob as glob

DATA_PATH = os.path.expanduser('~/.antspymm/')

def version( ):
    """
    report versions of this package and primary dependencies

    Arguments
    ---------
    None

    Returns
    -------
    a dictionary with package name and versions

    Example
    -------
    >>> import antspymm
    >>> antspymm.version()
    """
    import pkg_resources
    return {
              'tensorflow': pkg_resources.require("tensorflow")[0].version,
              'antspyx': pkg_resources.require("antspyx")[0].version,
              'antspynet': pkg_resources.require("antspynet")[0].version,
              'antspyt1w': pkg_resources.require("antspyt1w")[0].version,
              'antspymm': pkg_resources.require("antspymm")[0].version
              }

def get_valid_modalities( long=False, asString=False, qc=False ):
    """
    return a list of valid modality identifiers used in NRG modality designation
    and that can be processed by this package.

    long - return the long version

    asString - concat list to string
    """
    if long:
        mymod = ["T1w", "NM2DMT", "rsfMRI", "rsfMRI_LR", "rsfMRI_RL", "DTI", "DTI_LR","DTI_RL","T2Flair", "dwi", "func" ]
    elif qc:
        mymod = [ 'T1w', 'T2Flair', 'rsfMRI', 'NM2DMT','DTIdwi','DTIb0']
    else:
        mymod = ["T1w", "NM2DMT", "rsfMRI","DTI","T2Flair" ]
    if not asString:
        return mymod
    else:
        mymodchar=""
        for x in mymod:
            mymodchar = mymodchar + " " + str(x)
        return mymodchar

def generate_mm_dataframe(
        projectID,
        subjectID,
        date,
        imageUniqueID,
        modality,
        source_image_directory,
        output_image_directory,
        t1_filename,
        flair_filename=[],
        rsf_filenames=[],
        dti_filenames=[],
        nm_filenames=[]
):
    from os.path import exists
    valid_modalities = get_valid_modalities()
    if not isinstance(t1_filename, str):
        raise ValueError("t1_filename is not a string")
    if not exists(t1_filename):
        raise ValueError("t1_filename does not exist")
    if modality not in valid_modalities:
        raise ValueError('modality ' + str(modality) + " not a valid mm modality:  " + get_valid_modalities(asString=True))
    # if not exists( output_image_directory ):
    #    raise ValueError("output_image_directory does not exist")
    if not exists( source_image_directory ):
        raise ValueError("source_image_directory does not exist")
    if len( rsf_filenames ) < 2:
        for k in range(len(rsf_filenames),2):
            rsf_filenames.append(None)
    if len( dti_filenames ) < 2:
        for k in range(len(dti_filenames),2):
            dti_filenames.append(None)
    if len( nm_filenames ) < 10:
        for k in range(len(nm_filenames),10):
            nm_filenames.append(None)
    # check modality names
    if not "T1w" in t1_filename:
        raise ValueError("T1w is not in t1 filename " + t1_filename)
    if flair_filename is not None:
        if isinstance(flair_filename,list):
            if (len(flair_filename) == 0):
                flair_filename=None
            else:
                print("Take first entry from flair_filename list")
                flair_filename=flair_filename[0]
    if flair_filename is not None and not "lair" in flair_filename:
            raise ValueError("flair is not flair filename " + flair_filename)
    for k in nm_filenames:
        if k is not None:
            if not "NM" in k:
                raise ValueError("NM is not flair filename " + k)
    for k in dti_filenames:
        if k is not None:
            if not "DTI" in k and not "dwi" in k:
                raise ValueError("DTI/DWI is not dti filename " + k)
    for k in rsf_filenames:
        if k is not None:
            if not "fMRI" in k and not "func" in k:
                raise ValueError("rsfMRI/func is not rsfmri filename " + k)
    allfns = [t1_filename] + [flair_filename] + nm_filenames + dti_filenames + rsf_filenames
    for k in allfns:
        if k is not None:
            if not isinstance(k, str):
                raise ValueError(str(k) + " is not a string")
            if not exists( k ):
                raise ValueError( "image " + k + " does not exist")
    coredata = [
        projectID,
        subjectID,
        date,
        imageUniqueID,
        modality,
        source_image_directory,
        output_image_directory,
        t1_filename,
        flair_filename]
    mydata0 = coredata +  rsf_filenames + dti_filenames
    mydata = mydata0 + nm_filenames
    corecols = [
        'projectID',
        'subjectID',
        'date',
        'imageID',
        'modality',
        'sourcedir',
        'outputdir',
        'filename',
        'flairid']
    mycols0 = corecols + [
        'rsfid1', 'rsfid2',
        'dtid1', 'dtid2']
    nmext = [
        'nmid1', 'nmid2' 'nmid3', 'nmid4', 'nmid5',
        'nmid6', 'nmid7','nmid8', 'nmid9', 'nmid10', 'nmid11'
    ]
    mycols = mycols0 + nmext
    print(len(mydata0))
    print(len(nm_filenames))
    print(len(mycols0))
    print(len(nmext))
    studycsv = pd.DataFrame([ mydata ],
        columns=mycols)
    return studycsv

def parse_nrg_filename( x, separator='-' ):
    """
    split a NRG filename into its named parts
    """
    temp = x.split( separator )
    if len(temp) != 5:
        raise ValueError(x + " not a valid NRG filename")
    return {
        'project':temp[0],
        'subjectID':temp[1],
        'date':temp[2],
        'modality':temp[3],
        'imageID':temp[4]
    }



def nrg_2_bids( nrg_filename ):
    """
    Convert an NRG filename to BIDS path/filename.

    Parameters:
    nrg_filename (str): The NRG filename to convert.

    Returns:
    str: The BIDS path/filename.
    """

    # Split the NRG filename into its components
    nrg_dirname, nrg_basename = os.path.split(nrg_filename)
    nrg_suffix = '.' + nrg_basename.split('.',1)[-1]
    nrg_basename = nrg_basename.replace(nrg_suffix, '') # remove ext
    nrg_parts = nrg_basename.split('-')
    nrg_subject_id = nrg_parts[1]
    nrg_modality = nrg_parts[3]
    nrg_repeat= nrg_parts[4]

    # Build the BIDS path/filename
    bids_dirname = os.path.join(nrg_dirname, 'bids')
    bids_subject = f'sub-{nrg_subject_id}'
    bids_session = f'ses-{nrg_repeat}'

    valid_modalities = get_valid_modalities()
    if nrg_modality is not None:
        if not nrg_modality in valid_modalities:
            raise ValueError('nrg_modality ' + str(nrg_modality) + " not a valid mm modality:  " + get_valid_modalities(asString=True))

    if nrg_modality == 'T1w' :
        bids_modality_folder = 'anat'
        bids_modality_filename = 'T1w'

    if nrg_modality == 'T2Flair' :
        bids_modality_folder = 'anat'
        bids_modality_filename = 'flair'

    if nrg_modality == 'NM2DMT' :
        bids_modality_folder = 'anat'
        bids_modality_filename = 'nm2dmt'

    if nrg_modality == 'DTI' or nrg_modality == 'DTI_RL' or nrg_modality == 'DTI_LR' :
        bids_modality_folder = 'dwi'
        bids_modality_filename = 'dwi'

    if nrg_modality == 'rsfMRI' or nrg_modality == 'rsfMRI_RL' or nrg_modality == 'rsfMRI_LR' :
        bids_modality_folder = 'func'
        bids_modality_filename = 'func'

    bids_suffix = nrg_suffix[1:]
    bids_filename = f'{bids_subject}_{bids_session}_{bids_modality_filename}.{bids_suffix}'

    # Return bids filepath/filename
    return os.path.join(bids_dirname, bids_subject, bids_session, bids_modality_folder, bids_filename)


def bids_2_nrg( bids_filename, project_name, date, nrg_modality=None ):
    """
    Convert a BIDS filename to NRG path/filename.

    Parameters:
    bids_filename (str): The BIDS filename to convert
    project_name (str) : Name of project (i.e. PPMI)
    date (str) : Date of image acquisition


    Returns:
    str: The NRG path/filename.
    """

    bids_dirname, bids_basename = os.path.split(bids_filename)
    bids_suffix = '.'+ bids_basename.split('.',1)[-1]
    bids_basename = bids_basename.replace(bids_suffix, '') # remove ext
    bids_parts = bids_basename.split('_')
    nrg_subject_id = bids_parts[0].replace('sub-','')
    nrg_image_id = bids_parts[1].replace('ses-', '')
    bids_modality = bids_parts[2]
    valid_modalities = get_valid_modalities()
    if nrg_modality is not None:
        if not nrg_modality in valid_modalities:
            raise ValueError('nrg_modality ' + str(nrg_modality) + " not a valid mm modality: " + get_valid_modalities(asString=True))

    if bids_modality == 'anat' and nrg_modality is None :
        nrg_modality = 'T1w'

    if bids_modality == 'dwi' and nrg_modality is None  :
        nrg_modality = 'DTI'

    if bids_modality == 'func' and nrg_modality is None  :
        nrg_modality = 'rsfMRI'

    nrg_suffix = bids_suffix[1:]
    nrg_filename = f'{project_name}-{nrg_subject_id}-{date}-{nrg_modality}-{nrg_image_id}.{nrg_suffix}'

    return os.path.join(project_name, nrg_subject_id, date, nrg_modality, nrg_image_id,nrg_filename)

def collect_blind_qc_by_modality( modality_path, set_index_to_fn=True ):
    """
    Collects blind QC data from multiple CSV files with the same modality.

    Args:

    modality_path (str): The path to the folder containing the CSV files.

    set_index_to_fn: boolean

    Returns:
    Pandas DataFrame: A DataFrame containing all the blind QC data from the CSV files.
    """
    import glob as glob
    fns = glob.glob( modality_path )
    fns.sort()
    jdf = pd.DataFrame()
    for k in range(len(fns)):
        temp=pd.read_csv(fns[k])
        if not 'fn' in temp.keys():
            temp['fn']=fns[k]
        jdf=pd.concat( [jdf,temp])
    if set_index_to_fn:
        jdf.reset_index(drop=True)
        if "Unnamed: 0" in jdf.columns:
            holder=jdf.pop( "Unnamed: 0" )
        jdf.set_index('fn')
    return jdf


def outlierness_by_modality( qcdf, uid='fn', outlier_columns = ['noise', 'snr', 'cnr', 'psnr', 'ssim', 'mi','reflection_err', 'EVR', 'msk_vol'], verbose=False ):
    """
    Calculates outlierness scores for each modality in a dataframe based on given outlier columns using antspyt1w.loop_outlierness() and LOF.  LOF appears to be more conservative.

    Args:
    - qcdf: (Pandas DataFrame) Dataframe containing columns with outlier information for each modality.
    - uid: (str) Unique identifier for a subject. Default is 'fn'.
    - outlier_columns: (list) List of columns containing outlier information. Default is ['noise', 'snr', 'cnr', 'psnr', 'ssim', 'mi', 'reflection_err', 'EVR', 'msk_vol'].
    - verbose: (bool) If True, prints information for each modality. Default is False.

    Returns:
    - qcdf: (Pandas DataFrame) Updated dataframe with outlierness scores for each modality in the 'ol_loop' and 'ol_lof' column.  Higher values near 1 are more outlying.

    Raises:
    - ValueError: If uid is not present in the dataframe.

    Example:
    >>> df = pd.read_csv('data.csv')
    >>> outlierness_by_modality(df)
    """
    from PyNomaly import loop
    from sklearn.neighbors import LocalOutlierFactor
    qcdfout = qcdf.copy()
    if uid not in qcdfout.keys():
        raise ValueError(uid + " not in dataframe")
    if 'ol_loop' not in qcdfout.keys():
        qcdfout['ol_loop']=math.nan
    if 'ol_lof' not in qcdfout.keys():
        qcdfout['ol_lof']=math.nan
    for mod in get_valid_modalities( qc=True ):
        lof = LocalOutlierFactor()
        locsel = qcdfout["modality"] == mod
        rr = qcdfout[locsel][outlier_columns]
        if rr.shape[0] > 1:
            if verbose:
                print(mod)
            myneigh = np.min( [24, int(np.round(rr.shape[0]*0.5)) ] )
            temp = antspyt1w.loop_outlierness(rr.astype(float), standardize=True, extent=3, n_neighbors=myneigh, cluster_labels=None)
            qcdfout.loc[locsel,'ol_loop']=temp
            yhat = lof.fit_predict(rr)
            temp = lof.negative_outlier_factor_*(-1.0)
            temp = temp - temp.min()
            yhat[ yhat == 1] = 0
            yhat[ yhat == -1] = 1 # these are outliers
            qcdfout.loc[locsel,'ol_lof_decision']=yhat
            qcdfout.loc[locsel,'ol_lof']=temp/temp.max()
    return qcdfout


def nrg_format_path( projectID, subjectID, date, modality, imageID, separator='-' ):
    """
    create the NRG path on disk given the project, subject id, date, modality and image id

    Arguments
    ---------

    projectID : string for the project e.g. PPMI

    subjectID : string uniquely identifying the subject e.g. 0001

    date : string for the date usually 20550228 ie YYYYMMDD format

    modality : string should be one of T1w, T2Flair, rsfMRI, NM2DMT and DTI ... rsfMRI and DTI may also be DTI_LR, DTI_RL, rsfMRI_LR and rsfMRI_RL where the RL / LR relates to phase encoding direction (even if it is AP/PA)

    imageID : string uniquely identifying the specific image

    separator : default to -

    Returns
    -------
    the path where one would write the image on disk

    """
    thedirectory = os.path.join( str(projectID), str(subjectID), str(date), str(modality), str(imageID) )
    thefilename = str(projectID) + separator + str(subjectID) + separator + str(date) + separator + str(modality) + separator + str(imageID)
    return os.path.join( thedirectory, thefilename )

def highest_quality_repeat(mxdfin, idvar, visitvar, qualityvar):
    """
    This function returns a subset of the input dataframe that retains only the rows
    that correspond to the highest quality observation for each combination of ID and visit.

    Parameters:
    ----------
    mxdfin: pandas.DataFrame
        The input dataframe.
    idvar: str
        The name of the column that contains the ID variable.
    visitvar: str
        The name of the column that contains the visit variable.
    qualityvar: str
        The name of the column that contains the quality variable.

    Returns:
    -------
    pandas.DataFrame
        A subset of the input dataframe that retains only the rows that correspond
        to the highest quality observation for each combination of ID and visit.
    """
    if visitvar not in mxdfin.columns:
        raise ValueError("visitvar not in dataframe")
    if idvar not in mxdfin.columns:
        raise ValueError("idvar not in dataframe")
    if qualityvar not in mxdfin.columns:
        raise ValueError("qualityvar not in dataframe")

    vizzes = mxdfin[visitvar].unique()
    uids = mxdfin[idvar].unique()
    useit = np.zeros(mxdfin.shape[0], dtype=bool)

    for u in uids:
        losel = mxdfin[idvar] == u
        vizzesloc = mxdfin[losel][visitvar].unique()

        for v in vizzesloc:
            losel = (mxdfin[idvar] == u) & (mxdfin[visitvar] == v)
            mysnr = mxdfin.loc[losel, qualityvar]
            myw = np.where(losel)[0]

            if len(myw) > 1:
                if any(~np.isnan(mysnr)):
                    useit[myw[np.argmax(mysnr)]] = True
                else:
                    useit[myw] = True
            else:
                useit[myw] = True

    return mxdfin[useit]


def match_modalities( qc_dataframe, unique_identifier='fn', outlier_column='ol_loop',  verbose=False ):
    """
    Find the best multiple modality dataset at each time point

    :param qc_dataframe: quality control data frame with
    :param unique_identifier : the unique NRG filename for each image
    :param outlier_column: outlierness score used to identify the best image (pair) at a given date
    :param verbose: boolean
    :return: filtered matched modality data frame
    """
    import pandas as pd
    import numpy as np
    mmdf = best_mmm( qc_dataframe, 'T1w', outlier_column=outlier_column )['filt']
    fldf = best_mmm( qc_dataframe, 'T2Flair', outlier_column=outlier_column )['filt']
    nmdf = best_mmm( qc_dataframe, 'NM2DMT', outlier_column=outlier_column )['filt']
    rsdf = best_mmm( qc_dataframe, 'rsfMRI', outlier_column=outlier_column )['filt']
    dtdf = best_mmm( qc_dataframe, 'DTI', outlier_column=outlier_column )['filt']
    mmdf['flairid'] = np.nan
    mmdf['flairfn'] = np.nan
    mmdf['flairloop'] = np.nan
    mmdf['flairlof'] = np.nan
    mmdf['dtid1'] = np.nan
    mmdf['dtfn1'] = np.nan
    mmdf['dtloop1'] = np.nan
    mmdf['dtlof1'] = np.nan
    mmdf['dtid2'] = np.nan
    mmdf['dtfn2'] = np.nan
    mmdf['dtloop2'] = np.nan
    mmdf['dtlof2'] = np.nan
    mmdf['rsfid1'] = np.nan
    mmdf['rsffn1'] = np.nan
    mmdf['rsfloop1'] = np.nan
    mmdf['rsflof1'] = np.nan
    mmdf['rsfid2'] = np.nan
    mmdf['rsffn2'] = np.nan
    mmdf['rsfloop2'] = np.nan
    mmdf['rsflof2'] = np.nan
    for k in range(1,11):
        myid='nmid'+str(k)
        mmdf[myid] = np.nan
        myid='nmfn'+str(k)
        mmdf[myid] = np.nan
        myid='nmloop'+str(k)
        mmdf[myid] = np.nan
        myid='nmlof'+str(k)
        mmdf[myid] = np.nan
    if verbose:
        print( mmdf.shape )
    for k in range(mmdf.shape[0]):
        if verbose:
            if k % 100 == 0:
                progger = str( k ) # np.round( k / mmdf.shape[0] * 100 ) )
                print( progger, end ="...", flush=True)
        if dtdf is not None:
            locsel = (dtdf["subjectIDdate"] == mmdf["subjectIDdate"].iloc[k]) & (dtdf[outlier_column] < 0.5)
            if sum(locsel) == 1:
                mmdf.iloc[k, mmdf.columns.get_loc("dtid1")] = dtdf["imageID"][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("dtfn1")] = dtdf["fn"][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("dtloop1")] = dtdf[outlier_column][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("dtlof1")] = dtdf['ol_lof_decision'][locsel].values[0]
            elif sum(locsel) > 1:
                locdf = dtdf[locsel]
                dedupe = locdf[["snr","cnr"]].duplicated()
                locdf = locdf[~dedupe]
                if locdf.shape[0] > 1:
                    locdf = locdf.sort_values(outlier_column).iloc[:2]
                mmdf.iloc[k, mmdf.columns.get_loc("dtid1")] = locdf["imageID"].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("dtfn1")] = locdf["fn"].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("dtloop1")] = locdf[outlier_column].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("dtlof1")] = locdf['ol_lof_decision'][locsel].values[0]
                if locdf.shape[0] > 1:
                    mmdf.iloc[k, mmdf.columns.get_loc("dtid2")] = locdf["imageID"].values[1]
                    mmdf.iloc[k, mmdf.columns.get_loc("dtfn2")] = locdf["fn"].values[1]
                    mmdf.iloc[k, mmdf.columns.get_loc("dtloop2")] = locdf[outlier_column].values[1]
                    mmdf.iloc[k, mmdf.columns.get_loc("dtlof2")] = locdf['ol_lof_decision'][locsel].values[1]
        if rsdf is not None:
            locsel = (rsdf["subjectIDdate"] == mmdf["subjectIDdate"].iloc[k]) & (rsdf[outlier_column] < 0.5)
            if sum(locsel) == 1:
                mmdf.iloc[k, mmdf.columns.get_loc("rsfid1")] = rsdf["imageID"][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("rsffn1")] = rsdf["fn"][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("rsfloop1")] = rsdf[outlier_column][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("rsflof1")] = rsdf['ol_lof_decision'][locsel].values[0]
            elif sum(locsel) > 1:
                locdf = rsdf[locsel]
                dedupe = locdf[["snr","cnr"]].duplicated()
                locdf = locdf[~dedupe]
                if locdf.shape[0] > 1:
                    locdf = locdf.sort_values(outlier_column).iloc[:2]
                mmdf.iloc[k, mmdf.columns.get_loc("rsfid1")] = locdf["imageID"].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("rsffn1")] = locdf["fn"].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("rsfloop1")] = locdf[outlier_column].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("rsflof1")] = locdf['ol_lof_decision'].values[0]
                if locdf.shape[0] > 1:
                    mmdf.iloc[k, mmdf.columns.get_loc("rsfid2")] = locdf["imageID"].values[1]
                    mmdf.iloc[k, mmdf.columns.get_loc("rsffn2")] = locdf["fn"].values[1]
                    mmdf.iloc[k, mmdf.columns.get_loc("rsfloop2")] = locdf[outlier_column].values[1]
                    mmdf.iloc[k, mmdf.columns.get_loc("rsflof2")] = locdf['ol_lof_decision'].values[1]

        if fldf is not None:
            locsel = fldf['subjectIDdate'] == mmdf['subjectIDdate'].iloc[k]
            if locsel.sum() == 1:
                mmdf.iloc[k, mmdf.columns.get_loc("flairid")] = fldf['imageID'][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("flairfn")] = fldf['fn'][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("flairloop")] = fldf[outlier_column][locsel].values[0]
                mmdf.iloc[k, mmdf.columns.get_loc("flairlof")] = fldf['ol_lof_decision'][locsel].values[0]

        if nmdf is not None:
            locsel = nmdf['subjectIDdate'] == mmdf['subjectIDdate'].iloc[k]
            if locsel.sum() > 0:
                locdf = nmdf[locsel]
                for i in range(np.min( [10,locdf.shape[0]])):
                    nmid = "nmid"+str(i+1)
                    mmdf[nmid].iloc[k] = locdf['imageID'].iloc[i]
                    nmfn = "nmfn"+str(i+1)
                    mmdf[nmfn].iloc[k] = locdf['imageID'].iloc[i]
                    nmloop = "nmloop"+str(i+1)
                    mmdf[nmloop].iloc[k] = locdf[outlier_column].iloc[i]
                    nmloop = "nmlof"+str(i+1)
                    mmdf[nmloop].iloc[k] = locdf['ol_lof_decision'].iloc[i]

    return mmdf

def best_mmm( mmdf, wmod, mysep='-', outlier_column='ol_loop', verbose=False):
    """
    Selects the best repeats per modality.

    Args:
    wmod (str): the modality of the image ( 'T1w', 'T2Flair', 'NM2DMT' 'rsfMRI', 'DTI')

    mysep (str, optional): the separator used in the image file names. Defaults to '-'.

    outlier_name : column name for outlier score

    verbose (bool, optional): default True

    Returns:

    list: a list containing two metadata dataframes - raw and filt. raw contains all the metadata for the selected modality and filt contains the metadata filtered for highest quality repeats.

    """
    msel = mmdf['modality'] == wmod
    if wmod == 'rsfMRI':
        msel1 = mmdf['modality'] == 'rsfMRI'
        msel2 = mmdf['modality'] == 'rsfMRI_LR'
        msel3 = mmdf['modality'] == 'rsfMRI_RL'
        msel = msel1 | msel2
        msel = msel | msel3
    if wmod == 'DTI':
        msel1 = mmdf['modality'] == 'DTI'
        msel2 = mmdf['modality'] == 'DTI_LR'
        msel3 = mmdf['modality'] == 'DTI_RL'
        msel4 = mmdf['modality'] == 'DTIdwi'
        msel = msel1 | msel2 | msel3 | msel4
    if sum(msel) == 0:
        return {'raw': None, 'filt': None}
    uids = list(mmdf['fn'][msel])
    metasub = mmdf[msel]

    if verbose:
        print(f"{wmod} {(metasub.shape[0])} pre")

    metasub['subjectID']=math.nan
    metasub['date']=math.nan
    metasub['subjectIDdate']=math.nan
    metasub['imageID']=math.nan
    for k in range(len(uids)):
        temp = uids[k].split( mysep )
        metasub['subjectID'].iloc[k] = temp[1]
        metasub['date'].iloc[k] = temp[2]
        metasub['subjectIDdate'].iloc[k] = temp[1] + mysep + temp[2]
        metasub['imageID'].iloc[k] = temp[4]

    metasub['negol'] = metasub[outlier_column].max() - metasub[outlier_column]
    if 'date' not in metasub.keys():
        metasub['date']='NA'
    metasubq = highest_quality_repeat(metasub, 'fn', 'date', 'negol')

    if verbose:
        print(f"{wmod} {metasubq.shape[0]} post")

    return {'raw': metasub, 'filt': metasubq}

def mm_read( x, standardize_intensity=False, modality='' ):
    """
    read an image from a filename - same as ants.image_read (for now)

    standardize_intensity : boolean ; if True will set negative values to zero and normalize into the range of zero to one

    modality : not used
    """
    img = ants.image_read( x, reorient=False )
    if standardize_intensity:
        img[img<0.0]=0.0
        img=ants.iMath(img,'Normalize')
    return img

def mm_read_to_3d( x, slice=None, modality='' ):
    """
    read an image from a filename - and return as 3d or None if that is not possible
    """
    img = ants.image_read( x, reorient=False )
    if img.dimension < 3:
        return None
    elif img.dimension == 4:
        nslices = img.shape[3]
        if slice is None:
            sl = np.round( nslices * 0.5 )
        else:
            sl = slice
        if sl > nslices:
            sl = nslices-1
        return ants.slice_image( img, axis=3, idx=int(sl) )
    elif img.dimension == 3:
        return img
    return None

def image_write_with_thumbnail( x,  fn, y=None, thumb=True ):
    """
    will write the image and (optionally) a png thumbnail with (optional) overlay/underlay
    """
    ants.image_write( x, fn )
    if not thumb or x.components > 1:
        return
    thumb_fn=re.sub(".nii.gz","_3dthumb.png",fn)
    if thumb and x.dimension == 3:
        if y is None:
            try:
                ants.plot_ortho( x, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0 )
            except:
                pass
        else:
            try:
                ants.plot_ortho( y, x, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0 )
            except:
                pass
    if thumb and x.dimension == 4:
        thumb_fn=re.sub(".nii.gz","_4dthumb.png",fn)
        nslices = x.shape[3]
        sl = np.round( nslices * 0.5 )
        if sl > nslices:
            sl = nslices-1
        xview = ants.slice_image( x, axis=3, idx=int(sl) )
        if y is None:
            try:
                ants.plot_ortho( xview, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0 )
            except:
                pass
        else:
            if y.dimension == 3:
                try:
                    ants.plot_ortho(y, xview, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0 )
                except:
                    pass
    return


def mc_resample_image_to_target( x , y, interp_type='linear' ):
    """
    multichannel version of resample_image_to_target
    """
    xx=ants.split_channels( x )
    yy=ants.split_channels( y )[0]
    newl=[]
    for k in range(len(xx)):
        newl.append(  ants.resample_image_to_target( xx[k], yy, interp_type=interp_type ) )
    return ants.merge_channels( newl )

def nrg_filelist_to_dataframe( filename_list, myseparator="-" ):
    """
    convert a list of files in nrg format to a dataframe

    Arguments
    ---------
    filename_list : globbed list of files

    myseparator : string separator between nrg parts

    Returns
    -------

    df : pandas data frame

    """
    def getmtime(x):
        x= dt.datetime.fromtimestamp(os.path.getmtime(x)).strftime("%Y-%m-%d %H:%M:%d")
        return x
    df=pd.DataFrame(columns=['filename','file_last_mod_t','else','sid','visitdate','modality','uid'])
    df.set_index('filename')
    df['filename'] = pd.Series([file for file in filename_list ])
    # I applied a time modified file to df['file_last_mod_t'] by getmtime function
    df['file_last_mod_t'] = df['filename'].apply(lambda x: getmtime(x))
    for k in range(df.shape[0]):
        locfn=df['filename'].iloc[k]
        splitter=os.path.basename(locfn).split( myseparator )
        df['sid'].iloc[k]=splitter[1]
        df['visitdate'].iloc[k]=splitter[2]
        df['modality'].iloc[k]=splitter[3]
        temp = os.path.splitext(splitter[4])[0]
        df['uid'].iloc[k]=os.path.splitext(temp)[0]
    return df


def merge_timeseries_data( img_LR, img_RL, allow_resample=True ):
    """
    merge time series data into space of reference_image

    img_LR : image

    img_RL : image

    allow_resample : boolean

    """
    # concatenate the images into the reference space
    mimg=[]
    for kk in range( img_LR.shape[3] ):
        temp = ants.slice_image( img_LR, axis=3, idx=kk )
        mimg.append( temp )
    for kk in range( img_RL.shape[3] ):
        temp = ants.slice_image( img_RL, axis=3, idx=kk )
        if kk == 0:
            insamespace = ants.image_physical_space_consistency( temp, mimg[0] )
        if allow_resample and not insamespace :
            temp = ants.resample_image_to_target( temp, mimg[0] )
        mimg.append( temp )
    return ants.list_to_ndimage( img_LR, mimg )


def timeseries_reg(
    image,
    avg_b0,
    type_of_transform="Rigid",
    total_sigma=1.0,
    fdOffset=2.0,
    trim = 0,
    output_directory=None,
    verbose=False, **kwargs
):
    """
    Correct time-series data for motion - with deformation.

    Arguments
    ---------
    image: antsImage, usually ND where D=4.

    avg_b0: Fixed image b0 image

    type_of_transform : string
            A linear or non-linear registration type. Mutual information metric and rigid transformation by default.
            See ants registration for details.

    fdOffset: offset value to use in framewise displacement calculation

    trim : integer - trim this many images off the front of the time series

    output_directory : string
            output will be placed in this directory plus a numeric extension.

    verbose: boolean

    kwargs: keyword args
            extra arguments - these extra arguments will control the details of registration that is performed. see ants registration for more.

    Returns
    -------
    dict containing follow key/value pairs:
        `motion_corrected`: Moving image warped to space of fixed image.
        `motion_parameters`: transforms for each image in the time series.
        `FD`: Framewise displacement generalized for arbitrary transformations.

    Notes
    -----
    Control extra arguments via kwargs. see ants.registration for details.

    Example
    -------
    >>> import ants
    """
    idim = image.dimension
    ishape = image.shape
    nTimePoints = ishape[idim - 1]
    FD = np.zeros(nTimePoints)
    if type_of_transform is None:
        return {
            "motion_corrected": image,
            "motion_parameters": None,
            "FD": FD
        }

    remove_it=False
    if output_directory is None:
        remove_it=True
        output_directory = tempfile.mkdtemp()
    output_directory_w = output_directory + "/ts_reg/"
    os.makedirs(output_directory_w,exist_ok=True)
    ofnG = tempfile.NamedTemporaryFile(delete=False,suffix='global_deformation',dir=output_directory_w).name
    ofnL = tempfile.NamedTemporaryFile(delete=False,suffix='local_deformation',dir=output_directory_w).name
    if verbose:
        print('bold motcorr with ' + type_of_transform)
        print(output_directory_w)
        print(ofnG)
        print(ofnL)
        print("remove_it " + str( remove_it ) )

    # get a local deformation from slice to local avg space
    motion_parameters = list()
    motion_corrected = list()
    mask = ants.get_mask( avg_b0 )
    centerOfMass = mask.get_center_of_mass()
    npts = pow(2, idim - 1)
    pointOffsets = np.zeros((npts, idim - 1))
    myrad = np.ones(idim - 1).astype(int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(
        mask1, mask1, radius=myrad, spatial_info=True
    )["offsets"]
    mycols = list("xy")
    if idim - 1 == 3:
        mycols = list("xyz")
    useinds = list()
    for k in range(myoffsets.shape[0]):
        if abs(myoffsets[k, :]).sum() == (idim - 2):
            useinds.append(k)
        myoffsets[k, :] = myoffsets[k, :] * fdOffset / 2.0 + centerOfMass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)
    if verbose:
        print("Progress:")
    counter = round( nTimePoints / 10 )
    for k in range( nTimePoints):
        if verbose and ( ( k % counter ) ==  0 ) or ( k == (nTimePoints-1) ):
            myperc = round( k / nTimePoints * 100)
            print(myperc, end="%.", flush=True)
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.n4_bias_field_correction( temp )
        temp = ants.iMath(temp, "Normalize")
        txprefix = ofnL+str(k % 2).zfill(4)+"_"
        if temp.numpy().var() > 0:
            myrig = ants.registration(
                    avg_b0, temp,
                    type_of_transform='BOLDRigid',
                    outprefix=txprefix
                )
            if type_of_transform == 'SyN':
                myreg = ants.registration(
                    avg_b0, temp,
                    type_of_transform='SyNOnly',
                    total_sigma=total_sigma,
                    initial_transform=myrig['fwdtransforms'][0],
                    outprefix=txprefix,
                    **kwargs
                )
            else:
                myreg = myrig
            fdptsTxI = ants.apply_transforms_to_points(
                idim - 1, fdpts, myrig["fwdtransforms"]
            )
            if k > 0 and motion_parameters[k - 1] != "NA":
                fdptsTxIminus1 = ants.apply_transforms_to_points(
                    idim - 1, fdpts, motion_parameters[k - 1]
                )
            else:
                fdptsTxIminus1 = fdptsTxI
            # take the absolute value, then the mean across columns, then the sum
            FD[k] = (fdptsTxIminus1 - fdptsTxI).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
        else:
            motion_parameters.append("NA")

        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        if temp.numpy().var() > 0:
            img1w = ants.apply_transforms( avg_b0,
                temp,
                motion_parameters[k] )
            motion_corrected.append(img1w)
        else:
            motion_corrected.append(avg_b0)

    if remove_it:
        import shutil
        shutil.rmtree(output_directory, ignore_errors=True )

    if verbose:
        print("Done")
    d4siz = list(avg_b0.shape)
    d4siz.append( 2 )
    spc = list(ants.get_spacing( avg_b0 ))
    spc.append( ants.get_spacing(image)[3] )
    mydir = ants.get_direction( avg_b0 )
    mydir4d = ants.get_direction( image )
    mydir4d[0:3,0:3]=mydir
    myorg = list(ants.get_origin( avg_b0 ))
    myorg.append( 0.0 )
    avg_b0_4d = ants.make_image(d4siz,0,spacing=spc,origin=myorg,direction=mydir4d)
    return {
        "motion_corrected": ants.list_to_ndimage(avg_b0_4d, motion_corrected[trim:len(motion_corrected)]),
        "motion_parameters": motion_parameters[trim:len(motion_parameters)],
        "FD": FD[trim:len(FD)]
    }


def merge_dwi_data( img_LRdwp, bval_LR, bvec_LR, img_RLdwp, bval_RL, bvec_RL ):
    """
    merge motion and distortion corrected data if possible

    img_LRdwp : image

    bval_LR : array

    bvec_LR : array

    img_RLdwp : image

    bval_RL : array

    bvec_RL : array

    """
    insamespace = ants.image_physical_space_consistency( img_LRdwp, img_RLdwp )
    if not insamespace :
        raise ValueError('not insamespace ... corrected image pair should occupy the same physical space')

    bval_LR = np.concatenate([bval_LR,bval_RL])
    bvec_LR = np.concatenate([bvec_LR,bvec_RL])
    # concatenate the images
    mimg=[]
    for kk in range( img_LRdwp.shape[3] ):
            mimg.append( ants.slice_image( img_LRdwp, axis=3, idx=kk ) )
    for kk in range( img_RLdwp.shape[3] ):
            mimg.append( ants.slice_image( img_RLdwp, axis=3, idx=kk ) )
    img_LRdwp = ants.list_to_ndimage( img_LRdwp, mimg )
    return img_LRdwp, bval_LR, bvec_LR

def bvec_reorientation( motion_parameters, bvecs ):
    if motion_parameters is None:
        return bvecs
    n = len(motion_parameters)
    if n < 1:
        return bvecs
    from scipy.linalg import inv, polar
    from dipy.core.gradients import reorient_bvecs
    dipymoco = np.zeros( [n,3,3] )
    for myidx in range(n):
        if myidx < bvecs.shape[0]:
            dipymoco[myidx,:,:] = np.eye( 3 )
            if motion_parameters[myidx] != 'NA':
                temp = motion_parameters[myidx]
                if len(temp) == 4 :
                    temp1=temp[3] # FIXME should be composite of index 1 and 3
                    temp2=temp[1] # FIXME should be composite of index 1 and 3
                    txparam1 = ants.read_transform(temp1)
                    txparam1 = ants.get_ants_transform_parameters(txparam1)[0:9].reshape( [3,3])
                    txparam2 = ants.read_transform(temp2)
                    txparam2 = ants.get_ants_transform_parameters(txparam2)[0:9].reshape( [3,3])
                    Rinv = inv( np.dot( txparam2, txparam1 ) )
                elif len(temp) == 2 :
                    temp=temp[1] # FIXME should be composite of index 1 and 3
                    txparam = ants.read_transform(temp)
                    txparam = ants.get_ants_transform_parameters(txparam)[0:9].reshape( [3,3])
                    Rinv = inv( txparam )
                elif len(temp) == 3 :
                    temp1=temp[2] # FIXME should be composite of index 1 and 3
                    temp2=temp[1] # FIXME should be composite of index 1 and 3
                    txparam1 = ants.read_transform(temp1)
                    txparam1 = ants.get_ants_transform_parameters(txparam1)[0:9].reshape( [3,3])
                    txparam2 = ants.read_transform(temp2)
                    txparam2 = ants.get_ants_transform_parameters(txparam2)[0:9].reshape( [3,3])
                    Rinv = inv( np.dot( txparam2, txparam1 ) )
                else:
                    temp=temp[0]
                    txparam = ants.read_transform(temp)
                    txparam = ants.get_ants_transform_parameters(txparam)[0:9].reshape( [3,3])
                    Rinv = inv( txparam )
                bvecs[myidx,:] = np.dot( Rinv, bvecs[myidx,:] )
    return bvecs


def dti_reg(
    image,
    avg_b0,
    avg_dwi,
    bvals=None,
    bvecs=None,
    b0_idx=None,
    type_of_transform="Rigid",
    total_sigma=3.0,
    fdOffset=2.0,
    mask_csf=False,
    output_directory=None,
    verbose=False, **kwargs
):
    """
    Correct time-series data for motion - with optional deformation.

    Arguments
    ---------
        image: antsImage, usually ND where D=4.

        avg_b0: Fixed image b0 image

        avg_dwi: Fixed dwi same space as b0 image

        bvals: bvalues (file or array)

        bvecs: bvecs (file or array)

        b0_idx: indices of b0

        type_of_transform : string
            A linear or non-linear registration type. Mutual information metric and rigid transformation by default.
            See ants registration for details.

        fdOffset: offset value to use in framewise displacement calculation

        mask_csf: boolean

        output_directory : string
            output will be placed in this directory plus a numeric extension.

        verbose: boolean

        kwargs: keyword args
            extra arguments - these extra arguments will control the details of registration that is performed. see ants registration for more.

    Returns
    -------
    dict containing follow key/value pairs:
        `motion_corrected`: Moving image warped to space of fixed image.
        `motion_parameters`: transforms for each image in the time series.
        `FD`: Framewise displacement generalized for arbitrary transformations.

    Notes
    -----
    Control extra arguments via kwargs. see ants.registration for details.

    Example
    -------
    >>> import ants
    """
    idim = image.dimension
    ishape = image.shape
    nTimePoints = ishape[idim - 1]
    FD = np.zeros(nTimePoints)
    if bvals is not None and bvecs is not None:
        if isinstance(bvecs, str):
            bvals, bvecs = read_bvals_bvecs( bvals , bvecs  )
        else: # assume we already read them
            bvals = bvals.copy()
            bvecs = bvecs.copy()
    if type_of_transform is None:
        return {
            "motion_corrected": image,
            "motion_parameters": None,
            "FD": FD,
            'bvals':bvals,
            'bvecs':bvecs
        }

    from scipy.linalg import inv, polar
    from dipy.core.gradients import reorient_bvecs

    remove_it=False
    if output_directory is None:
        remove_it=True
        output_directory = tempfile.mkdtemp()
    output_directory_w = output_directory + "/dti_reg/"
    os.makedirs(output_directory_w,exist_ok=True)
    ofnG = tempfile.NamedTemporaryFile(delete=False,suffix='global_deformation',dir=output_directory_w).name
    ofnL = tempfile.NamedTemporaryFile(delete=False,suffix='local_deformation',dir=output_directory_w).name
    if verbose:
        print(output_directory_w)
        print(ofnG)
        print(ofnL)
        print("remove_it " + str( remove_it ) )

    if b0_idx is None:
        b0_idx = segment_timeseries_by_meanvalue( image )['highermeans']
    # first get a local deformation from slice to local avg space
    # then get a global deformation from avg to ref space
    ab0, adw = get_average_dwi_b0( image )
    mask = ants.get_mask(adw)
    motion_parameters = list()
    motion_corrected = list()
    centerOfMass = mask.get_center_of_mass()
    npts = pow(2, idim - 1)
    pointOffsets = np.zeros((npts, idim - 1))
    myrad = np.ones(idim - 1).astype(int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(
        mask1, mask1, radius=myrad, spatial_info=True
    )["offsets"]
    mycols = list("xy")
    if idim - 1 == 3:
        mycols = list("xyz")
    useinds = list()
    for k in range(myoffsets.shape[0]):
        if abs(myoffsets[k, :]).sum() == (idim - 2):
            useinds.append(k)
        myoffsets[k, :] = myoffsets[k, :] * fdOffset / 2.0 + centerOfMass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)


    if verbose:
        print("begin global distortion correction")
    # initrig = tra_initializer(avg_b0, ab0, max_rotation=60, transform=['rigid'], verbose=verbose)
    if mask_csf:
        bcsf = ants.threshold_image( avg_b0,"Otsu",2).threshold_image(1,1).morphology("open",1).iMath("GetLargestComponent")
    else:
        bcsf = avg_b0[0] * 0 + 1

    initrig = ants.registration( avg_b0, ab0*bcsf,'BOLDRigid',outprefix=ofnG)
    deftx = ants.registration( avg_dwi, adw*bcsf, 'SyNOnly',
        syn_metric='CC', syn_sampling=2,
        reg_iterations=[50,50,20],
        multivariate_extras=[ [ "CC", avg_b0, ab0*bcsf, 1, 2 ]],
        initial_transform=initrig['fwdtransforms'][0],
        outprefix=ofnG
        )['fwdtransforms']
    if verbose:
        print("end global distortion correction")

    if verbose:
        print("Progress:")
    counter = round( nTimePoints / 10 )
    for k in range(nTimePoints):
        if verbose and ( ( k % counter ) ==  0 ) or ( k == (nTimePoints-1) ):
            myperc = round( k / nTimePoints * 100)
            print(myperc, end="%.", flush=True)
        if k in b0_idx:
            fixed=ants.image_clone( ab0 )
        else:
            fixed=ants.image_clone( adw )
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.n4_bias_field_correction( temp )
        temp = ants.iMath(temp, "Normalize")
        txprefix = ofnL+str(k).zfill(4)+"rig_"
        txprefix2 = ofnL+str(k % 2).zfill(4)+"def_"
        if temp.numpy().var() > 0:
            myrig = ants.registration(
                    fixed, temp,
                    type_of_transform='BOLDRigid',
                    outprefix=txprefix,
                    **kwargs
                )
            if type_of_transform == 'SyN':
                myreg = ants.registration(
                    fixed, temp,
                    type_of_transform='SyNOnly',
                    total_sigma=total_sigma, grad_step=0.1,
                    initial_transform=myrig['fwdtransforms'][0],
                    outprefix=txprefix2,
                    **kwargs
                )
            else:
                myreg = myrig
            fdptsTxI = ants.apply_transforms_to_points(
                idim - 1, fdpts, myrig["fwdtransforms"]
            )
            if k > 0 and motion_parameters[k - 1] != "NA":
                fdptsTxIminus1 = ants.apply_transforms_to_points(
                    idim - 1, fdpts, motion_parameters[k - 1]
                )
            else:
                fdptsTxIminus1 = fdptsTxI
            # take the absolute value, then the mean across columns, then the sum
            FD[k] = (fdptsTxIminus1 - fdptsTxI).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
        else:
            motion_parameters.append("NA")

        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        if k in b0_idx:
            fixed=ants.image_clone( ab0 )
        else:
            fixed=ants.image_clone( adw )
        if temp.numpy().var() > 0:
            motion_parameters[k]=deftx+motion_parameters[k]
            img1w = ants.apply_transforms( avg_dwi,
                ants.slice_image(image, axis=idim - 1, idx=k),
                motion_parameters[k] )
            motion_corrected.append(img1w)
        else:
            motion_corrected.append(fixed)

    if verbose:
        print("Reorient bvecs")
    if bvecs is not None:
        bvecs = bvec_reorientation( motion_parameters, bvecs )

    if remove_it:
        import shutil
        shutil.rmtree(output_directory, ignore_errors=True )

    if verbose:
        print("Done")
    d4siz = list(avg_b0.shape)
    d4siz.append( 2 )
    spc = list(ants.get_spacing( avg_b0 ))
    spc.append( 1.0 )
    mydir = ants.get_direction( avg_b0 )
    mydir4d = ants.get_direction( image )
    mydir4d[0:3,0:3]=mydir
    myorg = list(ants.get_origin( avg_b0 ))
    myorg.append( 0.0 )
    avg_b0_4d = ants.make_image(d4siz,0,spacing=spc,origin=myorg,direction=mydir4d)
    return {
        "motion_corrected": ants.list_to_ndimage(avg_b0_4d, motion_corrected),
        "motion_parameters": motion_parameters,
        "FD": FD,
        'bvals':bvals,
        'bvecs':bvecs
    }


def mc_reg(
    image,
    fixed=None,
    type_of_transform="Rigid",
    mask=None,
    total_sigma=3.0,
    fdOffset=2.0,
    output_directory=None,
    verbose=False, **kwargs
):
    """
    Correct time-series data for motion - with deformation.

    Arguments
    ---------
        image: antsImage, usually ND where D=4.

        fixed: Fixed image to register all timepoints to.  If not provided,
            mean image is used.

        type_of_transform : string
            A linear or non-linear registration type. Mutual information metric and rigid transformation by default.
            See ants registration for details.

        fdOffset: offset value to use in framewise displacement calculation

        output_directory : string
            output will be named with this prefix plus a numeric extension.

        verbose: boolean

        kwargs: keyword args
            extra arguments - these extra arguments will control the details of registration that is performed. see ants registration for more.

    Returns
    -------
    dict containing follow key/value pairs:
        `motion_corrected`: Moving image warped to space of fixed image.
        `motion_parameters`: transforms for each image in the time series.
        `FD`: Framewise displacement generalized for arbitrary transformations.

    Notes
    -----
    Control extra arguments via kwargs. see ants.registration for details.

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('ch2'))
    >>> mytx = ants.motion_correction( fi )
    """
    remove_it=False
    if output_directory is None:
        remove_it=True
        output_directory = tempfile.mkdtemp()
    output_directory_w = output_directory + "/mc_reg/"
    os.makedirs(output_directory_w,exist_ok=True)
    ofnG = tempfile.NamedTemporaryFile(delete=False,suffix='global_deformation',dir=output_directory_w).name
    ofnL = tempfile.NamedTemporaryFile(delete=False,suffix='local_deformation',dir=output_directory_w).name
    if verbose:
        print(output_directory_w)
        print(ofnG)
        print(ofnL)

    idim = image.dimension
    ishape = image.shape
    nTimePoints = ishape[idim - 1]
    if fixed is None:
        fixed = ants.get_average_of_timeseries( image )
    if mask is None:
        mask = ants.get_mask(fixed)
    FD = np.zeros(nTimePoints)
    motion_parameters = list()
    motion_corrected = list()
    centerOfMass = mask.get_center_of_mass()
    npts = pow(2, idim - 1)
    pointOffsets = np.zeros((npts, idim - 1))
    myrad = np.ones(idim - 1).astype(int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(
        mask1, mask1, radius=myrad, spatial_info=True
    )["offsets"]
    mycols = list("xy")
    if idim - 1 == 3:
        mycols = list("xyz")
    useinds = list()
    for k in range(myoffsets.shape[0]):
        if abs(myoffsets[k, :]).sum() == (idim - 2):
            useinds.append(k)
        myoffsets[k, :] = myoffsets[k, :] * fdOffset / 2.0 + centerOfMass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)
    if verbose:
        print("Progress:")
    counter = 0
    for k in range(nTimePoints):
        mycount = round(k / nTimePoints * 100)
        if verbose and mycount == counter:
            counter = counter + 10
            print(mycount, end="%.", flush=True)
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "Normalize")
        if temp.numpy().var() > 0:
            myrig = ants.registration(
                    fixed, temp,
                    type_of_transform='Rigid',
                    outprefix=ofnL+str(k).zfill(4)+"_",
                    **kwargs
                )
            if type_of_transform == 'SyN':
                myreg = ants.registration(
                    fixed, temp,
                    type_of_transform='SyNOnly',
                    total_sigma=total_sigma,
                    initial_transform=myrig['fwdtransforms'][0],
                    outprefix=ofnL+str(k).zfill(4)+"_",
                    **kwargs
                )
            else:
                myreg = myrig
            fdptsTxI = ants.apply_transforms_to_points(
                idim - 1, fdpts, myreg["fwdtransforms"]
            )
            if k > 0 and motion_parameters[k - 1] != "NA":
                fdptsTxIminus1 = ants.apply_transforms_to_points(
                    idim - 1, fdpts, motion_parameters[k - 1]
                )
            else:
                fdptsTxIminus1 = fdptsTxI
            # take the absolute value, then the mean across columns, then the sum
            FD[k] = (fdptsTxIminus1 - fdptsTxI).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
            img1w = ants.apply_transforms( fixed,
                ants.slice_image(image, axis=idim - 1, idx=k),
                myreg["fwdtransforms"] )
            motion_corrected.append(img1w)
        else:
            motion_parameters.append("NA")
            motion_corrected.append(temp)

    if remove_it:
        import shutil
        shutil.rmtree(output_directory, ignore_errors=True )

    if verbose:
        print("Done")
    return {
        "motion_corrected": ants.list_to_ndimage(image, motion_corrected),
        "motion_parameters": motion_parameters,
        "FD": FD,
    }


def get_data( name=None, force_download=False, version=11, target_extension='.csv' ):
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


def get_models( version=3, force_download=True ):
    """
    Get ANTsPyMM data models

    force_download: boolean

    Returns
    -------
    None

    """
    os.makedirs(DATA_PATH, exist_ok=True)

    def download_data( version ):
        url = "https://figshare.com/ndownloader/articles/21718412/versions/"+str(version)
        target_file_name = "21718412.zip"
        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
            cache_subdir=DATA_PATH, extract = True )
        os.remove( DATA_PATH + target_file_name )

    if force_download:
        download_data( version = version )
    return



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


def super_res_mcimage( image,
    srmodel,
    truncation=[0.0001,0.995],
    poly_order='hist',
    target_range=[0,1],
    isotropic = False,
    verbose=False ):
    """
    Super resolution on a timeseries or multi-channel image

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
    """
    idim = image.dimension
    ishape = image.shape
    nTimePoints = ishape[idim - 1]
    mcsr = list()
    for k in range(nTimePoints):
        if verbose and (( k % 5 ) == 0 ):
            mycount = round(k / nTimePoints * 100)
            print(mycount, end="%.", flush=True)
        temp = ants.slice_image( image, axis=idim - 1, idx=k )
        temp = ants.iMath( temp, "TruncateIntensity", truncation[0], truncation[1] )
        mysr = antspynet.apply_super_resolution_model_to_image( temp, srmodel,
            target_range = target_range )
        if poly_order is not None:
            bilin = ants.resample_image_to_target( temp, mysr )
            if poly_order == 'hist':
                mysr = ants.histogram_match_image( mysr, bilin )
            else:
                mysr = antspynet.regression_match_image( mysr, bilin, poly_order = poly_order )
        if isotropic:
            mysr = down2iso( mysr )
        if k == 0:
            upshape = list()
            for j in range(len(ishape)-1):
                upshape.append( mysr.shape[j] )
            upshape.append( ishape[ idim-1 ] )
            if verbose:
                print("SR will be of voxel size:" + str(upshape) )
        mcsr.append( mysr )

    upshape = list()
    for j in range(len(ishape)-1):
        upshape.append( mysr.shape[j] )
    upshape.append( ishape[ idim-1 ] )
    if verbose:
        print("SR will be of voxel size:" + str(upshape) )

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


def get_average_rsf( x ):
    """
    automatically generates the average bold image with quick registration

    returns:
        avg_bold
    """
    output_directory = tempfile.mkdtemp()
    ofn = output_directory + "/w"
    bavg = ants.slice_image( x, axis=3, idx=0 ) * 0.0
    oavg = ants.slice_image( x, axis=3, idx=0 )
    for myidx in range(x.shape[3]):
        b0 = ants.slice_image( x, axis=3, idx=myidx)
        bavg = bavg + ants.registration(oavg,b0,'Rigid',outprefix=ofn)['warpedmovout']
    bavg = ants.iMath( bavg, 'Normalize' )
    oavg = ants.image_clone( bavg )
    bavg = oavg * 0.0
    for myidx in range(x.shape[3]):
        b0 = ants.slice_image( x, axis=3, idx=myidx)
        bavg = bavg + ants.registration(oavg,b0,'Rigid',outprefix=ofn)['warpedmovout']
    import shutil
    shutil.rmtree(output_directory, ignore_errors=True )
    return ants.n4_bias_field_correction(bavg)


def get_average_dwi_b0( x, fixed_b0=None, fixed_dwi=None, fast=False ):
    """
    automatically generates the average b0 and dwi and outputs both;
    maps dwi to b0 space at end.

    x : input image

    fixed_b0 : alernative reference space

    fixed_dwi : alernative reference space

    fast : boolean

    returns:
        avg_b0, avg_dwi
    """
    output_directory = tempfile.mkdtemp()
    ofn = output_directory + "/w"
    temp = segment_timeseries_by_meanvalue( x )
    b0_idx = temp['highermeans']
    non_b0_idx = temp['lowermeans']
    if ( fixed_b0 is None and fixed_dwi is None ) or fast:
        xavg = ants.slice_image( x, axis=3, idx=0 ) * 0.0
        bavg = ants.slice_image( x, axis=3, idx=0 ) * 0.0
        fixed_b0_use = ants.slice_image( x, axis=3, idx=b0_idx[0] )
        fixed_dwi_use = ants.slice_image( x, axis=3, idx=non_b0_idx[0] )
    else:
        temp_b0 = ants.slice_image( x, axis=3, idx=b0_idx[0] )
        temp_dwi = ants.slice_image( x, axis=3, idx=non_b0_idx[0] )
        xavg = fixed_b0 * 0.0
        bavg = fixed_b0 * 0.0
        tempreg = ants.registration( fixed_b0, temp_b0,'BOLDRigid')
        fixed_b0_use = tempreg['warpedmovout']
        fixed_dwi_use = ants.apply_transforms( fixed_b0, temp_dwi, tempreg['fwdtransforms'] )
    for myidx in range(x.shape[3]):
        b0 = ants.slice_image( x, axis=3, idx=myidx)
        if not fast:
            if not myidx in b0_idx:
                xavg = xavg + ants.registration(fixed_dwi_use,b0,'Rigid',outprefix=ofn)['warpedmovout']
            else:
                bavg = bavg + ants.registration(fixed_b0_use,b0,'Rigid',outprefix=ofn)['warpedmovout']
        else:
            if not myidx in b0_idx:
                xavg = xavg + b0
            else:
                bavg = bavg + b0
    bavg = ants.iMath( bavg, 'Normalize' )
    xavg = ants.iMath( xavg, 'Normalize' )
    import shutil
    shutil.rmtree(output_directory, ignore_errors=True )
    avgb0=ants.n4_bias_field_correction(bavg)
    avgdwi=ants.n4_bias_field_correction(xavg)
    avgdwi=ants.registration( avgb0, avgdwi, 'Rigid' )['warpedmovout']
    return avgb0, avgdwi

def dti_template(
    b_image_list=None,
    w_image_list=None,
    iterations=5,
    gradient_step=0.5,
    mask_csf=False,
    average_both=True,
    verbose=False
):
    """
    two channel version of build_template

    returns:
        avg_b0, avg_dwi
    """
    output_directory = tempfile.mkdtemp()
    mydeftx = tempfile.NamedTemporaryFile(delete=False,dir=output_directory).name
    tmp = tempfile.NamedTemporaryFile(delete=False,dir=output_directory,suffix=".nii.gz")
    wavgfn = tmp.name
    tmp2 = tempfile.NamedTemporaryFile(delete=False,dir=output_directory)
    comptx = tmp2.name
    weights = np.repeat(1.0 / len(b_image_list), len(b_image_list))
    weights = [x / sum(weights) for x in weights]
    w_initial_template = w_image_list[0]
    b_initial_template = b_image_list[0]
    b_initial_template = ants.iMath(b_initial_template,"Normalize")
    w_initial_template = ants.iMath(w_initial_template,"Normalize")
    if mask_csf:
        bcsf0 = ants.threshold_image( b_image_list[0],"Otsu",2).threshold_image(1,1).morphology("open",1).iMath("GetLargestComponent")
        bcsf1 = ants.threshold_image( b_image_list[1],"Otsu",2).threshold_image(1,1).morphology("open",1).iMath("GetLargestComponent")
    else:
        bcsf0 = b_image_list[0] * 0 + 1
        bcsf1 = b_image_list[1] * 0 + 1
    bavg = b_initial_template.clone() * bcsf0
    wavg = w_initial_template.clone() * bcsf0
    bcsf = [ bcsf0, bcsf1 ]
    for i in range(iterations):
        for k in range(len(w_image_list)):
            fimg=wavg
            mimg=w_image_list[k] * bcsf[k]
            fimg2=bavg
            mimg2=b_image_list[k] * bcsf[k]
            w1 = ants.registration(
                fimg, mimg, type_of_transform='antsRegistrationSyNQuick[s]',
                    multivariate_extras= [ [ "mattes", fimg2, mimg2, 1, 32 ]],
                    outprefix=mydeftx,
                    verbose=0 )
            txname = ants.apply_transforms(wavg, wavg,
                w1["fwdtransforms"], compose=comptx )
            if k == 0:
                txavg = ants.image_read(txname) * weights[k]
                wavgnew = ants.apply_transforms( wavg,
                    w_image_list[k] * bcsf[k], txname ).iMath("Normalize")
                bavgnew = ants.apply_transforms( wavg,
                    b_image_list[k] * bcsf[k], txname ).iMath("Normalize")
            else:
                txavg = txavg + ants.image_read(txname) * weights[k]
                if i >= (iterations-2) and average_both:
                    wavgnew = wavgnew+ants.apply_transforms( wavg,
                        w_image_list[k] * bcsf[k], txname ).iMath("Normalize")
                    bavgnew = bavgnew+ants.apply_transforms( wavg,
                        b_image_list[k] * bcsf[k], txname ).iMath("Normalize")
        if verbose:
            print("iteration:",str(i),str(txavg.abs().mean()))
        wscl = (-1.0) * gradient_step
        txavg = txavg * wscl
        ants.image_write( txavg, wavgfn )
        wavg = ants.apply_transforms(wavg, wavgnew, wavgfn).iMath("Normalize")
        bavg = ants.apply_transforms(bavg, bavgnew, wavgfn).iMath("Normalize")
    import shutil
    shutil.rmtree( output_directory, ignore_errors=True )
    if verbose:
        print("done")
    return bavg, wavg

def t1_based_dwi_brain_extraction(
    t1w_head,
    t1w,
    dwi,
    b0_idx = None,
    transform='Rigid',
    deform=None,
    verbose=False
):
    """
    Map a t1-based brain extraction to b0 and return a mask and average b0

    Arguments
    ---------
    t1w_head : an antsImage of the hole head

    t1w : an antsImage probably but not necessarily T1-weighted

    dwi : an antsImage holding B0 and DWI

    b0_idx : the indices of the B0; if None, use segment_timeseries_by_meanvalue to guess

    transform : string Rigid or other ants.registration tx type

    deform : follow up transform with deformation

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
    reg = tra_initializer( b0_avg, t1w, n_simulations=12,   verbose=verbose )
    if deform is not None:
        reg = ants.registration( b0_avg, t1w,
            'SyNOnly',
            total_sigma=0.5,
            initial_transform=reg['fwdtransforms'][0],
            verbose=False )
    outmsk = ants.apply_transforms( b0_avg, t1bxt, reg['fwdtransforms'], interpolator='linear').threshold_image( 0.5, 1.0 )
    return  {
    'b0_avg':b0_avg,
    'b0_mask':outmsk }

def mc_denoise( x, ratio = 0.5 ):
    """
    ants denoising for timeseries (4D)

    Arguments
    ---------
    x : an antsImage 4D

    ratio : weight between 1 and 0 - lower weights bring result closer to initial image

    Returns
    -------
    denoised time series

    """
    dwpimage = []
    for myidx in range(x.shape[3]):
        b0 = ants.slice_image( x, axis=3, idx=myidx)
        dnzb0 = ants.denoise_image( b0, p=1,r=1,noise_model='Gaussian' )
        dwpimage.append( dnzb0 * ratio + b0 * (1.0-ratio) )
    return ants.list_to_ndimage( x, dwpimage )

def tsnr( x, mask, indices=None ):
    """
    3D temporal snr image from a 4D time series image ... the matrix is normalized to range of 0,1

    x: image

    mask : mask

    indices: indices to use

    returns a 3D image
    """
    M = ants.timeseries_to_matrix( x, mask )
    M = M - M.min()
    M = M / M.max()
    if indices is not None:
        M=M[indices,:]
    stdM = np.std(M, axis=0 )
    stdM[np.isnan(stdM)] = 0
    tt = round( 0.975*100 )
    threshold_std = np.percentile( stdM, tt )
    tsnrimage = ants.make_image( mask, stdM )
    return tsnrimage

def dvars( x,  mask, indices=None ):
    """
    dvars on a time series image ... the matrix is normalized to range of 0,1

    x: image

    mask : mask

    indices: indices to use

    returns an array
    """
    M = ants.timeseries_to_matrix( x, mask )
    M = M - M.min()
    M = M / M.max()
    if indices is not None:
        M=M[indices,:]
    DVARS = np.zeros( M.shape[0] )
    for i in range(1, M.shape[0] ):
        vecdiff = M[i-1,:] - M[i,:]
        DVARS[i] = np.sqrt( ( vecdiff * vecdiff ).mean() )
    DVARS[0] = DVARS.mean()
    return DVARS


def slice_snr( x,  background_mask, foreground_mask, indices=None ):
    """
    slice-wise SNR on a time series image

    x: image

    background_mask : mask - maybe CSF or background or dilated brain mask minus original brain mask

    foreground_mask : mask - maybe cortex or WM or brain mask

    indices: indices to use

    returns an array
    """
    xuse=ants.iMath(x,"Normalize")
    MB = ants.timeseries_to_matrix( xuse, background_mask )
    MF = ants.timeseries_to_matrix( xuse, foreground_mask )
    if indices is not None:
        MB=MB[indices,:]
        MF=MF[indices,:]
    ssnr = np.zeros( MB.shape[0] )
    for i in range( MB.shape[0] ):
        ssnr[i]=MF[i,:].mean()/MB[i,:].std()
    ssnr[np.isnan(ssnr)] = 0
    return ssnr


def impute_fa( fa, md ):
    """
    impute bad values in dti, fa, md
    """
    def imputeit( x, fa ):
        badfa=ants.threshold_image(fa,1,1)
        if badfa.max() == 1:
            temp=ants.image_clone(x)
            temp[badfa==1]=0
            temp=ants.iMath(temp,'GD',2)
            x[ badfa==1 ]=temp[badfa==1]
        return x
    md=imputeit( md, fa )
    fa=imputeit( ants.image_clone(fa), fa )
    return fa, md

def trim_dti_mask( fa, mask, param=4.0 ):
    """
    trim the dti mask to get rid of bright fa rim

    this function erodes the famask by param amount then segments the rim into
    bright and less bright parts.  the bright parts are trimmed from the mask
    and the remaining edges are cleaned up a bit with closing.

    param: closing radius unit is in physical space
    """
    spacing = ants.get_spacing(mask)
    spacing_product = np.prod( spacing )
    spcmin = min( spacing )
    paramVox = int(np.round( param / spcmin ))
    trim_mask = ants.image_clone( mask )
    trim_mask = ants.iMath( trim_mask, "FillHoles" )
    edgemask = trim_mask - ants.iMath( trim_mask, "ME", paramVox )
    maxk=4
    edgemask = ants.threshold_image( fa * edgemask, "Otsu", maxk )
    edgemask = ants.threshold_image( edgemask, maxk-1, maxk )
    trim_mask[edgemask >= 1 ]=0
    trim_mask = ants.iMath(trim_mask,"ME",paramVox-1)
    trim_mask = ants.iMath(trim_mask,'GetLargestComponent')
    trim_mask = ants.iMath(trim_mask,"MD",paramVox-1)
    return trim_mask

def dipy_dti_recon(
    image,
    bvalsfn,
    bvecsfn,
    mask = None,
    b0_idx = None,
    mask_dilation = 2,
    mask_closing = 5,
    fit_method='WLS',
    trim_the_mask=2,
    verbose=False ):
    """
    DiPy DTI reconstruction - building on the DiPy basic DTI example

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
    """


    if b0_idx is None:
        b0_idx = segment_timeseries_by_meanvalue( image )['highermeans']

    if isinstance(bvecsfn, str):
        bvals, bvecs = read_bvals_bvecs( bvalsfn , bvecsfn   )
    else: # assume we already read them
        bvals = bvalsfn.copy()
        bvecs = bvecsfn.copy()

    b0 = ants.slice_image( image, axis=3, idx=b0_idx[0] )
    bxtmod='bold'
    bxtmod='t2'
    constant_mask=False
    if mask is not None:
        constant_mask=True
        mask = ants.resample_image_to_target( mask, b0, interp_type='nearestNeighbor')
    else:
        mask = antspynet.brain_extraction( b0, bxtmod ).threshold_image(0.5,1).iMath("GetLargestComponent").morphology("close",2).iMath("FillHoles")
    if mask_closing > 0 and not constant_mask :
        mask = ants.morphology( mask, "close", mask_closing ) # good
    maskdil = ants.iMath( mask, "MD", mask_dilation )

    if verbose:
        print("recon dti.TensorModel",flush=True)

    def justthefit( gtab, fit_method, imagein, maskin ):
        if fit_method is None:
            return None, None, None, None
        maskedimage=[]
        for myidx in range(imagein.shape[3]):
            b0 = ants.slice_image( imagein, axis=3, idx=myidx)
            maskedimage.append( b0 * maskin )
        maskedimage = ants.list_to_ndimage( imagein, maskedimage )
        maskdata = maskedimage.numpy()
        tenmodel = dti.TensorModel(gtab,fit_method=fit_method)
        tenfit = tenmodel.fit(maskdata)
        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 1
        FA = np.clip(FA, 0, 1)
        MD1 = dti.mean_diffusivity(tenfit.evals)
        MD1 = ants.copy_image_info( b0, ants.from_numpy( MD1.astype(np.float32) ) )
        FA = ants.copy_image_info(  b0, ants.from_numpy( FA.astype(np.float32) ) )
        FA, MD1 = impute_fa( FA, MD1 )
        RGB = color_fa(FA.numpy(), tenfit.evecs)
        RGB = ants.from_numpy( RGB.astype(np.float32) )
        RGB0 = ants.copy_image_info( b0, ants.slice_image( RGB, axis=3, idx=0 ) )
        RGB1 = ants.copy_image_info( b0, ants.slice_image( RGB, axis=3, idx=1 ) )
        RGB2 = ants.copy_image_info( b0, ants.slice_image( RGB, axis=3, idx=2 ) )
        RGB = ants.merge_channels( [RGB0,RGB1,RGB2] )
        return tenfit, FA, MD1, RGB

    gtab = gradient_table(bvals, bvecs)
    tenfit, FA, MD1, RGB = justthefit( gtab, fit_method, image, maskdil )
    if verbose:
        print("recon dti.TensorModel done",flush=True)

    # change the brain mask based on high FA values
    if trim_the_mask > 0 and fit_method is not None:
        mask = trim_dti_mask( FA, mask, trim_the_mask )
        tenfit, FA, MD1, RGB = justthefit( gtab, fit_method, image, mask )

    return {
        'tensormodel' : tenfit,
        'MD' : MD1 ,
        'FA' : FA ,
        'RGB' : RGB,
        'dwi_mask':mask,
        'bvals':bvals,
        'bvecs':bvecs
        }


def concat_dewarp(
        refimg,
        originalDWI,
        physSpaceDWI,
        dwpTx,
        motion_parameters,
        motion_correct=True,
        verbose=False ):
    """
    Apply concatentated motion correction and dewarping transforms to timeseries image.

    Arguments
    ---------

    refimg : an antsImage defining the reference domain (3D)

    originalDWI : the antsImage in original (not interpolated space) (4D)

    physSpaceDWI : ants antsImage defining the physical space of the mapping (4D)

    dwpTx : dewarping transform

    motion_parameters : previously computed list of motion parameters

    motion_correct : boolean

    verbose : boolean

    """
    # apply the dewarping tx to the original dwi and reconstruct again
    # NOTE: refimg must be in the same space for this to work correctly
    # due to the use of ants.list_to_ndimage( originalDWI, dwpimage )
    dwpimage = []
    for myidx in range(originalDWI.shape[3]):
        b0 = ants.slice_image( originalDWI, axis=3, idx=myidx)
        concatx = dwpTx.copy()
        if motion_correct:
            concatx = concatx + motion_parameters[myidx]
        if verbose and myidx == 0:
            print("dwp parameters")
            print( dwpTx )
            print("Motion parameters")
            print( motion_parameters[myidx] )
            print("concat parameters")
            print(concatx)
        warpedb0 = ants.apply_transforms( refimg, b0, concatx,
            interpolator='nearestNeighbor' )
        dwpimage.append( warpedb0 )
    return ants.list_to_ndimage( physSpaceDWI, dwpimage )


def joint_dti_recon(
    img_LR,
    bval_LR,
    bvec_LR,
    jhu_atlas,
    jhu_labels,
    reference_B0,
    reference_DWI,
    srmodel = None,
    img_RL = None,
    bval_RL = None,
    bvec_RL = None,
    t1w = None,
    brain_mask = None,
    motion_correct = None,
    dewarp_modality = 'FA',
    denoise=False,
    fit_method='WLS',
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

    Arguments
    ---------

    img_LR : an antsImage holding B0 and DWI LR acquisition

    bval_LR : bvalue filename LR

    bvec_LR : bvector filename LR

    jhu_atlas : atlas FA image

    jhu_labels : atlas labels

    reference_B0 : the "target" B0 image space

    reference_DWI : the "target" DW image space

    srmodel : optional h5 (tensorflow) model

    img_RL : an antsImage holding B0 and DWI RL acquisition

    bval_RL : bvalue filename RL

    bvec_RL : bvector filename RL

    t1w : antsimage t1w neuroimage (brain-extracted)

    brain_mask : mask for the DWI - just 3D

    motion_correct : None Rigid or SyN

    dewarp_modality : string average_dwi, average_b0, MD or FA

    denoise: boolean

    fit_method : string one of WLS LS NLLS or restore - see import dipy.reconst.dti as dti and help(dti.TensorModel)

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

    def fix_dwi_shape( img, bvalfn, bvecfn ):
        if isinstance(bvecfn, str):
            bvals, bvecs = read_bvals_bvecs( bvalfn , bvecfn   )
        if bvecs.shape[0] < img.shape[3]:
            imgout = ants.from_numpy( img[:,:,:,0:bvecs.shape[0]] )
            imgout = ants.copy_image_info( img, imgout )
            return( imgout )
        else:
            return( img )

    img_LR = fix_dwi_shape( img_LR, bval_LR, bvec_LR )
    if denoise and srmodel is None:
        img_LR = mc_denoise( img_LR )
    if img_RL is not None:
        img_RL = fix_dwi_shape( img_RL, bval_RL, bvec_RL )
        if denoise and srmodel is None:
            img_RL = mc_denoise( img_RL )

    if brain_mask is not None:
        maskInRightSpace = ants.image_physical_space_consistency( brain_mask, reference_B0 )
        if not maskInRightSpace :
            raise ValueError('not maskInRightSpace ... provided brain mask should be in reference_B0 space')

    if img_RL is not None :
        if verbose:
            print("img_RL correction")
        reg_RL = dti_reg(
            img_RL,
            avg_b0=reference_B0,
            avg_dwi=reference_DWI,
            bvals=bval_RL,
            bvecs=bvec_RL,
            type_of_transform=motion_correct,
            verbose=True )
    else:
        reg_RL=None


    if verbose:
        print("img_LR correction")
    reg_LR = dti_reg(
            img_LR,
            avg_b0=reference_B0,
            avg_dwi=reference_DWI,
            bvals=bval_LR,
            bvecs=bvec_LR,
            type_of_transform=motion_correct,
            verbose=True )

    ts_LR_avg = None
    ts_RL_avg = None
    reg_its = [100,50,10]
    img_LRdwp = ants.image_clone( reg_LR[ 'motion_corrected' ] )
    if img_RL is not None:
        img_RLdwp = ants.image_clone( reg_RL[ 'motion_corrected' ] )
        if srmodel is not None:
            if verbose:
                print("convert img_RL_dwp to img_RL_dwp_SR")
            img_RLdwp = super_res_mcimage( img_RLdwp, srmodel, isotropic=True,
                        verbose=verbose )
    if srmodel is not None:
        reg_its = [100] + reg_its
        if verbose:
            print("convert img_LR_dwp to img_LR_dwp_SR")
        img_LRdwp = super_res_mcimage( img_LRdwp, srmodel, isotropic=True,
                verbose=verbose )
    if verbose:
        print("recon after distortion correction", flush=True)

    if img_RL is not None:
        img_LRdwp, bval_LR, bvec_LR = merge_dwi_data(
            img_LRdwp, reg_LR['bvals'], reg_LR['bvecs'],
            img_RLdwp, reg_RL['bvals'], reg_RL['bvecs']
        )
    else:
        bval_LR=reg_LR['bvals']
        bvec_LR=reg_LR['bvecs']

    if verbose:
        print("final recon", flush=True)
        print(img_LRdwp)
    recon_LR_dewarp = dipy_dti_recon(
            img_LRdwp, bval_LR, bvec_LR,
            mask = brain_mask,
            fit_method=fit_method,
            mask_dilation=0, verbose=True )
    if verbose:
        print("recon done", flush=True)

    if img_RL is not None:
        fdjoin = [ reg_LR['FD'],
                   reg_RL['FD'] ]
        framewise_displacement=np.concatenate( fdjoin )
    else:
        framewise_displacement=reg_LR['FD']

    motion_count = ( framewise_displacement > 1.5  ).sum()
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
        'MD_JHU_labels_edited',
        reconMD,
        OR_FA_jhulabels)
    df_MD_JHU_ORRL_bfwide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
            {'df_MD_JHU_ORRL' : df_MD_JHU_ORRL},
            col_names = ['Mean'] )

    temp = segment_timeseries_by_meanvalue( img_LRdwp )
    b0_idx = temp['highermeans']
    non_b0_idx = temp['lowermeans']

    nonbrainmask = ants.iMath( recon_LR_dewarp['dwi_mask'], "MD",2) - recon_LR_dewarp['dwi_mask']
    fgmask = ants.threshold_image( reconFA, 0.5 , 1.0).iMath("GetLargestComponent")
    bgmask = ants.threshold_image( reconFA, 1e-4 , 0.1)
    fa_SNR = 0.0
    fa_SNR = mask_snr( reconFA, bgmask, fgmask, bias_correct=False )
    fa_evr = antspyt1w.patch_eigenvalue_ratio( reconFA, 512, [16,16,16], evdepth = 0.9, mask=recon_LR_dewarp['dwi_mask'] )

    return {
        'recon_fa':reconFA,
        'recon_fa_summary':df_FA_JHU_ORRL_bfwide,
        'recon_md':reconMD,
        'recon_md_summary':df_MD_JHU_ORRL_bfwide,
        'jhu_labels':OR_FA_jhulabels,
        'jhu_registration':OR_FA2JHUreg,
        'reg_LR':reg_LR,
        'reg_RL':reg_RL,
        'dtrecon_LR_dewarp':recon_LR_dewarp,
        'dwi_LR_dewarped':img_LRdwp,
        'bval_LR':bval_LR,
        'bvec_LR':bvec_LR,
        'bval_RL':bval_RL,
        'bvec_RL':bvec_RL,
        'b0avg': reference_B0,
        'dwiavg': reference_DWI,
        'framewise_displacement':framewise_displacement,
        'high_motion_count': motion_count,
        'tsnr_b0': tsnr( img_LRdwp, recon_LR_dewarp['dwi_mask'], b0_idx),
        'tsnr_dwi': tsnr( img_LRdwp, recon_LR_dewarp['dwi_mask'], non_b0_idx),
        'dvars_b0': dvars( img_LRdwp, recon_LR_dewarp['dwi_mask'], b0_idx),
        'dvars_dwi': dvars( img_LRdwp, recon_LR_dewarp['dwi_mask'], non_b0_idx),
        'ssnr_b0': slice_snr( img_LRdwp, bgmask , fgmask, b0_idx),
        'ssnr_dwi': slice_snr( img_LRdwp, bgmask, fgmask, non_b0_idx),
        'fa_evr': fa_evr,
        'fa_SNR': fa_SNR
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
    if foreground_mask.sum() <= 1 or background_mask.sum() <= 1:
        return 0
    xbc = ants.iMath( x - x.min(), "Normalize" )
    if bias_correct:
        xbc = ants.n3_bias_field_correction( xbc )
    xbc = ants.iMath( xbc - xbc.min(), "Normalize" )
    signal = (xbc[ foreground_mask == 1] ).mean()
    noise = (xbc[ background_mask == 1] ).std()
    return signal / noise


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
    fit_method='WLS',
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

    fit_method : string one of WLS LS NLLS or restore - see import dipy.reconst.dti as dti and help(dti.TensorModel)

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
    dti_model = dti.TensorModel(gtab,fit_method=fit_method)
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
        # if os.getenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        #    mynump = os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']
        # current_openblas = os.environ.get('OPENBLAS_NUM_THREADS', '')
        # current_mkl = os.environ.get('MKL_NUM_THREADS', '')
        # os.environ['DIPY_OPENBLAS_NUM_THREADS'] = current_openblas
        # os.environ['DIPY_MKL_NUM_THREADS'] = current_mkl
        # os.environ['OPENBLAS_NUM_THREADS'] = '1'
        # os.environ['MKL_NUM_THREADS'] = '1'
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
        if False:
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
        target_image_mask = ants.image_clone( target_image ) * 0.0
        target_image_mask[ target_image != 0 ] = 1
        cortmapped = ants.apply_transforms(
            target_image,
            seg,
            mytx, interpolator='nearestNeighbor' ) * target_image_mask
        mapped = antspyt1w.map_intensity_to_dataframe(
            mapname,
            target_image,
            cortmapped )
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


def tra_initializer( fixed, moving, n_simulations=32, max_rotation=30,
    transform=['rigid'], verbose=False ):
    """
    multi-start multi-transform registration solution - based on ants.registration

    fixed: fixed image

    moving: moving image

    n_simulations : number of simulations

    max_rotation : maximum rotation angle

    transform : list of transforms to loop through

    verbose : boolean

    """
    if True:
        output_directory = tempfile.mkdtemp()
        output_directory_w = output_directory + "/tra_reg/"
        os.makedirs(output_directory_w,exist_ok=True)
        bestmi = math.inf
        myorig = list(ants.get_origin( fixed ))
        mymax = 0;
        for k in range(len( myorig ) ):
            if abs(myorig[k]) > mymax:
                mymax = abs(myorig[k])
        maxtrans = mymax * 0.05
        bestreg=ants.registration( fixed,moving,'Translation',
            outprefix=output_directory_w+"trans")
        initx = ants.read_transform( bestreg['fwdtransforms'][0] )
        for mytx in transform:
            regtx = 'Rigid'
            with tempfile.NamedTemporaryFile(suffix='.h5') as tp:
                if mytx == 'translation':
                    regtx = 'Translation'
                    rRotGenerator = ants.contrib.RandomTranslate3D( ( maxtrans*(-1.0), maxtrans ), reference=fixed )
                elif mytx == 'affine':
                    regtx = 'Affine'
                    rRotGenerator = ants.contrib.RandomRotate3D( ( maxtrans*(-1.0), maxtrans ), reference=fixed )
                else:
                    rRotGenerator = ants.contrib.RandomRotate3D( ( max_rotation*(-1.0), max_rotation ), reference=fixed )
                for k in range(n_simulations):
                    simtx = ants.compose_ants_transforms( [rRotGenerator.transform(), initx] )
                    ants.write_transform( simtx, tp.name )
                    if k > 0:
                        reg = ants.registration( fixed, moving, regtx,
                            initial_transform=tp.name,
                            outprefix=output_directory_w+"reg"+str(k),
                            verbose=False )
                    else:
                        reg = ants.registration( fixed, moving,
                            regtx,
                            outprefix=output_directory_w+"reg"+str(k),
                            verbose=False )
                    mymi = math.inf
                    temp = reg['warpedmovout']
                    myvar = temp.numpy().var()
                    if verbose:
                        print( str(k) + " : " + regtx  + " : " + mytx + " _var_ " + str( myvar ) )
                    if myvar > 0 :
                        mymi = ants.image_mutual_information( fixed, temp )
                        if mymi < bestmi:
                            if verbose:
                                print( "mi @ " + str(k) + " : " + str(mymi), flush=True)
                            bestmi = mymi
                            bestreg = reg
        return bestreg

def neuromelanin( list_nm_images, t1, t1_head, t1lab, brain_stem_dilation=8,
    bias_correct=True,
    denoise=None,
    srmodel=None,
    target_range=[0,1],
    poly_order='hist',
    normalize_nm = False,
    verbose=False ) :

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

  """

  fnt=os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni.nii.gz" )
  fntNM=os.path.expanduser("~/.antspymm/CIT168_T1w_700um_pad_adni_NM_norm_avg.nii.gz" )
  fntbst=os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni_brainstem.nii.gz")
  fnslab=os.path.expanduser("~/.antspyt1w/CIT168_MT_Slab_adni.nii.gz")
  fntseg=os.path.expanduser("~/.antspyt1w/det_atlas_25_pad_LR_adni.nii.gz")

  template = mm_read( fnt )
  templateNM = ants.iMath( mm_read( fntNM ), "Normalize" )
  templatebstem = mm_read( fntbst ).threshold_image( 1, 1000 )
  # reg = ants.registration( t1, template, 'antsRegistrationSyNQuickRepro[s]' )
  reg = ants.registration( t1, template, 'SyN' )
  # map NM avg to t1 for neuromelanin processing
  nmavg2t1 = ants.apply_transforms( t1, templateNM,
    reg['fwdtransforms'], interpolator='linear' )
  slab2t1 = ants.threshold_image( nmavg2t1, "Otsu", 2 ).threshold_image(1,2).iMath("MD",1).iMath("FillHoles")
  # map brain stem and slab to t1 for neuromelanin processing
  bstem2t1 = ants.apply_transforms( t1, templatebstem,
    reg['fwdtransforms'],
    interpolator='nearestNeighbor' ).iMath("MD",1)
  slab2t1B = ants.apply_transforms( t1, mm_read( fnslab ),
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
    if verbose:
        print(str(k) + " of " + str(len( list_nm_images ) ) )
    current_image = ants.registration( list_nm_images[k], nm_avg,
        type_of_transform = 'Rigid' )
    txlist.append( current_image['fwdtransforms'][0] )
    current_image = current_image['warpedfixout']
    nm_avg_new = nm_avg_new + current_image / len( list_nm_images )
  nm_avg = nm_avg_new

  if verbose:
      print("do slab registration to map anatomy to NM space")
  t1c = ants.crop_image( t1_head, slab2t1 ).iMath("Normalize") # old way
  nmavg2t1c = ants.crop_image( nmavg2t1, slab2t1 ).iMath("Normalize")
  # slabreg = ants.registration( nm_avg, nmavg2t1c, 'Rigid' )
  slabreg = tra_initializer( nm_avg, t1c, verbose=verbose )
  if False:
      slabregT1 = tra_initializer( nm_avg, t1c, verbose=verbose  )
      miNM = ants.image_mutual_information( ants.iMath(nm_avg,"Normalize"),
            ants.iMath(slabreg0['warpedmovout'],"Normalize") )
      miT1 = ants.image_mutual_information( ants.iMath(nm_avg,"Normalize"),
            ants.iMath(slabreg1['warpedmovout'],"Normalize") )
      if miT1 < miNM:
        slabreg = slabregT1
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
          temp = antspynet.apply_super_resolution_model_to_image(
                crop_nm_list[k], srmodel, target_range=target_range,
                regression_order=None )
          if poly_order is not None:
              bilin = ants.resample_image_to_target( crop_nm_list[k], temp )
              if poly_order == 'hist':
                  temp = ants.histogram_match_image( temp, bilin )
              else:
                  temp = antspynet.regression_match_image( temp, bilin, poly_order = poly_order )
          crop_nm_list[k] = temp

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
                'BOLDRigid' )
            warpednext = ants.apply_transforms(
                nm_avg_cropped_new,
                crop_nm_list[k],
                myreg['fwdtransforms'] )
            nm_avg_cropped_new = nm_avg_cropped_new + warpednext
      nm_avg_cropped = nm_avg_cropped_new / len( crop_nm_list )

  slabregUpdated = tra_initializer( nm_avg_cropped, t1c, verbose=verbose  )
  tempOrig = ants.apply_transforms( nm_avg_cropped_new, t1c, slabreg['fwdtransforms'] )
  tempUpdate = ants.apply_transforms( nm_avg_cropped_new, t1c, slabregUpdated['fwdtransforms'] )
  miUpdate = ants.image_mutual_information(
    ants.iMath(nm_avg_cropped,"Normalize"), ants.iMath(tempUpdate,"Normalize") )
  miOrig = ants.image_mutual_information(
    ants.iMath(nm_avg_cropped,"Normalize"), ants.iMath(tempOrig,"Normalize") )
  if miUpdate < miOrig :
      slabreg = slabregUpdated

  if normalize_nm:
      nm_avg_cropped = ants.iMath( nm_avg_cropped, "Normalize" )
      nm_avg_cropped = ants.iMath( nm_avg_cropped, "TruncateIntensity",0.05,0.95)
      nm_avg_cropped = ants.iMath( nm_avg_cropped, "Normalize" )

  labels2nm = ants.apply_transforms( nm_avg_cropped, t1lab,
        slabreg['fwdtransforms'], interpolator='nearestNeighbor' )

  # fix the reference region - keep top two parts
  def get_biggest_part( x, labeln ):
      temp33 = ants.threshold_image( x, labeln, labeln ).iMath("GetLargestComponent")
      x[ x == labeln] = 0
      x[ temp33 == 1 ] = labeln

  get_biggest_part( labels2nm, 33 )
  get_biggest_part( labels2nm, 34 )

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
  snvol = np.prod( ants.get_spacing(sn_mask) ) * sn_mask.sum()

  # get the mean voxel position of the SN
  if snvol > 0:
      sn_z = ants.transform_physical_point_to_index( sn_mask, ants.get_center_of_mass(sn_mask ))[2]
      sn_z = sn_z/sn_mask.shape[2] # around 0.5 would be nice
  else:
      sn_z = math.nan

  nm_evr = antspyt1w.patch_eigenvalue_ratio( nm_avg, 512, [6,6,6], evdepth = 0.9, mask=cropper2nm )

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
      'NM_volume_substantianigra' : snvol,
      'NM_avg_refregion' : rravg,
      'NM_std_refregion' : rrstd,
      'NM_min' : nm_avg_cropped.min(),
      'NM_max' : nm_avg_cropped.max(),
      'NM_mean' : nm_avg_cropped.numpy().mean(),
      'NM_sd' : math.sqrt( nm_avg_cropped.numpy().mean() ),
      'NM_q0pt05' : np.quantile( nm_avg_cropped.numpy(), 0.05 ),
      'NM_q0pt10' : np.quantile( nm_avg_cropped.numpy(), 0.10 ),
      'NM_q0pt90' : np.quantile( nm_avg_cropped.numpy(), 0.90 ),
      'NM_q0pt95' : np.quantile( nm_avg_cropped.numpy(), 0.95 ),
      'NM_substantianigra_z_coordinate' : sn_z,
      'NM_evr' : nm_evr,
      'NM_count': len( list_nm_images )
       }

def resting_state_fmri_networks( fmri, fmri_template, t1, t1segmentation,
    f=[0.03,0.08], FD_threshold=0.5, spa = 1.5, spt = 0.5, nc = 6, type_of_transform='SyN',
    verbose=False ):

  """
  Compute resting state network correlation maps based on the J Power labels.
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

  """
  import numpy as np
  import pandas as pd
  import re
  import math
  A = np.zeros((1,1))
  powers_areal_mni_itk = pd.read_csv( get_data('powers_mni_itk', target_extension=".csv")) # power coordinates
  fmri = ants.iMath( fmri, 'Normalize' )
  bmask = antspynet.brain_extraction( fmri_template, 'bold' ).threshold_image(0.5,1).iMath("FillHoles")
  if verbose:
      print("Begin rsfmri motion correction")
  corrmo = timeseries_reg(
    fmri, fmri_template,
    type_of_transform=type_of_transform,
    total_sigma=0.0,
    fdOffset=2.0,
    trim = 8,
    output_directory=None,
    verbose=False,
    syn_metric='cc',
    syn_sampling=2,
    reg_iterations=[40,20,5] )
  if verbose:
      print("End rsfmri motion correction")
      # ants.image_write( corrmo['motion_corrected'], '/tmp/temp.nii.gz' )

  mytsnr = tsnr( corrmo['motion_corrected'], bmask )
  mytsnrThresh = np.quantile( mytsnr.numpy(), 0.995 )
  tsnrmask = ants.threshold_image( mytsnr, 0, mytsnrThresh ).morphology("close",2)
  bmask = bmask * tsnrmask
  und = fmri_template * bmask
  t1reg = ants.registration( und, t1, "SyNBold" )
  if verbose:
      print("t1 2 bold done")
#      ants.image_write( und, '/tmp/template_bold_masked.nii.gz' )
#      ants.image_write( t1reg['warpedmovout'], '/tmp/t1tobold.nii.gz' )
  boldseg = ants.apply_transforms( und, t1segmentation,
    t1reg['fwdtransforms'], interpolator = 'genericLabel' ) * bmask
  gmseg = ants.threshold_image( t1segmentation, 2, 2 )
  gmseg = gmseg + ants.threshold_image( t1segmentation, 4, 4 )
  gmseg = ants.threshold_image( gmseg, 1, 4 )
  gmseg = ants.iMath( gmseg, 'MD', 1 )
  gmseg = ants.apply_transforms( und, gmseg,
    t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' ) * bmask
  csfAndWM = ( ants.threshold_image( t1segmentation, 1, 1 ) +
               ants.threshold_image( t1segmentation, 3, 3 ) ).morphology("erode",1)
  csfAndWM = ants.apply_transforms( und, csfAndWM,
    t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' )  * bmask

  # get falff and alff
  mycompcor = ants.compcor( corrmo['motion_corrected'],
    ncompcor=nc, quantile=0.90, mask = csfAndWM,
    filter_type='polynomial', degree=2 )

  nt = corrmo['motion_corrected'].shape[3]

  myvoxes = range(powers_areal_mni_itk.shape[0])
  anat = powers_areal_mni_itk['Anatomy']
  syst = powers_areal_mni_itk['SystemName']
  Brod = powers_areal_mni_itk['Brodmann']
  xAAL  = powers_areal_mni_itk['AAL']
  ch2 = mm_read( ants.get_ants_data( "ch2" ) )
  treg = ants.registration( t1, ch2, 'SyN' )
  concatx2 = treg['invtransforms'] + t1reg['invtransforms']
  pts2bold = ants.apply_transforms_to_points( 3, powers_areal_mni_itk, concatx2,
    whichtoinvert = ( True, False, True, False ) )
  locations = pts2bold.iloc[:,:3].values
  ptImg = ants.make_points_image( locations, bmask, radius = 2 )

  tr = ants.get_spacing( corrmo['motion_corrected'] )[3]
  highMotionTimes = np.where( corrmo['FD'] >= 1.0 )
  goodtimes = np.where( corrmo['FD'] < 0.5 )
  smth = ( spa, spa, spa, spt ) # this is for sigmaInPhysicalCoordinates = F
  simg = ants.smooth_image(corrmo['motion_corrected'], smth, sigma_in_physical_coordinates = False )

  nuisance = mycompcor[ 'components' ]
  nuisance = np.c_[ nuisance, mycompcor['basis'] ]
  nuisance = np.c_[ nuisance, corrmo['FD'] ]

  gmmat = ants.timeseries_to_matrix( simg, gmseg )
  gmmat = ants.bandpass_filter_matrix( gmmat, tr = tr, lowf=f[0], highf=f[1] ) # some would argue against this
  gmsignal = gmmat.mean( axis = 1 )
  nuisance = np.c_[ nuisance, gmsignal ]
  gmmat = ants.regress_components( gmmat, nuisance )
  # turn data following nuisance and gsr back to image format
  gsrbold = ants.matrix_to_timeseries(simg, gmmat, gmseg)

  myfalff=alff_image( simg, bmask, flo=f[0], fhi=f[1], nuisance=nuisance )

  outdict = {}
  outdict['meanBold'] = und
  outdict['pts2bold'] = pts2bold

  # add correlation matrix that captures each node pair
  # some of the spheres overlap so extract separately from each ROI
  nPoints = pts2bold['ROI'].max()
  nVolumes = simg.shape[3]
  meanROI = np.zeros([nVolumes, nPoints])
  roiNames = []
  for i in range(nPoints):
    # specify name for matrix entries that's links back to ROI number and network; e.g., ROI1_Uncertain
    netLabel = re.sub( " ", "", pts2bold.loc[i,'SystemName'])
    netLabel = re.sub( "-", "", netLabel )
    netLabel = re.sub( "/", "", netLabel )
    roiLabel = "ROI" + str(pts2bold.loc[i,'ROI']) + '_' + netLabel
    roiNames.append( roiLabel )
    ptImage = ants.make_points_image(pts2bold.iloc[[i],:3].values, bmask, radius=1).threshold_image( 1, 1e9 )
    meanROI[:,i] = ants.timeseries_to_matrix( gsrbold, ptImage).mean(axis=1)

  # get full correlation matrix
  corMat = np.corrcoef(meanROI, rowvar=False)
  outputMat = pd.DataFrame(corMat)
  outputMat.columns = roiNames
  outputMat['ROIs'] = roiNames
  # add to dictionary
  outdict['fullCorrMat'] = outputMat

  networks = powers_areal_mni_itk['SystemName'].unique()

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
  outdict['alff'] = myfalff['alff']
  outdict['falff'] = myfalff['falff']
  # add global mean and standard deviation for post-hoc z-scoring
  outdict['alff_mean'] = (myfalff['alff'][myfalff['alff']!=0]).mean()
  outdict['alff_sd'] = (myfalff['alff'][myfalff['alff']!=0]).std()
  outdict['falff_mean'] = (myfalff['falff'][myfalff['falff']!=0]).mean()
  outdict['falff_sd'] = (myfalff['falff'][myfalff['falff']!=0]).std()

  for k in range(1,270):
    anatname=( pts2bold['AAL'][k] )
    if isinstance(anatname, str):
        anatname = re.sub("_","",anatname)
    else:
        anatname='Unk'
    fname='falffPoint'+str(k)+anatname
    aname='alffPoint'+str(k)+anatname
    outdict[fname]=(outdict['falff'][ptImg==k]).mean()
    outdict[aname]=(outdict['alff'][ptImg==k]).mean()

  rsfNuisance = pd.DataFrame( nuisance )
  rsfNuisance['FD']=corrmo['FD']

  nonbrainmask = ants.iMath( bmask, "MD",2) - bmask
  trimmask = ants.iMath( bmask, "ME",2)
  edgemask = ants.iMath( bmask, "ME",1) - trimmask
  outdict['motion_corrected'] = corrmo['motion_corrected']
  outdict['brain_mask'] = bmask
  outdict['nuisance'] = rsfNuisance
  outdict['tsnr'] = mytsnr
  outdict['ssnr'] = slice_snr( corrmo['motion_corrected'], csfAndWM, gmseg )
  outdict['dvars'] = dvars( corrmo['motion_corrected'], gmseg )
  outdict['high_motion_count'] = (rsfNuisance['FD'] > FD_threshold ).sum()
  outdict['high_motion_pct'] = (rsfNuisance['FD'] > FD_threshold ).sum() / rsfNuisance.shape[0]
  outdict['FD_max'] = rsfNuisance['FD'].max()
  outdict['FD_mean'] = rsfNuisance['FD'].mean()
  outdict['bold_evr'] =  antspyt1w.patch_eigenvalue_ratio( und, 512, [16,16,16], evdepth = 0.9, mask = bmask )
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

def crop_mcimage( x, mask, padder=None ):
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
            temp = ants.crop_image( temp, mask )
            if padder is not None:
                temp = ants.pad_image( temp, pad_width=padder )
            croplist.append( temp )
        temp = ants.list_to_ndimage( x, croplist )
        temp.set_origin( myorig )
        return temp
    else:
        return( ants.crop_image( x, mask ) )


def mm(
    t1_image,
    hier,
    rsf_image=[],
    flair_image=None,
    nm_image_list=None,
    dw_image=[], bvals=[], bvecs=[],
    srmodel=None,
    do_tractography = False,
    do_kk = False,
    do_normalization = None,
    target_range = [0,1],
    dti_motion_correct = 'Rigid',
    dti_denoise = False,
    test_run = False,
    verbose = False ):
    """
    Multiple modality processing and normalization

    aggregates modality-specific processing under one roof.  see individual
    modality specific functions for details.

    Parameters
    -------------

    t1_image : raw t1 image

    hier  : output of antspyt1w.hierarchical ( see read hierarchical )

    rsf_image : list of resting state fmri

    flair_image : flair

    nm_image_list : list of neuromelanin images

    dw_image : list of diffusion weighted images

    bvals : list of bvals file names

    bvecs : list of bvecs file names

    srmodel : optional srmodel

    do_tractography : boolean

    do_kk : boolean to control whether we compute kelly kapowski thickness image (slow)

    do_normalization : template transformation if available

    target_range : 2-element tuple
        a tuple or array defining the (min, max) of the input image
        (e.g., [-127.5, 127.5] or [0,1]).  Output images will be scaled back to original
        intensity. This range should match the mapping used in the training
        of the network.
    
    dti_motion_correct : None Rigid or SyN

    dti_denoise : boolean

    test_run : boolean 

    verbose : boolean

    """
    from os.path import exists
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    ex_path_mm = os.path.expanduser( "~/.antspymm/" )
    mycsvfn = ex_path + "FA_JHU_labels_edited.csv"
    citcsvfn = ex_path + "CIT168_Reinf_Learn_v1_label_descriptions_pad.csv"
    dktcsvfn = ex_path + "dkt.csv"
    cnxcsvfn = ex_path + "dkt_cortex_cit_deep_brain.csv"
    JHU_atlasfn = ex_path + 'JHU-ICBM-FA-1mm.nii.gz' # Read in JHU atlas
    JHU_labelsfn = ex_path + 'JHU-ICBM-labels-1mm.nii.gz' # Read in JHU labels
    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
    if not exists( mycsvfn ) or not exists( citcsvfn ) or not exists( cnxcsvfn ) or not exists( dktcsvfn ) or not exists( JHU_atlasfn ) or not exists( JHU_labelsfn ) or not exists( templatefn ):
        print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
        raise ValueError('**missing files** => call get_data from latest antspyt1w and antspymm.')
    mycsv = pd.read_csv(  mycsvfn )
    citcsv = pd.read_csv(  os.path.expanduser( citcsvfn ) )
    dktcsv = pd.read_csv(  os.path.expanduser( dktcsvfn ) )
    cnxcsv = pd.read_csv(  os.path.expanduser( cnxcsvfn ) )
    JHU_atlas = mm_read( JHU_atlasfn ) # Read in JHU atlas
    JHU_labels = mm_read( JHU_labelsfn ) # Read in JHU labels
    template = mm_read( templatefn ) # Read in template
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
        'alff_norm' : None,
        'falff_norm' : None,
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
    if test_run:
        return output_dict, normalization_dict

    if do_kk:
        if verbose:
            print('kk')
        output_dict['kk'] = antspyt1w.kelly_kapowski_thickness( hier['brain_n4_dnz'],
            labels=hier['dkt_parc']['dkt_cortex'], iterations=45 )
    ################################## do the rsf .....
    if len(rsf_image) > 0:
        rsf_image = [i for i in rsf_image if i is not None]
        if verbose:
            print('rsf length ' + str( len( rsf_image ) ) )
        if len( rsf_image ) >= 2: # assume 2 is the largest possible value
            rsf_image1 = rsf_image[0]
            rsf_image2 = rsf_image[1]
            # build a template then join the images
            if verbose:
                print("initial average for rsf")
            rsfavg1=get_average_rsf(rsf_image1)
            rsfavg2=get_average_rsf(rsf_image2)
            if verbose:
                print("template average for rsf")
            init_temp = ants.image_clone( rsfavg1 )
            if rsf_image1.shape[3] < rsf_image2.shape[3]:
                init_temp = ants.image_clone( rsfavg2 )
            boldTemplate = ants.build_template(
                initial_template = init_temp,
                image_list=[rsfavg1,rsfavg2],
                iterations=5, verbose=False )
            if verbose:
                print("join the 2 rsf")
            if rsf_image1.shape[3] > 10 and rsf_image2.shape[3] > 10:
                rsf_image = merge_timeseries_data( rsf_image1, rsf_image2 )
            elif rsf_image1.shape[3] > rsf_image2.shape[3]:
                rsf_image = rsf_image1
            else:
                rsf_image = rsf_image2
        elif len( rsf_image ) == 1:
            rsf_image = rsf_image[0]
            boldTemplate=get_average_rsf(rsf_image)
        if rsf_image.shape[3] > 10: # FIXME - better heuristic?
            output_dict['rsf'] = resting_state_fmri_networks(
                rsf_image,
                boldTemplate,
                hier['brain_n4_dnz'],
                t1atropos,
                f=[0.03,0.08],
                spa = 1.0,
                spt = 0.5,
                nc = 6, verbose=verbose )
    if nm_image_list is not None:
        if verbose:
            print('nm')
        if srmodel is None:
            output_dict['NM'] = neuromelanin( nm_image_list, t1imgbrn, t1_image, hier['deep_cit168lab'], verbose=verbose )
        else:
            output_dict['NM'] = neuromelanin( nm_image_list, t1imgbrn, t1_image, hier['deep_cit168lab'], srmodel=srmodel, target_range=target_range, verbose=verbose  )
################################## do the dti .....
    if len(dw_image) > 0 :
        if verbose:
            print('dti-x')
        if len( dw_image ) == 1: # use T1 for distortion correction and brain extraction
            if verbose:
                print("We have only one DTI: " + str(len(dw_image)))
            dw_image = dw_image[0]
            btpB0,btpDW=get_average_dwi_b0(dw_image)
            initrig = ants.registration( btpDW, hier['brain_n4_dnz'], 'BOLDRigid' )['fwdtransforms'][0]
            tempreg = ants.registration( btpDW, hier['brain_n4_dnz'], 'SyNOnly',
                syn_metric='mattes', syn_sampling=32,
                reg_iterations=[50,50,20],
                multivariate_extras=[ [ "mattes", btpB0, hier['brain_n4_dnz'], 1, 32 ]],
                initial_transform=initrig
                )
            mybxt = ants.threshold_image( ants.iMath(hier['brain_n4_dnz'], "Normalize" ), 0.001, 1 )
            btpDW = ants.apply_transforms( btpDW, btpDW,
                tempreg['invtransforms'][1], interpolator='linear')
            btpB0 = ants.apply_transforms( btpB0, btpB0,
                tempreg['invtransforms'][1], interpolator='linear')
            dwimask = ants.apply_transforms( btpDW, mybxt, tempreg['fwdtransforms'][1], interpolator='nearestNeighbor')
            # dwimask = ants.iMath(dwimask,'MD',1)
            t12dwi = ants.apply_transforms( btpDW, hier['brain_n4_dnz'], tempreg['fwdtransforms'][1], interpolator='linear')
            output_dict['DTI'] = joint_dti_recon(
                dw_image,
                bvals[0],
                bvecs[0],
                jhu_atlas=JHU_atlas,
                jhu_labels=JHU_labels,
                brain_mask = dwimask,
                reference_B0 = btpB0,
                reference_DWI = btpDW,
                srmodel=srmodel,
                motion_correct=dti_motion_correct, # set to False if using input from qsiprep
                denoise=dti_denoise,
                verbose = verbose)
        else :  # use phase encoding acquisitions for distortion correction and T1 for brain extraction
            if verbose:
                print("We have both DTI_LR and DTI_RL: " + str(len(dw_image)))
            a1b,a1w=antspymm.get_average_dwi_b0(dw_image[0])
            a2b,a2w=antspymm.get_average_dwi_b0(dw_image[1],fixed_b0=a1b,fixed_dwi=a1w)
            btpB0, btpDW = dti_template(
                b_image_list=[a1b,a2b],
                w_image_list=[a1w,a2w],
                iterations=7, verbose=verbose )
            initrig = ants.registration( btpDW, hier['brain_n4_dnz'], 'BOLDRigid' )['fwdtransforms'][0]
            tempreg = ants.registration( btpDW, hier['brain_n4_dnz'], 'SyNOnly',
                syn_metric='mattes', syn_sampling=32,
                reg_iterations=[50,50,20],
                multivariate_extras=[ [ "mattes", btpB0, hier['brain_n4_dnz'], 1, 32 ]],
                initial_transform=initrig
                )
            mybxt = ants.threshold_image( ants.iMath(hier['brain_n4_dnz'], "Normalize" ), 0.001, 1 )
            dwimask = ants.apply_transforms( btpDW, mybxt, tempreg['fwdtransforms'], interpolator='nearestNeighbor')
            output_dict['DTI'] = joint_dti_recon(
                dw_image[0],
                bvals[0],
                bvecs[0],
                jhu_atlas=JHU_atlas,
                jhu_labels=JHU_labels,
                brain_mask = dwimask,
                reference_B0 = btpB0,
                reference_DWI = btpDW,
                srmodel=srmodel,
                img_RL=dw_image[1],
                bval_RL=bvals[1],
                bvec_RL=bvecs[1],
                motion_correct='SyN', # set to False if using input from qsiprep
                denoise=True,
                verbose = verbose)
        mydti = output_dict['DTI']
        # summarize dwi with T1 outputs
        # first - register ....
        reg = ants.registration( mydti['recon_fa'], hier['brain_n4_dnz'], 'SyNBold', total_sigma=1.0 )
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
        citmapped = ants.apply_transforms(
            mydti['recon_fa'],
            hier['cit168lab'],
            reg['fwdtransforms'], interpolator='nearestNeighbor' )
        dktmapped[ citmapped > 0]=0
        mask = ants.threshold_image( mydti['recon_fa'], 0.01, 2.0 ).iMath("GetLargestComponent")
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
            output_dict['tractography_connectivity'] = dwi_streamline_connectivity( mystr['streamlines'], dktmapped+citmapped, cnxcsv, verbose=verbose )
    ################################## do the flair .....
    if flair_image is not None:
        if verbose:
            print('flair')
        wmhprior = None
        priorfn = ex_path_mm + 'CIT168_wmhprior_700um_pad_adni.nii.gz'
        if ( exists( priorfn ) ):
            wmhprior = ants.image_read( priorfn )
            wmhprior = ants.apply_transforms( t1_image, wmhprior, do_normalization['invtransforms'] )
        output_dict['flair'] = boot_wmh( flair_image, t1_image, t1atropos,
            prior_probability=wmhprior, verbose=verbose )
    #################################################################
    ### NOTES: deforming to a common space and writing out images ###
    ### images we want come from: DTI, NM, rsf, thickness ###########
    #################################################################
    if do_normalization is not None:
        if verbose:
            print('normalization')
        # might reconsider this template space - cropped and/or higher res?
        template = ants.resample_image( template, [1,1,1], use_voxels=False )
        # t1reg = ants.registration( template, hier['brain_n4_dnz'], "antsRegistrationSyNQuickRepro[s]")
        t1reg = do_normalization
        if do_kk:
            normalization_dict['kk_norm'] = ants.apply_transforms( template, output_dict['kk']['thickness_image'], t1reg['fwdtransforms'])
        if output_dict['DTI'] is not None:
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
                image_write_with_thumbnail( mm_norm[mykey], tempfn )
    thkderk = None
    if t1wide is not None:
        thkderk = t1wide.iloc[: , 1:]
    kkderk = None
    if mm['kk'] is not None:
        kkderk = mm['kk']['thickness_dataframe'].iloc[: , 1:]
        mykey='thickness_image'
        tempfn = output_prefix + separator + mykey + '.nii.gz'
        image_write_with_thumbnail( mm['kk'][mykey], tempfn )
    nmderk = None
    if mm['NM'] is not None:
        nmderk = mm['NM']['NM_dataframe_wide'].iloc[: , 1:]
        for mykey in ['NM_avg_cropped', 'NM_avg', 'NM_labels' ]:
            tempfn = output_prefix + separator + mykey + '.nii.gz'
            image_write_with_thumbnail( mm['NM'][mykey], tempfn, thumb=False )

    faderk = mdderk = fat1derk = mdt1derk = None
    if mm['DTI'] is not None:
        mydti = mm['DTI']
        myop = output_prefix + separator
        write_bvals_bvecs( mydti['bval_LR'], mydti['bvec_LR'], myop + 'reoriented' )
        image_write_with_thumbnail( mydti['dwi_LR_dewarped'],  myop + 'dwi.nii.gz' )
        image_write_with_thumbnail( mydti['dtrecon_LR_dewarp']['RGB'] ,  myop + 'DTIRGB.nii.gz' )
        image_write_with_thumbnail( mydti['jhu_labels'],  myop+'dtijhulabels.nii.gz', mydti['recon_fa'] )
        image_write_with_thumbnail( mydti['recon_fa'],  myop+'dtifa.nii.gz' )
        image_write_with_thumbnail( mydti['recon_md'],  myop+'dtimd.nii.gz' )
        image_write_with_thumbnail( mydti['b0avg'],  myop+'b0avg.nii.gz' )
        image_write_with_thumbnail( mydti['dwiavg'],  myop+'dwiavg.nii.gz' )
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
        ofn = output_prefix + separator + 'dtistreamlineconn.csv'
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
        mm_wide['NM_volume_substantianigra'] = mm['NM']['NM_volume_substantianigra']
        mm_wide['NM_avg_refregion'] = mm['NM']['NM_avg_refregion']
        mm_wide['NM_std_refregion'] = mm['NM']['NM_std_refregion']
        mm_wide['NM_evr'] = mm['NM']['NM_evr']
        mm_wide['NM_count'] = mm['NM']['NM_count']
        mm_wide['NM_min'] = mm['NM']['NM_min']
        mm_wide['NM_max'] = mm['NM']['NM_max']
        mm_wide['NM_mean'] = mm['NM']['NM_mean']
        mm_wide['NM_sd'] = mm['NM']['NM_sd']
        mm_wide['NM_q0pt05'] = mm['NM']['NM_q0pt05']
        mm_wide['NM_q0pt10'] = mm['NM']['NM_q0pt10']
        mm_wide['NM_q0pt90'] = mm['NM']['NM_q0pt90']
        mm_wide['NM_q0pt95'] = mm['NM']['NM_q0pt95']
        mm_wide['NM_substantianigra_z_coordinate'] = mm['NM']['NM_substantianigra_z_coordinate']
    if mm['flair'] is not None:
        myop = output_prefix + separator + 'wmh.nii.gz'
        if mm['flair']['WMH_probability_map'] is not None:
            image_write_with_thumbnail( mm['flair']['WMH_probability_map'], myop, thumb=False )
        mm_wide['flair_wmh'] = mm['flair']['wmh_mass']
        mm_wide['flair_wmh_prior'] = mm['flair']['wmh_mass_prior']
        mm_wide['flair_evr'] = mm['flair']['wmh_evr']
        mm_wide['flair_SNR'] = mm['flair']['wmh_SNR']
    if mm['rsf'] is not None:
        mynets = list([ 'meanBold', 'brain_mask', 'motion_corrected', 'alff', 'falff',
            'CinguloopercularTaskControl', 'DefaultMode', 'MemoryRetrieval',
            'VentralAttention', 'Visual', 'FrontoparietalTaskControl', 'Salience',
            'Subcortical', 'DorsalAttention', 'tsnr'] )
        rsfpro = mm['rsf']
        for mykey in mynets:
            myop = output_prefix + separator + mykey + '.nii.gz'
            image_write_with_thumbnail( rsfpro[mykey], myop, thumb=True )
        rsfpro['corr_wide'].set_index( mm_wide.index, inplace=True )
        mm_wide = pd.concat( [ mm_wide, rsfpro['corr_wide'] ], axis=1 )
        # falff and alff
        search_key='alffPoint'
        alffkeys = [key for key, val in rsfpro.items() if search_key in key]
        for myalf in alffkeys:
            mm_wide[ myalf ]=rsfpro[myalf]
        mm_wide['rsf_tsnr_mean'] =  rsfpro['tsnr'].mean()
        mm_wide['rsf_dvars_mean'] =  rsfpro['dvars'].mean()
        mm_wide['rsf_ssnr_mean'] =  rsfpro['ssnr'].mean()
        mm_wide['rsf_high_motion_count'] =  rsfpro['high_motion_count']
        # mm_wide['rsf_high_motion_pct'] = rsfpro['rsf_high_motion_pct'] # BUG : rsf_high_motion_pct does not exist
        mm_wide['rsf_evr'] =  rsfpro['bold_evr']
        mm_wide['rsf_FD_mean'] = rsfpro['FD_mean']
        mm_wide['rsf_FD_max'] = rsfpro['FD_max']
        mm_wide['rsf_alff_mean'] = rsfpro['alff_mean']
        mm_wide['rsf_alff_sd'] = rsfpro['alff_sd']
        mm_wide['rsf_falff_mean'] = rsfpro['falff_mean']
        mm_wide['rsf_falff_sd'] = rsfpro['falff_sd']
        ofn = output_prefix + separator + 'rsfcorr.csv'
        rsfpro['corr'].to_csv( ofn )
        # apply same principle to new correlation matrix, doesn't need to be incorporated with mm_wide
        ofn2 = output_prefix + separator + 'nodescorr.csv'
        rsfpro['fullCorrMat'].to_csv( ofn2 )
    if mm['DTI'] is not None:
        mydti = mm['DTI']
        mm_wide['dti_tsnr_b0_mean'] =  mydti['tsnr_b0'].mean()
        mm_wide['dti_tsnr_dwi_mean'] =  mydti['tsnr_dwi'].mean()
        mm_wide['dti_dvars_b0_mean'] =  mydti['dvars_b0'].mean()
        mm_wide['dti_dvars_dwi_mean'] =  mydti['dvars_dwi'].mean()
        mm_wide['dti_ssnr_b0_mean'] =  mydti['ssnr_b0'].mean()
        mm_wide['dti_ssnr_dwi_mean'] =  mydti['ssnr_dwi'].mean()
        mm_wide['dti_fa_evr'] =  mydti['fa_evr']
        mm_wide['dti_fa_SNR'] =  mydti['fa_SNR']
        if mydti['framewise_displacement'] is not None:
            mm_wide['dti_high_motion_count'] =  mydti['high_motion_count']
            mm_wide['dti_FD_mean'] = mydti['framewise_displacement'].mean()
            mm_wide['dti_FD_max'] = mydti['framewise_displacement'].max()
            fdfn = output_prefix + separator + '_fd.csv'
            # mm_wide.to_csv( fdfn )
        else:
            mm_wide['dti_FD_mean'] = mm_wide['dti_FD_max'] = 'NA'
    mmwidefn = output_prefix + separator + 'mmwide.csv'
    mm_wide.to_csv( mmwidefn )
    return


def mm_nrg(
    studyid,   # pandas data frame
    sourcedir = os.path.expanduser( "~/data/PPMI/MV/example_s3_b/images/PPMI/" ),
    sourcedatafoldername = 'images', # root for source data
    processDir = "processed", # where output will go - parallel to sourcedatafoldername
    mysep = '-', # define a separator for filename components
    srmodel_T1 = False, # optional - will add a great deal of time
    srmodel_NM = False, # optional - will add a great deal of time
    srmodel_DTI = False, # optional - will add a great deal of time
    visualize = True,
    nrg_modality_list = ["T1w", "NM2DMT", "rsfMRI","DTI","T2Flair" ],
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

    studyid : must have columns 1. subjectID 2. date (in form 20220228) and 3. imageID
        other relevant columns include nmid1-10, rsfid1, rsfid2, dtid1, dtid2, flairid;
        these provide unique image IDs for these modalities: nm=neuromelanin, dti=diffusion tensor,
        rsf=resting state fmri, flair=T2Flair.  none of these are required. only
        t1 is required.  rsfid1/rsfid2 will be processed jointly. same for dtid1/dtid2 and nmid*.  see antspymm.generate_mm_dataframe

    sourcedir : a study specific folder containing individual subject folders

    sourcedatafoldername : root for source data e.g. "images"

    processDir : where output will go - parallel to sourcedatafoldername e.g.
        "processed"

    mysep : define a character separator for filename components

    srmodel_T1 : False (default) - will add a great deal of time - or h5 filename, 2 chan

    srmodel_NM : False (default) - will add a great deal of time - or h5 filename, 1 chan

    srmodel_DTI : False (default) - will add a great deal of time - or h5 filename, 1 chan

    visualize : True - will plot some results to png

    nrg_modality_list : list of permissible modalities - always include [T1w] as base

    verbose : boolean

    Returns
    ---------

    writes output to disk and potentially produces figures that may be
    captured in a ipynb / html file.

    """
    studyid = studyid.dropna(axis=1)
    if studyid.shape[0] < 1:
        raise ValueError('studyid has no rows')
    musthavecols = ['subjectID','date','imageID']
    for k in range(len(musthavecols)):
        if not musthavecols[k] in studyid.keys():
            raise ValueError('studyid is missing column ' +musthavecols[k] )
    def makewideout( x, separator = '-' ):
        return x + separator + 'mmwide.csv'
    if nrg_modality_list[0] != 'T1w':
        nrg_modality_list.insert(0, "T1w" )
    testloop = False
    counter=0
    import glob as glob
    from os.path import exists
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    ex_pathmm = os.path.expanduser( "~/.antspymm/" )
    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
    if not exists( templatefn ):
        print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
        antspyt1w.get_data( force_download=True )
        get_data( force_download=True )
    temp = sourcedir.split( "/" )
    splitCount = len( temp )
    template = mm_read( templatefn ) # Read in template
    test_run = False
    if test_run:
        visualize=False
    # get sid and dtid from studyid
    sid = str(studyid['subjectID'].iloc[0])
    dtid = str(studyid['date'].iloc[0])
    iid = str(studyid['imageID'].iloc[0])
    subjectrootpath = os.path.join(sourcedir,sid, dtid)
    if verbose:
        print("subjectrootpath: "+ subjectrootpath )
    myimgsInput = glob.glob( subjectrootpath+"/*" )
    myimgsInput.sort( )
    if verbose:
        print( myimgsInput )
    # hierarchical
    # NOTE: if there are multiple T1s for this time point, should take
    # the one with the highest resnetGrade
    t1_search_path = os.path.join(subjectrootpath, "T1w", iid, "*nii.gz")
    if verbose:
        print(f"t1 search path: {t1_search_path}")
    t1fn = glob.glob(t1_search_path)
    t1fn.sort()
    if len( t1fn ) < 1:
        raise ValueError('mm_nrg cannot find the T1w with uid ' + iid + ' @ ' + subjectrootpath )
    t1fn = t1fn[0]
    t1 = mm_read( t1fn )
    hierfn0 = re.sub( sourcedatafoldername, processDir, t1fn)
    hierfn0 = re.sub( ".nii.gz", "", hierfn0)
    hierfn = re.sub( "T1w", "T1wHierarchical", hierfn0)
    hierfn = hierfn + mysep
    hierfntest = hierfn + 'snseg.csv'
    regout = hierfn0 + mysep + "syn"
    templateTx = {
        'fwdtransforms': [ regout+'1Warp.nii.gz', regout+'0GenericAffine.mat'],
        'invtransforms': [ regout+'0GenericAffine.mat', regout+'1InverseWarp.nii.gz']  }
    if verbose:
        print( "REGISTRATION EXISTENCE: " +
            str(exists( templateTx['fwdtransforms'][0])) + " " +
            str(exists( templateTx['fwdtransforms'][1])) + " " +
            str(exists( templateTx['invtransforms'][0])) + " " +
            str(exists( templateTx['invtransforms'][1])) )
    if verbose:
        print( hierfntest )
    hierexists = exists( hierfntest ) # FIXME should test this explicitly but we assume it here
    hier = None
    if not hierexists and not testloop:
        subjectpropath = os.path.dirname( hierfn )
        if verbose:
            print( subjectpropath )
        os.makedirs( subjectpropath, exist_ok=True  )
        hier = antspyt1w.hierarchical( t1, hierfn, labels_to_register=None )
        antspyt1w.write_hierarchical( hier, hierfn )
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
        t1wide.to_csv( hierfn + 'mmwide.csv' )
    ################# read the hierarchical data ###############################
    hier = antspyt1w.read_hierarchical( hierfn )
    if exists( hierfn + 'mmwide.csv' ) :
        t1wide = pd.read_csv( hierfn + 'mmwide.csv' )
    elif not testloop:
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
    if srmodel_T1 is not False :
        hierfnSR = re.sub( sourcedatafoldername, processDir, t1fn)
        hierfnSR = re.sub( "T1w", "T1wHierarchicalSR", hierfnSR)
        hierfnSR = re.sub( ".nii.gz", "", hierfnSR)
        hierfnSR = hierfnSR + mysep
        hierfntest = hierfnSR + 'mtl.csv'
        if verbose:
            print( hierfntest )
        hierexists = exists( hierfntest ) # FIXME should test this explicitly but we assume it here
        if not hierexists:
            subjectpropath = os.path.dirname( hierfnSR )
            if verbose:
                print( subjectpropath )
            os.makedirs( subjectpropath, exist_ok=True  )
            # hierarchical_to_sr(t1hier, sr_model, tissue_sr=False, blending=0.5, verbose=False)
            bestup = siq.optimize_upsampling_shape( ants.get_spacing(t1), modality='T1' )
            mdlfn = ex_pathmm + "siq_default_sisr_" + bestup + "_2chan_featvggL6_postseg_best_mdl.h5"
            if isinstance( srmodel_T1, str ):
                mdlfn = os.path.join( ex_pathmm, srmodel_T1 )
            if verbose:
                print( mdlfn )
            if exists( mdlfn ):
                srmodel_T1_mdl = tf.keras.models.load_model( mdlfn, compile=False )
            else:
                print( mdlfn + " does not exist - will not run.")
            hierSR = antspyt1w.hierarchical_to_sr( hier, srmodel_T1_mdl, blending=None, tissue_sr=False )
            antspyt1w.write_hierarchical( hierSR, hierfnSR )
            t1wideSR = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                    hierSR['dataframes'], identifier=None )
            t1wideSR.to_csv( hierfnSR + 'mmwide.csv' )
    hier = antspyt1w.read_hierarchical( hierfn )
    if exists( hierfn + 'mmwide.csv' ) :
        t1wide = pd.read_csv( hierfn + 'mmwide.csv' )
    elif not testloop:
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
    if not testloop:
        t1imgbrn = hier['brain_n4_dnz']
        t1atropos = hier['dkt_parc']['tissue_segmentation']
    # loop over modalities and then unique image IDs
    # we treat NM in a "special" way -- aggregating repeats
    # other modalities (beyond T1) are treated individually
    nimages = len(myimgsInput)
    if verbose:
        print(  " we have : " + str(nimages) + " modalities.")
    for overmodX in nrg_modality_list:
        counter=counter+1
        if counter > (len(nrg_modality_list)+1):
            print("This is weird. " + str(counter))
            return
        if overmodX == 'T1w':
            iidOtherMod = iid
            mod_search_path = os.path.join(subjectrootpath, overmodX, iidOtherMod, "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
        elif overmodX == 'NM2DMT' and ('nmid1' in studyid.keys() ):
            iidOtherMod = str( int(studyid['nmid1'].iloc[0]) )
            mod_search_path = os.path.join(subjectrootpath, overmodX, iidOtherMod, "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
            for nmnum in range(2,11):
                locnmnum = 'nmid'+str(nmnum)
                if locnmnum in studyid.keys() :
                    iidOtherMod = str( int(studyid[locnmnum].iloc[0]) )
                    mod_search_path = os.path.join(subjectrootpath, overmodX, iidOtherMod, "*nii.gz")
                    myimgsr.append( glob.glob(mod_search_path)[0] )
        elif 'rsfMRI' in overmodX and ( ( 'rsfid1' in studyid.keys() ) or ('rsfid2' in studyid.keys() ) ):
            myimgsr = []
            if  'rsfid1' in studyid.keys():
                iidOtherMod = str( int(studyid['rsfid1'].iloc[0]) )
                mod_search_path = os.path.join(subjectrootpath, overmodX+"*", iidOtherMod, "*nii.gz")
                myimgsr.append( glob.glob(mod_search_path)[0] )
            if  'rsfid2' in studyid.keys():
                iidOtherMod = str( int(studyid['rsfid2'].iloc[0]) )
                mod_search_path = os.path.join(subjectrootpath, overmodX+"*", iidOtherMod, "*nii.gz")
                myimgsr.append( glob.glob(mod_search_path)[0] )
        elif 'DTI' in overmodX and (  'dtid1' in studyid.keys() or  'dtid2' in studyid.keys() ):
            myimgsr = []
            if  'dtid1' in studyid.keys():
                iidOtherMod = str( int(studyid['dtid1'].iloc[0]) )
                mod_search_path = os.path.join(subjectrootpath, overmodX+"*", iidOtherMod, "*nii.gz")
                myimgsr.append( glob.glob(mod_search_path)[0] )
            if  'dtid2' in studyid.keys():
                iidOtherMod = str( int(studyid['dtid2'].iloc[0]) )
                mod_search_path = os.path.join(subjectrootpath, overmodX+"*", iidOtherMod, "*nii.gz")
                myimgsr.append( glob.glob(mod_search_path)[0] )
        elif 'T2Flair' in overmodX and ('flairid' in studyid.keys() ):
            iidOtherMod = str( int(studyid['flairid'].iloc[0]) )
            mod_search_path = os.path.join(subjectrootpath, overmodX, iidOtherMod, "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
        if verbose:
            print( "overmod " + overmodX + " " + iidOtherMod )
            print(f"modality search path: {mod_search_path}")
        myimgsr.sort()
        if len(myimgsr) > 0:
            overmodXx = str(overmodX)
            dowrite=False
            if verbose:
                print( 'overmodX is : ' + overmodXx )
                print( 'example image name is : '  )
                print( myimgsr )
            if overmodXx == 'NM2DMT':
                myimgsr2 = myimgsr
                myimgsr2.sort()
                is4d = False
                temp = ants.image_read( myimgsr2[0] )
                if temp.dimension == 4:
                    is4d = True
                if len( myimgsr2 ) == 1 and not is4d: # check dimension
                    myimgsr2 = myimgsr2 + myimgsr2
                subjectpropath = os.path.dirname( myimgsr2[0] )
                subjectpropath = re.sub( sourcedatafoldername, processDir,subjectpropath )
                if verbose:
                    print( "subjectpropath " + subjectpropath )
                mysplit = subjectpropath.split( "/" )
                os.makedirs( subjectpropath, exist_ok=True  )
                mysplitCount = len( mysplit )
                project = mysplit[mysplitCount-5]
                subject = mysplit[mysplitCount-4]
                date = mysplit[mysplitCount-3]
                modality = mysplit[mysplitCount-2]
                uider = mysplit[mysplitCount-1]
                identifier = mysep.join([project, subject, date, modality ])
                identifier = identifier + "_" + iid
                mymm = subjectpropath + "/" + identifier
                mymmout = makewideout( mymm )
                if verbose and not exists( mymmout ):
                    print( "NM " + mymm  + ' execution ')
                elif verbose and exists( mymmout ) :
                    print( "NM " + mymm + ' complete ' )
                if exists( mymmout ):
                    continue
                if is4d:
                    nmlist = ants.ndimage_to_list( mm_read( myimgsr2[0] ) )
                else:
                    nmlist = []
                    for zz in myimgsr2:
                        nmlist.append( mm_read( zz ) )
                srmodel_NM_mdl = None
                if srmodel_NM is not False:
                    bestup = siq.optimize_upsampling_shape( ants.get_spacing(nmlist[0]), modality='NM', roundit=True )
                    mdlfn = ex_pathmm + "siq_default_sisr_" + bestup + "_1chan_featvggL6_best_mdl.h5"
                    if isinstance( srmodel_NM, str ):
                        srmodel_NM = re.sub( "bestup", bestup, srmodel_NM )
                        mdlfn = os.path.join( ex_pathmm, srmodel_NM )
                    if exists( mdlfn ):
                        if verbose:
                            print(mdlfn)
                        srmodel_NM_mdl = tf.keras.models.load_model( mdlfn, compile=False  )
                    else:
                        print( mdlfn + " does not exist - wont use SR")
                if not testloop:
                    tabPro, normPro = mm( t1, hier,
                            nm_image_list = nmlist,
                            srmodel=srmodel_NM_mdl,
                            do_tractography=False,
                            do_kk=False,
                            do_normalization=templateTx,
                            test_run=test_run,
                            verbose=True )
                    if not test_run:
                        write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=None, separator=mysep )
                        nmpro = tabPro['NM']
                        mysl = range( nmpro['NM_avg'].shape[2] )
                    if visualize:
                        mysl = range( nmpro['NM_avg'].shape[2] )
                        ants.plot( nmpro['NM_avg'],  nmpro['t1_to_NM'], slices=mysl, axis=2, title='nm + t1', filename=mymm+mysep+"NMavg.png" )
                        mysl = range( nmpro['NM_avg_cropped'].shape[2] )
                        ants.plot( nmpro['NM_avg_cropped'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop', filename=mymm+mysep+"NMavgcrop.png" )
                        ants.plot( nmpro['NM_avg_cropped'], nmpro['t1_to_NM'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop + t1', filename=mymm+mysep+"NMavgcropt1.png" )
                        ants.plot( nmpro['NM_avg_cropped'], nmpro['NM_labels'], axis=2, slices=mysl, title='nm crop + labels', filename=mymm+mysep+"NMavgcroplabels.png" )
            else :
                if len( myimgsr ) > 0:
                    dowrite=False
                    myimgcount = 0
                    if len( myimgsr ) > 0 :
                        myimg = myimgsr[myimgcount]
                        subjectpropath = os.path.dirname( myimg )
                        subjectpropath = re.sub( sourcedatafoldername, processDir, subjectpropath )
                        mysplit = subjectpropath.split("/")
                        mysplitCount = len( mysplit )
                        project = mysplit[mysplitCount-5]
                        date = mysplit[mysplitCount-4]
                        subject = mysplit[mysplitCount-3]
                        mymod = mysplit[mysplitCount-2] # FIXME system dependent
                        uid = mysplit[mysplitCount-1] # unique image id
                        os.makedirs( subjectpropath, exist_ok=True  )
                        if mymod == 'T1w':
                            identifier = mysep.join([project, date, subject, mymod, uid])
                        else:  # add the T1 unique id since that drives a lot of the analysis
                            identifier = mysep.join([project, date, subject, mymod, uid ])
                            identifier = identifier + "_" + iid
                        mymm = subjectpropath + "/" + identifier
                        mymmout = makewideout( mymm )
                        if verbose and not exists( mymmout ):
                            print("Modality specific processing: " + mymod + " execution " )
                            print( mymm )
                        elif verbose and exists( mymmout ) :
                            print("Modality specific processing: " + mymod + " complete " )
                        if exists( mymmout ) :
                            continue
                        if verbose:
                            print(subjectpropath)
                            print(identifier)
                            print( myimg )
                        if not testloop:
                            img = mm_read( myimg )
                            ishapelen = len( img.shape )
                            if mymod == 'T1w' and ishapelen == 3: # for a real run, set to True
                                if not exists( regout + "logjacobian.nii.gz" ) or not exists( regout+'1Warp.nii.gz' ):
                                    if verbose:
                                        print('start t1 registration')
                                    ex_path = os.path.expanduser( "~/.antspyt1w/" )
                                    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
                                    template = mm_read( templatefn )
                                    template = ants.resample_image( template, [1,1,1], use_voxels=False )
                                    t1reg = ants.registration( template, hier['brain_n4_dnz'],
                                        "antsRegistrationSyNQuickRepro[s]", outprefix = regout, verbose=False )
                                    myjac = ants.create_jacobian_determinant_image( template,
                                        t1reg['fwdtransforms'][0], do_log=True, geom=True )
                                    image_write_with_thumbnail( myjac, regout + "logjacobian.nii.gz", thumb=False )
                                    if visualize:
                                        ants.plot( ants.iMath(t1reg['warpedmovout'],"Normalize"),  axis=2, nslices=21, ncol=7, crop=True, title='warped to template', filename=regout+"totemplate.png" )
                                        ants.plot( ants.iMath(myjac,"Normalize"),  axis=2, nslices=21, ncol=7, crop=True, title='jacobian', filename=regout+"jacobian.png" )
                                if not exists( mymm + mysep + "kk_norm.nii.gz" ):
                                    dowrite=True
                                    if verbose:
                                        print('start kk')
                                    tabPro, normPro = mm( t1, hier,
                                        srmodel=None,
                                        do_tractography=False,
                                        do_kk=True,
                                        do_normalization=templateTx,
                                        test_run=test_run,
                                        verbose=True )
                                    if visualize:
                                        maxslice = np.min( [21, hier['brain_n4_dnz'].shape[2] ] )
                                        ants.plot( hier['brain_n4_dnz'],  axis=2, nslices=maxslice, ncol=7, crop=True, title='brain extraction', filename=mymm+mysep+"brainextraction.png" )
                                        ants.plot( tabPro['kk']['thickness_image'], axis=2, nslices=maxslice, ncol=7, crop=True, title='kk',
                                        cmap='plasma', filename=mymm+mysep+"kkthickness.png" )
                            if mymod == 'T2Flair' and ishapelen == 3:
                                dowrite=True
                                tabPro, normPro = mm( t1, hier,
                                    flair_image = img,
                                    srmodel=None,
                                    do_tractography=False,
                                    do_kk=False,
                                    do_normalization=templateTx,
                                    test_run=test_run,
                                    verbose=True )
                                if visualize:
                                    maxslice = np.min( [21, img.shape[2] ] )
                                    ants.plot_ortho( img, crop=True, title='Flair', filename=mymm+mysep+"flair.png", flat=True )
                                    ants.plot_ortho( img, tabPro['flair']['WMH_probability_map'], crop=True, title='Flair + WMH', filename=mymm+mysep+"flairWMH.png", flat=True )
                                    if tabPro['flair']['WMH_posterior_probability_map'] is not None:
                                        ants.plot_ortho( img, tabPro['flair']['WMH_posterior_probability_map'],  crop=True, title='Flair + prior WMH', filename=mymm+mysep+"flairpriorWMH.png", flat=True )
                            if ( mymod == 'rsfMRI_LR' or mymod == 'rsfMRI_RL' or mymod == 'rsfMRI' )  and ishapelen == 4:
                                img2 = None
                                if len( myimgsr ) > 1:
                                    img2 = mm_read( myimgsr[myimgcount+1] )
                                    ishapelen2 = len( img2.shape )
                                    if ishapelen2 != 4 :
                                        img2 = None
                                dowrite=True
                                tabPro, normPro = mm( t1, hier,
                                    rsf_image=[img,img2],
                                    srmodel=None,
                                    do_tractography=False,
                                    do_kk=False,
                                    do_normalization=templateTx,
                                    test_run=test_run,
                                    verbose=True )
                                if tabPro['rsf'] is not None and visualize:
                                    maxslice = np.min( [21, tabPro['rsf']['meanBold'].shape[2] ] )
                                    ants.plot( tabPro['rsf']['meanBold'],
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='meanBOLD', filename=mymm+mysep+"meanBOLD.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], ants.iMath(tabPro['rsf']['alff'],"Normalize"),
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='ALFF', filename=mymm+mysep+"boldALFF.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], ants.iMath(tabPro['rsf']['falff'],"Normalize"),
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='fALFF', filename=mymm+mysep+"boldfALFF.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['DefaultMode'],
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='DefaultMode', filename=mymm+mysep+"boldDefaultMode.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['FrontoparietalTaskControl'],
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='FrontoparietalTaskControl', filename=mymm+mysep+"boldFrontoparietalTaskControl.png"  )
                            if ( mymod == 'DTI_LR' or mymod == 'DTI_RL' or mymod == 'DTI' ) and ishapelen == 4:
                                dowrite=True
                                bvalfn = re.sub( '.nii.gz', '.bval' , myimg )
                                bvecfn = re.sub( '.nii.gz', '.bvec' , myimg )
                                imgList = [ img ]
                                bvalfnList = [ bvalfn ]
                                bvecfnList = [ bvecfn ]
                                if len( myimgsr ) > 1:  # find DTI_RL
                                    dtilrfn = myimgsr[myimgcount+1]
                                    if len( dtilrfn ) == 1:
                                        bvalfnRL = re.sub( '.nii.gz', '.bval' , dtilrfn )
                                        bvecfnRL = re.sub( '.nii.gz', '.bvec' , dtilrfn )
                                        imgRL = ants.image_read( dtilrfn )
                                        imgList.append( imgRL )
                                        bvalfnList.append( bvalfnRL )
                                        bvecfnList.append( bvecfnRL )
                                srmodel_DTI_mdl=None
                                if srmodel_DTI is not False:
                                    temp = ants.get_spacing(img)
                                    dtspc=[temp[0],temp[1],temp[2]]
                                    bestup = siq.optimize_upsampling_shape( dtspc, modality='DTI' )
                                    mdlfn = ex_pathmm + "siq_default_sisr_" + bestup + "_1chan_featvggL6_best_mdl.h5"
                                    if isinstance( srmodel_DTI, str ):
                                        srmodel_DTI = re.sub( "bestup", bestup, srmodel_DTI )
                                        mdlfn = os.path.join( ex_pathmm, srmodel_DTI )
                                    if exists( mdlfn ):
                                        if verbose:
                                            print(mdlfn)
                                        srmodel_DTI_mdl = tf.keras.models.load_model( mdlfn, compile=False )
                                    else:
                                        print(mdlfn + " does not exist - wont use SR")
                                tabPro, normPro = mm( t1, hier,
                                    dw_image=imgList,
                                    bvals = bvalfnList,
                                    bvecs = bvecfnList,
                                    srmodel=srmodel_DTI_mdl,
                                    do_tractography=not test_run,
                                    do_kk=False,
                                    do_normalization=templateTx,
                                    test_run=test_run,
                                    verbose=True )
                                mydti = tabPro['DTI']
                                if visualize:
                                    maxslice = np.min( [21, mydti['recon_fa'] ] )
                                    ants.plot( mydti['recon_fa'],  axis=2, nslices=maxslice, ncol=7, crop=True, title='FA (supposed to be better)', filename=mymm+mysep+"FAbetter.png"  )
                                    ants.plot( mydti['recon_fa'], mydti['jhu_labels'], axis=2, nslices=maxslice, ncol=7, crop=True, title='FA + JHU', filename=mymm+mysep+"FAJHU.png"  )
                                    ants.plot( mydti['recon_md'],  axis=2, nslices=maxslice, ncol=7, crop=True, title='MD', filename=mymm+mysep+"MD.png"  )
                            if dowrite:
                                write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=t1wide, separator=mysep )
                                for mykey in normPro.keys():
                                    if normPro[mykey] is not None:
                                        if visualize:
                                            ants.plot( template, normPro[mykey], axis=2, nslices=21, ncol=7, crop=True, title=mykey, filename=mymm+mysep+mykey+".png"   )
        if overmodX == nrg_modality_list[ len( nrg_modality_list ) - 1 ]:
            return
        if verbose:
            print("done with " + overmodX )
    if verbose:
        print("mm_nrg complete.")
    return



def mm_csv(
    studycsv,   # pandas data frame
    mysep = '-', # or "_" for BIDS
    srmodel_T1 = False, # optional - will add a great deal of time
    srmodel_NM = False, # optional - will add a great deal of time
    srmodel_DTI = False, # optional - will add a great deal of time
    dti_motion_correct = 'Rigid',
    dti_denoise = False,
    nrg_modality_list = None
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

    this function does not assume NRG format for the input data ....

    Parameters
    -------------

    studycsv : must have columns:
        - subjectID
        - date or session
        - imageID
        - modality
        - sourcedir
        - outputdir
        - filename (path to the t1 image)
        other relevant columns include nmid1-10, rsfid1, rsfid2, dtid1, dtid2, flairid;
        these provide filenames for these modalities: nm=neuromelanin, dti=diffusion tensor,
        rsf=resting state fmri, flair=T2Flair.  none of these are required. only
        t1 is required. rsfid1/rsfid2 will be processed jointly. same for dtid1/dtid2 and nmid*.
        see antspymm.generate_mm_dataframe

    sourcedir : a study specific folder containing individual subject folders

    outputdir : a study specific folder where individual output subject folders will go

    filename : the raw image filename (full path)

    srmodel_T1 : False (default) - will add a great deal of time - or h5 filename, 2 chan

    srmodel_NM : False (default) - will add a great deal of time - or h5 filename, 1 chan

    srmodel_DTI : False (default) - will add a great deal of time - or h5 filename, 1 chan

    dti_motion_correct : None, Rigid or SyN

    dti_denoise : boolean

    nrg_modality_list : optional; defaults to None; use to focus on a given modality

    Returns
    ---------

    writes output to disk and produces figures

    """
    visualize = True
    verbose = True
    if nrg_modality_list is None:
        nrg_modality_list = get_valid_modalities()
    if studycsv.shape[0] < 1:
        raise ValueError('studycsv has no rows')
    musthavecols = ['projectID', 'subjectID','date','imageID','modality','sourcedir','outputdir','filename']
    for k in range(len(musthavecols)):
        if not musthavecols[k] in studycsv.keys():
            raise ValueError('studycsv is missing column ' +musthavecols[k] )
    def makewideout( x, separator = mysep ):
        return x + separator + 'mmwide.csv'
    testloop = False
    counter=0
    import glob as glob
    from os.path import exists
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    ex_pathmm = os.path.expanduser( "~/.antspymm/" )
    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
    if not exists( templatefn ):
        print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
        antspyt1w.get_data( force_download=True )
        get_data( force_download=True )
    template = mm_read( templatefn ) # Read in template
    test_run = False
    if test_run:
        visualize=False
    # get sid and dtid from studycsv
    # musthavecols = ['projectID','subjectID','date','imageID','modality','sourcedir','outputdir','filename']
    projid = str(studycsv['projectID'].iloc[0])
    sid = str(studycsv['subjectID'].iloc[0])
    dtid = str(studycsv['date'].iloc[0])
    iid = str(studycsv['imageID'].iloc[0])
    t1iidUse=iid
    modality = str(studycsv['modality'].iloc[0])
    sourcedir = str(studycsv['sourcedir'].iloc[0])
    outputdir = str(studycsv['outputdir'].iloc[0])
    filename = str(studycsv['filename'].iloc[0])
    if not exists(filename):
            raise ValueError('mm_nrg cannot find filename ' + filename + ' in mm_csv' )
    def docsamson( locmod, t1iid=None, verbose=True ):
        myimgsInput = []
        myoutputPrefix = None
        imfns = [ 'filename', 'rsfid1', 'rsfid2', 'dtid1', 'dtid2', 'flairid' ]
        if locmod == 'T1w':
            imfns=['filename']
        elif locmod == 'T2Flair':
            imfns=['flairid']
        elif locmod == 'NM2DMT':
            imfns=[]
            for i in range(11):
                imfns.append( 'nmid' + str(i) )
        elif locmod == 'rsfMRI':
            imfns=[]
            for i in range(3):
                imfns.append( 'rsfid' + str(i) )
        elif locmod == 'DTI':
            imfns=[]
            for i in range(3):
                imfns.append( 'dtid' + str(i) )
        for i in imfns:
            if verbose:
                print( i + " " + locmod )
            if i in studycsv.keys():
                fni=str(studycsv[i].iloc[0])
                if verbose:
                    print( i + " " + fni + ' exists ' + str( exists( fni ) ) )
                if exists( fni ):
                    myimgsInput.append( fni )
                    temp = os.path.basename( fni )
                    mysplit = temp.split( mysep )
                    iid = re.sub( ".nii.gz", "", mysplit[len(mysplit)-1] )
                    iid = re.sub( ".mha", "", iid )
                    iid = re.sub( ".nii", "", iid )
                    iid2 = iid
                    if locmod != 'T1w' and t1iid is not None:
                        iid2=iid+"_"+t1iid
                    myoutputPrefix = outputdir + "/" + projid + "/" + sid + "/" + dtid + "/" + locmod + '/' + iid + "/" + projid + mysep + sid + mysep + dtid + mysep + locmod + mysep + iid2
        if verbose:
            print("VERBOSE in docsamson")
            print( locmod )
            print( myimgsInput )
            print( myoutputPrefix )
        return {
            'modality': locmod,
            'outprefix': myoutputPrefix,
            'images': myimgsInput
            }
    # hierarchical
    # NOTE: if there are multiple T1s for this time point, should take
    # the one with the highest resnetGrade
    t1fn = filename
    if not exists( t1fn ):
        raise ValueError('mm_nrg cannot find the T1w with uid ' + t1fn )
    t1 = mm_read( t1fn )
    hierfn = outputdir + "/"  + projid + "/" + sid + "/" + dtid + "/" + "T1wHierarchical" + '/' + iid + "/" + projid + mysep + sid + mysep + dtid + mysep + "T1wHierarchical" + mysep + iid + mysep
    hierfnSR = outputdir + "/" + projid + "/"  + sid + "/" + dtid + "/" + "T1wHierarchicalSR" + '/' + iid + "/" + projid + mysep + sid + mysep + dtid + mysep + "T1wHierarchicalSR" + mysep + iid + mysep
    hierfntest = hierfn + 'snseg.csv'
    if verbose:
        print( hierfntest )
    regout = hierfn + "syn"
    templateTx = {
        'fwdtransforms': [ regout+'1Warp.nii.gz', regout+'0GenericAffine.mat'],
        'invtransforms': [ regout+'0GenericAffine.mat', regout+'1InverseWarp.nii.gz']  }
    if verbose:
        print( "REGISTRATION EXISTENCE: " +
            str(exists( templateTx['fwdtransforms'][0])) + " " +
            str(exists( templateTx['fwdtransforms'][1])) + " " +
            str(exists( templateTx['invtransforms'][0])) + " " +
            str(exists( templateTx['invtransforms'][1])) )
    if verbose:
        print( hierfntest )
    hierexists = exists( hierfntest ) # FIXME should test this explicitly but we assume it here
    hier = None
    if not hierexists and not testloop:
        subjectpropath = os.path.dirname( hierfn )
        if verbose:
            print( subjectpropath )
        os.makedirs( subjectpropath, exist_ok=True  )
        hier = antspyt1w.hierarchical( t1, hierfn, labels_to_register=None )
        antspyt1w.write_hierarchical( hier, hierfn )
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
        t1wide.to_csv( hierfn + 'mmwide.csv' )
    ################# read the hierarchical data ###############################
    hier = antspyt1w.read_hierarchical( hierfn )
    if exists( hierfn + 'mmwide.csv' ) :
        t1wide = pd.read_csv( hierfn + 'mmwide.csv' )
    elif not testloop:
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
    if srmodel_T1 is not False :
        hierfntest = hierfnSR + 'mtl.csv'
        if verbose:
            print( hierfntest )
        hierexists = exists( hierfntest ) # FIXME should test this explicitly but we assume it here
        if not hierexists:
            subjectpropath = os.path.dirname( hierfnSR )
            if verbose:
                print( subjectpropath )
            os.makedirs( subjectpropath, exist_ok=True  )
            # hierarchical_to_sr(t1hier, sr_model, tissue_sr=False, blending=0.5, verbose=False)
            bestup = siq.optimize_upsampling_shape( ants.get_spacing(t1), modality='T1' )
            mdlfn = ex_pathmm + "siq_default_sisr_" + bestup + "_2chan_featvggL6_postseg_best_mdl.h5"
            if isinstance( srmodel_T1, str ):
                mdlfn = os.path.join( ex_pathmm, srmodel_T1 )
            if verbose:
                print( mdlfn )
            if exists( mdlfn ):
                srmodel_T1_mdl = tf.keras.models.load_model( mdlfn, compile=False )
            else:
                print( mdlfn + " does not exist - will not run.")
            hierSR = antspyt1w.hierarchical_to_sr( hier, srmodel_T1_mdl, blending=None, tissue_sr=False )
            antspyt1w.write_hierarchical( hierSR, hierfnSR )
            t1wideSR = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                    hierSR['dataframes'], identifier=None )
            t1wideSR.to_csv( hierfnSR + 'mmwide.csv' )
    hier = antspyt1w.read_hierarchical( hierfn )
    if exists( hierfn + 'mmwide.csv' ) :
        t1wide = pd.read_csv( hierfn + 'mmwide.csv' )
    elif not testloop:
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
                hier['dataframes'], identifier=None )
    if not testloop:
        t1imgbrn = hier['brain_n4_dnz']
        t1atropos = hier['dkt_parc']['tissue_segmentation']
    # loop over modalities and then unique image IDs
    # we treat NM in a "special" way -- aggregating repeats
    # other modalities (beyond T1) are treated individually
    for overmodX in nrg_modality_list:
        # define 1. input images 2. output prefix
        mydoc = docsamson( overmodX, t1iid=t1iidUse )
        myimgsr = mydoc['images']
        mymm = mydoc['outprefix']
        mymod = mydoc['modality']
        if verbose:
            print( mydoc )
        if len(myimgsr) > 0:
            dowrite=False
            if verbose:
                print( 'overmodX is : ' + overmodX )
                print( 'example image name is : '  )
                print( myimgsr )
            if overmodX == 'NM2DMT':
                subjectpropath = os.path.dirname( mydoc['outprefix'] )
                if verbose:
                    print("subjectpropath is")
                    print(subjectpropath)
                    os.makedirs( subjectpropath, exist_ok=True  )
                myimgsr2 = myimgsr
                myimgsr2.sort()
                is4d = False
                temp = ants.image_read( myimgsr2[0] )
                if temp.dimension == 4:
                    is4d = True
                if len( myimgsr2 ) == 1 and not is4d: # check dimension
                    myimgsr2 = myimgsr2 + myimgsr2
                mymmout = makewideout( mymm )
                if verbose and not exists( mymmout ):
                    print( "NM " + mymm  + ' execution ')
                elif verbose and exists( mymmout ) :
                    print( "NM " + mymm + ' complete ' )
                if exists( mymmout ):
                    continue
                if is4d:
                    nmlist = ants.ndimage_to_list( mm_read( myimgsr2[0] ) )
                else:
                    nmlist = []
                    for zz in myimgsr2:
                        nmlist.append( mm_read( zz ) )
                srmodel_NM_mdl = None
                if srmodel_NM is not False:
                    bestup = siq.optimize_upsampling_shape( ants.get_spacing(nmlist[0]), modality='NM', roundit=True )
                    mdlfn = ex_pathmm + "siq_default_sisr_" + bestup + "_1chan_featvggL6_best_mdl.h5"
                    if isinstance( srmodel_NM, str ):
                        srmodel_NM = re.sub( "bestup", bestup, srmodel_NM )
                        mdlfn = os.path.join( ex_pathmm, srmodel_NM )
                    if exists( mdlfn ):
                        if verbose:
                            print(mdlfn)
                        srmodel_NM_mdl = tf.keras.models.load_model( mdlfn, compile=False  )
                    else:
                        print( mdlfn + " does not exist - wont use SR")
                if not testloop:
                    tabPro, normPro = mm( t1, hier,
                            nm_image_list = nmlist,
                            srmodel=srmodel_NM_mdl,
                            do_tractography=False,
                            do_kk=False,
                            do_normalization=templateTx,
                            test_run=test_run,
                            verbose=True )
                    if not test_run:
                        write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=None, separator=mysep )
                        nmpro = tabPro['NM']
                        mysl = range( nmpro['NM_avg'].shape[2] )
                    if visualize:
                        mysl = range( nmpro['NM_avg'].shape[2] )
                        ants.plot( nmpro['NM_avg'],  nmpro['t1_to_NM'], slices=mysl, axis=2, title='nm + t1', filename=mymm+mysep+"NMavg.png" )
                        mysl = range( nmpro['NM_avg_cropped'].shape[2] )
                        ants.plot( nmpro['NM_avg_cropped'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop', filename=mymm+mysep+"NMavgcrop.png" )
                        ants.plot( nmpro['NM_avg_cropped'], nmpro['t1_to_NM'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop + t1', filename=mymm+mysep+"NMavgcropt1.png" )
                        ants.plot( nmpro['NM_avg_cropped'], nmpro['NM_labels'], axis=2, slices=mysl, title='nm crop + labels', filename=mymm+mysep+"NMavgcroplabels.png" )
            else :
                if len( myimgsr ) > 0:
                    dowrite=False
                    myimgcount=0
                    if len( myimgsr ) > 0 :
                        myimg = myimgsr[myimgcount]
                        subjectpropath = os.path.dirname( mydoc['outprefix'] )
                        if verbose:
                            print("subjectpropath is")
                            print(subjectpropath)
                        os.makedirs( subjectpropath, exist_ok=True  )
                        mymmout = makewideout( mymm )
                        if verbose and not exists( mymmout ):
                            print("Modality specific processing: " + mymod + " execution " )
                            print( mymm )
                        elif verbose and exists( mymmout ) :
                            print("Modality specific processing: " + mymod + " complete " )
                        if exists( mymmout ) :
                            continue
                        if verbose:
                            print(subjectpropath)
                            print( myimg )
                        if not testloop:
                            img = mm_read( myimg )
                            ishapelen = len( img.shape )
                            if mymod == 'T1w' and ishapelen == 3: # for a real run, set to True
                                if not exists( regout + "logjacobian.nii.gz" ) or not exists( regout+'1Warp.nii.gz' ):
                                    if verbose:
                                        print('start t1 registration')
                                    ex_path = os.path.expanduser( "~/.antspyt1w/" )
                                    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
                                    template = mm_read( templatefn )
                                    template = ants.resample_image( template, [1,1,1], use_voxels=False )
                                    t1reg = ants.registration( template, hier['brain_n4_dnz'],
                                        "antsRegistrationSyNQuickRepro[s]", outprefix = regout, verbose=False )
                                    myjac = ants.create_jacobian_determinant_image( template,
                                        t1reg['fwdtransforms'][0], do_log=True, geom=True )
                                    image_write_with_thumbnail( myjac, regout + "logjacobian.nii.gz", thumb=False )
                                    if visualize:
                                        ants.plot( ants.iMath(t1reg['warpedmovout'],"Normalize"),  axis=2, nslices=21, ncol=7, crop=True, title='warped to template', filename=regout+"totemplate.png" )
                                        ants.plot( ants.iMath(myjac,"Normalize"),  axis=2, nslices=21, ncol=7, crop=True, title='jacobian', filename=regout+"jacobian.png" )
                                if not exists( mymm + mysep + "kk_norm.nii.gz" ):
                                    dowrite=True
                                    if verbose:
                                        print('start kk')
                                    tabPro, normPro = mm( t1, hier,
                                        srmodel=None,
                                        do_tractography=False,
                                        do_kk=True,
                                        do_normalization=templateTx,
                                        test_run=test_run,
                                        verbose=True )
                                    if visualize:
                                        maxslice = np.min( [21, hier['brain_n4_dnz'].shape[2] ] )
                                        ants.plot( hier['brain_n4_dnz'],  axis=2, nslices=maxslice, ncol=7, crop=True, title='brain extraction', filename=mymm+mysep+"brainextraction.png" )
                                        ants.plot( tabPro['kk']['thickness_image'], axis=2, nslices=maxslice, ncol=7, crop=True, title='kk',
                                        cmap='plasma', filename=mymm+mysep+"kkthickness.png" )
                            if mymod == 'T2Flair' and ishapelen == 3:
                                dowrite=True
                                tabPro, normPro = mm( t1, hier,
                                    flair_image = img,
                                    srmodel=None,
                                    do_tractography=False,
                                    do_kk=False,
                                    do_normalization=templateTx,
                                    test_run=test_run,
                                    verbose=True )
                                if visualize:
                                    maxslice = np.min( [21, img.shape[2] ] )
                                    ants.plot_ortho( img, crop=True, title='Flair', filename=mymm+mysep+"flair.png", flat=True )
                                    ants.plot_ortho( img, tabPro['flair']['WMH_probability_map'], crop=True, title='Flair + WMH', filename=mymm+mysep+"flairWMH.png", flat=True )
                                    if tabPro['flair']['WMH_posterior_probability_map'] is not None:
                                        ants.plot_ortho( img, tabPro['flair']['WMH_posterior_probability_map'],  crop=True, title='Flair + prior WMH', filename=mymm+mysep+"flairpriorWMH.png", flat=True )
                            if ( mymod == 'rsfMRI_LR' or mymod == 'rsfMRI_RL' or mymod == 'rsfMRI' )  and ishapelen == 4:
                                img2 = None
                                if len( myimgsr ) > 1:
                                    img2 = mm_read( myimgsr[myimgcount+1] )
                                    ishapelen2 = len( img2.shape )
                                    if ishapelen2 != 4 :
                                        img2 = None
                                dowrite=True
                                tabPro, normPro = mm( t1, hier,
                                    rsf_image=[img,img2],
                                    srmodel=None,
                                    do_tractography=False,
                                    do_kk=False,
                                    do_normalization=templateTx,
                                    test_run=test_run,
                                    verbose=True )
                                if tabPro['rsf'] is not None and visualize:
                                    maxslice = np.min( [21, tabPro['rsf']['meanBold'].shape[2] ] )
                                    ants.plot( tabPro['rsf']['meanBold'],
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='meanBOLD', filename=mymm+mysep+"meanBOLD.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], ants.iMath(tabPro['rsf']['alff'],"Normalize"),
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='ALFF', filename=mymm+mysep+"boldALFF.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], ants.iMath(tabPro['rsf']['falff'],"Normalize"),
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='fALFF', filename=mymm+mysep+"boldfALFF.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['DefaultMode'],
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='DefaultMode', filename=mymm+mysep+"boldDefaultMode.png" )
                                    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['FrontoparietalTaskControl'],
                                        axis=2, nslices=maxslice, ncol=7, crop=True, title='FrontoparietalTaskControl', filename=mymm+mysep+"boldFrontoparietalTaskControl.png"  )
                            if ( mymod == 'DTI_LR' or mymod == 'DTI_RL' or mymod == 'DTI' ) and ishapelen == 4:
                                dowrite=True
                                bvalfn = re.sub( '.nii.gz', '.bval' , myimg )
                                bvecfn = re.sub( '.nii.gz', '.bvec' , myimg )
                                imgList = [ img ]
                                bvalfnList = [ bvalfn ]
                                bvecfnList = [ bvecfn ]
                                if len( myimgsr ) > 1:  # find DTI_RL
                                    dtilrfn = myimgsr[myimgcount+1]
                                    if exists( dtilrfn ):
                                        bvalfnRL = re.sub( '.nii.gz', '.bval' , dtilrfn )
                                        bvecfnRL = re.sub( '.nii.gz', '.bvec' , dtilrfn )
                                        imgRL = ants.image_read( dtilrfn )
                                        imgList.append( imgRL )
                                        bvalfnList.append( bvalfnRL )
                                        bvecfnList.append( bvecfnRL )
                                # check existence of all files expected ...
                                for dtiex in bvalfnList+bvecfnList+myimgsr:
                                    if not exists(dtiex):
                                        raise ValueError('mm_csv dti data ' + dtiex )
                                srmodel_DTI_mdl=None
                                if srmodel_DTI is not False:
                                    temp = ants.get_spacing(img)
                                    dtspc=[temp[0],temp[1],temp[2]]
                                    bestup = siq.optimize_upsampling_shape( dtspc, modality='DTI' )
                                    mdlfn = ex_pathmm + "siq_default_sisr_" + bestup + "_1chan_featvggL6_best_mdl.h5"
                                    if isinstance( srmodel_DTI, str ):
                                        srmodel_DTI = re.sub( "bestup", bestup, srmodel_DTI )
                                        mdlfn = os.path.join( ex_pathmm, srmodel_DTI )
                                    if exists( mdlfn ):
                                        if verbose:
                                            print(mdlfn)
                                        srmodel_DTI_mdl = tf.keras.models.load_model( mdlfn, compile=False )
                                    else:
                                        print(mdlfn + " does not exist - wont use SR")
                                tabPro, normPro = mm( t1, hier,
                                    dw_image=imgList,
                                    bvals = bvalfnList,
                                    bvecs = bvecfnList,
                                    srmodel=srmodel_DTI_mdl,
                                    do_tractography=not test_run,
                                    do_kk=False,
                                    do_normalization=templateTx,
                                    dti_motion_correct = dti_motion_correct,
                                    dti_denoise = dti_denoise,
                                    test_run=test_run,
                                    verbose=True )
                                mydti = tabPro['DTI']
                                if visualize:
                                    maxslice = np.min( [21, mydti['recon_fa'] ] )
                                    ants.plot( mydti['recon_fa'],  axis=2, nslices=maxslice, ncol=7, crop=True, title='FA (supposed to be better)', filename=mymm+mysep+"FAbetter.png"  )
                                    ants.plot( mydti['recon_fa'], mydti['jhu_labels'], axis=2, nslices=maxslice, ncol=7, crop=True, title='FA + JHU', filename=mymm+mysep+"FAJHU.png"  )
                                    ants.plot( mydti['recon_md'],  axis=2, nslices=maxslice, ncol=7, crop=True, title='MD', filename=mymm+mysep+"MD.png"  )
                            if dowrite:
                                write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=t1wide, separator=mysep )
                                for mykey in normPro.keys():
                                    if normPro[mykey] is not None:
                                        if visualize:
                                            ants.plot( template, normPro[mykey], axis=2, nslices=21, ncol=7, crop=True, title=mykey, filename=mymm+mysep+mykey+".png"   )
        if overmodX == nrg_modality_list[ len( nrg_modality_list ) - 1 ]:
            return
        if verbose:
            print("done with " + overmodX )
    if verbose:
        print("mm_nrg complete.")
    return

def spec_taper(x, p=0.1):
    from scipy import stats, signal, fft
    from statsmodels.regression.linear_model import yule_walker
    """
    Computes a tapered version of x, with tapering p.

    Adapted from R's stats::spec.taper at https://github.com/telmo-correa/time-series-analysis/blob/master/Python/spectrum.py

    """

    p = np.r_[p]
    assert np.all((p >= 0) & (p < 0.5)), "'p' must be between 0 and 0.5"

    x = np.r_[x].astype('float64')
    original_shape = x.shape

    assert len(original_shape) <= 2, "'x' must have at most 2 dimensions"
    while len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)

    nr, nc = x.shape
    if len(p) == 1:
        p = p * np.ones(nc)
    else:
        assert len(p) == nc, "length of 'p' must be 1 or equal the number of columns of 'x'"

    for i in range(nc):
        m = int(np.floor(nr * p[i]))
        if m == 0:
            continue
        w = 0.5 * (1 - np.cos(np.pi * np.arange(1, 2 * m, step=2)/(2 * m)))
        x[:, i] = np.r_[w, np.ones(nr - 2 * m), w[::-1]] * x[:, i]

    x = np.reshape(x, original_shape)
    return x

def plot_spec(spec_res, coverage=None, ax=None, title=None):
    import matplotlib.pyplot as plt
    """Convenience plotting method, also includes confidence cross in the same style as R.

    Note that the location of the cross is irrelevant; only width and height matter."""
    f, Pxx = spec_res['freq'], spec_res['spec']

    if coverage is not None:
        ci = spec_ci(spec_res['df'], coverage=coverage)
        conf_x = (max(spec_res['freq']) - spec_res['bandwidth']) + np.r_[-0.5, 0.5] * spec_res['bandwidth']
        conf_y = max(spec_res['spec']) / ci[1]

    if ax is None:
        ax = plt.gca()

    ax.plot(f, Pxx, color='C0')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Log Spectrum')
    ax.set_yscale('log')
    if coverage is not None:
        ax.plot(np.mean(conf_x) * np.r_[1, 1], conf_y * ci, color='red')
        ax.plot(conf_x, np.mean(conf_y) * np.r_[1, 1], color='red')

    ax.set_title(spec_res['method'] if title is None else title)

def spec_ci(df, coverage=0.95):
    from scipy import stats, signal, fft
    from statsmodels.regression.linear_model import yule_walker
    """
    Computes the confidence interval for a spectral fit, based on the number of degrees of freedom.

    Adapted from R's stats::plot.spec at https://github.com/telmo-correa/time-series-analysis/blob/master/Python/spectrum.py

    """

    assert coverage >= 0 and coverage < 1, "coverage probability out of range [0, 1)"

    tail = 1 - coverage

    phi = stats.chi2.cdf(x=df, df=df)
    upper_quantile = 1 - tail * (1 - phi)
    lower_quantile = tail * phi

    return df / stats.chi2.ppf([upper_quantile, lower_quantile], df=df)

def spec_pgram(x, xfreq=1, spans=None, kernel=None, taper=0.1, pad=0, fast=True, demean=False, detrend=True,
               plot=True, **kwargs):
    """
    Computes the spectral density estimate using a periodogram.  Optionally, it also:
    - Uses a provided kernel window, or a sequence of spans for convoluted modified Daniell kernels.
    - Tapers the start and end of the series to avoid end-of-signal effects.
    - Pads the provided series before computation, adding pad*(length of series) zeros at the end.
    - Pads the provided series before computation to speed up FFT calculation.
    - Performs demeaning or detrending on the series.
    - Plots results.

    Implemented to ensure compatibility with R's spectral functions, as opposed to reusing scipy's periodogram.

    Adapted from R's stats::spec.pgram at https://github.com/telmo-correa/time-series-analysis/blob/master/Python/spectrum.py

    example:

    import numpy as np
    import antspymm
    myx = np.random.rand(100,1)
    myspec = antspymm.spec_pgram(myx,0.5)

    """
    from scipy import stats, signal, fft
    from statsmodels.regression.linear_model import yule_walker
    def daniell_window_modified(m):
        """ Single-pass modified Daniell kernel window.

        Weight is normalized to add up to 1, and all values are the same, other than the first and the
        last, which are divided by 2.
        """
        def w(k):
            return np.where(np.abs(k) < m, 1 / (2*m), np.where(np.abs(k) == m, 1/(4*m), 0))

        return w(np.arange(-m, m+1))

    def daniell_window_convolve(v):
        """ Convolved version of multiple modified Daniell kernel windows.

        Parameter v should be an iterable of m values.
        """

        if len(v) == 0:
            return np.r_[1]

        if len(v) == 1:
            return daniell_window_modified(v[0])

        return signal.convolve(daniell_window_modified(v[0]), daniell_window_convolve(v[1:]))

    # Ensure we can store non-integers in x, and that it is a numpy object
    x = np.r_[x].astype('float64')
    original_shape = x.shape

    # Ensure correct dimensions
    assert len(original_shape) <= 2, "'x' must have at most 2 dimensions"
    while len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)

    N, nser = x.shape
    N0 = N

    # Ensure only one of spans, kernel is provided, and build the kernel window if needed
    assert (spans is None) or (kernel is None), "must specify only one of 'spans' or 'kernel'"
    if spans is not None:
        kernel = daniell_window_convolve(np.floor_divide(np.r_[spans], 2))

    # Detrend or demean the series
    if detrend:
        t = np.arange(N) - (N - 1)/2
        sumt2 = N * (N**2 - 1)/12
        x -= (np.repeat(np.expand_dims(np.mean(x, axis=0), 0), N, axis=0) + np.outer(np.sum(x.T * t, axis=1), t/sumt2).T)
    elif demean:
        x -= np.mean(x, axis=0)

    # Compute taper and taper adjustment variables
    x = spec_taper(x, taper)
    u2 = (1 - (5/8) * taper * 2)
    u4 = (1 - (93/128) * taper * 2)

    # Pad the series with copies of the same shape, but filled with zeroes
    if pad > 0:
        x = np.r_[x, np.zeros((pad * x.shape[0], x.shape[1]))]
        N = x.shape[0]

    # Further pad the series to accelerate FFT computation
    if fast:
        newN = fft.next_fast_len(N, True)
        x = np.r_[x, np.zeros((newN - N, x.shape[1]))]
        N = newN

    # Compute the Fourier frequencies (R's spec.pgram convention style)
    Nspec = int(np.floor(N/2))
    freq = (np.arange(Nspec) + 1) * xfreq / N

    # Translations to keep same row / column convention as stats::mvfft
    xfft = fft.fft(x.T).T

    # Compute the periodogram for each i, j
    pgram = np.empty((N, nser, nser), dtype='complex')
    for i in range(nser):
        for j in range(nser):
            pgram[:, i, j] = xfft[:, i] * np.conj(xfft[:, j]) / (N0 * xfreq)
            pgram[0, i, j] = 0.5 * (pgram[1, i, j] + pgram[-1, i, j])

    if kernel is None:
        # Values pre-adjustment
        df = 2
        bandwidth = np.sqrt(1 / 12)
    else:
        def conv_circular(signal, kernel):
            """
            Performs 1D circular convolution, in the same style as R::kernapply,
            assuming the kernel window is centered at 0.
            """
            pad = len(signal) - len(kernel)
            half_window = int((len(kernel) + 1) / 2)
            indexes = range(-half_window, len(signal) - half_window)
            orig_conv = np.real(fft.ifft(fft.fft(signal) * fft.fft(np.r_[np.zeros(pad), kernel])))
            return orig_conv.take(indexes, mode='wrap')

        # Convolve pgram with kernel with circular conv
        for i in range(nser):
            for j in range(nser):
                pgram[:, i, j] = conv_circular(pgram[:, i, j], kernel)

        df = 2 / np.sum(kernel**2)
        m = (len(kernel) - 1)/2
        k = np.arange(-m, m+1)
        bandwidth = np.sqrt(np.sum((1/12 + k**2) * kernel))

    df = df/(u4/u2**2)*(N0/N)
    bandwidth = bandwidth * xfreq/N

    # Remove padded results
    pgram = pgram[1:(Nspec+1), :, :]

    spec = np.empty((Nspec, nser))
    for i in range(nser):
        spec[:, i] = np.real(pgram[:, i, i])

    if nser == 1:
        coh = None
        phase = None
    else:
        coh = np.empty((Nspec, int(nser * (nser - 1)/2)))
        phase = np.empty((Nspec, int(nser * (nser - 1)/2)))
        for i in range(nser):
            for j in range(i+1, nser):
                index = int(i + j*(j-1)/2)
                coh[:, index] = np.abs(pgram[:, i, j])**2 / (spec[:, i] * spec[:, j])
                phase[:, index] = np.angle(pgram[:, i, j])

    spec = spec / u2
    spec = spec.squeeze()

    results = {
        'freq': freq,
        'spec': spec,
        'coh': coh,
        'phase': phase,
        'kernel': kernel,
        'df': df,
        'bandwidth': bandwidth,
        'n.used': N,
        'orig.n': N0,
        'taper': taper,
        'pad': pad,
        'detrend': detrend,
        'demean': demean,
        'method': 'Raw Periodogram' if kernel is None else 'Smoothed Periodogram'
    }

    if plot:
        plot_spec(results, coverage=0.95, **kwargs)

    return results

def alffmap( x, flo=0.01, fhi=0.1, tr=1, detrend = True ):
    """
    Amplitude of Low Frequency Fluctuations (ALFF; Zang et al., 2007) and
    fractional Amplitude of Low Frequency Fluctuations (f/ALFF; Zou et al., 2008)
    are related measures that quantify the amplitude of low frequency
    oscillations (LFOs).  This function outputs ALFF and fALFF for the input.
    same function in ANTsR.

    x input vector for the time series of interest
    flo low frequency, typically 0.01
    fhi high frequency, typically 0.1
    tr the period associated with the vector x (inverse of frequency)
    detrend detrend the input time series

    return vector is output showing ALFF and fALFF values
    """
    temp = spec_pgram( x, xfreq=1.0/tr, demean=False, detrend=detrend, taper=0, fast=True, plot=False )
    fselect = np.logical_and( temp['freq'] >= flo, temp['freq'] <= fhi )
    denom = (temp['spec']).sum()
    numer = (temp['spec'][fselect]).sum()
    return {  'alff':numer, 'falff': numer/denom }


def alff_image( x, mask, flo=0.01, fhi=0.1, nuisance=None ):
    """
    Amplitude of Low Frequency Fluctuations (ALFF; Zang et al., 2007) and
    fractional Amplitude of Low Frequency Fluctuations (f/ALFF; Zou et al., 2008)
    are related measures that quantify the amplitude of low frequency
    oscillations (LFOs).  This function outputs ALFF and fALFF for the input.

    x - input clean resting state fmri
    mask - mask over which to compute f/alff
    flo - low frequency, typically 0.01
    fhi - high frequency, typically 0.1
    nuisance - optional nuisance matrix

    return dictionary with ALFF and fALFF images
    """
    xmat = ants.timeseries_to_matrix( x, mask )
    if nuisance is not None:
        xmat = ants.regress_components( xmat, nuisance )
    alffvec = xmat[0,:]*0
    falffvec = xmat[0,:]*0
    mytr = ants.get_spacing( x )[3]
    for n in range( xmat.shape[1] ):
        temp = alffmap( xmat[:,n], flo=flo, fhi=fhi, tr=mytr )
        alffvec[n]=temp['alff']
        falffvec[n]=temp['falff']
    alffi=ants.make_image( mask, alffvec )
    falffi=ants.make_image( mask, falffvec )
    return {  'alff': alffi, 'falff': falffi }


def down2iso( x, interpolation='linear', takemin=False ):
    """
    will downsample an anisotropic image to an isotropic resolution

    x: input image

    interpolation: linear or nearestneighbor

    takemin : boolean map to min space; otherwise max

    return image downsampled to isotropic resolution
    """
    spc = ants.get_spacing( x )
    if takemin:
        newspc = np.asarray(spc).min()
    else:
        newspc = np.asarray(spc).max()
    newspc = np.repeat( newspc, x.dimension )
    if interpolation == 'linear':
        xs = ants.resample_image( x, newspc, interp_type=0)
    else:
        xs = ants.resample_image( x, newspc, interp_type=1)
    return xs


def read_mm_csv( x, is_t1=False, colprefix=None, separator='-', verbose=False ):
    splitter=os.path.basename(x).split( separator )
    lensplit = len( splitter )-1
    temp = os.path.basename(x)
    temp = os.path.splitext(temp)[0]
    temp = re.sub(separator+'mmwide','',temp)
    idcols = ['u_hier_id','sid','visitdate','modality','mmimageuid','t1imageuid']
    df = pd.DataFrame( columns = idcols, index=range(1) )
    valstoadd = [temp] + splitter[1:(lensplit-1)]
    if is_t1:
        valstoadd = valstoadd + [splitter[(lensplit-1)],splitter[(lensplit-1)]]
    else:
        split2=splitter[(lensplit-1)].split( "_" )
        if len(split2) == 1:
            split2.append( split2[0] )
        if len(valstoadd) == 3:
            valstoadd = valstoadd + [split2[0]] + [math.nan] + [split2[1]]
        else:
            valstoadd = valstoadd + [split2[0],split2[1]]
    if verbose:
        print( valstoadd )
    df.iloc[0] = valstoadd
    if verbose:
        print( "read xdf: " + x )
    xdf = pd.read_csv( x )
    df.reset_index()
    xdf.reset_index(drop=True)
    if "Unnamed: 0" in xdf.columns:
        holder=xdf.pop( "Unnamed: 0" )
    if "Unnamed: 1" in xdf.columns:
        holder=xdf.pop( "Unnamed: 1" )
    if "u_hier_id.1" in xdf.columns:
        holder=xdf.pop( "u_hier_id.1" )
    if "u_hier_id" in xdf.columns:
        holder=xdf.pop( "u_hier_id" )
    if not is_t1:
        if 'resnetGrade' in xdf.columns:
            index_no = xdf.columns.get_loc('resnetGrade')
            xdf = xdf.drop( xdf.columns[range(index_no+1)] , axis=1)

    if xdf.shape[0] == 2:
        xdfcols = xdf.columns
        xdf = xdf.iloc[1]
        ddnum = xdf.to_numpy()
        ddnum = ddnum.reshape([1,ddnum.shape[0]])
        newcolnames = xdf.index.to_list()
        if len(newcolnames) != ddnum.shape[1]:
            print("Cannot Merge : Shape MisMatch " + str( len(newcolnames) ) + " " + str(ddnum.shape[1]))
        else:
            xdf = pd.DataFrame(ddnum, columns=xdfcols )
    if xdf.shape[1] == 0:
        return None
    if colprefix is not None:
        xdf.columns=colprefix + xdf.columns
    return pd.concat( [df,xdf], axis=1 )

def assemble_modality_specific_dataframes( mm_wide_csvs, hierdfin, nrg_modality, separator='-', progress=None, verbose=False ):
    moddersub = re.sub( "[*]","",nrg_modality)
    nmdf=pd.DataFrame()
    for k in range( hierdfin.shape[0] ):
        if progress is not None:
            if k % progress == 0:
                progger = str( np.round( k / hierdfin.shape[0] * 100 ) )
                print( progger, end ="...", flush=True)
        temp = mm_wide_csvs[k]
        mypartsf = temp.split("T1wHierarchical")
        myparts = mypartsf[0]
        t1iid = str(mypartsf[1].split("/")[1])
        fnsnm = glob.glob(myparts+"/" + nrg_modality + "/*/*" + t1iid + "*wide.csv")
        if len( fnsnm ) > 0 :
            for y in fnsnm:
                temp=read_mm_csv( y, colprefix=moddersub+'_', is_t1=False, separator=separator, verbose=verbose )
                if temp is not None:
                    nmdf=pd.concat( [nmdf, temp], axis=0)
    return nmdf

def bind_wide_mm_csvs( mm_wide_csvs, merge=True, separator='-', verbose = 0 ) :
    """
    will convert a list of t1w hierarchical csv filenames to a merged dataframe

    returns a pair of data frames, the left side having all entries and the
        right side having row averaged entries i.e. unique values for each visit

    set merge to False to return individual dataframes ( for debugging )

    return alldata, row_averaged_data
    """
    mm_wide_csvs.sort()
    if not mm_wide_csvs:
        print("No files found with specified pattern")
        return
    # 1. row-bind the t1whier data
    # 2. same for each other modality
    # 3. merge the modalities by the keys
    hierdf = pd.DataFrame()
    for y in mm_wide_csvs:
        temp=read_mm_csv( y, colprefix='T1Hier_', separator=separator, is_t1=True )
        if temp is not None:
            hierdf=pd.concat( [hierdf, temp], axis=0)
    if verbose > 0:
        mypro=50
    else:
        mypro=None
    if verbose > 0:
        print("thickness")
    thkdf = assemble_modality_specific_dataframes( mm_wide_csvs, hierdf, 'T1w', progress=mypro, verbose=verbose==2)
    if verbose > 0:
        print("flair")
    flairdf = assemble_modality_specific_dataframes( mm_wide_csvs, hierdf, 'T2Flair', progress=mypro, verbose=verbose==2)
    if verbose > 0:
        print("NM")
    nmdf = assemble_modality_specific_dataframes( mm_wide_csvs, hierdf, 'NM2DMT', progress=mypro, verbose=verbose==2)
    if verbose > 0:
        print("rsf")
    rsfdf = assemble_modality_specific_dataframes( mm_wide_csvs, hierdf, 'rsfMRI*', progress=mypro, verbose=verbose==2)
    if verbose > 0:
        print("dti")
    dtidf = assemble_modality_specific_dataframes( mm_wide_csvs, hierdf, 'DTI*', progress=mypro, verbose=verbose==2 )
    if not merge:
        return hierdf, thkdf, flairdf, nmdf, rsfdf, dtidf
    hierdfmix = hierdf.copy()
    modality_df_suffixes = [
        (thkdf, "_thk"),
        (flairdf, "_flair"),
        (nmdf, "_nm"),
        (rsfdf, "_rsf"),
        (dtidf, "_dti"),
    ]
    for pair in modality_df_suffixes:
        hierdfmix = merge_mm_dataframe(hierdfmix, pair[0], pair[1])
    hierdfmix = hierdfmix.replace(r'^\s*$', np.nan, regex=True)
    return hierdfmix, hierdfmix.groupby("u_hier_id", as_index=False).mean(numeric_only=True)

def merge_mm_dataframe(hierdf, mmdf, mm_suffix):
    try:
        hierdf = hierdf.merge(mmdf, on=['sid', 'visitdate', 't1imageuid'], suffixes=("",mm_suffix),how='left')
        return hierdf
    except KeyError:
        return hierdf

def augment_image( x,  max_rot=10, nzsd=1 ):
    rRotGenerator = ants.contrib.RandomRotate3D( ( max_rot*(-1.0), max_rot ), reference=x )
    tx = rRotGenerator.transform()
    itx = ants.invert_ants_transform(tx)
    y = ants.apply_ants_transform_to_image( tx, x, x, interpolation='linear')
    y = ants.add_noise_to_image( y,'additivegaussian', [0,nzsd] )
    return y, tx, itx

def boot_wmh( flair, t1, t1seg, mmfromconvexhull = 0.0, strict=True,
        probability_mask=None, prior_probability=None, n_simulations=16,
        verbose=False ) :
    if verbose and prior_probability is None:
        print("augmented flair")
    if verbose and prior_probability is not None:
        print("augmented flair with prior")
    wmh_sum_aug = 0
    wmh_sum_prior_aug = 0
    augprob = flair * 0.0
    augprob_prior = None
    if prior_probability is not None:
        augprob_prior = flair * 0.0
    for n in range(n_simulations):
        augflair, tx, itx = augment_image( flair, 5 )
        locwmh = wmh( augflair, t1, t1seg, mmfromconvexhull = mmfromconvexhull,
            strict=strict, probability_mask=None, prior_probability=prior_probability )
        if verbose:
            print( "flair sim: " + str(n) + " vol: " + str( locwmh['wmh_mass'] )+ " vol-prior: " + str( locwmh['wmh_mass_prior'] )+ " snr: " + str( locwmh['wmh_SNR'] ) )
        wmh_sum_aug = wmh_sum_aug + locwmh['wmh_mass']
        wmh_sum_prior_aug = wmh_sum_prior_aug + locwmh['wmh_mass_prior']
        temp = locwmh['WMH_probability_map']
        augprob = augprob + ants.apply_ants_transform_to_image( itx, temp, flair, interpolation='linear')
        if prior_probability is not None:
            temp = locwmh['WMH_posterior_probability_map']
            augprob_prior = augprob_prior + ants.apply_ants_transform_to_image( itx, temp, flair, interpolation='linear')
    augprob = augprob * (1.0/float( n_simulations ))
    if prior_probability is not None:
        augprob_prior = augprob_prior * (1.0/float( n_simulations ))
    wmh_sum_aug = wmh_sum_aug / float( n_simulations )
    wmh_sum_prior_aug = wmh_sum_prior_aug / float( n_simulations )
    return{
      'WMH_probability_map' : augprob,
      'WMH_posterior_probability_map' : augprob_prior,
      'wmh_mass': wmh_sum_aug,
      'wmh_mass_prior': wmh_sum_prior_aug,
      'wmh_evr': locwmh['wmh_evr'],
      'wmh_SNR': locwmh['wmh_SNR']  }


def threaded_bind_wide_mm_csvs( mm_wide_csvs, n_workers ):
    from concurrent.futures import as_completed
    from concurrent import futures
    import concurrent.futures
    def chunks(l, n):
        """Yield n number of sequential chunks from l."""
        d, r = divmod(len(l), n)
        for i in range(n):
            si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
            yield l[si:si+(d+1 if i < r else d)]
    import numpy as np
    newx = list( chunks( mm_wide_csvs, n_workers ) )
    import pandas as pd
    alldf = pd.DataFrame()
    alldfavg = pd.DataFrame()
    with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        to_do = []
        for group in range(len(newx)) :
            future = executor.submit(bind_wide_mm_csvs, newx[group] )
            to_do.append(future)
        results = []
        for future in futures.as_completed(to_do):
            res0, res1 = future.result()
            alldf=pd.concat(  [alldf, res0 ], axis=0 )
            alldfavg=pd.concat(  [alldfavg, res1 ], axis=0 )
    return alldf, alldfavg


def get_names_from_data_frame(x, demogIn, exclusions=None):
    """
    data = {'Name':['Tom', 'nick', 'krish', 'jack'], 'Age':[20, 21, 19, 18]}
    antspymm.get_names_from_data_frame( ['e'], df )
    antspymm.get_names_from_data_frame( ['a','e'], df )
    antspymm.get_names_from_data_frame( ['e'], df, exclusions='N' )
    """
    def get_unique( qq ):
        unique = []
        for number in qq:
            if number in unique:
                continue
            else:
                unique.append(number)
        return unique
    outnames = list(demogIn.columns[demogIn.columns.str.contains(x[0])])
    if len(x) > 1:
        for y in x[1:]:
            outnames = [i for i in outnames if y in i]
    outnames = get_unique( outnames )
    if exclusions is not None:
        toexclude = [name for name in outnames if exclusions[0] in name ]
        if len(exclusions) > 1:
            for zz in exclusions[1:]:
                toexclude.extend([name for name in outnames if zz in name ])
        if len(toexclude) > 0:
            outnames = [name for name in outnames if name not in toexclude]
    return outnames


def average_mm_df( jmm_in, diagnostic_n=25, corr_thresh=0.9, verbose=False ):
    """
    jmrowavg, jmmcolavg, diagnostics = antspymm.average_mm_df( jmm_in, verbose=True )
    """

    jmm = jmm_in.copy()
    dxcols=['subjectid1','subjectid2','modalityid','joinid','correlation','distance']
    joinDiagnostics = pd.DataFrame( columns = dxcols )
    nanList=[math.nan]
    def rob(x, y=0.99):
        x[x > np.quantile(x, y, nan_policy="omit")] = np.nan
        return x

    jmm = jmm.replace(r'^\s*$', np.nan, regex=True)

    if verbose:
        print("do rsfMRI")
    # here - we first have to average within each row
    dt0 = get_names_from_data_frame(["rsfMRI"], jmm, exclusions=["Unnamed", "rsfMRI_LR", "rsfMRI_RL"])
    dt1 = get_names_from_data_frame(["rsfMRI_RL"], jmm, exclusions=["Unnamed"])
    if len( dt0 ) > 0 and len( dt1 ) > 0:
        flid = dt0[0]
        wrows = []
        for i in range(jmm.shape[0]):
            if not pd.isna(jmm[dt0[1]][i]) or not pd.isna(jmm[dt1[1]][i]) :
                wrows.append(i)
        for k in wrows:
            v1 = jmm.iloc[k][dt0[1:]].astype(float)
            v2 = jmm.iloc[k][dt1[1:]].astype(float)
            vvec = [v1[0], v2[0]]
            if any(~np.isnan(vvec)):
                mynna = [i for i, x in enumerate(vvec) if ~np.isnan(x)]
                jmm.iloc[k][dt0[0]] = 'rsfMRI'
                if len(mynna) == 1:
                    if mynna[0] == 0:
                        jmm.iloc[k][dt0[1:]] = v1
                    if mynna[0] == 1:
                        jmm.iloc[k][dt0[1:]] = v2
                elif len(mynna) > 1:
                    if len(v2) > diagnostic_n:
                        v1dx=v1[0:diagnostic_n]
                        v2dx=v2[0:diagnostic_n]
                    else :
                        v1dx=v1
                        v2dx=v2
                    joinDiagnosticsLoc = pd.DataFrame( columns = dxcols, index=range(1) )
                    mycorr = np.corrcoef( v1dx.values, v2dx.values )[0,1]
                    myerr=np.sqrt(np.mean((v1dx.values - v2dx.values)**2))
                    joinDiagnosticsLoc.iloc[0] = [jmm.loc[k,'u_hier_id'],math.nan,'rsfMRI','colavg',mycorr,myerr]
                    if mycorr > corr_thresh:
                        jmm.loc[k, dt0[1:]] = v1.values*0.5 + v2.values*0.5
                    else:
                        jmm.loc[k, dt0[1:]] = nanList * len(v1)
                    if verbose:
                        print( joinDiagnosticsLoc )
                    joinDiagnostics = pd.concat( [joinDiagnostics, joinDiagnosticsLoc], axis=0)

    if verbose:
        print("do DTI")
    # here - we first have to average within each row
    dt0 = get_names_from_data_frame(["DTI"], jmm, exclusions=["Unnamed", "DTI_LR", "DTI_RL"])
    dt1 = get_names_from_data_frame(["DTI_LR"], jmm, exclusions=["Unnamed"])
    dt2 = get_names_from_data_frame( ["DTI_RL"], jmm, exclusions=["Unnamed"])
    flid = dt0[0]
    wrows = []
    for i in range(jmm.shape[0]):
        if not pd.isna(jmm[dt0[1]][i]) or not pd.isna(jmm[dt1[1]][i]) or not pd.isna(jmm[dt2[1]][i]):
            wrows.append(i)
    for k in wrows:
        v1 = jmm.loc[k, dt0[1:]].astype(float)
        v2 = jmm.loc[k, dt1[1:]].astype(float)
        v3 = jmm.loc[k, dt2[1:]].astype(float)
        checkcol = dt0[5]
        if not np.isnan(v1[checkcol]):
            if v1[checkcol] < 0.25:
                v1.replace(np.nan, inplace=True)
        checkcol = dt1[5]
        if not np.isnan(v2[checkcol]):
            if v2[checkcol] < 0.25:
                v2.replace(np.nan, inplace=True)
        checkcol = dt2[5]
        if not np.isnan(v3[checkcol]):
            if v3[checkcol] < 0.25:
                v3.replace(np.nan, inplace=True)
        vvec = [v1[0], v2[0], v3[0]]
        if any(~np.isnan(vvec)):
            mynna = [i for i, x in enumerate(vvec) if ~np.isnan(x)]
            jmm.loc[k, dt0[0]] = 'DTI'
            if len(mynna) == 1:
                if mynna[0] == 0:
                    jmm.loc[k, dt0[1:]] = v1
                if mynna[0] == 1:
                    jmm.loc[k, dt0[1:]] = v2
                if mynna[0] == 2:
                    jmm.loc[k, dt0[1:]] = v3
            elif len(mynna) > 1:
                if mynna[0] == 0:
                    jmm.loc[k, dt0[1:]] = v1
                else:
                    joinDiagnosticsLoc = pd.DataFrame( columns = dxcols, index=range(1) )
                    mycorr = np.corrcoef( v2[0:diagnostic_n].values, v3[0:diagnostic_n].values )[0,1]
                    myerr=np.sqrt(np.mean((v2[0:diagnostic_n].values - v3[0:diagnostic_n].values)**2))
                    joinDiagnosticsLoc.iloc[0] = [jmm.loc[k,'u_hier_id'],math.nan,'DTI','colavg',mycorr,myerr]
                    if mycorr > corr_thresh:
                        jmm.loc[k, dt0[1:]] = v2.values*0.5 + v3.values*0.5
                    else: #
                        jmm.loc[k, dt0[1:]] = nanList * len( dt0[1:] )
                    if verbose:
                        print( joinDiagnosticsLoc )
                    joinDiagnostics = pd.concat( [joinDiagnostics, joinDiagnosticsLoc], axis=0)


    # first task - sort by u_hier_id
    jmm = jmm.sort_values( "u_hier_id" )
    # get rid of junk columns
    badnames = get_names_from_data_frame( ['Unnamed'], jmm )
    jmm=jmm.drop(badnames, axis=1)
    jmm=jmm.set_index("u_hier_id",drop=False)
    # 2nd - get rid of duplicated u_hier_id
    jmmUniq = jmm.drop_duplicates( subset="u_hier_id" ) # fast and easy
    # for each modality, count which ids have more than one
    mod_names = get_valid_modalities()
    for mod_name in mod_names:
        fl_names = get_names_from_data_frame([mod_name], jmm,
            exclusions=['Unnamed',"DTI_LR","DTI_RL","rsfMRI_RL","rsfMRI_LR"])
        if len( fl_names ) > 1:
            if verbose:
                print(mod_name)
                print(fl_names)
            fl_id = fl_names[0]
            n_names = len(fl_names)
            locvec = jmm[fl_names[n_names-1]].astype(float)
            boolvec=~pd.isna(locvec)
            jmmsub = jmm[boolvec][ ['u_hier_id']+fl_names]
            my_tbl = Counter(jmmsub['u_hier_id'])
            gtoavg = [name for name in my_tbl.keys() if my_tbl[name] == 1]
            gtoavgG1 = [name for name in my_tbl.keys() if my_tbl[name] > 1]
            if verbose:
                print("Join 1")
            jmmsub1 = jmmsub.loc[jmmsub['u_hier_id'].isin(gtoavg)][['u_hier_id']+fl_names]
            for u in gtoavg:
                jmmUniq.loc[u][fl_names[1:]] = jmmsub1.loc[u][fl_names[1:]]
            if verbose and len(gtoavgG1) > 1:
                print("Join >1")
            jmmsubG1 = jmmsub.loc[jmmsub['u_hier_id'].isin(gtoavgG1)][['u_hier_id']+fl_names]
            for u in gtoavgG1:
                temp = jmmsubG1.loc[u][ ['u_hier_id']+fl_names ]
                dropnames = get_names_from_data_frame( ['MM.ID'], temp )
                tempVec = temp.drop(columns=dropnames)
                joinDiagnosticsLoc = pd.DataFrame( columns = dxcols, index=range(1) )
                id1=temp[fl_id].iloc[0]
                id2=temp[fl_id].iloc[1]
                v1=tempVec.iloc[0][1:].astype(float).to_numpy()
                v2=tempVec.iloc[1][1:].astype(float).to_numpy()
                if len(v2) > diagnostic_n:
                    v1=v1[0:diagnostic_n]
                    v2=v2[0:diagnostic_n]
                mycorr = np.corrcoef( v1, v2 )[0,1]
                # mycorr=temparr[np.triu_indices_from(temparr, k=1)].mean()
                myerr=np.sqrt(np.mean((v1 - v2)**2))
                joinDiagnosticsLoc.iloc[0] = [id1,id2,mod_name,'rowavg',mycorr,myerr]
                if verbose:
                    print( joinDiagnosticsLoc )
                temp = jmmsubG1.loc[u][fl_names[1:]].astype(float)
                if mycorr > corr_thresh or len( v1 ) < 10:
                    jmmUniq.loc[u][fl_names[1:]] = temp.mean(axis=0)
                else:
                    jmmUniq.loc[u][fl_names[1:]] = nanList * temp.shape[1]
                joinDiagnostics = pd.concat( [joinDiagnostics, joinDiagnosticsLoc], axis=0)

    return jmmUniq, jmm, joinDiagnostics



def quick_viz_mm_nrg(
    sourcedir, # root folder
    projectid, # project name
    sid , # subject unique id
    dtid, # date
    extract_brain=True,
    slice_factor = 0.55,
    show_it = None, # output path
    verbose = True
):
    """
    This function creates visualizations of brain images for a specific subject in a project using ANTsPy.

    Args:

    sourcedir (str): Root folder.
    
    projectid (str): Project name.
    
    sid (str): Subject unique id.
    
    dtid (str): Date.
    
    extract_brain (bool): If True, the function extracts the brain from the T1w image. Default is True.
    
    slice_factor (float): The slice to be visualized is determined by multiplying the image size by this factor. Default is 0.55.
    
    show_it (str): Output path. If not None, the visualizations will be saved at this location. Default is None.
    
    verbose (bool): If True, information will be printed while running the function. Default is True.

    Returns:
    vizlist (list): List of image visualizations.

    """
    iid='*'
    import glob as glob
    from os.path import exists
    import ants
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    ex_pathmm = os.path.expanduser( "~/.antspymm/" )
    templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
    if not exists( templatefn ):
        print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
        antspyt1w.get_data( force_download=True )
        get_data( force_download=True )
    temp = sourcedir.split( "/" )
    splitCount = len( temp )
    template = mm_read( templatefn ) # Read in template
    subjectrootpath = os.path.join(sourcedir, projectid, sid, dtid)
    myimgsInput = glob.glob( subjectrootpath+"/*" )
    myimgsInput.sort( )
    if verbose:
        print( myimgsInput )
    t1_search_path = os.path.join(subjectrootpath, "T1w", "*", "*nii.gz")
    if verbose:
        print(f"t1 search path: {t1_search_path}")
    t1fn = glob.glob(t1_search_path)
    t1fn.sort()
    if len( t1fn ) < 1:
        raise ValueError('quick_viz_mm_nrg cannot find the T1w @ ' + subjectrootpath )
    t1fn = t1fn[0]
    t1 = mm_read( t1fn )
    nimages = len(myimgsInput)
    vizlist=[]
    if verbose:
        print(  " we have : " + str(nimages) + " modalities.  will visualize T1 NM rsfMRI DTIB0 DTIDWI FLAIR")
    # nrg_modality_list = ["T1w", "NM2DMT", "rsfMRI","rsfMRI_LR","rsfMRI_RL","DTI","DTI_LR", "T2Flair" ],
    nrg_modality_list = [ 'T1w', 'NM2DMT', 'rsfMRI', 'DWI1', 'DWI2', 'T2Flair' ]
    for nrgNum in [0,1,2,3,4,5]:
        overmodX = nrg_modality_list[nrgNum]
        if overmodX == 'T1w':
            mod_search_path = os.path.join(subjectrootpath, overmodX, iid, "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
            if len( myimgsr ) == 0:
                if verbose:
                    print("No t1 images: " + sid + dtid )
                return None
            myimgsr.sort()
            myimgsr=myimgsr[0]
            vimg=ants.image_read( myimgsr )
        elif overmodX == 'DWI1':
            mod_search_path = os.path.join(subjectrootpath, 'DTI*', "*", "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
            if len( myimgsr ) > 0:
                myimgsr.sort()
                myimgsr=myimgsr[0]
                vimg=ants.image_read( myimgsr )
            else:
                if verbose:
                    print("No " + overmodX)
                vimg = noizimg
        elif overmodX == 'DWI2':
            mod_search_path = os.path.join(subjectrootpath, 'DTI*', "*", "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
            if len( myimgsr ) > 0:
                myimgsr.sort()
                myimgsr=myimgsr[len(myimgsr)-1]
                vimg=ants.image_read( myimgsr )
            else:
                if verbose:
                    print("No " + overmodX)
                vimg = noizimg
        elif overmodX == 'NM2DMT':
            mod_search_path = os.path.join(subjectrootpath, overmodX, "*", "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
            if len( myimgsr ) > 0:
                myimgsr.sort()
                myimgsr0=myimgsr[0]
                vimg=ants.image_read( myimgsr0 )
                for k in range(1,len(myimgsr)):
                    temp = ants.image_read( myimgsr[k])
                    vimg=vimg+ants.resample_image_to_target(temp,vimg)
            else:
                if verbose:
                    print("No " + overmodX)
                vimg = noizimg
        elif overmodX == 'rsfMRI':
            mod_search_path = os.path.join(subjectrootpath, 'rsfMRI*', "*", "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
            if len( myimgsr ) > 0:
                myimgsr.sort()
                myimgsr=myimgsr[0]
                vimg=mm_read_to_3d( myimgsr )
            else:
                if verbose:
                    print("No " + overmodX)
                vimg = noizimg
        else :
            mod_search_path = os.path.join(subjectrootpath, overmodX, "*", "*nii.gz")
            myimgsr = glob.glob(mod_search_path)
            if len( myimgsr ) > 0:
                myimgsr.sort()
                myimgsr=myimgsr[0]
                vimg=ants.image_read( myimgsr )
            else:
                if verbose:
                    print("No " + overmodX)
                vimg = noizimg
        if True:
            if extract_brain and overmodX == 'T1w':
                vimg = vimg * antspyt1w.brain_extraction(vimg)
            if verbose:
                print(f"modality search path: {myimgsr}" + " num: " + str(nrgNum))
            if len( vimg.shape ) == 4 and ( overmodX == "DWI2"  ):
                ttb0, ttdw=get_average_dwi_b0(vimg)
                vimg = ttdw
            elif len( vimg.shape ) == 4 and overmodX == "DWI1":
                ttb0, ttdw=get_average_dwi_b0(vimg)
                vimg = ttb0
            elif len( vimg.shape ) == 4 :
                vimg=ants.get_average_of_timeseries(vimg)
            msk=ants.get_mask(vimg)
            vimg=ants.crop_image(vimg,msk)
            if overmodX == 'T1w':
                refimg=ants.image_clone( vimg )
                noizimg = ants.add_noise_to_image( refimg*0, 'additivegaussian', [100,1] )
                vizlist.append( vimg )
            else:
                vimg = ants.resample_image_to_target( vimg, refimg )
                vimg = ants.iMath( vimg, 'TruncateIntensity',0.01,0.98)
                vizlist.append( ants.iMath( vimg, 'Normalize' ) * 255 )

    listlen = len( vizlist )
    vizlist = np.asarray( vizlist )
    if show_it is not None:
        filenameout=None
        if verbose:
            print( show_it )
        for a in [0,1,2]:
            n=int(np.round( refimg.shape[a] * slice_factor ))
            slices=np.repeat( int(n), listlen  )
            if isinstance(show_it,str):
                filenameout=show_it+'_ax'+str(int(a))+'_sl'+str(n)+'.png'
                if verbose:
                    print( filenameout )
            ants.plot_grid(vizlist.reshape(2,3), slices.reshape(2,3), title='MM Subject ' + sid + ' ' + dtid, rfacecolor='white', axes=a, filename=filenameout )
    if verbose:
        print("viz complete.")
    return vizlist


def blind_image_assessment(
    image,
    viz_filename=None,
    title=False,
    pull_rank=False,
    resample='max',
    verbose=False
):
    """
    quick blind image assessment and triplanar visualization of an image ... 4D input will be visualized and assessed in 3D.  produces a png and csv where csv contains:

    * reflection error ( estimates asymmetry )

    * brisq ( blind quality assessment )

    * patch eigenvalue ratio ( blind quality assessment )

    * PSNR and SSIM vs a smoothed reference (4D or 3D appropriate)

    * mask volume ( estimates foreground object size )

    * spacing

    * dimension after cropping by mask

    image : character or image object usually a nifti image

    viz_filename : character for a png output image

    title : display a summary title on the png

    pull_rank : boolean

    resample : None, numeric max or min, resamples image to isotropy

    verbose : boolean

    """
    import glob as glob
    from os.path import exists
    import ants
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path
    import json
    import re
    mystem=''
    if isinstance(image,list):
        isfilename=isinstance( image[0], str)
        image = image[0]
    else:
        isfilename=isinstance( image, str)
    outdf = pd.DataFrame()
    mymeta = None
    image_filename=''
    if isfilename:
        image_filename = image
        if isinstance(image,list):
            image_filename=image[0]
        json_name = re.sub(".nii.gz",".json",image_filename)
        if exists( json_name ):
            with open(json_name, 'r') as fcc_file:
                mymeta = json.load(fcc_file)
                if verbose:
                    print(json.dumps(mymeta, indent=4))
        mystem=Path( image ).stem
        mystem=Path( mystem ).stem
        image_reference = ants.image_read( image )
        image = ants.image_read( image )
    else:
        image_reference = ants.image_clone( image )
    ntimepoints = 1
    if image_reference.dimension == 4:
        ntimepoints = image_reference.shape[3]
        if "DTI" in image_filename:
            myTSseg = segment_timeseries_by_meanvalue( image_reference )
            image_b0, image_dwi = get_average_dwi_b0( image_reference, fast=True )
            image_b0 = ants.iMath( image_b0, 'Normalize' )
            image_dwi = ants.iMath( image_dwi, 'Normalize' )
        else:
            image_b0 = ants.get_average_of_timeseries( image_reference ).iMath("Normalize")
    else:
        image_compare = ants.smooth_image( image_reference, 3, sigma_in_physical_coordinates=False )
    for jjj in range(ntimepoints):
        modality='unknown'
        if "rsfMRI" in image_filename:
            modality='rsfMRI'
        elif "T1w" in image_filename:
            modality='T1w'
        elif "T2Flair" in image_filename:
            modality='T2Flair'
        elif "NM2DMT" in image_filename:
            modality='NM2DMT'
        if image_reference.dimension == 4:
            image = ants.slice_image( image_reference, idx=int(jjj), axis=3 )
            if "DTI" in image_filename:
                if jjj in myTSseg['highermeans']:
                    image_compare = ants.image_clone( image_b0 )
                    modality='DTIb0'
                else:
                    image_compare = ants.image_clone( image_dwi )
                    modality='DTIdwi'
            else:
                image_compare = ants.image_clone( image_b0 )
        image = ants.iMath( image, 'TruncateIntensity',0.01,0.995)
        if resample is not None:
            if resample == 'min':
                newspc = np.repeat( np.min(ants.get_spacing(image)), 3 )
            elif resample == 'max':
                newspc = np.repeat( np.max(ants.get_spacing(image)), 3 )
            else:
                newspc = np.repeat( resample, 3 )
            image = ants.resample_image( image, newspc )
            image_compare = ants.resample_image( image_compare, newspc )
        # if "NM2DMT" in image_filename or "FIXME" in image_filename or "SPECT" in image_filename or "UNKNOWN" in image_filename:
        msk = ants.threshold_image( ants.iMath(image,'Normalize'), 0.15, 1.0 )
        # else:
        #    msk = ants.get_mask( image )
        msk = ants.morphology(msk, "close", 3 )
        bgmsk = msk*0+1-msk
        mskdil = ants.iMath(msk, "MD", 4 )
        # ants.plot_ortho( image, msk, crop=False )
        image = ants.crop_image( image, mskdil ).iMath("Normalize")
        msk = ants.crop_image( msk, mskdil ).iMath("Normalize")
        bgmsk = ants.crop_image( bgmsk, mskdil ).iMath("Normalize")
        image_compare = ants.crop_image( image_compare, mskdil ).iMath("Normalize")
        nvox = int( msk.sum() )
        minshp = np.min( image.shape )
        npatch = int( np.round(  0.1 * nvox ) )
        npatch = np.min(  [512,npatch ] )
        patch_shape = []
        for k in range( 3 ):
            p = int( 32.0 / ants.get_spacing( image  )[k] )
            if p > int( np.round( image.shape[k] * 0.5 ) ):
                p = int( np.round( image.shape[k] * 0.5 ) )
            patch_shape.append( p )
        if verbose:
            print(image)
            print( patch_shape )
            print( npatch )
        myevr = math.nan # dont want to fail if something odd happens in patch extraction
        try:
            myevr = antspyt1w.patch_eigenvalue_ratio( image, npatch, patch_shape,
                evdepth = 0.9, mask=msk )
        except:
            pass
        if pull_rank:
            image = ants.rank_intensity(image)
        imagereflect = ants.reflect_image(image, axis=0)
        asym_err = ( image - imagereflect ).abs().mean()
        # estimate noise by center cropping, denoizing and taking magnitude of difference
        mycc = antspyt1w.special_crop( image,
            ants.get_center_of_mass( msk *0 + 1 ), patch_shape )
        myccd = ants.denoise_image( mycc, p=2,r=2,noise_model='Gaussian' )
        noizlevel = ( mycc - myccd ).abs().mean()
#        ants.plot_ortho( image, crop=False, filename=viz_filename, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0 )
#        from brisque import BRISQUE
#        obj = BRISQUE(url=False)
#        mybrisq = obj.score( np.array( Image.open( viz_filename )) )
        spc = ants.get_spacing( image )
        msk_vol = msk.sum() * np.prod( spc )
        bgstd = image[ bgmsk == 1 ].std()
        fgmean = image[ msk == 1 ].mean()
        bgmean = image[ bgmsk == 1 ].mean()
        snrref = fgmean / bgstd
        cnrref = ( fgmean - bgmean ) / bgstd
        psnrref = antspynet.psnr(  image_compare, image  )
        ssimref = antspynet.ssim(  image_compare, image  )
        mymi = ants.image_mutual_information( image_compare, image )
        mriseries='NA'
        mrimfg='NA'
        mrimodel='NA'
        if mymeta is not None:
            # mriseries=mymeta['']
            try:
                mrimfg=mymeta['Manufacturer']
            except:
                pass
            try:
                mrimodel=mymeta['ManufacturersModelName']
            except:
                pass
        ttl=mystem + ' '
        ttl=''
        ttl=ttl + "NZ: " + "{:0.4f}".format(noizlevel) + " SNR: " + "{:0.4f}".format(snrref) + " CNR: " + "{:0.4f}".format(cnrref) + " PS: " + "{:0.4f}".format(psnrref)+ " SS: " + "{:0.4f}".format(ssimref) + " EVR: " + "{:0.4f}".format(myevr)+ " MI: " + "{:0.4f}".format(mymi)
        if viz_filename is not None and ( jjj == 0 or (jjj % 30 == 0) ):
            viz_filename_use = re.sub( ".png", "_slice"+str(jjj).zfill(4)+".png", viz_filename )
            ants.plot_ortho( image, crop=False, filename=viz_filename_use, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0,  title=ttl, titlefontsize=12, title_dy=-0.02,textfontcolor='red' )
        df = pd.DataFrame([[ mystem, noizlevel, snrref, cnrref, psnrref, ssimref, mymi, asym_err, myevr, msk_vol, spc[0], spc[1], spc[2], image.shape[0], image.shape[1], image.shape[2], jjj, modality, mriseries, mrimfg, mrimodel ]], columns=['fn', 'noise', 'snr', 'cnr', 'psnr', 'ssim', 'mi', 'reflection_err', 'EVR', 'msk_vol', 'spc0','spc1','spc2','dimx','dimy','dimz','slice','modality', 'mriseries', 'mrimfg', 'mrimodel' ])
        outdf = pd.concat( [outdf, df ], axis=0 )
        if verbose:
            print( outdf )
    if viz_filename is not None:
        csvfn = re.sub( "png", "csv", viz_filename )
        outdf.to_csv( csvfn )
    return outdf


def average_blind_qc_by_modality(qc_full,verbose=False):
    """
    Averages time series qc results to yield one entry per image. this also filters to "known" columns.

    Args:
    qc_full: pandas dataframe containing the full qc data.

    Returns:
    pandas dataframe containing the processed qc data.
    """
    # Get unique modalities
    modalities = qc_full['modality'].unique()
    modalities = modalities[modalities != 'unknown']
    # Get modalities to select
    m0sel = qc_full['modality'].isin(modalities)
    # Get unique ids
    uid = qc_full['fn'] + "_" + qc_full['modality'].astype(str)
    to_average = uid.unique()
    # Define column indices
    contcols = ['noise', 'snr', 'cnr', 'psnr', 'ssim', 'mi',
       'reflection_err', 'EVR', 'msk_vol', 'spc0', 'spc1', 'spc2', 'dimx',
       'dimy', 'dimz', 'slice']
    ocols = ['fn','modality', 'mriseries', 'mrimfg', 'mrimodel']
    # restrict to columns we "know"
    qc_full = qc_full[ocols+contcols]
    # Create empty meta dataframe
    meta = pd.DataFrame(columns=ocols+contcols)
    # Process each unique id
    n = len(to_average)
    for k in range(n):
        if verbose:
            if k % 100 == 0:
                progger = str( np.round( k / n * 100 ) )
                print( progger, end ="...", flush=True)
        m1sel = uid == to_average[k]
        if sum(m1sel) > 1:
            # If more than one entry for id, take the average of continuous columns,
            # maximum of the slice column, and the first entry of the other columns
            mfsub = qc_full[m1sel]
            if mfsub.shape[0] > 1:
                meta.loc[k, contcols] = mfsub.loc[:, contcols].mean(numeric_only=True)
                meta.loc[k, 'slice'] = mfsub['slice'].max()
                meta.loc[k, ocols] = mfsub[ocols].iloc[0]
        elif sum(m1sel) == 1:
            # If only one entry for id, just copy the entry
            mfsub = qc_full[m1sel]
            meta.loc[k] = mfsub.iloc[0]
    return meta

def wmh( flair, t1, t1seg,
    mmfromconvexhull = 3.0,
    strict=True,
    probability_mask=None,
    prior_probability=None,
    model='sysu',
    verbose=False ) :
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

    """
    import numpy as np
    import math
    t1_2_flair_reg = ants.registration(flair, t1, type_of_transform = 'Rigid') # Register T1 to Flair
    if probability_mask is None and model == 'sysu':
        if verbose:
            print('sysu')
        probability_mask = antspynet.sysu_media_wmh_segmentation( flair )
    elif probability_mask is None and model == 'hyper':
        if verbose:
            print('hyper')
        probability_mask = antspynet.hypermapp3r_segmentation( t1_2_flair_reg['warpedmovout'], flair )
    # t1_2_flair_reg = tra_initializer( flair, t1, n_simulations=4, max_rotation=5, transform=['rigid'], verbose=False )
    prior_probability_flair = None
    if prior_probability is not None:
        prior_probability_flair = ants.apply_transforms( flair, prior_probability,
            t1_2_flair_reg['fwdtransforms'] )
    wmseg_mask = ants.threshold_image( t1seg,
        low_thresh = 3, high_thresh = 3).iMath("FillHoles")
    wmseg_mask_use = ants.image_clone( wmseg_mask )
    distmask = None
    if mmfromconvexhull > 0:
            convexhull = ants.threshold_image( t1seg, 1, 4 )
            spc2vox = np.prod( ants.get_spacing( t1seg ) )
            voxdist = 0.0
            myspc = ants.get_spacing( t1seg )
            for k in range( t1seg.dimension ):
                voxdist = voxdist + myspc[k] * myspc[k]
            voxdist = math.sqrt( voxdist )
            nmorph = round( 2.0 / voxdist )
            convexhull = ants.morphology( convexhull, "close", nmorph ).iMath("FillHoles")
            dist = ants.iMath( convexhull, "MaurerDistance" ) * -1.0
            distmask = ants.threshold_image( dist, mmfromconvexhull, 1.e80 )
            wmseg_mask = wmseg_mask + distmask
            if strict:
                wmseg_mask_use = ants.threshold_image( wmseg_mask, 2, 2 )
            else:
                wmseg_mask_use = ants.threshold_image( wmseg_mask, 1, 2 )
    ##############################################################################
    wmseg_2_flair = ants.apply_transforms(flair, wmseg_mask_use,
        transformlist = t1_2_flair_reg['fwdtransforms'],
        interpolator = 'nearestNeighbor' )
    seg_2_flair = ants.apply_transforms(flair, t1seg,
        transformlist = t1_2_flair_reg['fwdtransforms'],
        interpolator = 'nearestNeighbor' )
    csfmask = ants.threshold_image(seg_2_flair,1,1)
    flairsnr = mask_snr( flair, csfmask, wmseg_2_flair, bias_correct = False )
    probability_mask_WM = wmseg_2_flair * probability_mask # Remove WMH signal outside of WM
    wmh_sum = np.prod( ants.get_spacing( flair ) ) * probability_mask_WM.sum()
    wmh_sum_prior = math.nan
    probability_mask_posterior = None
    if prior_probability_flair is not None:
        probability_mask_posterior = prior_probability_flair * probability_mask # use prior
        wmh_sum_prior = np.prod( ants.get_spacing(flair) ) * probability_mask_posterior.sum()
    if math.isnan( wmh_sum ):
        wmh_sum=0
    if math.isnan( wmh_sum_prior ):
        wmh_sum_prior=0
    flair_evr = antspyt1w.patch_eigenvalue_ratio( flair, 512, [16,16,16], evdepth = 0.9, mask=wmseg_2_flair )
    return{
        'WMH_probability_map_raw': probability_mask,
        'WMH_probability_map' : probability_mask_WM,
        'WMH_posterior_probability_map' : probability_mask_posterior,
        'wmh_mass': wmh_sum,
        'wmh_mass_prior': wmh_sum_prior,
        'wmh_evr' : flair_evr,
        'wmh_SNR' : flairsnr,
        'convexhull_mask': distmask }



def novelty_detection_ee(df_train, df_test, contamination=0.05):
    """
    This function performs novelty detection using Elliptic Envelope.

    Parameters:

    - df_train (pandas dataframe): training data used to fit the model

    - df_test (pandas dataframe): test data used to predict novelties

    - contamination (float): parameter controlling the proportion of outliers in the data (default: 0.05)

    Returns:

    predictions (pandas series): predicted labels for the test data (1 for novelties, 0 for inliers)
    """
    import pandas as pd
    from sklearn.covariance import EllipticEnvelope
    # Fit the model on the training data
    clf = EllipticEnvelope(contamination=contamination,support_fraction=1)
    df_train[ df_train == math.inf ] = 0
    df_test[ df_test == math.inf ] = 0
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train)
    clf.fit(scaler.transform(df_train))
    predictions = clf.predict(scaler.transform(df_test))
    predictions[predictions==1]=0
    predictions[predictions==-1]=1
    if str(type(df_train))=="<class 'pandas.core.frame.DataFrame'>":
        return pd.Series(predictions, index=df_test.index)
    else:
        return pd.Series(predictions)



def novelty_detection_svm(df_train, df_test, nu=0.05, kernel='rbf'):
    """
    This function performs novelty detection using One-Class SVM.

    Parameters:

    - df_train (pandas dataframe): training data used to fit the model

    - df_test (pandas dataframe): test data used to predict novelties

    - nu (float): parameter controlling the fraction of training errors and the fraction of support vectors (default: 0.05)

    - kernel (str): kernel type used in the SVM algorithm (default: 'rbf')

    Returns:

    predictions (pandas series): predicted labels for the test data (1 for novelties, 0 for inliers)
    """
    from sklearn.svm import OneClassSVM
    # Fit the model on the training data
    df_train[ df_train == math.inf ] = 0
    df_test[ df_test == math.inf ] = 0
    clf = OneClassSVM(nu=nu, kernel=kernel)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train)
    clf.fit(scaler.transform(df_train))
    predictions = clf.predict(scaler.transform(df_test))
    predictions[predictions==1]=0
    predictions[predictions==-1]=1
    if str(type(df_train))=="<class 'pandas.core.frame.DataFrame'>":
        return pd.Series(predictions, index=df_test.index)
    else:
        return pd.Series(predictions)



def novelty_detection_lof(df_train, df_test, n_neighbors=20):
    """
    This function performs novelty detection using Local Outlier Factor (LOF).

    Parameters:

    - df_train (pandas dataframe): training data used to fit the model

    - df_test (pandas dataframe): test data used to predict novelties

    - n_neighbors (int): number of neighbors used to compute the LOF (default: 20)

    Returns:

    - predictions (pandas series): predicted labels for the test data (1 for novelties, 0 for inliers)

    """
    from sklearn.neighbors import LocalOutlierFactor
    # Fit the model on the training data
    df_train[ df_train == math.inf ] = 0
    df_test[ df_test == math.inf ] = 0
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm='auto',contamination='auto', novelty=True)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train)
    clf.fit(scaler.transform(df_train))
    predictions = clf.predict(scaler.transform(df_test))
    predictions[predictions==1]=0
    predictions[predictions==-1]=1
    if str(type(df_train))=="<class 'pandas.core.frame.DataFrame'>":
        return pd.Series(predictions, index=df_test.index)
    else:
        return pd.Series(predictions)


def novelty_detection_loop(df_train, df_test, n_neighbors=20, distance_metric='minkowski'):
    """
    This function performs novelty detection using Local Outlier Factor (LOF).

    Parameters:

    - df_train (pandas dataframe): training data used to fit the model

    - df_test (pandas dataframe): test data used to predict novelties

    - n_neighbors (int): number of neighbors used to compute the LOOP (default: 20)

    - distance_metric : default minkowski

    Returns:

    - predictions (pandas series): predicted labels for the test data (1 for novelties, 0 for inliers)

    """
    from PyNomaly import loop
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train)
    data = np.vstack( [scaler.transform(df_test),scaler.transform(df_train)])
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric=distance_metric)
    neigh.fit(data)
    d, idx = neigh.kneighbors(data, return_distance=True)
    m = loop.LocalOutlierProbability(distance_matrix=d, neighbor_matrix=idx, n_neighbors=n_neighbors).fit()
    return m.local_outlier_probabilities[range(df_test.shape[0])]



def novelty_detection_quantile(df_train, df_test):
    """
    This function performs novelty detection using quantiles for each column.

    Parameters:

    - df_train (pandas dataframe): training data used to fit the model

    - df_test (pandas dataframe): test data used to predict novelties

    Returns:

    - quantiles for the test sample at each column where values range in [0,1]
        and higher values mean the column is closer to the edge of the distribution

    """
    myqs = df_test.copy()
    n = df_train.shape[0]
    df_trainkeys = df_train.keys()
    for k in range( df_train.shape[1] ):
        mykey = df_trainkeys[k]
        temp = (myqs[mykey][0] >  df_train[mykey]).sum() / n
        myqs[mykey] = abs( temp - 0.5 ) / 0.5
    return myqs
