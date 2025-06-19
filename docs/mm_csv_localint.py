##################################################################
# for easier to access data with a full mm_csv example, see:
# github.com:stnava/ANTPD_antspymm
##################################################################
import os
from os.path import exists
nthreads = str(8)
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import numpy as np
import glob as glob
import antspymm
import ants
import random
import re
rdir = os.path.expanduser( "~/Downloads/temp/shortrun/nrgdata/" )
mydir = rdir + "PPMI/"
outdir = re.sub( 'nrgdata', 'antspymmoutput', rdir )
################
import antspymm
import pandas as pd
import glob as glob
t1fn=glob.glob(mydir+"101018/20210412/T1w/1496225/*.nii.gz")
if len(t1fn) > 0:
    t1fn=t1fn[0]
    print("Begin " + t1fn)
    flfn=glob.glob(mydir+"101018/20210412/T2Flair/*/*.nii.gz")[0]
    dtfn=glob.glob(mydir+"101018/20210412/DTI*/*/*.nii.gz")
    rsfn=glob.glob(mydir+"101018/20210412/rsfMRI*/*/*.nii.gz")
    nmfn=glob.glob(mydir+"101018/20210412/NM2DMT/*/*.nii.gz")
    studycsv = antspymm.generate_mm_dataframe( 
        projectID='PPMI',
        subjectID='101018', 
        date='20210412', 
        imageUniqueID='1496225', 
        modality='T1w', 
        source_image_directory=rdir, 
        output_image_directory=outdir, 
        t1_filename = t1fn,
        flair_filename=flfn,
        rsf_filenames=rsfn,
        dti_filenames=dtfn,
        nm_filenames=nmfn
    )
    studycsv2 = studycsv.dropna(axis=1)
    print( studycsv2 )
    mmrun = antspymm.mm_csv( studycsv2  )
else:
    print("T1w data is missing: see github.com:stnava/ANTPD_antspymm for a full integration study and container with more easily accessible data")