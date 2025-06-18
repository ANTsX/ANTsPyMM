##################################################################
# this assumes NRG format for the input data ....
# NRG = https://github.com/stnava/biomedicalDataOrganization
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
mydir = os.path.expanduser( "./example/nrgdata/PPMI/" )

import antspymm
import pandas as pd
import glob as glob
t1fn=glob.glob(mydir+"101018/20210412/T1w/1496225/*.nii.gz")[0]
flfn=glob.glob(mydir+"101018/20210412/T2Flair/*/*.nii.gz")[0]
dtfn=glob.glob(mydir+"101018/20210412/DTI*/*/*.nii.gz")
rsfn=glob.glob(mydir+"101018/20210412/rsfMRI*/*/*.nii.gz")
nmfn=glob.glob(mydir+"101018/20210412/NM2DMT/*/*.nii.gz")
studycsv = antspymm.generate_mm_dataframe( 
    subjectID='101018', 
    date='20210412', 
    imageUniqueID='1496225', 
    modality='T1w', 
    source_image_directory="./example/nrgdata/", 
    output_image_directory="./example/processedX/", 
    t1_filename = t1fn,
    flair_filename=flfn,
    rsf_filenames=rsfn,
    dti_filenames=dtfn,
    nm_filenames=nmfn
)
studycsv2 = studycsv.dropna(axis=1)
print( studycsv2 )
mmrun = antspymm.mm_csv( studycsv2  )
