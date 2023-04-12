import antspymm
import pandas as pd
import glob as glob
t1fn=glob.glob("imagesNRG/ANTPD/sub-RC4125/ses-*/*/*/*T1w*gz")[0]
# flair also takes a single image
dtfn=glob.glob("imagesNRG/ANTPD/sub-RC4125/ses-*/*/*/*DTI*gz")
rsfn=glob.glob("imagesNRG/ANTPD/sub-RC4125/ses-*/*/*/*rsfMRI*gz")
studycsv = antspymm.generate_mm_dataframe(
    'ANTPD',
    'sub-RC4125',
    'ses-1',
    '000',
    'T1w',
    '/Users/stnava/data/openneuro/imagesNRG/',
    '/Users/stnava/data/openneuro/processed/',
    t1fn,
    rsf_filenames=rsfn,
    dti_filenames=dtfn
)
studycsv2 = studycsv.dropna(axis=1)
mmrun = antspymm.mm_csv( studycsv2, mysep='_' )

