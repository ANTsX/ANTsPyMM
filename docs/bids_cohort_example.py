
import antspymm
import pandas as pd
import glob as glob
fns = glob.glob("imagesBIDS/ANTPD/sub-RC4125/ses-*/*/*gz")
fns.sort()
randid='000' # BIDS does not have unique image ids - so we assign one
studycsv = antspymm.generate_mm_dataframe(
    'ANTPD',
    'sub-RC4125',
    'ses-1',
    randid,
    'T1w',
    '/Users/stnava/data/openneuro/imagesBIDS/',
    '/Users/stnava/data/openneuro/processed/',
    t1_filename=fns[0],
    dti_filenames=[fns[1]],
    rsf_filenames=[fns[2]])
studycsv2 = studycsv.dropna(axis=1)
mmrun = antspymm.mm_csv( studycsv2, mysep='_' )

