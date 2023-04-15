import ants
import antspymm
import pandas as pd
import glob as glob
import os
from os.path import exists
import re
from dipy.io.gradients import read_bvals_bvecs
projectID='UKBB'
subjectID='example'
date='20220101'
imageID='zzz' # randomly assigned
rdir='./original/'
t1fn=rdir+"T1/T1.nii.gz"
ext='.nii.gz'
validmods = antspymm.get_valid_modalities()
symlink=False
if not exists(t1fn):
    print("t1 does not exist")
else:
    t1 = ants.image_read( t1fn )
    t1fno=antspymm.nrg_format_path(projectID, subjectID, date, 'T1w', imageID+'0', separator='-')
    subjectpropath = os.path.join( "nrg", t1fno )
    os.makedirs( os.path.dirname(subjectpropath), exist_ok=True  )
    nrgt1fn=subjectpropath+ext
    if symlink:
        os.symlink( t1fn, nrgt1fn )
    else: 
        ants.image_write( t1, nrgt1fn )
t2fn=rdir+"T2_FLAIR/T2_FLAIR.nii.gz"
if not exists(t2fn):
    print("t2 does not exist")
else:
    t1 = ants.image_read( t2fn )
    t1fno=antspymm.nrg_format_path(projectID, subjectID, date, 'T2Flair', imageID+'1', separator='-')
    subjectpropath = os.path.join( "nrg", t1fno )
    os.makedirs( os.path.dirname(subjectpropath), exist_ok=True  )
    nrgt2fn=subjectpropath+ext
    if symlink:
        os.symlink( t2fn, nrgt2fn )
    else: 
        ants.image_write( t1, nrgt2fn )
rfn=rdir+"fMRI/rfMRI.nii.gz"
if not exists(rfn):
    print("rsfmri does not exist")
else:
    t1 = ants.image_read( rfn )
    t1fno=antspymm.nrg_format_path(projectID, subjectID, date, 'rsfMRI', imageID+'2', separator='-')
    subjectpropath = os.path.join( "nrg", t1fno )
    os.makedirs( os.path.dirname(subjectpropath), exist_ok=True  )
    nrgrsfn = subjectpropath+ext
    if symlink:
        os.symlink( rfn, nrgrsfn )
    else: 
        ants.image_write( t1, nrgrsfn )
dfn="original/dMRI/dMRI/data_ud.nii.gz"
bvalfn="original/dMRI/dMRI/bvals"
bvecfn="original/dMRI/dMRI/data.eddy_rotated_bvecs"
if not exists(dfn):
    print("dti does not exist")
else:
    t1 = ants.image_read( dfn )
    t1fno=antspymm.nrg_format_path(projectID, subjectID, date, 'DTI', imageID+str(3), separator='-')
    subjectpropath = os.path.join( "nrg", t1fno )
    os.makedirs( os.path.dirname(subjectpropath), exist_ok=True  )
    nrgdtfn0 = subjectpropath+ext
    if symlink:
        os.symlink( dfn, nrgdtfn0 )
    else:
        ants.image_write( t1, nrgdtfn0 )
    bvals, bvecs = read_bvals_bvecs( bvalfn, bvecfn )
    antspymm.write_bvals_bvecs( bvals, bvecs, re.sub(".nii.gz","",nrgdtfn0) )
studycsv = antspymm.generate_mm_dataframe(
    projectID,
    subjectID,
    date,
    imageID+'0', # the T1 id
    'T1w',
    './nrg/',
    './processed/',
    t1_filename=nrgt1fn,
    flair_filename=[nrgt2fn],
#    dti_filenames=[nrgdtfn0,nrgdtfn1],
    dti_filenames=[nrgdtfn0],
    rsf_filenames=[nrgrsfn])
studycsv2 = studycsv.dropna(axis=1)

antspymm.quick_viz_mm_nrg(
    './nrg/', # root for source data
    projectID,
    subjectID , # subject unique id
    date, # date
    extract_brain=True,
    slice_factor = 0.55,
    show_it = "viz_it",
    verbose = True )

mmrun = antspymm.mm_csv( studycsv2, dti_motion_correct=None, dti_denoise=False )

