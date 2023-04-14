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
if not exists(t1fn):
    print("t1 does not exist")
else:
    t1 = ants.image_read( t1fn )
    t1fno=antspymm.nrg_format_path(projectID, subjectID, date, 'T1w', imageID+'0', separator='-')
    subjectpropath = os.path.join( "nrg", t1fno )
    os.makedirs( os.path.dirname(subjectpropath), exist_ok=True  )
    nrgt1fn=subjectpropath+ext
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
    ants.image_write( t1, nrgrsfn )
dfn=glob.glob("original/dMRI/raw/*gz")
if not exists(dfn[0]):
    print("dti does not exist")
else:
    t1 = ants.image_read( dfn[0] )
    t1fno=antspymm.nrg_format_path(projectID, subjectID, date, 'DTI_LR', imageID+str(3), separator='-')
    subjectpropath = os.path.join( "nrg", t1fno )
    os.makedirs( os.path.dirname(subjectpropath), exist_ok=True  )
    nrgdtfn0 = subjectpropath+ext
    ants.image_write( t1, nrgdtfn0 )
    t1 = ants.image_read( dfn[1] )
    t1fno=antspymm.nrg_format_path(projectID, subjectID, date, 'DTI_RL', imageID+str(4), separator='-')
    subjectpropath = os.path.join( "nrg", t1fno )
    os.makedirs( os.path.dirname(subjectpropath), exist_ok=True  )
    nrgdtfn1 = subjectpropath+ext
    ants.image_write( t1, nrgdtfn1 )
    bvals, bvecs = read_bvals_bvecs(
        re.sub("nii.gz","bval",dfn[0]), re.sub("nii.gz","bvec",dfn[0]))
    antspymm.write_bvals_bvecs( bvals, bvecs, re.sub(".nii.gz","",nrgdtfn0) )
    bvals, bvecs = read_bvals_bvecs(
        re.sub("nii.gz","bval",dfn[1]), re.sub("nii.gz","bvec",dfn[1]))
    antspymm.write_bvals_bvecs( bvals, bvecs, re.sub(".nii.gz","",nrgdtfn1) )
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
    dti_filenames=[nrgdtfn0,nrgdtfn1],
    rsf_filenames=[nrgrsfn])
studycsv2 = studycsv.dropna(axis=1)
derka
mmrun = antspymm.mm_csv( studycsv2, mysep='_' )


