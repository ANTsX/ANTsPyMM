# ANTsPyMM

## processing utilities for timeseries/multichannel images - mostly neuroimaging

the outputs of these processes can be used for data inspection/cleaning/triage
as well for interrogating hypotheses.

this package also keeps track of the latest preferred algorithm variations for
production environments.

install the `dev` version by calling (within the source directory):

```
python setup.py install
```

or install the latest release via 

```
pip install antspymm
```

# what this will do

ANTsPyMM will process several types of brain MRI into tabular form as well as normalized (standard template) space.  The processing includes:

* T1wHier uses hierarchical processing from ANTsPyT1w organized around these measurements

    * CIT168 template 10.1101/211201

    * Desikan Killiany Tourville (DKT) 10.3389/fnins.2012.00171

    * basal forebrain (Avants et al HBM 2022 abstract)

    * other regions (egMTL) 10.1101/2023.01.17.23284693

    * also produces jacobian data

* rsfMRI: resting state functional MRI

    * uses 10.1016/j.conb.2012.12.009 to estimate network specific correlations

    * f/ALFF 10.1016/j.jneumeth.2008.04.012

* NM2DMT: neuromelanin mid-brain images

    * CIT168 template 10.1101/211201

* DTI: DWI diffusion weighted images organized via:

    * CIT168 template 10.1101/211201

    * JHU atlases 10.1016/j.neuroimage.2008.07.009  10.1016/j.neuroimage.2007.07.053

    * DKT for cortical to cortical tractography estimates based on DiPy

* T2Flair: flair for white matter hyperintensity

    * https://pubmed.ncbi.nlm.nih.gov/30908194/

    * https://pubmed.ncbi.nlm.nih.gov/30125711/

    * https://pubmed.ncbi.nlm.nih.gov/35088930/

* T1w: voxel-based cortical thickness (DiReCT) 10.1016/j.neuroimage.2008.12.016

Results of these processes are plentiful; processing for a single subject
will all modalities will take around 2 hours on an average laptop.

documentation of functions [here](http://htmlpreview.github.io/?https://github.com/stnava/ANTsPyMM/blob/main/docs/antspymm/mm.html).

# first time setup

```python
import antspyt1w
import antspymm
antspyt1w.get_data(force_download=True)
antspymm.get_data(force_download=True)
```

NOTE: `get_data` has a `force_download` option to make sure the latest
package data is installed.

NOTE: some functions in `antspynet` will download deep network model weights on the fly.  if one is containerizing, then it would be worth running a test case through in the container to make sure all the relevant weights are pre-downloaded.

# example processing

see the latest help but this snippet gives an idea of how one might use the package:

```python
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import antspymm
import antspyt1w
import antspynet
import ants

... i/o code here ...

tabPro, normPro = antspymm.mm(
    t1,
    hier,
    nm_image_list = mynm,
    rsf_image = rsf,
    dw_image = dwi,
    bvals = bval_fname,
    bvecs = bvec_fname,
    flair_image = flair,
    do_tractography=False,
    do_kk=False,
    do_normalization=True,
    verbose=True )

antspymm.write_mm( '/tmp/test_output', t1wide, tabPro, normPro )

```

## blind quality control

this package also provides tools to identify the *best* multi-modality image set at a given visit.

the code below provides guidance on how to automatically qc, filter and match multiple modality images at each time point.  these tools are based on standard unsupervised approaches and are not perfect so we recommend using the associated plotting/visualization techniques to check the quality characterizations for each modality.

```python
## run the qc on all images - requires a relatively large sample per modality to be effective
## then aggregate
qcdf=pd.DataFrame()
for fn in fns:
  qcdf=pd.concat( [qcdf,antspymm.blind_image_assessment(fn)], axis=0)
qcdfa=antspymm.average_blind_qc_by_modality(qcdf,verbose=True) ## reduce the time series qc
qcdfaol=antspymm.outlierness_by_modality(qcdfa) # estimate outlier scores
print( qcdfaol.shape )
print( qcdfaol.keys )
matched_mm_data=antspymm.match_modalities( qcdfaol  )
```

or just get modality-specific outlierness "by hand" then match `mm`:

```python
import antspymm
import pandas as pd
mymods = antspymm.get_valid_modalities( )
alldf = pd.DataFrame()
for n in range(len(mymods)):
    m=mymods[n]
    jj=antspymm.collect_blind_qc_by_modality("qc/*"+m+"*csv")
    jjj=antspymm.average_blind_qc_by_modality(jj,verbose=False).dropna(axis=1) ## reduce the time series qc
    jjj=antspymm.outlierness_by_modality( jjj, verbose=False)
    alldf = pd.concat( [alldf, jjj ], axis=0 )
    jjj.to_csv( "mm_outlierness_"+m+".csv")
    print(m+" done")
# write the joined data out
alldf.to_csv( "mm_outlierness.csv", index=False )
# find the best mm collection
matched_mm_data=antspymm.match_modalities( alldf, verbose=True )
matched_mm_data.to_csv( "matched_mm_data.csv", index=False )
matched_mm_data['negative_outlier_factor'] = 1.0 - matched_mm_data['ol_loop'].astype("float")
matched_mm_data2 = antspymm.highest_quality_repeat( matched_mm_data, 'subjectID', 'date', qualityvar='negative_outlier_factor')
matched_mm_data2.to_csv( "matched_mm_data2.csv", index=False )
```

## an example on open neuro (BIDS) data

from : [ANT PD](https://openneuro.org/datasets/ds001907/versions/3.0.2)

```
imagesBIDS/
└── ANTPD
    └── sub-RC4125
        └── ses-1
            ├── anat
            │   ├── sub-RC4125_ses-1_T1w.json
            │   └── sub-RC4125_ses-1_T1w.nii.gz
            ├── dwi
            │   ├── sub-RC4125_ses-1_dwi.bval
            │   ├── sub-RC4125_ses-1_dwi.bvec
            │   ├── sub-RC4125_ses-1_dwi.json
            │   └── sub-RC4125_ses-1_dwi.nii.gz
            └── func
                ├── sub-RC4125_ses-1_task-ANT_run-1_bold.json
                ├── sub-RC4125_ses-1_task-ANT_run-1_bold.nii.gz
                └── sub-RC4125_ses-1_task-ANT_run-1_events.tsv
```

```python
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
```

## NRG example

NRG format details [here](https://htmlpreview.github.io/?https://github.com/stnava/biomedicalDataOrganization/blob/master/src/nrg_data_organization_summary.html)

```
imagesNRG/
└── ANTPD
    └── sub-RC4125
        └── ses-1
            ├── DTI
            │   └── 000
            │       ├── ANTPD_sub-RC4125_ses-1_DTI_000.bval
            │       ├── ANTPD_sub-RC4125_ses-1_DTI_000.bvec
            │       ├── ANTPD_sub-RC4125_ses-1_DTI_000.json
            │       └── ANTPD_sub-RC4125_ses-1_DTI_000.nii.gz
            ├── T1w
            │   └── 000
            │       └── ANTPD_sub-RC4125_ses-1_T1w_000.nii.gz
            └── rsfMRI
                └── 000
                    └── ANTPD_sub-RC4125_ses-1_rsfMRI_000.nii.gz
```



```python
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
```

## useful tools for converting dicom to nifti

* [dcm2niix](https://github.com/rordenlab/dcm2niix)

* [dicom2nifti](https://dicom2nifti.readthedocs.io/en/latest/)

```python
import dicom2nifti

dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)
```

* [simpleitk](https://pypi.org/project/SimpleITK/)

```python
import SimpleITK as sitk
import sys
import os
import glob as glob
import ants
dd='dicom'
oo='dicom2nifti'
folders=glob.glob('dicom/*')
k=0
for f in folders:
    print(f)    
    reader = sitk.ImageSeriesReader()
    ff=glob.glob(f+"/*")
    dicom_names = reader.GetGDCMSeriesFileNames(ff[0])
    if len(ff) > 0:
        fnout='dicom2nifti/image_'+str(k).zfill(4)+'.nii.gz'
        if not exists(fnout):
            failed=False
            reader.SetFileNames(dicom_names)
            try:
                image = reader.Execute()
            except:
                failed=True
                pass
            if not failed:
                size = image.GetSpacing()
                print( image.GetMetaDataKeys( ) )
                print( size )
                sitk.WriteImage(image, fnout )
                img=ants.image_read( fnout )
                img=ants.iMath(img,'TruncateIntensity',0.02,0.98)
                ants.plot( img, nslices=21,ncol=7,axis=2, crop=True )
        else:
            print(f+ ": "+'empty')
    k=k+1
```

## build docs

```
pdoc -o ./docs antspymm --html
```

## to publish a release

```
rm -r -f build/ antspymm.egg-info/ dist/
python3 setup.py sdist bdist_wheel
python3 -m twine upload -u username -p password  dist/*
```
