# ANTsPyMM

![mapping](https://i.imgur.com/qKqYjU9.jpeg)

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

    * other regions (eg MTL) 10.1101/2023.01.17.23284693

    * also produces jacobian data

* rsfMRI: resting state functional MRI

    * uses [a recent homotopic parcellation](https://pubmed.ncbi.nlm.nih.gov/36918136/) to estimate network specific correlations

    * f/ALFF 10.1016/j.jneumeth.2008.04.012

    * [percent absolute fluctuation](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.888174/full)

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

* documentation of functions [here](http://htmlpreview.github.io/?https://github.com/stnava/ANTsPyMM/blob/main/docs/antspymm/mm.html).

* data dictionary [here](http://htmlpreview.github.io/?https://github.com/stnava/ANTsPyMM/blob/main/docs/make_dict_table.html).

* notes on QC [here](http://htmlpreview.github.io/?https://github.com/stnava/ANTsPyMM/blob/main/docs/blind_qc.html)


achieved through four steps (recommended approach):

1. organize data in NRG format

2. perform blind QC

3. compute outlierness per modality and select optimally matched modalities ( steps 3.1 and 3.2 )

4. run the main antspymm function


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

NOTE: an example process for BIDS data on a cluster is [here](https://github.com/stnava/ANTPD_antspymm).  this repo is also a good place to try to learn how to use this tool.

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

from : [ANT PD](https://openneuro.org/datasets/ds001907/versions/3.0.2)   see also [this repo](https://github.com/stnava/ANTPD_antspymm).

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

# aggregate the data after you've run on many subjects
# studycsv_all would be the vstacked studycsv2 data frames
zz=antspymm.aggregate_antspymm_results_sdf( studycsv_all, 
    subject_col='subjectID', date_col='date', image_col='imageID',  base_path=bd, 
    splitsep='_', idsep='-', wild_card_modality_id=True, verbose=True)

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


## Population studies

Large population studies may need more care to ensure everything is reproducibly organized and processed.  In this case, we recommend:

### 1. blind qc

first run the blind qc function that would look like `tests/blind_qc.py`. 
this gives a quick view of the relevant data to be processed. it provides 
both figures and summary data for each 3D and 4D (potential) input image.

### 2. collect outlierness measurements

the outlierness function gives one an idea of how each image relates to 
the others in terms of similarity.  it may or may not succeed in detecting 
true outliers but does a reasonable job of providing some rank ordering 
of quality when there is repeated data.  see `tests/outlierness.py`.

### 3. match the modalities for each subject and timepoint

this occurs at the end of `tests/outlierness.py`.  the output of the 
function will select the best quality time point multiple modality 
collection and will define the antspymm cohort in a reproducible manner.

### 4. run the antspymm processing

for each subject/timepoint, one would run:

```python
# ... imports above ...
studyfn="matched_mm_data2.csv"
df=pd.read_csv( studyfn )
index = 20 # 20th subject/timepoint
csvfns = df['filename']
csvrow = df[ df['filename'] == csvfns[index] ]
csvrow['projectID']='MyStudy'

############################################################################################
template = ants.image_read("~/.antspymm/PPMI_template0.nii.gz")
bxt = ants.image_read("~/.antspymm/PPMI_template0_brainmask.nii.gz")
template = template * bxt
template = ants.crop_image( template, ants.iMath( bxt, "MD", 12 ) )
studycsv2 = antspymm.study_dataframe_from_matched_dataframe(
        csvrow,
        rootdir + "nrgdata/data/",
        rootdir + "processed/", verbose=True)
mmrun = antspymm.mm_csv( studycsv2,
                        dti_motion_correct='SyN',
                        dti_denoise=True,
                        normalization_template=template,
                        normalization_template_output='ppmi',
                        normalization_template_transform_type='antsRegistrationSyNQuickRepro[s]',
                        normalization_template_spacing=[1,1,1])
```

### 5. aggregate results

if you have a large population study then the last step would look like this:

```python
import antspymm
import glob as glob
import re
import pandas as pd
import os
df = pd.read_csv( "matched_mm_data2.csv" )
pdir='./processed/'
df['projectID']='MYSTUDY'
merged = antspymm.merge_wides_to_study_dataframe( df, pdir, verbose=False, report_missing=False, progress=100 )
print(merged.shape)
merged.to_csv("mystudy_results_antspymm.csv")
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

## ssl error 

if you get an odd certificate error when calling `force_download`, try:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## to publish a release

```
rm -r -f build/ antspymm.egg-info/ dist/
python3 setup.py sdist bdist_wheel
twine upload --repository antspymm dist/*
```

