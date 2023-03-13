# ANTsPyMM

## processing utilities for timeseries/multichannel images - mostly neuroimaging

the outputs of these processes can be used for data inspection/cleaning/triage
as well for interrogating hypotheses.

this package also keeps track of the latest preferred algorithm variations for
production environments.

install by calling (within the source directory):

```
python setup.py install
```

or install via `pip install antspymm` **FIXME**

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
import antspymm
antspymm.get_data()
```

NOTE: `get_data` has a `force_download` option to make sure the latest
package data is installed.

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

automatically qc, filter and match multiple modality images at each time point.

```python
qcdf=antspymm.blind_image_assessment(fns) ## run the qc on all images - requires a relatively large sample per modality to be effective
qcdfa=antspymm.average_blind_qc_by_modality(qcdf,verbose=True) ## reduce the time series qc
qcdfaol=antspymm.outlierness_by_modality(qcdfa) # estimate outlier scores
print( qcdfaol.shape )
print( qcdfaol.keys )
matched_mm_data=antspymm.match_modalities( qcdfaol  )
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
    'sub-RC4125', 
    'ses-1', 
    randid, 
    'T1w', 
    '/Users/stnava/data/openneuro/imagesBIDS/', 
    '/Users/stnava/data/openneuro/processed/', 
    t1_filename=fns[0], 
    dti_filenames=[fns[2]], 
    rsf_filenames=[fns[1]])
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
dtfn=glob.glob("imagesNRG/ANTPD/sub-RC4125/ses-*/*/*/*DTI*gz")
rsfn=glob.glob("imagesNRG/ANTPD/sub-RC4125/ses-*/*/*/*rsfMRI*gz")
studycsv = antspymm.generate_mm_dataframe( 
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

## build docs

```
pdoc -o ./docs antspymm --html 
```

## to publish a release

```
python3 -m build
python -m twine upload -u username -p password  dist/*
```
