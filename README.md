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


## to publish a release

```
python3 -m build
python -m twine upload -u username -p password  dist/*
```
