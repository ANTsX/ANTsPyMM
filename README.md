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

process:

* resting state

* neuromelanin mid-brain images

* DWI

* flair

* voxel-based cortical thickness

into tabular and template-based voxel-wise format.

# first time setup

```python
import antspymm
antspymm.get_data()
```

NOTE: `get_data` has a `force_download` option to make sure the latest
package data is installed.

# example processing

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
