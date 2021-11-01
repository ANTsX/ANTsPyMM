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

- FIXME


the processes FIXME

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


img1 = ants.image_read( antspymm.get_data( "I1499279_Anon_20210819142214_5", target_extension=".nii.gz") )
img2 = ants.image_read( antspymm.get_data( "I1499337_Anon_20210819142214_6", target_extension=".nii.gz") )
dwp = antspymm.dewarp_imageset( [img1,img2] )
# now write out the mean and dewarped images for further processing, eg with dipy

```


## to publish a release

```
python3 -m build
python -m twine upload -u username -p password  dist/*
```
