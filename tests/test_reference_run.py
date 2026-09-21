import sys, os
import unittest
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import tempfile
import shutil
import antspyt1w
import antspynet
import antspymm
import ants
import re
import pandas as pd
def test_reference_run():
    import antspymm
    v = antspymm.version()
    assert isinstance(v, dict)
    assert 'antspymm' in v
    assert hasattr(antspymm, 'mm')
    assert hasattr(antspymm, 'validate_nrg_file_format')
    assert callable(antspymm.generate_voxelwise_bvecs)
