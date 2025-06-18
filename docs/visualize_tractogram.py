import os

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.io.utils import (create_nifti_header, get_reference_info,
                           is_header_compatible)
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.utils import density_map

from dipy.data.fetcher import (fetch_file_formats,
                               get_file_formats)

if False:
    ref_anat_filename = 'processed_or/data/PPMI/100018/20210202/DTI_LR/1497589/PPMI-100018-20210202-DTI_LR-1497589-I1497578-DTIRGB.nii.gz'
    trkfn = 'processed_or/data/PPMI/100018/20210202/DTI_LR/1497589/PPMI-100018-20210202-DTI_LR-1497589-I1497578-tractogram.trk'
    reference_anatomy = nib.load(ref_anat_filename)

    if not 'trk' in locals():
        trk = load_tractogram(trkfn, reference_anatomy)

    streamlines = select_random_set_of_streamlines(trk.streamlines, 10000)

    from fury import actor, colormap
    from dipy.tracking import utils
    # from utils.visualization_utils import generate_anatomical_volume_figure
    streamlines_actor = actor.line(streamlines, colormap.line_colors(streamlines))

    from dipy.viz import window, actor, colormap, has_fury
    scene = window.Scene()
    scene.add(actor.line(streamlines,
            colormap.line_colors(streamlines)))
    window.show(scene, size=[2000,2000])
