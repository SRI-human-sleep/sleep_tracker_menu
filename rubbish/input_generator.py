# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:35:39 2022

@author: E34476
"""

import os
import glob as glb
from itertools import repeat
import pandas as pd

from staging_performance.utils import (xml_processing,
                           correct_import_verification,
                           import_file_to_analysis,
                           hypno_plot,
                           get_metrics
                           )


os.chdir(r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\staging_routine_implementation')

ground_truth = glb.glob(os.path.join(os.getcwd(), 'ground_truth\*.XML'))
prediction = glb.glob(os.path.join(os.getcwd(), 'prediction\*.csv'))

files = list(map(import_file_to_analysis, ground_truth, prediction))

to_concat = pd.concat(map(lambda x: pd.concat([pd.Series(repeat(x[0],
                                                    len(x[1])),
                                          name='ID'), x[1]], axis=1),
                     files))

to_concat.to_csv('single_file.csv')


# =============================================================================
# list(map(lambda x: x[1].to_csv(x[0], index=None),
#          files))
# =============================================================================

