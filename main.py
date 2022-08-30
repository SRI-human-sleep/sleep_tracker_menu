# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:00:31 2022

@author: E34476
"""

import pandas as pd

from sleep_tracker_menu import SleepTrackerMenu

file = r''

file = pd.read_csv(file)

saving_path = r''

iclass = SleepTrackerMenu(
    file,
    id_col='ID',
    reference_col='ground_truth',
    device_col=['prediction'],
    sleep_scoring={
        'Wake': 'W',
        'Sleep': ["N1", "N2", "N3", "R"]
    },
    sleep_stages = {
        'REM': "R",
        'NREM': ["N1", "N2", "N3"]
    },
    save_path=saving_path,
    digit=2,
    ci_level=0.95
)

# iclass = InterfaceClass(
#     file,
#     id_col='subject',
#     reference_col='reference',
#     device_col=['device'],
#     sleep_scoring={
#         'Wake': 0,
#         'Sleep': [1, 2, 3]
#     },
#     sleep_stages=None,
#     save_path=saving_path,
#     digit=2,
#     ci_level=0.95
# )


