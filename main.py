# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:00:31 2022

@author: Davide Benedetti
"""

import os
import pandas as pd

from sleep_tracker_menu import SleepTrackerMenu

os.chdir(r'D:\sri\old\yasa_validation')
file = r'2022_04_10.csv'

file = pd.read_csv(file)

saving_path = r'D:\sri\old\yasa_validation\debug_sleep_tracker_menu'

file.replace('W', "WAKE", inplace=True)  # FOR DEBUGGING!!!!

iclass = SleepTrackerMenu(
    file,
    id_col='ID',
    reference_col='Human',
    device_col=['YASA'],
    sleep_scores={
        'Wake': 'WAKE',
        'Sleep': ['N1', 'N2', 'N3', 'R']
    },
    sleep_stages={'REM': 'R', 'NREM': ['N1', 'N2', 'N3']},
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

# to generate Bland Altman plots

iclass.bland_altman_plot()

# # to produce hypnogram plots:

iclass.hypnogram_plot()

# # to generate confusion matrices
iclass.standard_confusion_matrix_plot()
iclass.proportional_confusion_matrix_plot()

# # to_retrieve performance metrics to be manipulated in the code
# pm_overall = iclass.performance_metrics_overall
# pm_by_sleep_stage = iclass.performance_metrics_each_sleep_stage
# pm_mean_ci = iclass.performance_metrics_mean_ci
#
# # # to generate performance metrics plots
# #
# iclass.performance_metrics_heatmap_plot()
# iclass.boxplot_swarmplot_performance_metrics_each_device()
#
# # to generate discrepancy plots
#
# iclass.discrepancy_plot()
