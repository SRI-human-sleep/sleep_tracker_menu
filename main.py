# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:00:31 2022

@author: E34476
"""
import os
import pandas as pd

from sleep_tracker_menu import SleepTrackerMenu

# night_in_processing = "overall"
# print(night_in_processing)
#
# file = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\2022_04_10.csv'
#
# to_drop = r"C:\Users\e34476\OneDrive - SRI International\NCANDA\data\spreadsheets\yasa_to_drop.xlsx"
# to_drop = pd.read_excel(to_drop).loc[:, "ID_to_drop"]
# to_drop = to_drop.to_list()
#
# file = pd.read_csv(file)
# file.reset_index(inplace=True)
# id_to_keep = [i[0] for i in file.iterrows() if i[1].ID not in to_drop]
#
# file = file.iloc[id_to_keep, :]
#
# dict_to_rep = {
#     "Devika": "devika", "Fiona": 'fiona', "Fiona_FINAL": 'fiona', 'JG': 'justin',
#     "Justin Greco": "justin", "Lena": 'lena', "Leo": "leo", "Max": "max",
#     "Max_FINAL": "max", "Max_Final": "max", "SC": "sc", "Sarah": "sarah"
# }
#
# file["Scorer"] = file["Scorer"].replace(dict_to_rep)
# file = file.where(file["Scorer"] != 'devika').dropna(how="all")
# file = file.where(file["Scorer"] != 'leo').dropna(how='all')
# file = file.where(file["ID"] != "B-00350-M-2_N3_0YFU").dropna(how='all')
# file = file.where(file["ID"] != "A-00101-M-8_N2_0YFU").dropna(how='all')
#
# file.night = file.night.replace({'architecture': 'first_night', 'adaptation': 'first_night'})
# if night_in_processing == "overall":
#     pass
# else:
#     file = file.where(file.night == night_in_processing).dropna(how='all')
#
# saving_path = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\pipeline_run'
#
# file = file.dropna()
#
# iclass = SleepTrackerMenu(
#     file,
#     id_col='ID',
#     reference_col='Human',
#     device_col=['YASA'],
#     sleep_scoring={
#         'Wake': 'W',
#         'Sleep': ["N1", "N2", "N3", "R"]
#     },
#     sleep_stages={
#         'REM': "R",
#         'NREM': ["N1", "N2", "N3"]
#     },
#     save_path=saving_path,
#     digit=4,
#     plot_dpi=1500,
#     ci_level=0.95
# )

# %%

file = pd.read_csv(r"C:\Users\e34476\OneDrive - SRI International\fitbit\script\dev_python\src\training\deep_training\data\perf_eval.csv")

file = file.replace({0: "S", 1: "W"})

sleep_tracker = SleepTrackerMenu(
    file,
    id_col="ID",
    reference_col="gold_standard",
    device_col=["python_algorithm"],
    save_path=r"C:\Users\e34476\OneDrive - SRI International\fitbit\script\dev_python\src\training\deep_training\data\fitbit\to_performance_evaluation\analyzes",
    sleep_scoring={
        'Wake': "W",
        'Sleep': ["S"]
    }
)

# sleep_tracker.proportional_confusion_matrix_plot(
#     figsize=(12, 12),
#     annot_fontsize=17,
#     axis_label_fontsize=20,
#     axis_ticks_fontsize=25,
#     title_fontsize=20
# )

# sleep_tracker.performance_metrics_heatmap_plot(
#     figsize=(10, 60),
#     title_fontsize=15,
#     axis_label_fontsize=15,
#     axis_ticks_fontsize=15
# )
# sleep_tracker.standard_confusion_matrix_plot(
#     absolute=False,
#     figsize=(13, 13),
#     annot_fontsize=20,
#     axis_label_fontsize=22,
#     axis_ticks_fontsize=25,
#     title_fontsize=20
# )
# sleep_tracker.performance_metrics_heatmap_plot(
#     figsize=(10, 60),
#     title_fontsize=15,
#     axis_label_fontsize=15,
#     axis_ticks_fontsize=15
# )
# performance_metrics = iclass.performance_metrics_each_sleep_stage_moments
# performance_metrics.to_excel(os.path.join(saving_path, f"{night_in_processing}_mean_ci.xlsx"))

# if night_in_processing == 'first_night':
#     night_in_processing = "Standard PSG night"
# elif night_in_processing == "ERP":
#     night_in_processing = "ERP/PSG night"
# elif night_in_processing == "overall":
#     night_in_processing = " "
# elif night_in_processing == "architecture":
#     night_in_processing = "Architecture"
# elif night_in_processing == "adaptation":
#     night_in_processing = "Adaptation"
# # %%
# #test = iclass.sleep_parameters_calculation()
# iclass.bland_altman_plot(
#     log_transform=False,
#     augmentation_factor_ylimits=0.1,
#     title_fontsize=30,
#     axis_label_fontsize=30,
#     axis_ticks_fontsize=10
# )
# # %%
# p_metrics = iclass.performance_metrics_each_sleep_stage
# iclass.standard_confusion_matrix_plot(
#     figsize=(13, 13),
#     annot_fontsize=20,
#     axis_label_fontsize=22,
#     axis_ticks_fontsize=25,
#     title_text=night_in_processing,
#     title_fontsize=20
# )
#
# iclass.proportional_confusion_matrix_plot(
#      figsize=(12, 12),
#     annot_fontsize=17,
#     axis_label_fontsize=20,
#     axis_ticks_fontsize=25,
#     title_text=night_in_processing,
#     title_fontsize=20
# )
# # %%
# iclass.sleep_parameters_calculation()
# iclass.boxplot_swarmplot_performance_metrics_each_device(
#     size=3,
#     figsize=(10, 10),
#     title_text=night_in_processing,
#     title_fontsize=20,
#     axis_label_fontsize=22,
#     axis_ticks_fontsize=22
# )


