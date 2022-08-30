# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:56:13 2022

@author: bened
"""

import os
import glob as glb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             matthews_corrcoef
                             )

from staging.utils import (xml_processing,
                           correct_import_verification,
                           import_file_to_analysis,
                           hypno_plot,
                           get_metrics
                           )

os.chdir(r"C:\Users\bened\OneDrive\SRI\NCANDA\data\staging_perfomance\performance_analysis")

ground_path = r"C:\Users\bened\OneDrive\SRI\NCANDA\data\staging_perfomance\staging_groundtruth\*.xml"
ground_path = glb.glob(ground_path)

pred_path = r"C:\Users\bened\OneDrive\SRI\NCANDA\data\staging_perfomance\staging_algorithm\*.csv"
pred_path = glb.glob(pred_path)

savepath_hypno = r"C:\Users\bened\OneDrive\SRI\NCANDA\data\staging_perfomance\hypnogram"

files = list(map(import_file_to_analysis, ground_path, pred_path))

del ground_path, pred_path

metrics = list(map(lambda x: (x[0],
                              classification_report(x[1].ground_truth,
                                                    x[1].prediction,
                                                    target_names=["W", "N1", "N2", "N3", "R"],
                                                    output_dict=True
                                                    )
                              ),
                    files
                    )
            )

accuracy = pd.concat(map(lambda x: pd.DataFrame([x[0], round(x[1].get("accuracy"),3)]
                                                ).transpose(),
                         metrics
                         ),
                         )

# =============================================================================
# def iqr_calculation(x):
#   return np.subtract(*np.percentile(x, [75, 25]))
# =============================================================================

# =============================================================================
# accuracy.columns = ["ID", "accuracy"]
# accuracy_mean_sem = pd.DataFrame(["mean ± sem", f"{round(accuracy.mean()[0], 3)} ± {round(iqr_calculation(accuracy.accuracy), 3)}"]).transpose()
# accuracy_mean_sem.columns = ["ID", "accuracy"]
# accuracy = pd.concat([accuracy, accuracy_mean_sem],
#                      axis=0
#                      )
# accuracy.index = list(range(len(accuracy)))
# 
# del accuracy_mean_sem
# =============================================================================

# =============================================================================
# weighted_avg = get_metrics(metrics, "weighted avg")
# wake = get_metrics(metrics, "W").to_excel("wake.xlsx", index=False)
# R = get_metrics(metrics, "R").to_excel("rem.xlsx", index=False)
# n1 = get_metrics(metrics, "N1").to_excel("n1.xlsx", index=False)
# n2 = get_metrics(metrics, "N2").to_excel("n2.xlsx", index=False)
# n3 = get_metrics(metrics, "N3").to_excel("n3.xlsx", index=False)
# macro_avg = get_metrics(metrics, "macro avg").to_excel("macro_avg.xlsx", index=False)
# output = accuracy.merge(weighted_avg)
# output = output.rename(columns={"support": "# epochs"})
# =============================================================================

to_confusion = pd.concat(map(lambda x: x[1], files))
to_confusion.index = list(range(len(to_confusion)))
mcc = matthews_corrcoef(y_true=to_confusion.ground_truth,
                        y_pred=to_confusion.prediction
                        )
mcc = round(mcc, 3)

#%% hypnogram
# =============================================================================
# 
# list(map(hypno_plot, files , repeat(savepath_hypno)))
# =============================================================================
# =============================================================================
# to_confusion = pd.concat(map(lambda x: x[1], files))
# to_confusion.index = list(range(len(to_confusion)))
# 
# conf = confusion_matrix(y_true=to_confusion.ground_truth,
#                         y_pred=to_confusion.prediction,
#                         labels=["N1", "N2", "N3", "R", "W"],
#                         )
# 
# conf = ConfusionMatrixDisplay(conf,
#                               display_labels=["N1", "N2", "N3", "R","W"]
#                               )
# conf.plot()
# 
# plt.savefig("confusion_matrix.png",
#             dpi=1000)
# 
# plt.show()
# 
# #%% modified confusion matrix 1
# 
# to_confusion = to_confusion.replace({"N2": "N2/N3",
#                                      "N3": "N2/N3"}
#                                     )
# 
# conf = confusion_matrix(y_true=to_confusion.ground_truth,
#                         y_pred=to_confusion.prediction,
#                         labels=["N1", "N2/N3", "R", "W"]
#                         )
# 
# conf = ConfusionMatrixDisplay(conf,
#                               display_labels=["N1", "N2/N3", "R", "W"]
#                               )
# conf.plot()
# 
# plt.savefig("modified_confusion_matrix.png",
#             dpi=1000)
# plt.show()
# 
# to_confusion = to_confusion.replace({"N2": "N2/N3",
#                                      "N3": "N2/N3",
#                                      "N1": "NOT N2/N3",
#                                      "W": "NOT N2/N3",
#                                      "R": "NOT N2/N3"
#                                      }
#                                     )
# conf = confusion_matrix(y_true=to_confusion.ground_truth,
#                         y_pred=to_confusion.prediction,
#                         labels=["N2/N3", "NOT N2/N3"]
#                         )
# 
# conf = ConfusionMatrixDisplay(conf,
#                               display_labels=["N2/N3", "NOT N2/N3"]
#                               )
# conf.plot()
# 
# 
# plt.savefig("n2_n3_vs_non.png",
#             dpi=1000)
# plt.show()
# 
# # %%
# 
# to_confusion = pd.concat(map(lambda x: x[1], files))
# to_confusion.index = list(range(len(to_confusion)))
# 
# conf = confusion_matrix(y_true=to_confusion.ground_truth,
#                         y_pred=to_confusion.prediction,
#                         labels=["N1", "N2", "N3", "R", "W"],
#                         normalize="true"
#                         )
# 
# conf = ConfusionMatrixDisplay(conf,
#                               display_labels=["N1", "N2", "N3", "R", "W"]
#                               )
# conf.plot()
# 
# plt.savefig("confusion_matrix_norm.png",
#             dpi=1000
#             )
# 
# plt.show()
# 
# #%% modified confusion matrix 1
# 
# to_confusion = to_confusion.replace({"N2": "N2/N3",
#                                      "N3": "N2/N3"}
#                                     )
# 
# conf = confusion_matrix(y_true=to_confusion.ground_truth,
#                         y_pred=to_confusion.prediction,
#                         labels=["N1", "N2/N3", "R", "W"],
#                         normalize="true"
#                         )
# 
# conf = ConfusionMatrixDisplay(conf,
#                               display_labels=["N1", "N2/N3", "R", "W"]
#                               )
# conf.plot()
# 
# plt.savefig("modified_confusion_matrix_norm.png",
#             dpi=1000)
# plt.show()
# 
# to_confusion = to_confusion.replace({"N2": "N2/N3",
#                                      "N3": "N2/N3",
#                                      "N1": "NOT N2/N3",
#                                      "W": "NOT N2/N3",
#                                      "R": "NOT N2/N3"
#                                      }
#                                     )
# conf = confusion_matrix(y_true=to_confusion.ground_truth,
#                         y_pred=to_confusion.prediction,
#                         labels=["N2/N3", "NOT N2/N3"],
#                         normalize="true"
#                         )
# 
# conf = ConfusionMatrixDisplay(conf,
#                               display_labels=["N2/N3", "NOT N2/N3"]
#                               )
# conf.plot()
# 
# 
# plt.savefig("n2_n3_vs_non_norm.png",
#             dpi=1000)
# plt.show()
# =============================================================================
