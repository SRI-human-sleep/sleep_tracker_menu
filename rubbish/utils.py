# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:15:24 2022

@author: bened
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import repeat

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

def xml_processing(xml_path):
    """
    Imports and preprocesses human staging

    Parameters
    ----------
    xml_path : Text
        file path derived through glob

    Returns
    -------
    arr : pd.Series
        imported and preprocessed sleep staging

    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.keys()

    event_settings = root[4] #sleepstages

    arr = []
    for i in event_settings:
        arr.append(int(i.text))

    arr = pd.Series(arr,
                    name="ground_truth"
                    )
    arr = arr.replace({0: "W",
                       1: "N1",
                       2: "N2",
                       3: "N3",
                       5:"R"}
                      )
    return arr

def correct_import_verification(path_to_ver):
    return Path(path_to_ver).parts[-1]

def import_file_to_analysis(ground_path_in, pred_path_in):

    prediction = pd.read_csv(pred_path_in,
                             header=None
                             )
    prediction.columns = ["prediction"]

    ground_truth = xml_processing(ground_path_in)

    ground_path_in = correct_import_verification(ground_path_in).replace(".XML", "")
    pred_path_in = correct_import_verification(pred_path_in).replace(".csv", "")

    if ground_path_in != pred_path_in:
        raise ValueError("ground_truth's file does not correspond to prediction's file")

    to_analysis = pd.concat([ground_truth, prediction], axis=1)

    return ground_path_in, to_analysis

def iqr_calculation(x):
  return np.subtract(*np.percentile(x, [75, 25]))



def hypno_plot(to_hypno, hypno_path):
    dict_to_hypno = {"W": 5,
                     "R": 4,
                     "N1": 3,
                     "N2": 2,
                     "N3": 1
                     }

    ground_truth = to_hypno[1]["ground_truth"]
    ground_truth = ground_truth.replace(dict_to_hypno)

    prediction = to_hypno[1]["prediction"]
    prediction = prediction.replace(dict_to_hypno)

    difference = to_hypno[1]
    difference = difference.replace(dict_to_hypno)
    difference = Series(
        difference.where(difference.ground_truth != difference.prediction
                         ).dropna().index,
        name="error"
        )

    rem_ground = Series(ground_truth.where(ground_truth == 4).dropna().index)
    rem_ground = pd.concat([rem_ground, Series(repeat(4.1, len(rem_ground)))],
                           axis=1
                           )
    rem_ground.columns = ["x", "y"]
    rem_pred = Series(prediction.where(prediction == 4).dropna().index)
    rem_pred = pd.concat([rem_pred, Series(repeat(4.1, len(rem_pred)))],
                         axis=1
                         )
    rem_pred.columns = ["x", "y"]
    del dict_to_hypno


    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=1000)
    ax1.plot(prediction, color="black")
    ax1.scatter(rem_pred.x,
                rem_pred.y,
                marker="*",
                s=0.3,
                color="red"
                )
    for i in difference:
        ax1.axvspan(i-0.1,i+0.1,
                    alpha=1,
                    facecolor='blue'
                    )
        ax2.axvspan(i-0.1,i+0.1,
                    alpha=1,
                    facecolor='blue',
                    label="misclassified epochs"
                    )

    ax2.plot(ground_truth, color="black")
    ax2.scatter(rem_ground.x,
                rem_ground.y,
                marker="*",
                s=0.3,
                color="red"
                )

    ax1.set_yticks(list(range(1,6)), ["N3", "N2", "N1", "R", "W"])
    ax2.set_yticks(list(range(1,6)), ["N3", "N2", "N1", "R", "W"])

    ax1.set_ylabel("Prediction")
    ax2.set_ylabel("Ground Truth")
    ax2.set_xlabel("Time (min)")

    fig.tight_layout()
    fig.savefig(os.path.join(hypno_path, f"{to_hypno[0]}.png"), dpi=1000)
    fig.show()
    return None