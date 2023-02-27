# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:00:31 2022

@author: E34476
"""

from itertools import repeat

import numpy as np
import pandas as pd
import multiprocessing
from sklearn.metrics import confusion_matrix
from utils.sanity_check import sanity_check
from sleep_tracker_menu import SleepTrackerMenu

file = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\2022_04_10.csv'

to_drop = r"C:\Users\e34476\OneDrive - SRI International\NCANDA\data\spreadsheets\yasa_to_drop.xlsx"
to_drop = pd.read_excel(to_drop).loc[:, "ID_to_drop"]
to_drop = to_drop.to_list()

file = pd.read_csv(file)
file.reset_index(inplace=True)
id_to_keep = [i[0] for i in file.iterrows() if i[1].ID not in to_drop]

file = file.iloc[id_to_keep, :]

dict_to_rep = {
    "Devika": "devika", "Fiona": 'fiona', "Fiona_FINAL": 'fiona', 'JG': 'justin',
    "Justin Greco": "justin", "Lena": 'lena', "Leo": "leo", "Max": "max",
    "Max_FINAL": "max", "Max_Final": "max", "SC": "sc", "Sarah": "sarah"
}

file["Scorer"] = file["Scorer"].replace(dict_to_rep)
file = file.where(file["Scorer"] != 'devika').dropna(how="all")
file = file.where(file["Scorer"] != 'leo').dropna(how='all')
file = file.where(file["ID"] != "B-00350-M-2_N3_0YFU").dropna(how='all')
file = file.where(file["ID"] != "A-00101-M-8_N2_0YFU").dropna(how='all')
file = list(file.groupby("ID"))
file = list(map(lambda x: x[1], file))
file = file[:10]
file = pd.concat(file)

saving_path = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\pipeline_run'

def concatenate_single_participant_grouped(to_clean):
    clean_df = [i for i in to_clean if isinstance(i, pd.DataFrame)]
    clean_df = pd.concat(clean_df)
    return clean_df


def calculate_mean_sem_for_plotting(to_plot):
    to_plot = to_plot[1].drop(columns=['threshold'], inplace=True)
    mean = to_plot.mean()
    sem = to_plot.sem()
    return mean, sem


# to_roc = SleepTrackerMenu(
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
#     digit=2,
#     plot_dpi=500,
#     ci_level=0.95
# )




#%%

class RocCurveParameters:

    def calculate_parameters_roc_each_stage(self):

        single_participant = list(
            map(
                self._RocCurveParameters__roc_each_stage,
                self.file.groupby(self.reference_header)
            )
        )  # grouping by sleep stage according to the reference
        sleep_stages_to_roc = list(
            map(
                self.__RocCurveParameters__concatenate_participants,
                single_participant
            )
        )

        to_plot = sleep_stages_to_roc[0]

        to_plot = list(
            map(
                self.__RocCurveParameters__calculate_mean_sem,
                to_plot.groupby('threshold')
            )
        )
        return to_plot

    def __roc_each_stage(self, single_stage):
        name_stage = single_stage[0]
        data_to_roc = single_stage[1]
        device = self._RocCurveParameters__relabel(data_to_roc[self.reference_header], name_stage)
        reference = self._RocCurveParameters__relabel(data_to_roc[self.device_header], name_stage)
        single_participant_grouped = pd.concat(
            [reference, device, data_to_roc[self.probability_header]],
            axis=1
        )
        single_participant_grouped = single_participant_grouped.groupby(level=0)

        roc_data = list(
            map(
                self._RocCurveParameters__roc_each_participant,
                single_participant_grouped
            )
        )
        return roc_data

    def __roc_each_participant(self, single_participant):

        name_participant = single_participant[0]
        print(name_participant)
        data_to_confusion = single_participant[1]

        participant = list(
            map(
                self._RocCurveParameters__roc_single_participant,
                repeat(data_to_confusion),
                self.thresholds
            )
        )
        participant = pd.DataFrame(participant)
        participant.columns = ['fpr', 'tpr', 'threshold']

        return participant

    def __roc_single_participant(self, data_to_confusion_matrix, threshold_in):

        data_to_confusion_matrix = data_to_confusion_matrix.where(
            data_to_confusion_matrix[self.probability_header] > threshold_in
        ).dropna(how='all')

        conf_matr = confusion_matrix(
            y_true=data_to_confusion_matrix[self.reference_header],
            y_pred=data_to_confusion_matrix[self.device_header]
        )
        try:
            fpr = conf_matr[0, 0]
            tpr = conf_matr[0, 1]

            fpr_tpr_to_return = (fpr, tpr, threshold_in)
        except IndexError:
            fpr_tpr_to_return = (np.nan, np.nan, np.nan)

        return fpr_tpr_to_return

    @staticmethod
    def __concatenate_participants(to_clean):
        clean_df = [i for i in to_clean if isinstance(i, pd.DataFrame)]
        clean_df = pd.concat(clean_df)
        return clean_df

    @staticmethod
    def __calculate_mean_sem(to_mean_sem):
        to_mean_sem = to_mean_sem[1].drop(columns=['threshold'])
        mean = to_mean_sem.mean()
        sem = to_mean_sem.sem()
        to_ret = pd.concat([mean, sem], axis=1)
        to_ret.columns = ['mean', 'sem']
        return to_ret

    @staticmethod
    def __relabel(to_roc: pd.DataFrame, name_stage: str):
        """
        Relabel dataframe comprising data of each stage
        Parameters
        Parameters
        ----------
        to_roc
        name_stage

        Returns
        -------
        to_roc_relabeled: pd.DataFrame

        """
        labels = [i for i in set(to_roc) if i != name_stage]
        labels = dict(zip(labels, repeat(0, len(labels))))
        labels = labels | {name_stage: 1}
        to_roc_relabeled = to_roc.replace(labels)
        return to_roc_relabeled


class RocCurve(RocCurveParameters):
    def __init__(
            self,
            file: pd.DataFrame,
            sleep_scoring: dict,
            id_col: str = "ID",
            device_header: str = 'device',
            reference_header: str = 'reference',
            probability_header: str = 'predict_proba',
            upper_limit_thresholds: int|float = 0,
            lower_limit_thresholds: int|float = 1.999,
            number_of_thresholds: int = 300,
            n_jobs: None | int = None
    ):
        assert isinstance(device_header, str), \
            "device_header must be a string. " \
            "Multiple devices are not supported by RocCurve. " \
            "You should create an instance of SleepTrackerMenu for each device " \
            "you would like to process. "

        assert isinstance(reference_header, str), \
           f"reference_header must be a string, {type(reference_header)} is not allowed"

        file, wrong_epochs = sanity_check(
            file_in_process=file,
            sleep_scoring=sleep_scoring,
            reference_col=reference_header,
            device_col=[device_header],
            drop_wrong_labels=True
        )
        # sleep scoring is only used to check the quality of the dataframe
        # fed into RocCurve
        file.index = file[id_col]
        file = file.drop(columns=[id_col])

        self.wrong_epochs = wrong_epochs

        self.file = file
        self.device_header = device_header
        self.reference_header = reference_header
        self.probability_header = probability_header
        self.thresholds = pd.Series(
            np.linspace(
                upper_limit_thresholds,
                lower_limit_thresholds,
                num=number_of_thresholds
            )
        )

        if n_jobs is None:
            self.n_jobs = 1
        else:
            assert n_jobs <= multiprocessing.cpu_count(), \
                f"n_jobs (passed:{n_jobs}) must be equal or lower" \
                f"than available CPUs (available: {multiprocessing.cpu_count()})"
            self.n_jobs = n_jobs



test = RocCurve(
    file,
    sleep_scoring={
        'Wake': 'W',
        'Sleep': ["N1", "N2", "N3", "R"]
    },
    id_col="ID",
    device_header="YASA",
    reference_header='Human',
    probability_header='YASA_confidence',
    upper_limit_thresholds=0,
    lower_limit_thresholds=1.999,
    number_of_thresholds=300,
    n_jobs=5

)

test.calculate_parameters_roc_each_stage()
