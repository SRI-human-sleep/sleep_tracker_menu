import os
import glob as glb
from pathlib import PurePath

import numpy as np
import pandas as pd

file = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\staging_routine_implementation\single_file_to_process\single_file.csv'

file = pd.read_csv(file)

from typing import List, Text, Tuple, TypedDict

from datetime import datetime

from sleep_parameters import SleepParameters
from utils.save_directory_generation import save_directory_generation


class InterfaceClass(SleepParameters):
    def __init__(self,
                 file: pd.DataFrame,
                 id_col: Text,
                 reference_col: Text,
                 device_col: List,
                 save_path: Text,
                 target_names: List[Text],
                 sleep_scoring: TypedDict = None,
                 epoch_length: int = 30,
                 round_factor: int = 3
                 ) -> None:
        '''


        Parameters
        ----------
        file : pd.DataFrame
        id_col : Text
            IDs
        reference_col : Text
            gold standard/ ground truth.
        device_col : List[Text]
            list of devices that undergo the performance evaluation.
        target_names : List[Text]
            labels to process. If sleep_parameters is True, pass a dictionary
            that indicates the sleep stages. Otherwise, a list containing the labels
            to process is passed. If binary classification of sleep stages is passed,
            the following dictionary should be passed
        sleep_parameters : TypedDict, optional
            If None, sleep parameters are not calculated.
            The default is None.

        epoch_length : int, optional
            Single epoch duration in seconds.
            The default is 30 seconds.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        self.file = file
        self.id = id_col

        self.reference = self.file.loc[:, [id_col, reference_col]]
        self._reference_col = reference_col

        self.device = list(map(lambda x: self.file.loc[:, [id_col, x]],
                               device_col
                               )
                           )
        self._device_col = device_col

        self.tnames = target_names

        self.sleep_scoring = sleep_scoring

        self.epoch_length = epoch_length

        self.round_factor = round_factor

        save_name = str(datetime.now())
        save_name = save_name.replace(':', '_')
        save_name = save_name.replace('-', '_')
        save_name = save_name.replace(' ', '_')
        save_name = save_name[:save_name.find('.')]
        self._save_name = save_name

        # create saving directory (sub)folders

        savepath_sleep_parameters = save_directory_generation(save_path, 'sleep_parameters')
        self._savepath_sleep_parameters = os.path.join(savepath_sleep_parameters[0], f'{save_name}.xlsx')

        savepath_discrepancies_plot = save_directory_generation(save_path, 'discrepancies_plot')
        self._savepath_discrepancies_plot = os.path.join(savepath_discrepancies_plot[0], f'{save_name}.png')

        # savepath_discrepancies_plot = save_directory_generation(save_path, 'max', ['max1', 'max2'])


saving_path = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\staging_routine_implementation\test_saving_folder'

iclass = InterfaceClass(
    file,
    id_col='ID',
    reference_col='ground_truth',
    device_col=['prediction'],
    target_names=["W", "N1", "N2", "N3", "R"],
    sleep_scoring={'Wake': 'W',
                   'Sleep': ["N1", "N2", "N3", "R"]
                   },
    save_path=saving_path,
    round_factor=3
)

idc = iclass.id
# file = iclass.file
# reference = iclass.reference
# device = iclass.device
# tnames = iclass.tnames
# sleep_scoring = iclass.sleep_scoring
# epoch_length = iclass.epoch_length
# round_factor = iclass.round_factor
# # savepath_sleep_parameters = iclass.savepath_sleep_parameters
# #
sleep_par = iclass.sleep_parameters_calculation()

sleep_parameters = iclass.sleep_parameters
sleep_stages = iclass.sleep_stages




