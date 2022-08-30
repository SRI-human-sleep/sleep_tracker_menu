# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:49:52 2022

@author: E34476
"""
from functools import reduce
from itertools import repeat
from typing import Text, Tuple

import numpy as np
import pandas as pd

from sleep_parameters.discrepancy_plot import DiscrepancyPlot

class SleepParameters(DiscrepancyPlot):
    def sleep_parameters_calculation(self):
        """
        Calculatesthe following metrics on both reference
        and device(s): tst, se, waso, sol.

        These four metrics are retained in two properties:
        self.sleep_parameters, self.sleep_parameters_log,
        where the postfix indicates parameters have been
        log-transformed.

        The difference between each sleep parameter,
        measured on reference and device(s) data,
        is calculated and retained in the following
        two properties: self.sleep_parameters_difference and
        self.sleep_parameters_difference_log, where the postfix
        log indicates that data has been log-transformed.

        Returns
        -------
        None

        """
        id_col = self.id
        epoch_len_in = self.epoch_length
        wake_label = self.sleep_scoring.get("Wake")

        # calculating TST, WASO, SE, SOL
        ref_sleep_par = pd.concat(
            map(
                lambda x: self._SleepParameters__par_calculation_each_recording(x, id_col, epoch_len_in, wake_label),
                self.reference.groupby(id_col)
            )
        )
        ref_sleep_par.index = list(range(len(ref_sleep_par)))

        dev_sleep_par = [
            pd.concat(
                map(
                    lambda x: self._SleepParameters__par_calculation_each_recording(x, id_col, epoch_len_in, wake_label),
                    i.groupby(id_col)
                )
            )
            for i in self.device
        ]
        # the outer list comprehension deals with the fact that multile devices are supported
        # the inner mapping functions is used to calculated data on each participant.

        dev_sleep_par = pd.concat(dev_sleep_par)
        dev_sleep_par.index = list(range(len(dev_sleep_par)))

        sleep_parameters = ref_sleep_par.merge(dev_sleep_par)
        sleep_parameters = self._SleepParameters__reindex_dataframe(
            sleep_parameters,
            self.id
        )

        # Counting sleep stages for each participant
        ref_sleep_stages = pd.concat(
            map(
                lambda x: self._SleepParameters__sleep_stages_counting(x, id_col, epoch_len_in),
                self.reference.groupby(id_col)
            )
        )  # reference calculation

        dev_sleep_stages = [
            pd.concat(
                map(
                    lambda x: self._SleepParameters__sleep_stages_counting(x, id_col, epoch_len_in),
                    i.groupby(id_col))  # i corresponds to a single device under study
                )
            for i in self.device
        ]
        dev_sleep_stages = reduce(lambda left, right: pd.merge(left, right, on=[id_col]), dev_sleep_stages)
        # concatenates all the devices in one single dataframe

        sleep_stages = ref_sleep_stages.merge(
            dev_sleep_stages,
            on=[self.id, "parameter"]
            )
        sleep_stages = self._SleepParameters__reindex_dataframe(
            sleep_stages,
            self.id
        )  # to multi index for easier data handling

        # generating attributes.

        self.sleep_parameters = pd.concat([sleep_parameters, sleep_stages], axis=0)

        self.sleep_parameters_difference = self._SleepParameters__parameter_difference_calculation(
            self.sleep_parameters
        )

        # log transformed metrics
        self.sleep_parameters_log = self.sleep_parameters.apply(np.log10)
        self.sleep_parameters_difference_log = self._SleepParameters__parameter_difference_calculation(
            self.sleep_parameters_log
        )
        return None

    @staticmethod
    def __par_calculation_each_recording(
            to_metrics: Tuple[Text, pd.Series],
            id_column: Text,
            epoch_length: int,
            wake_label: Text
            ) -> pd.DataFrame:
        """
        Calculates tst, se, waso, sol in minutes.

        Parameters
        ----------
        to_metrics : Tuple[Text, pd.Series]
            Tuple resulting from a groupby on file. The first element contains the
            participant's name, the second element contains data of the single
            participant
        id_column : Text
            self.id
        epoch_length : int
            self.epoch_length
        wake_label: Text
            Wake label in self.sleep_scoring

        Returns
        -------
        pd.DataFrame
            Calculated sleep parameters

        """

        participant_id = to_metrics[0]
        to_metrics = to_metrics[1]
        col_processed = str(to_metrics.columns[-1])
        # postfix to name of the metrics

        to_sleep_onset_offset = to_metrics.where(to_metrics != wake_label).dropna()
        sleep_onset = to_sleep_onset_offset.index[0]
        sleep_offset = to_sleep_onset_offset.index[-1]

        sol = to_metrics.loc[to_metrics.index[0]:(sleep_onset-1)]
        sol = len(sol) *epoch_length/60
        # returns the sleep onset latency
        sol = pd.DataFrame(
            {id_column: participant_id, "parameter": "SOL", col_processed: sol},
            index=[0]
        )

        to_tst = len(to_sleep_onset_offset) # number of epochs
        se = to_tst/len(to_metrics)*100 # returns the total sleep time in minutes
        se = pd.DataFrame(
            {id_column: participant_id, "parameter": "SE", col_processed: se},
            index=[0]
        )

        tst = to_tst*epoch_length/60  # returns the total sleep time in minutes
        tst = pd.DataFrame(
            {id_column: participant_id, "parameter": "TST", col_processed: tst},
            index=[0]
        )

        to_waso = to_metrics.loc[sleep_onset: sleep_offset]
        to_waso = to_waso.where(to_waso.iloc[:, 1] == wake_label).dropna()
        waso = len(to_waso)  # number of epochs scored as wake
        waso = waso*epoch_length/60  # returns the wake aftersleep onset in minutes
        waso = pd.DataFrame(
            {id_column: participant_id, "parameter": "WASO", col_processed: waso},
            index=[0]
        )

        return pd.concat([tst, se, waso, sol])

    @staticmethod
    def __parameter_difference_calculation(sleep_par_in):
        """
        Generates the difference between the parameter
        estimated through reference, and the parameter
        calculated by each device tested.

        Parameters
        ----------
        sleep_par_in : pd.DataFrame
            Dataframe on which the difference is
            calculated.

        Returns
        -------
        pd.DataFrame
            dataframe with the difference between
            each device and the reference.

        """
        ref = sleep_par_in.iloc[:, 0]
        dev = sleep_par_in.iloc[:, 1:]

        diff = dev.apply(lambda x: x-ref)
        diff.columns = dev.columns

        return diff

    @staticmethod
    def __sleep_stages_counting(
            to_count: Tuple[Text, pd.DataFrame],
            id_col_in:Text,
            epoch_length: int,
            ) -> pd.DataFrame:
        """
        Count the number of epochs for self.ref and self.device
    
        Parameters
        ----------
        to_count : Tuple[Text, pd.DataFrame]
            tuple resulting from a .groupby called on self.reference or each
            element of self.device
            (applied through mapping)
        id_col_in : Text
            self.id
        epoch_length : int
            self.epoch_length
    
        Returns
        -------
        stages_count : pd.DataFrame
            DESCRIPTION.
        """
        id_part = to_count[0]
        to_count = to_count[1].drop(columns=[id_col_in])
        stages_count = list(
            map(
                lambda x: (x[0], len(x[1])*epoch_length/60),
                to_count.groupby(to_count.columns[0])
            )
        )  # counts the number of epochs of each sleep stage as
        # specified by the user in sleep_parameters
        stages_count = pd.DataFrame(stages_count, columns=["parameter", to_count.columns[0]])

        stages_count = pd.concat(
            [pd.Series(repeat(id_part, len(stages_count)), name=id_col_in), stages_count],
            axis=1
        )
        return stages_count

    @staticmethod
    def __reindex_dataframe(
            to_multindex: pd.DataFrame,
            id_column
    ):
        """
        Transform in a multindex dataframe.

        It helps to manage data for bland-altman plots
        and other calculations

        Parameters
        ----------

        to_multindex: pd.DataFrame
            dataframe on which reindexing
            is applied

        Returns
        -------
        multindex_dataframe
            Log transformed dataframe

        """
        to_multindex.index = pd.MultiIndex.from_frame(
            to_multindex.iloc[:, :2],
            names=[id_column, "parameter"]
        )
        multidex_to_dataframe = to_multindex.drop(
            columns=[id_column, "parameter"]
        )
        return multidex_to_dataframe
