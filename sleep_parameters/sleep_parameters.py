from typing import Any
from itertools import repeat, chain
from functools import reduce, cached_property

import numpy as np
import pandas as pd

from sleep_parameters.discrepancy_plot import DiscrepancyPlot


class SleepParameters(DiscrepancyPlot):
    def __init__(self):
        """
        Constructor used to preallocate
        fields for class attributes.
        They will be updated in sleep_parameters_calculation

        Returns
        -------
        None

        """
        self.sleep_parameters = None
        self.sleep_parameters_difference = None
        self.sleep_parameters_log = None
        self.sleep_parameters_difference_log = None

    @cached_property
    def calculate_sleep_parameters(self):
        """
        Calculate Total Sleep Time (TST), Sleep Efficiency (SE),
        Wake After Sleep Onset (WASO) and Sleep Onset Latency (SOL)
        and the total amount of time (in minutes) spent in each stage
        on both reference and device (s). log-transform calculated
        parameters and store the result as a separate attribute

        Here follows parameters definition:
        1)  TST is defined as the total amount of sleep time
            scored during the total recording time.
            Unit of measurement: minutes (min)
        2)  WASO is defined as the period of wakefulness
            that occurs after sleep onset.
            Unit of measurement: minutes (min)
        3)  SE is defined as the ratio of TST
            to time in bed (from lights out to lights-on). This
            routine considers lights out as the first timepoint
            passed via the DataFrame during the construction
            of the instance (SleepTrackerMenu.__init__)
        4)  SOL, is defined as the duration of time from lights-out
            to the first sleep epoch  as determined by PSG.
        5)  The total amount of time (in minutes) spent
            in each stage depends on how many stages are
            allowed by the device in study. For example,
            if only binary classification is allowed (Sleep vs Wake),
            the routine calculates the total amount of
            minutes spent in Sleep and Wake.

        These metrics are stored in two properties:
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
        id_col: str = self.id
        epoch_len_in: int = self.epoch_length
        wake_label: str = self.sleep_scoring.get("Wake")

        # calculating TST, WASO, SE, SOL
        ref_sleep_par: list[list[tuple]] = list(
            map(
                lambda x: self._SleepParameters__par_calculation_each_recording(
                    x,  epoch_len_in, wake_label
                ),
                self.reference.groupby(id_col)
            )
        )
        ref_sleep_par: pd.DataFrame = pd.DataFrame(
            reduce(lambda x, y: x+y, ref_sleep_par),
            columns=[id_col, "parameter", self._reference_col]
        )

        dev_sleep_par: list[list[list[tuple]]] = [
            list(
                map(
                    lambda x: self._SleepParameters__par_calculation_each_recording(
                        x,
                        epoch_len_in,
                        wake_label
                    ),
                    i.groupby(id_col)
                )
            )
            for i in self.device
        ]
        # the outer list comprehension deals with the fact
        # that multiple devices are supported the inner
        # mapping functions is used to calculated parameters
        # on each participant.

        dev_sleep_par: list[list[tuple]] = list(
            map(
                lambda x: reduce(lambda y, z: y+z, x),
                dev_sleep_par
            )
        )
        # the outer list corresponds to each reference
        # the inner list contains a tuple, one for each
        # parameter (tst, waso, sol, se). The latter
        # list contains parameters for all participants
        # processed.

        dev_sleep_par: list[pd.DataFrame] = list(
            map(
                lambda x, y: pd.DataFrame(
                    x,
                    columns=[id_col, "parameter", y]
                ),
                dev_sleep_par,
                self._device_col
            )
        )
        dev_sleep_par: pd.DataFrame = pd.concat(
            dev_sleep_par
        )
        dev_sleep_par.reset_index(
            drop=True,
            inplace=True
        )

        sleep_parameters: pd.DataFrame = ref_sleep_par.merge(dev_sleep_par)
        sleep_parameters: pd.DataFrame = self._SleepParameters__reindex_dataframe(
            sleep_parameters,
            id_col
        )

        # Counting sleep stages for each participant
        ref_sleep_stages: list[list[tuple]] = list(
            map(
                lambda x: self._SleepParameters__sleep_stages_duration(
                    x,
                    id_col,
                    epoch_len_in
                ),
                self.reference.groupby(id_col)
            )
        )  # reference calculation

        ref_sleep_stages: list[tuple] = reduce(
            lambda x, y: x + y,
            ref_sleep_stages
        )

        ref_sleep_stages: pd.DataFrame = pd.DataFrame(
            ref_sleep_stages,
            columns=[id_col, "parameter", self._reference_col]
        )

        dev_sleep_stages: list[list[list[tuple]]] = [
            # devices[participant[stage]]]
            list(
                map(
                    lambda x: self._SleepParameters__sleep_stages_duration(
                        x,
                        id_col,
                        epoch_len_in
                    ),
                    i.groupby(id_col)
                )
            )
            for i in self.device
        ]  # i used to iterate over multiple devices
        # list comprehension preferred over for-loop
        # for performance

        dev_sleep_stages: list[pd.DataFrame] = list(
            map(
                lambda x, y: pd.DataFrame(
                    chain.from_iterable(x),
                    columns=[id_col, "parameter", y]
                ),
                dev_sleep_stages,
                self._device_col
            )
        )
        dev_sleep_stages: pd.DataFrame = reduce(
            lambda left, right: pd.merge(left, right, on=[id_col]),
            dev_sleep_stages
        )
        # concatenates all the devices in one single dataframe

        sleep_stages: pd.DataFrame = ref_sleep_stages.merge(
            dev_sleep_stages,
            on=[self.id, "parameter"]
            )
        sleep_stages = self._SleepParameters__reindex_dataframe(
            sleep_stages,
            self.id
        )  # to multi index for easier data handling

        self.sleep_parameters: pd.DataFrame = pd.concat(
            [sleep_parameters, sleep_stages],
            axis=0
        )

        self.sleep_parameters_difference: pd.DataFrame =\
            self._SleepParameters__parameter_difference_calculation(
                self.sleep_parameters
            )

        # log transformed metrics
        self.sleep_parameters_log = self.sleep_parameters.apply(np.log10)

        self.sleep_parameters_difference_log =\
            self._SleepParameters__parameter_difference_calculation(
                self.sleep_parameters_log
            )

        test = self.sleep_parameters_difference
        test_log = self.sleep_parameters_difference_log

        return self.sleep_parameters, self.sleep_parameters_difference

    @staticmethod
    def __par_calculation_each_recording(
            to_metrics: tuple[str, pd.Series],
            epoch_length: int,
            wake_label: str
            ) -> list[tuple]:
        """
        Calculates parameters for each recording.

        Parameters
        ----------
        to_metrics : tuple[str, pd.Series]
            Tuple resulting from a groupby on file. The first element
            contains the participant's name, the second element
            contains data of the single participant
        id_column : str
            column header that contains IDs
        epoch_length : int
            self.epoch_length
        wake_label: str
            wake label

        Returns
        -------
        pd.DataFrame
            Calculated sleep parameters

        """

        # col_processed: str = str(to_metrics.columns[-1])

        participant_id: str = to_metrics[0]
        to_metrics: pd.DataFrame = to_metrics[1]
        col_processed: str = str(to_metrics.columns[-1])
        # postfix to name of the metrics

        to_sleep_onset_offset: pd.DataFrame =\
            to_metrics.where(to_metrics != wake_label).dropna()
        # to_sleep_onset_offset contains all the epochs that
        # are not classified as wake.
        sleep_onset: np.int64 = to_sleep_onset_offset.index[0]
        # the first epoch scored as not wake is considered as sleep
        # onset
        sleep_offset: np.int64 = to_sleep_onset_offset.index[-1]
        # the last epoch scored as not wake is considered as
        # sleep offset

        to_sol: pd.DataFrame = to_metrics.loc[to_metrics.index[0]:(sleep_onset-1)]
        sol: float = len(to_sol) * epoch_length/60
        sol: tuple[Any] = (participant_id, "SOL", sol)
        # returns the sleep onset latency

        to_tst: int = len(to_sleep_onset_offset)
        # number of epochs that are not wake
        se: float = to_tst/len(to_metrics) * 100
        # returns the total sleep time in minutes
        # as percentage
        se: tuple[Any] = (participant_id, "SE", se)

        tst: float = to_tst * epoch_length/60
        # returns the total sleep time in minutes
        tst: tuple[Any] = (participant_id, "TST", tst)

        to_waso: pd.DataFrame = to_metrics.loc[sleep_onset: sleep_offset]
        to_waso = to_waso.where(to_waso.iloc[:, 1] == wake_label).dropna()
        waso: int = len(to_waso)  # number of epochs scored as wake
        waso: float = waso*epoch_length/60  # returns the wake aftersleep onset in minutes
        waso: tuple[Any] = (participant_id, "WASO", waso)

        sleep_parameters: list[tuple] = [tst, se, waso, sol]

        return sleep_parameters

    @staticmethod
    def __parameter_difference_calculation(
            sleep_par_in: pd.DataFrame
    ):
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
        diff: pd.DataFrame
            dataframe with the difference between
            each device and the reference.

        """
        ref: pd.Series = sleep_par_in.iloc[:, 0]
        dev: pd.DataFrame = sleep_par_in.iloc[:, 1:]

        diff: pd.DataFrame = dev.apply(lambda x: x-ref)
        diff.columns = dev.columns

        return diff

    @staticmethod
    def __sleep_stages_duration(
            to_duration: tuple[str, pd.DataFrame],
            id_col_in: str,
            epoch_length: int,
            ) -> list[tuple]:
        """
        Count the number of epochs for self.ref and self.device
    
        Parameters
        ----------
        to_duration : Tuple[Text, pd.DataFrame]
            tuple resulting from a .groupby called on self.reference or each
            element of self.device
            (applied through mapping)
        id_col_in : Text
            self.id
        epoch_length : int
            self.epoch_length
    
        Returns
        -------
        stages_count : list[tuple]
            number of minutes spent
            in each stage for one
            participant
        """
        id_participant: str = [to_duration[0]]
        to_duration: pd.DataFrame = to_duration[1].drop(
            columns=[id_col_in]
        )
        stages_count: list[list] = list(
            map(
                lambda x: [x[0], len(x[1]) * epoch_length / 60],
                to_duration.groupby(to_duration.columns[0])
            )
        )
        stages_count: list[tuple] = list(
            map(
                lambda x: tuple(id_participant + x),
                stages_count
            )
        )

        return stages_count

    @staticmethod
    def __reindex_dataframe(
            to_multindex: pd.DataFrame,
            id_column: str
    ) -> pd.DataFrame:
        """
        Transform in a multindex dataframe.

        It helps to manage data for bland-altman plots
        and other calculations

        Parameters
        ----------

        to_multindex: pd.DataFrame
            dataframe on which reindexing
            is applied
        id_column: str
            header of the column
            containing data

        Returns
        -------
        multindex_dataframe: pd.DataFrame
            Log transformed dataframe

        """
        to_multindex.index = pd.MultiIndex.from_frame(
            to_multindex.iloc[:, :2],
            names=[id_column, "parameter"]
        )
        multidex_to_dataframe: pd.DataFrame = to_multindex.drop(
            columns=[id_column, "parameter"]
        )
        return multidex_to_dataframe
