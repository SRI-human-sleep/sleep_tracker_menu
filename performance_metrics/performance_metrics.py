from itertools import repeat
from typing import Callable, Text

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    confusion_matrix,
    recall_score
)

from performance_metrics.performance_metrics_plot import PerformanceMetricsPlot
from utils.to_single_stage_performance_evaluation import to_single_stage_performance_evaluation


class PerformanceMetrics(PerformanceMetricsPlot):

    @property
    def performance_metrics_overall(self) -> pd.DataFrame:
        """
        for each device, the following  metrics are here calculated:
        accuracy cohen's kappa, f1-score MCC NPV PPV sensitivity specificity

         Parameters
         ----------
            self

         Returns
         -------
         pd.DataFrame
             Calculated metrics

        """
        reference = self.reference
        device = self.device

        metrics_function = self._PerformanceMetrics__metrics_calculation_single_participant
        metrics_overall = map(
            lambda ref, dev: (
                self._PerformanceMetrics__metrics_calculation_each_device(
                    ref, dev.iloc[:, -1], metrics_function, self.id
                )
            ),
            repeat(reference),
            device
            )

        metrics_overall = pd.concat(metrics_overall, axis=0)

        metrics_overall = metrics_overall.apply(lambda x: round(x, self.digit))
        metrics_overall.to_csv(self._savepath_metrics_plot)

        return metrics_overall

    @property
    def performance_metrics_each_sleep_stage(self) -> pd.DataFrame:
        """
        for each device and participant, the following
        metrics are here calculated: accuracy cohen's kappa,
        f1-score MCC NPV PPV sensitivity specificity.

        Parameters
        ----------
            self
        Returns
        -------
        pd.DataFrame
            Calculated metrics
        """

        id_col = self.id
        reference = self.reference
        device = self.device

        metrics_each_sleep_stage = map(
            lambda ref, dev: (self._PerformanceMetrics__metrics_calculation_each_device(
                ref,
                dev.iloc[:, -1],
                self._PerformanceMetrics__metrics_calculation_single_participant,
                id_col,
                sleep_stage=True
            )
            ),
            repeat(reference),
            device
        )
        del reference, device

        metrics_each_sleep_stage = pd.concat(metrics_each_sleep_stage, axis=0)
        metrics_each_sleep_stage = metrics_each_sleep_stage.apply(lambda x: round(x, self.digit))
        return metrics_each_sleep_stage

    @staticmethod
    def __metrics_calculation_each_device(
            reference_to_metrics: pd.DataFrame,
            device_to_metrics: pd.DataFrame,
            metrics_func: Callable,
            idc: Text,
            sleep_stage: bool = False
    ) -> pd.DataFrame:
        """
        Calculates the performance metrics of a single device.
        self.metrics_calculation_single_participant is
        here used (passed as metrics_func) to calculate the metrics
        of performance of the device passed to metrics_calculation_each_device.
        for each participant and for the overall sample,
        i.e. without grouping by device.

         Parameters
         ----------
         reference_to_metrics : Text
             self.reference

         device_to_metrics : pd.DataFrame
             one element of self.device
             (function applied to each element by iteration)

         metrics_func : Callable

         idc : Text
            self.id

         sleep_stage:bool = False
            if True, metrics are calculated for each
            sleep stage. if False, they are calculated
            without grouping by sleep stage

         Returns
         -------
         pd.DataFrame
             Calculated metrics

         """
        device_name = str(device_to_metrics.name)
        to_metrics = pd.concat([reference_to_metrics, device_to_metrics], axis=1)

        if sleep_stage is True:

            metrics_output = []
            for i in to_metrics.groupby(reference_to_metrics.columns[-1]):

                sleep_metrics = to_single_stage_performance_evaluation(to_metrics, i[0], idc)

                try:
                    [i for i in set(sleep_metrics) if i != "Any"][0]

                    metrics_overall = metrics_func("all", sleep_metrics, idc, sleep_stage=True)
                    metrics_single_participant = map(
                        lambda x: metrics_func(x[0], x[1], idc, sleep_stage=True),
                        sleep_metrics.groupby(idc)
                    )
                    metrics_single_participant = pd.concat(metrics_single_participant)

                    to_append = pd.concat([metrics_single_participant, metrics_overall], axis=0)
                    metrics_output.append(to_append)
                except IndexError:
                    pass


            metrics_output = pd.concat(metrics_output, axis=1)

        else:
            metrics_overall = metrics_func("all", to_metrics, idc)

            metrics_single_participant = map(
                lambda x: metrics_func(x[0], x[1], idc),
                to_metrics.groupby(idc)
            )
            metrics_single_participant = pd.concat(metrics_single_participant)

            metrics_output = pd.concat([metrics_single_participant, metrics_overall], axis=0)

        return metrics_output

    @staticmethod
    def __metrics_calculation_single_participant(
            participant_id,
            to_metrics,
            id_column,
            sleep_stage: bool = False
    ) -> pd.DataFrame:
        """
        Calculates the performance metrics for each
        participant.

        It is passed to __metrics_calculation_each_device

         Parameters
         ----------
         participant_id : Text
             participant_id

         to_metrics : pd.DataFrame
             Dataframe on which metrics will be calculated

         id_column : Text
            self.id

         sleep_stage: bool = False
            only used if metrics_calculation_single_participant
            is used to evaluate the performance of each device

         Returns
         -------
         pd.DataFrame
             Calculated metrics

         """
        y_reference = to_metrics.iloc[:, 1]
        y_device = to_metrics.iloc[:, 2]

        if sleep_stage is True:
            stage_name = set(y_reference)

            stage_name = [i for i in stage_name if i != "Any"][0]
        else:
            pass

        conf_matrix = confusion_matrix(
            y_true=y_reference,
            y_pred=y_device,
            labels=list(set(y_reference))
        )

        tp = np.diag(conf_matrix).flatten()  # true positive
        fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  # false positive
        fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)  # false negative
        tn = conf_matrix.sum() - (tp + fp + fn)  # true negative

        if np.shape(conf_matrix)[1] == 2 and sleep_stage is True:

            metrics_output = pd.DataFrame(
                {
                    f"{id_column}": participant_id,
                    "accuracy": accuracy_score(
                        y_true=y_reference,
                        y_pred=y_device
                    ),
                    "cohen_kappa": cohen_kappa_score(
                        y1=y_reference,
                        y2=y_device
                    ),
                    "f1_score": f1_score(
                        y_true=y_reference,
                        y_pred=y_device,
                        pos_label=stage_name
                    ),
                    "MCC": matthews_corrcoef(
                        y_true=y_reference,
                        y_pred=y_device
                    ),
                    "NPV": pd.Series(
                        tn / (tn + fn),
                        name="NPV"
                    ),
                    "PPV": precision_score(
                        y_true=y_reference,
                        y_pred=y_device,
                        pos_label=stage_name
                    ),
                    "sensitivity": recall_score(
                        y_true=y_reference,
                        y_pred=y_device,
                        pos_label=stage_name
                    ),
                    "specificity": pd.Series(
                        tn / (tn + fp),
                        name="NPV"
                    )
                },
                index=[0]
            )

        else:
            metrics_output = pd.DataFrame(
                {
                    f"{id_column}": participant_id,
                    "accuracy": accuracy_score(
                        y_true=y_reference,
                        y_pred=y_device
                    ),
                    "cohen_kappa": cohen_kappa_score(
                        y1=y_reference,
                        y2=y_device
                    ),
                    "f1_score_micro": f1_score(
                        y_true=y_reference,
                        y_pred=y_device,
                        average="micro"
                    ),
                    "MCC": matthews_corrcoef(
                        y_true=y_reference,
                        y_pred=y_device
                    ),
                    # "NPV": pd.Series(
                    #     tn / (tn + fn),
                    #     name="NPV"
                    # ),
                    "PPV_micro": precision_score(
                        y_true=y_reference,
                        y_pred=y_device,
                        average="micro"
                    ),
                    "sensitivity_micro": recall_score(
                        y_true=y_reference,
                        y_pred=y_device,
                        average="macro"
                    ),
                    # "specificity": pd.Series(
                    #     tn / (tn + fp),
                    #     name="NPV"
                    # )
                },
                index=[0]
            )

        metrics_output.index = pd.MultiIndex.from_product(
            [
                [y_device.name],
                metrics_output[id_column]
            ],
            names=["device", id_column]
        )
        metrics_output.drop(columns=[id_column], inplace=True)
        if sleep_stage is False:
            pass
        else:
            metrics_output.columns = pd.MultiIndex.from_product(
                [
                    [stage_name],
                    metrics_output.columns
                ],
                names=["stage", "metric"]
            )
        return metrics_output
