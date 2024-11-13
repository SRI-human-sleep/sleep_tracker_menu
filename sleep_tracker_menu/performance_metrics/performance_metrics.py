import os
from itertools import repeat
from functools import cached_property
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    confusion_matrix,
    recall_score
)

from .performance_metrics_plot import PerformanceMetricsPlot
from ..utils import to_single_stage_performance_evaluation


class PerformanceMetrics(PerformanceMetricsPlot):

    @cached_property
    def performance_metrics_overall(self) -> pd.DataFrame:
        """
        Calculates performance metrics for each device.

        This property computes a range of performance metrics for each device, including:
        accuracy, Cohen's kappa, F1-score, Matthews correlation coefficient (MCC),
        negative predictive value (NPV), positive predictive value (PPV),
        sensitivity, and specificity. The resulting metrics are rounded to the specified (during class construction)
        number of decimal places and saved to a CSV file.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated metrics for each device, with devices
        as rows and metric names as columns.

        Notes
        -----
        The calculation for each device relies on `reference` as the ground truth and
        `device` as the predicted values. Metrics are computed using the
        `_PerformanceMetrics__metrics_calculation_each_device` method, which applies
        `metrics_calculation_single_participant` to generate participant-level metrics
        before aggregating them at the device level. The results are saved to the path
        specified in `_savepath_metrics_plot`.
        The metrics are calculated through scikit-learn methods.

        Example
        -------
        >>> overall_metrics = iclass.performance_metrics_overall
        >>> print(overall_metrics)
        device,ID,accuracy,cohen_kappa,f1_score_micro,MCC,PPV_micro,sensitivity_micro
        device_1,1,0.87,0.83,0.87,0.83,0.87,0.82
        device_1,2,0.87,0.82,0.87,0.82,0.87,0.81
        device_1,3,0.88,0.83,0.88,0.83,0.88,0.8
        device_1,4,0.86,0.8,0.86,0.8,0.86,0.76
        device_1,5,0.81,0.73,0.81,0.73,0.81,0.74
        device_1,6,0.86,0.77,0.86,0.77,0.86,0.74

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
        if not os.path.exists(self._savepath_metrics_csv):
            os.makedirs(self._savepath_metrics_csv, exist_ok=True)
        metrics_overall.to_csv(os.path.join(self._savepath_metrics_csv, 'performance_metrics_overall.csv'))

        return metrics_overall

    @cached_property
    def performance_metrics_each_sleep_stage(self) -> pd.DataFrame:

        """
        Calculates the mean and 95% confidence interval (CI) for performance metrics per sleep stage.

        This property computes the mean and 95% CI for each performance metric across all participants
        and devices, segmented by sleep stage. The resulting DataFrame provides a statistical summary
        of each metric, including the lower and upper bounds of the CI.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the mean and 95% confidence intervals for each performance metric,
            with columns:
                - "mean": the mean of the metric across participants.
                - "lower_ci": the lower bound of the 95% confidence interval.
                - "upper_ci": the upper bound of the 95% confidence interval.

        Notes
        -----
        - The calculation is based on the `performance_metrics_each_sleep_stage` property, excluding
          any aggregate or summary rows.
        - The 95% confidence interval for each metric is calculated using a t-distribution.
        - This method rounds each result to three decimal places for clarity.

        Example
        -------
        >>> metrics_summary = iclass.performance_metrics_mean_ci
        >>> print(metrics_summary)

        stage,N1, ..., W
        metric,accuracy,cohen_kappa, ..., sensitivity,specificity
        device ID,
        device_1,1, ..., 0.79, 0.97
        device_1,2, ..., 0.94, 0.98
        device_1,3, ..., 0.98, 0.97
        device_1,4, ..., 0.74, 0.99
        device_1,5, ..., 0.81, 0.97
        ...
        device_2,1, ..., 0.94, 0.89
        device_2,2, ..., 0.98, 0.75
        device_2,3, ..., 0.96, 0.97
        device_2,4, ..., 0.95, 0.94
        device_2,all, ..., 0.93, 0.96

        [352 rows x 35 columns]

        This property provides a concise summary of the central tendency and variability
        of each performance metric, aiding in the assessment of metric stability across sleep stages.
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

        if not os.path.exists(self._savepath_metrics_csv):
            os.makedirs(self._savepath_metrics_csv, exist_ok=True)
        metrics_each_sleep_stage.to_csv(os.path.join(self._savepath_metrics_csv, 'metrics_each_sleep_stage.csv'))

        return metrics_each_sleep_stage

    @property
    def performance_metrics_mean_ci(self):
        """
        Computes the mean and 95% confidence interval (CI) for performance metrics across sleep stages.

        This property calculates the mean and 95% confidence interval for each performance metric
        (e.g., accuracy, sensitivity) across all participants and devices for each sleep stage.
        The output provides a statistical summary that includes the mean as well as the lower and
        upper bounds of the 95% CI for each metric.

        Returns
        -------
        pd.DataFrame
            A DataFrame with each performance metric as a row and three columns:
                - "mean": the mean of the metric across participants.
                - "lower_ci": the lower bound of the 95% confidence interval.
                - "upper_ci": the upper bound of the 95% confidence interval.

        Notes
        -----
        - The method relies on `performance_metrics_each_sleep_stage`, which contains
          per-participant, per-device metrics segmented by sleep stage.
        - The final row of the `performance_metrics_each_sleep_stage` DataFrame is excluded
          from the calculation.
        - The 95% CI for each metric is computed using a t-distribution and `scipy.stats.sem`
          for the standard error, providing an estimate of metric variability.

        Example
        -------
        >>> metrics_summary = iclass.performance_metrics_mean_ci
        >>> print(metrics_summary)
        stage,metric,mean,lower_ci,upper_ci
        N1,accuracy,0.944,0.942,0.946
        N1,cohen_kappa,0.363,0.348,0.377
        N1,f1_score,0.390,0.376,0.405
        N1,MCC,0.374,0.359,0.388
        N1,PPV,0.431,0.412,0.450
        N1,sensitivity,0.395,0.379,0.411
        N1,specificity,0.972,0.970,0.974
        N2,accuracy,0.874,0.868,0.880
        N2,cohen_kappa,0.735,0.722,0.748
        N2,f1_score,0.841,0.832,0.850
        N2,MCC,0.742,0.730,0.754
        N2,PPV,0.828,0.816,0.841
        N2,sensitivity,0.866,0.857,0.876
        N2,specificity,0.884,0.877,0.892
        N3,accuracy,0.935,0.930,0.941
        N3,cohen_kappa,0.811,0.796,0.826
        N3,f1_score,0.852,0.839,0.865
        N3,MCC,0.823,0.809,0.836
        N3,PPV,0.908,0.895,0.922
        N3,sensitivity,0.828,0.811,0.844
        N3,specificity,0.979,0.976,0.982
        R,accuracy,0.952,0.948,0.955
        R,cohen_kappa,0.822,0.808,0.835
        R,f1_score,0.850,0.837,0.862
        R,MCC,0.828,0.816,0.841
        R,PPV,0.888,0.877,0.898
        R,sensitivity,0.835,0.818,0.852
        R,specificity,0.977,0.975,0.979
        W,accuracy,0.953,0.947,0.959
        W,cohen_kappa,0.807,0.793,0.821
        W,f1_score,0.832,0.820,0.844
        W,MCC,0.819,0.807,0.831
        W,PPV,0.774,0.757,0.791
        W,sensitivity,0.930,0.924,0.936
        W,specificity,0.957,0.950,0.964

        This property provides an overview of the average performance metrics and their
        confidence intervals, helping to assess metric stability across sleep stages.
        """
        performance_metrics = self.performance_metrics_each_sleep_stage
        performance_metrics = performance_metrics.droplevel(level=0, axis=0)
        performance_metrics = performance_metrics.iloc[:-1, :]

        performance_metrics_mean = performance_metrics.mean(axis=0)
        performance_metrics_mean = performance_metrics_mean.round(3)
        performance_metrics_mean.name = 'mean'
        performance_metrics_ci = performance_metrics.apply(
            lambda x: st.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=st.sem(x)),
            axis=0
        )
        performance_metrics_ci = performance_metrics_ci.round(3)
        performance_metrics_ci = performance_metrics_ci.transpose()
        performance_metrics_ci.columns = ["lower_ci", "upper_ci"]

        performance_metrics_moments = pd.concat([performance_metrics_mean, performance_metrics_ci], axis=1)

        if not os.path.exists(self._savepath_metrics_csv):
            os.makedirs(self._savepath_metrics_csv, exist_ok=True)
        performance_metrics_moments.to_csv(os.path.join(self._savepath_metrics_csv, 'performance_metrics_moments.csv'))

        return performance_metrics_moments

    @staticmethod
    def __metrics_calculation_each_device(
            reference_to_metrics: pd.DataFrame,
            device_to_metrics: pd.DataFrame,
            metrics_func: object,
            idc: str,
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
         reference_to_metrics : str
             self.reference

         device_to_metrics : pd.DataFrame
             one element of self.device
             (function applied to each element by iteration)

         metrics_func : object

         idc : str
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
         participant_id : str
             participant_id

         to_metrics : pd.DataFrame
             Dataframe on which metrics will be calculated

         id_column : str
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

        labels = [i for i in list(set(y_reference)) if i != "Any"]

        if sleep_stage is True:
            stage_name = set(y_reference)

            stage_name = [i for i in stage_name if i != "Any"][0]
            labels = labels + ["Any"]
        else:
            pass

        conf_matrix = confusion_matrix(
            y_true=y_reference,
            y_pred=y_device,
            labels=labels
        )


        if np.shape(conf_matrix)[1] == 2 and sleep_stage is True:

            tp, fn, fp, tn = conf_matrix.ravel()

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
                    # "NPV": pd.Series(
                    #     tn / (tn + fn),
                    #     name="NPV"
                    # ),
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
                        name="specificity"
                    )
                },
                index=[0]
            )

        else:
            tp = np.diag(conf_matrix).flatten()  # true positive
            fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  # false positive
            fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)  # false negative
            tn = conf_matrix.sum() - (tp + fp + fn)  # true negative

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
