import os
from datetime import datetime
from typing import List, Text, TypedDict

import pandas as pd

from hypnograms.hypnograms import HypnogramPlot
from confusion_matrix.confusion_matrix import ConfusionMatrix
from sleep_parameters.sleep_parameters import SleepParameters
from bland_altman.bland_altman_plot import BlandAltmanPlot
from performance_metrics.performance_metrics import PerformanceMetrics

from utils.sanity_check import sanity_check
from utils.save_directory_generation import save_directory_generation


class SleepTrackerMenu(
    SleepParameters, PerformanceMetrics, ConfusionMatrix, BlandAltmanPlot, HypnogramPlot
):
    def __init__(
            self,
            file: pd.DataFrame,
            id_col: Text,
            reference_col: Text,
            device_col: List,
            save_path: Text,
            drop_wrong_labels: bool = True,
            sleep_scoring: TypedDict = None,
            sleep_stages: TypedDict = None,
            epoch_length: int = 30,
            digit: int = 2,
            ci_level: float = 0.95,
            plot_dpi: int = 500,
            ci_bootstrapping: bool = False,
            boot_method: Text = 'basic',
            boot_n_resamples: int = 10000
    ) -> None:
        """
        Interface class used to evaluate the performance
        of sleep-tracker devices.

        It implements methods to conduct epoch-by-epoch (EBE)
        and discrepancy analyzes, in accordance with Menghini et al.
        (2021). DOI: 10.1093/sleep/zsaa170

        This class supports the evaluation of performance of multiple
        devices (for example, two, three or more smartbands) against
        one reference device (for example, Polysomnography).
        Measurements of both the reference and devices under evaluation
        must be temporally aligned to the reference and between each others
        before being passed to the class. This implies that the number of
        samples for the reference and each device must be equal. If you are
        using sleep trackers with different sampling frequencies, you can either
        resample your dataset or instantiate SleepTrackerMenu separately for
        each device.

        This class does not automatically align time series. Time signals
        are assumed to be already aligned by the user.

        If you use this tool, please cite one of these
        two publications:
            1)
            2)

        Args:
            file: pd.DataFrame
                Two-dimensional table containing data to be analyzed.
                Each row corresponds to an observation.
                It must include at least three columns:
                1)  a column containing the IDs for each observation.
                    If multiple observations are collected for the same subject,
                    ID must be repeated for each observation.
                2)  a column containing measurements collected by the
                    reference signal. Only one reference technique
                    is supported by the routine.
                3)  a column containing measurements collected by the
                    device being evaluated. If the performance of multiple
                    devices is assessed, measurements of each device must
                    be passed in different columns.
            drop_wrong_labels: bool
                if True, any wrong values passed are dropped. If false,
                values are not dropped but the routine might run
                into errors. A wrong value is defined as a value
                that is not represented in sleep_scoring
            id_col: Text
                header of the column containing IDs
            reference_col: Text
                header of the column containing
                measurements collected by the device
                used as reference
            device_col: List
                List containing the header(s) of the device(s)
                whose performance is evaluated.
            save_path: Text
                directory in which save plots. Multiple
                folders are automatically generated in
                save_path.
            sleep_scoring: TypedDict
                dictionary specifying the label
                assigned to Wake, and Sleep.
            sleep_stages: TypedDict| bool
                It can be either a boolean (False),
                if the device(s) evaluated
                do(es) not make distinction between
                REM and NREM sleep.
                Pass the following dictionary if
                sleep distinction is supported:
                sleep_stages = {
                    'REM': "R",
                    'NREM': ["N1", "N2", "N3"]
                }
            epoch_length: int
                specifies the length of each epoch
                in seconds.
            digit: int
                significant figures.
            ci_level: float
                lambda of the confidence interval.
            plot_dpi: int
                dpi at which plots will be saved
            ci_bootstrapping: bool
                specifies if the confidence interval
                will be calculated through bootstrapping
            boot_method: Text
                type of bootstrap method to calculate confidence
                intervals.
            boot_n_resamples: int
                number of resampling when calculating
                confidence intervals via bootstrapping
        """
        self.file, self.wrong_epochs = sanity_check(
            file,
            sleep_scoring,
            reference_col,
            device_col,
            drop_wrong_labels
        )
        self.id = id_col

        self.reference = self.file.loc[:, [id_col, reference_col]]
        self._reference_col = reference_col

        self.device = list(
            map(
                lambda x: self.file.loc[:, [id_col, x]],
                device_col
            )
        )
        self._device_col = device_col

        self.sleep_scoring = sleep_scoring

        self.epoch_length = epoch_length

        self.digit = digit

        self.sleep_stages = sleep_stages

        self.ci_level = ci_level

        self.plot_dpi = plot_dpi

        self.ci_bootstrapping = ci_bootstrapping

        self.boot_method = boot_method
        self.boot_n_resamples = boot_n_resamples

        save_name = str(datetime.now())
        save_name = save_name.replace(':', '_')
        save_name = save_name.replace('-', '_')
        save_name = save_name.replace(' ', '_')
        save_name = save_name[:save_name.find('.')]
        self._save_name = save_name

        # create directories for saving data and plots

        savepath_sleep_parameters = save_directory_generation(
            save_path,
            'sleep_parameters'
        )
        self._savepath_sleep_parameters = os.path.join(
            savepath_sleep_parameters[0],
            f'{save_name}.xlsx'
        )

        savepath_discrepancies_plot = save_directory_generation(
            save_path,
            'discrepancies_plot'
        )
        self._savepath_discrepancies_plot = os.path.join(
            savepath_discrepancies_plot[0],
            f'{save_name}'
        )

        savepath_metrics = save_directory_generation(
            save_path,
            'performance_metrics',
            ['plot', 'csv_export']
        )
        self._savepath_metrics_plot = os.path.join(
            savepath_metrics[1][0],
            f'{save_name}'
        )
        self._savepath_metrics_csv = os.path.join(
            savepath_metrics[1][1],
            f'{save_name}.csv'
        )

        savepath_standard_absolute_confusion_matrix = save_directory_generation(
            save_path,
            'standard_absolute_confusion_matrix',
            ['plot', 'excel_export']
        )
        self._savepath_standard_absolute_confusion_matrix_plot = os.path.join(
            savepath_standard_absolute_confusion_matrix[1][0],
            f'{save_name}'
        )
        self._savepath_standard_absolute_confusion_matrix_xlsx = os.path.join(
            savepath_standard_absolute_confusion_matrix[1][1],
            f'{save_name}'
        )

        savepath_standard_normalized_confusion_matrix = save_directory_generation(
            save_path,
            'standard_normalized_confusion_matrix',
            ['plot', 'excel_export']
        )
        self._savepath_standard_normalized_confusion_matrix_plot = os.path.join(
            savepath_standard_normalized_confusion_matrix[1][0],
            f'{save_name}'
        )
        self._savepath_standard_normalized_confusion_matrix_xlsx = os.path.join(
            savepath_standard_normalized_confusion_matrix[1][1],
            f'{save_name}'
        )

        savepath_proportional_confusion_matrix = save_directory_generation(
            save_path,
            'proportional_confusion_matrix',
            ['plot', 'excel_export']
        )
        self._savepath_proportional_confusion_matrix_plot = os.path.join(
            savepath_proportional_confusion_matrix[1][0],
            f'{save_name}'
        )
        self._savepath_proportional_confusion_matrix_xlsx = os.path.join(
            savepath_proportional_confusion_matrix[1][1],
            f'{save_name}'
        )

        self._savepath_performance_metrics_boxplots = save_directory_generation(
            save_path,
            'performance_metrics_boxplots',
            [
                'accuracy',
                'cohen_kappa',
                'f1_score',
                'MCC',
                'NPV',
                'PPV',
                'sensitivity',
                'specificity'
            ]
        )
        self._savepath_performance_metrics_boxplots = \
            self._savepath_performance_metrics_boxplots + \
            [save_name]

        self._savepath_performance_metrics_heatmaps = save_directory_generation(
            save_path,
            'performance_metrics_heatmaps',
            [
                'accuracy',
                'cohen_kappa',
                'f1_score',
                'MCC',
                'NPV',
                'PPV',
                'sensitivity'
            ]
        )
        self._savepath_performance_metrics_heatmaps = \
            self._savepath_performance_metrics_heatmaps + \
            [save_name]

        self._savepath_hypnograms_plot = save_directory_generation(
            save_path,
            'hypnograms_plot',
            [save_name]
        )[1][0]

        self._savepath_bland_altman_plots = save_directory_generation(
            save_path,
            'bland_altman_plots',
            [save_name]
        )[1][0]
