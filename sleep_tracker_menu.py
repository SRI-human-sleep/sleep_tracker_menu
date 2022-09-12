import os
from datetime import datetime
from typing import List, Text, TypedDict

import pandas as pd

from bland_altman.bland_altman import BlandAltman
from hypnograms.hypnograms import HypnogramPlot
from confusion_matrix.confusion_matrix import ConfusionMatrix
from sleep_parameters.sleep_parameters import SleepParameters
from bland_altman.bland_altman_plot import BlandAltmanPlot
from performance_metrics.performance_metrics import PerformanceMetrics

from utils.sanity_check import sanity_check
from utils.save_directory_generation import save_directory_generation


class SleepTrackerMenu(
    SleepParameters, PerformanceMetrics, ConfusionMatrix, BlandAltman, HypnogramPlot, BlandAltmanPlot
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

        Args:
            file: pd.DataFrame
                file containing a column, containing the IDs,
                a column containing the reference, and columns
                containing data collected by device(s) under
                investigation.
            drop_wrong_labels: bool
                if True, the function checks for any wrong value
                passed with file and values are dropped. If false,
                values are not dropped but the routine might run
                into errors.
            id_col: Text
                header of the id column.
            reference_col: Text
                header of the reference column.
            device_col: List
                List containing the header(s) of the devices
                under investigation.
            save_path: Text
                directory in which save plots.
            sleep_scoring: TypedDict
                dictionary specifying the label
                assigned to Wake, and Sleep.
                Pass the following dictionary if
                sleep distinction is supported:
                sleep_scoring={
                    'Wake': 'W',
                    'Sleep': ["N1", "N2", "N3", "R"]
                    },
            sleep_stages: TypedDict| bool
                It can be either a boolean (False),
                if the device(s) under investigation
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
                type of
            boot_n_resamples: int
                number of resampling to apply during
                confidence interval calculation through
                evaluation.
        """
        self.file, self.wrong_epochs = sanity_check(file, sleep_scoring, reference_col, device_col, drop_wrong_labels)
        self.id = id_col

        self.reference = self.file.loc[:, [id_col, reference_col]]
        self._reference_col = reference_col

        self.device = list(map(lambda x: self.file.loc[:, [id_col, x]],
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

        savepath_absolute_confusion_matrix = save_directory_generation(
            save_path,
            'absolute_confusion_matrix',
            ['plot', 'excel_export']
        )
        self._savepath_absolute_confusion_matrix_plot = os.path.join(
            savepath_absolute_confusion_matrix[1][0],
            f'{save_name}'
        )
        self._savepath_absolute_confusion_matrix_xlsx = os.path.join(
            savepath_absolute_confusion_matrix[1][1],
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