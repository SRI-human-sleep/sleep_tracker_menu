import os
from datetime import datetime

import pandas as pd

from .hypnograms import HypnogramPlot
from .confusion_matrix import ConfusionMatrix
from .sleep_parameters import DiscrepancyPlot
from .bland_altman import BlandAltmanPlot
from .performance_metrics import PerformanceMetrics

from .utils import sanity_check
from .utils import save_directory_generation


class SleepTrackerMenu(
    DiscrepancyPlot, PerformanceMetrics, ConfusionMatrix, BlandAltmanPlot, HypnogramPlot
):
    def __init__(
            self,
            file: pd.DataFrame,
            id_col: str,
            reference_col: str,
            device_col: list,
            save_path: str,
            drop_wrong_labels: bool = True,
            sleep_scores: dict = None,
            sleep_stages: dict = None,
            epoch_length: int = 30,
            digit: int = 2,
            ci_level: float = 0.95,
            plot_dpi: int = 500,
            ci_bootstrapping: bool = False,
            boot_method: str = 'basic',
            boot_n_resamples: int = 10000
    ) -> None:
        """
        Initializes the SleepTrackerMenu class for evaluating sleep-tracker device performance.

        This class enables the analysis of sleep-tracker devices by comparing their outputs
        against a reference device, supporting epoch-by-epoch (EBE) and discrepancy analyses
        as described in Menghini et al. (2021). It allows for comparison of multiple devices
        against one reference device, with various options for result customization, including
        bootstrapping confidence intervals and saving metrics and plots to organized directories.

        Parameters
        ----------
        file : pd.DataFrame
            DataFrame containing observations with columns for IDs, reference device, and evaluation devices.
        id_col : str
            Column name that contains unique IDs for each observation.
        reference_col : str
            Column name with measurements from the reference device.
        device_col : list of str
            List of column names, each representing measurements from a device being evaluated.
        save_path : str
            Directory where results and plots will be saved. Subdirectories are created automatically.
        drop_wrong_labels : bool, optional
            Whether to drop rows with labels inconsistent with `sleep_scoring`. Default is True.
        sleep_scores : dict, optional
            Dictionary specifying labels for sleep scoring stages, e.g., `{'Wake': 0, 'Sleep': 1}`.
        sleep_stages : dict or bool, optional
            Dictionary specifying REM and NREM stages, or False if no distinction is provided. For example,
            `{'REM': 'R', 'NREM': ['N1', 'N2', 'N3']}`.
        epoch_length : int, optional
            Duration of each epoch in seconds. Default is 30.
        digit : int, optional
            Number of decimal places for rounding results. Default is 2.
        ci_level : float, optional
            Confidence interval level. Default is 0.95.
        plot_dpi : int, optional
            DPI setting for generated plots. Default is 500.
        ci_bootstrapping : bool, optional
            Whether to use bootstrapping for confidence intervals. Default is False.
        boot_method : str, optional
            Bootstrapping method used if `ci_bootstrapping` is True (e.g., 'basic', 'percentile'). Default is 'basic'.
        boot_n_resamples : int, optional
            Number of bootstrap resamples for confidence interval estimation if `ci_bootstrapping` is True. Default is 10000.

        Attributes
        ----------
        file : pd.DataFrame
            Processed DataFrame after validation.
        wrong_epochs : pd.DataFrame
            DataFrame containing rows with invalid labels, if `drop_wrong_labels` is True.
        reference : pd.DataFrame
            Subset of `file` with the reference column.
        device : list of pd.DataFrame
            List of DataFrames, each containing measurements from an evaluated device.
        epoch_length : int
            Length of each epoch in seconds.
        ci_level : float
            Confidence interval level.
        plot_dpi : int
            DPI setting for plot resolution.
        save_name : str
            Unique filename based on current timestamp, used to organize saved directories and files.
        savepath_* : str
            Paths for saving results, organized by metric type (e.g., confusion matrix, performance metrics).

        Raises
        ------
        ValueError
            If required columns are missing, if unsupported labels are found in `file`, or if device measurements
            are not aligned with the reference.

        Examples
        --------
        >>> sleep_menu = SleepTrackerMenu(
        ...     file=data,
        ...     id_col="ParticipantID",
        ...     reference_col="Polysomnography",
        ...     device_col=["Smartband1", "Smartband2"],
        ...     save_path="./results",
        ...     sleep_scores={"Wake": 0, "Sleep": 1},
        ...     sleep_stages={"REM": "R", "NREM": ["N1", "N2", "N3"]}
        ... )
        """
        self.file, self.wrong_epochs = sanity_check(
            file,
            sleep_scores,
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

        self.sleep_scores = sleep_scores

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
            f'{save_name}'
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
