
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import boxplot, heatmap, swarmplot

from ..utils import confidence_interval_calculation


class PerformanceMetricsPlot:
    def performance_metrics_heatmap_plot(self):
        """
        Generates heatmap plots for performance metrics for each device and sleep stage.

        This method visualizes the performance metrics calculated for each device
        and sleep stage, as stored in `performance_metrics_each_sleep_stage`. The
        resulting heatmaps provide an overview of various performance metrics per
        device (rows) and sleep stage (columns), with values annotated in each cell.

        Heatmaps are saved to the specified directory for each device-stage combination.

        Parameters
        ----------
        self : SleepTrackerMenu
            Instance of the SleepTrackerMenu class.

        Returns
        -------
        None
            The function does not return a value. Plots are displayed and saved to the
            appropriate directory.

        Notes
        -----
        If analyzing many participants, this methods might produce figures that are too
        large to be usable for either interpretation or publication. It better works with
        smaller datasets.

        Examples
        --------
        >>> iclass.performance_metrics_heatmap_plot()

        """
        print('Generating heatmaps')
        df_to_plot = self.performance_metrics_each_sleep_stage

        device_level = set(df_to_plot.index.get_level_values('device'))
        stage_level = set(df_to_plot.columns.get_level_values('stage'))

        for i in device_level:
            for j in stage_level:
                to_heatmap = df_to_plot.loc[i, j]
                data_heatmap, annot_heatmap = self._PerformanceMetricsPlot__proportional_metrics(
                    to_heatmap.drop("all"),
                    stage_name=j,
                    ci_level=self.ci_level,
                    digit_in=self.digit
                )

                data_heatmap = pd.concat([to_heatmap, data_heatmap], axis=0)
                annot_heatmap = pd.concat([to_heatmap, annot_heatmap], axis=0)

                fig, ax_to_plot = plt.subplots(nrows=1, ncols=1, figsize=(14, 11), dpi=self.plot_dpi)
                heatmap(
                    data=data_heatmap,
                    annot=annot_heatmap,
                    cmap="Blues",
                    fmt="",
                    annot_kws={"fontsize": "large", "in_layout": True},
                    ax=ax_to_plot
                )

                ax_to_plot.set_title(f"{j}, {i}", fontsize=12)

                plt.savefig(
                    f"{self._savepath_metrics_plot}_{i}_{j}.png",
                    dpi=self.plot_dpi
                )
                plt.tight_layout()
                plt.show(block=True)

        return None

    def boxplot_swarmplot_performance_metrics_each_device(
            self,
            size=None,
            figsize: tuple[int, int] = None,
            title_text: str = '',
            title_fontsize: int = 10,
            axis_label_fontsize: int =11,
            axis_ticks_fontsize: int=13
    ):
        """
        Generates boxplots with superimposed swarmplots for each device and metric.

        This method plots the performance metrics by sleep stage, with a boxplot and
        corresponding swarmplot for each metric (found in the first level of the
        `performance_metrics_each_sleep_stage` DataFrame). Each plot is generated for
        each device in the dataset.

        Parameters
        ----------
        size : float, optional
            Size of the swarmplot markers. Defaults to `5` if not specified.
        figsize : tuple of int, optional
            Figure size for the plot in inches as (width, height). Defaults to
            (6.4, 4.8) if not specified.
        title_text : str, optional
            Title text for the plots. If left empty, the device name will be used as
            the title. Default is an empty string.
        title_fontsize : int, optional
            Font size for the plot title. Default is `10`.
        axis_label_fontsize : int, optional
            Font size for the axis labels. Default is `11`.
        axis_ticks_fontsize : int, optional
            Font size for the axis tick labels. Default is `13`.

        Returns
        -------
        None
            The function generates and saves plots but does not return a value.

        Notes
        -----
        This function accesses the `performance_metrics_each_sleep_stage` property,
        which contains performance metrics data organized in a multi-level index DataFrame.
        The plots are saved to paths specified in `_savepath_performance_metrics_boxplots`.

        Example
        -------
        >>> iclass.boxplot_swarmplot_performance_metrics_each_device()

        Each plot is displayed and saved in the specified save directory for review.
        """
        print('Generating boxplots with swarmplots for every metric.')

        if size is None:
            size = 5  # seaborn default
        else:
            pass

        if figsize is None:
            figsize = [6.4, 4.8]
        else:
            pass

        df_to_plot = self.performance_metrics_each_sleep_stage

        device_level = set(df_to_plot.index.get_level_values("device"))
        metric_level = set(df_to_plot.columns.get_level_values("metric"))

        for i in device_level:
            for j in metric_level:
                print(f'    Plotting {i}: {j}')
                to_boxplot = df_to_plot.loc[i, :]
                to_boxplot = to_boxplot.stack()
                to_boxplot = to_boxplot.xs(
                    j,
                    axis=0,
                    level=1,
                    drop_level=False
                )  # gets only those rows having j (metric)
                # as level 1 index

                fig, ax_to_plot = plt.subplots(
                    ncols=1,
                    nrows=1,
                    dpi=self.plot_dpi,
                    figsize=figsize
                )
                boxplot(
                    ax=ax_to_plot,
                    data=to_boxplot
                )
                swarmplot(
                    edgecolor="white",
                    ax=ax_to_plot,
                    data=to_boxplot,
                    size=size
                )

                plt.xticks(fontsize=axis_ticks_fontsize)
                ax_to_plot.set_xlabel("Stage", fontsize=axis_label_fontsize)
                ax_to_plot.set_ylim(0, 1.2)
                plt.yticks(np.arange(0, 1.2, 0.2), fontsize=axis_ticks_fontsize)
                ax_to_plot.set_ylabel(j, fontsize=axis_label_fontsize)

                if title_text == '':
                    ax_to_plot.set_title(i, fontsize=title_fontsize)
                else:
                    ax_to_plot.set_title(title_text, fontsize=title_fontsize)

                plt.tight_layout()

                save_plot = [
                    k
                    for k in self._savepath_performance_metrics_boxplots[1]
                    if j in k
                ][0]

                save_plot = join(
                    save_plot,
                    f"{self._savepath_performance_metrics_boxplots[2]}_{i}_.png"
                )

                plt.tight_layout()

                plt.savefig(
                    save_plot,
                    dpi=self.plot_dpi
                )
                plt.show()

        return None

    @staticmethod
    def __proportional_metrics(
            to_boxplot_stat_in: pd.DataFrame,
            stage_name: str,
            ci_level: float,
            digit_in: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates the proportional metrics.

        For every performance metrics calculated
        (mean, std and confidence interval).
        confidence interval is calculated through t-test or
        bootstrapping.

        Parameters
        ----------
        to_boxplot_stat_in : pd.DataFrame
            pandas dataframe containing the performance metrics
            of every participant.

        stage_name : str
            corresponds to the sleep stage processed.

        ci_level_in: float.
            self.ci_level

        digit_in: float
            self.digit_in
            significant figures to keep

        Returns
        -------
        to_plot : tuple[pd.DataFrame, pd.DataFrame]
            mean, std and ci.

        """
        stat_mean = round(to_boxplot_stat_in.mean(axis=0), digit_in)
        stat_mean_annot = stat_mean.astype(str) + '\n'
        stat_std = '(' + round(to_boxplot_stat_in.std(axis=0), digit_in).astype(str) + ')' + '\n'
        stat_ci = confidence_interval_calculation(
            to_ci=to_boxplot_stat_in,
            stage_device_name=stage_name,
            return_annot_df=True,
            ci_level=ci_level,
            digit=digit_in,
            ci_bootstrapping=False
        )

        to_plot_annot = stat_mean_annot.add(stat_std).add(stat_ci)
        to_plot_annot.index = ["proportional metrics"]
        to_data_heatmap = pd.DataFrame(stat_mean).transpose()
        to_data_heatmap.index = ["proportional metrics"]

        return to_data_heatmap, to_plot_annot
