# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:02:41 2022

@author: E34476
"""

import matplotlib.pyplot as plt
from seaborn import heatmap
from pandas import DataFrame

from .sleep_parameters import SleepParameters


class DiscrepancyPlot(SleepParameters):
    def discrepancy_plot(
            self,
            plot_dpi: int = None,
            figsize: list[int] = None
    ):
        """
        Generates a discrepancy plot for each device, comparing sleep parameters.

        This method, used by the `sleep_parameters.SleepParameters` class, creates
        a discrepancy plot for each device, saving each plot to a predefined directory.

        Parameters
        ----------
        plot_dpi : int, optional
            Resolution of the plot in dots per inch. If not specified, defaults to
            `self.plot_dpi`.
        figsize : list of int, optional
            Dimensions of the plot in inches as [width, height]. If not provided,
            defaults to [6.4, 4.8], the standard size in matplotlib.

        Returns
        -------
        None
            This method generates and saves discrepancy plots but does not return a value.

        Notes
        -----
        - The discrepancy plot provides a visual comparison of sleep parameter differences
          across devices.
        - The method utilizes `self._DiscrepancyPlot__discrepancy_plot` for each device
          and passes in the required parameters, including plot resolution and figure size.

        Example
        -------
        >>> iclass.discrepancy_plot(plot_dpi=150, figsize=[8, 6])
        Generating discrepancy plots for each device...
        """
        save_path = self._savepath_discrepancies_plot

        if plot_dpi is None:
            plot_dpi = self.plot_dpi
        else:
            pass

        if figsize is None:
            figsize = [6.4, 4.8] # matplotlib defaults
        else:
            pass

        self.calculate_sleep_parameters

        for i in self.sleep_parameters_difference.loc[:, self._device_col].items():
            self._DiscrepancyPlot__discrepancy_plot(
                i,
                save_path,
                plot_dpi,
                figsize
            )
        return None

    @staticmethod
    def __discrepancy_plot(
            sleep_dev_in: DataFrame,
            save_path: str,
            plot_dpi_in: int,
            figsize: list[int]
    ):
        """
        Plot discrepancy plot.

        Is superclass to sleep_parameters.SleepParameters.
        discrepancy_plot method plots discrepamcy plot
        for each device, while plot of single device
        is carried out by _DiscrepanciesPlot__discrepancy_plot

        Parameters
        ----------
        sleep_dev_in : DataFrame
            sleep parameters of one device.
            The function is called through mapping

        save_path : str
            self._savepath_discrepancies_plot

        plot_dpi_in : int
            self.plot_dpi

        figsize: list[int, int]
            figsize to be passed to plt.subplots.

        Returns
        -------
        None

        """
        title_text = sleep_dev_in[0]
        # first element of iteritems' tuple

        to_heat = DataFrame(sleep_dev_in[1])

        to_heat = to_heat.unstack()
        to_heat.columns = to_heat.columns.droplevel(level=0)
        fig, ax_in_plot = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=figsize,
            dpi=plot_dpi_in
        )

        heatmap(
            to_heat,
            annot=True,
            cmap="Blues",
            ax=ax_in_plot,
        )

        fig.suptitle(
            title_text,
            fontsize=14
        )

        plt.tight_layout()

        save_path = f"{save_path}_{title_text}.png"
        plt.savefig(save_path,
                    dpi=plot_dpi_in
                    )

        plt.show(black=True)

        return None
