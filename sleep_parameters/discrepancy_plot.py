# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:02:41 2022

@author: E34476
"""

from typing import List, Text

import matplotlib.pyplot as plt
from seaborn import heatmap
from pandas import DataFrame


class DiscrepancyPlot:
    def discrepancy_plot(
            self,
            plot_dpi: int = None,
            figsize: List[int] = None
    ):
        """
        Plot discrepancy plot.

        It is superclass to sleep_parameters.SleepParameters.
        discrepancy_plot method plots.

        Parameters
        ----------
        plot_dpi: int
            self.plot_dpi
            The default is None
        figsize: List[int, int]
            specifies the dimension of discrepancy plot.
            If no argument is passed, figsize is set
            to be the default assigned in matplotlib.pyplot
            The default is None.

        Returns
        -------
        None

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

        for i in self.sleep_parameters_difference.loc[:, self._device_col].iteritems():
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
            save_path: Text,
            plot_dpi_in: int,
            figsize: List[int]
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

        save_path : Text
            self._savepath_discrepancies_plot

        plot_dpi_in : int
            self.plot_dpi

        figsize: List[int, int]
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
