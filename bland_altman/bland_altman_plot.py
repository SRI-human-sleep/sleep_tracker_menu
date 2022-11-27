import os
from math import inf
from typing import Text

import seaborn as sns
import matplotlib.pyplot as plt
from numpy import nan
from seaborn import JointGrid, scatterplot, kdeplot

from .bland_altman_parameters import BlandAltmanParameters


class BlandAltmanPlot(BlandAltmanParameters):

    def bland_altman_plot(
            self,
            log_transform: bool = False,
            parameter_to_plot: list[str] = None,
            device_to_plot: list[str] = None,
            ci_level: float = None,
            title_fontsize: int = 10,
            axis_label_fontsize: int = 11,
            axis_ticks_fontsize: int = 13,
            ci_bootstrapping: str = None,
            boot_n_resamples: int = None,
            joint_plot_ratio: int = 6,
            joint_plot_height: int = 10,
            augmentation_factor_ylimits: float = 0.3,
            augmentation_factor_xlimits: float = 0.1,  # not supported yet
            plot_dpi: int = None
    ):
        """
        Produce Bland Altman plot, modeling
        the bias and limits of agreement
        according to Bland and Altman 1999
        Args:
            log_transform: bool
                if to produce plots
                on log-transformed data
                The default is False
            parameter_to_plot: list[str]
                if a list of parameters is passed,
                only specified parameters will be plotted.
                if parameter_to_plot is not specified (default),
                all parameters are plotted
                The default is None
            device_to_plot: list[str]
                if a list of devices is passed,
                only specified devices will be plotted.
                if device_to_plot is not specified (default),
                all devices are plotted
                The default is None
            ci_level: float
                confidence level for confidence interval
                estimation. if not specified, ci_level specified
                when constructing SleepTrackerMenu is used.
                The default is None
            title_fontsize: int
                title fontsize
                The default is
            axis_label_fontsize: int
                axis labels fontsize
                The default is 11
            axis_ticks_fontsize: int
                axis labels fontsize
                The default is 13
            ci_bootstrapping: str
                if to calculate confidence
                intervals via bootstrapping
                If not specified, confidence
                intervals will be calculated
                according to what specified
                when constructing SleepTrackerMenu
                The default is None
            boot_n_resamples: int
                number of resamples when
                confidence intervals are calculated
                via bootstrapping.
                If not specified, the number of
                boostrapping used to produce
                Bland Altman plots will be the
                one specified when constructing
                SleepTrackerMenu
                The default is None
            joint_plot_ratio: int
                Ratio of joint axes height
                to marginal axes height
                The default is 6
            joint_plot_height: int
                Size of the figure.
                The default is 10
            augmentation_factor_ylimits: float
                used to widen y limits
                the formula applied is:
                max value along yaxis * (1+augmentation_factor_limit)
                The default is 0.3
            augmentation_factor_xlimits: float
                Not supported yet
                The default is 0.1
            plot_dpi: int
                dots per inch when saving
                Bland-Altman plots.
                if not specified, what
                specified when constructing
                SleepTrackerMenu is used.
                The default is None

        Returns:

        """

        if plot_dpi is None:
            plot_dpi = self.plot_dpi
        else:
            pass

        if ci_level is None:
            ci_level = self.ci_level
        else:
            pass

        if boot_n_resamples is None:
            boot_n_resamples = self.boot_n_resamples
        else:
            pass

        if hasattr(self, "sleep_parameters_difference") is False:
            self.calculate_sleep_parameters
        else:
            pass

        if log_transform is True:
            reference_to_scatter = self.sleep_parameters_log.loc[:, self._reference_col]
            device_to_scatter = self.sleep_parameters_difference_log
        else:
            reference_to_scatter = self.sleep_parameters.loc[:, self._reference_col]
            device_to_scatter = self.sleep_parameters_difference

        device_to_scatter = device_to_scatter.replace({-inf: nan})

        if device_to_plot is None:
            pass  # all devices are plotted.
        else:  # specifies those devices that the user would like to plot.
            reference_to_scatter = reference_to_scatter.loc[:, device_to_plot]

        if parameter_to_plot is None:
            pass  # all parameters are plotted.
        else:  # specifies those parameters that the user would like to plot.
            device_to_scatter = device_to_scatter.xs(
                parameter_to_plot,
                level='parameter',
                axis=0
            )
            reference_to_scatter = reference_to_scatter.xs(
                parameter_to_plot,
                level='parameter',
                axis=0
            )

        par_plot = self.calculate_parameters(
            reference_to_scatter,
            device_to_scatter,
            ci_level,
            augmentation_factor_ylimits,
            augmentation_factor_xlimits
        )

        sns.set_theme(
            context='notebook',
            style='darkgrid',
            palette='deep',
            font='sans-serif',
            font_scale=2,
            color_codes=True,
            rc=None
        )

        for i in par_plot:  # iterate over parameters
            parameter: str = i[0]
            unit_of_measurement = self._BlandAltmanPlot__unit_of_measurment_to_labels(parameter)

            for device in i[1]:  # iterate over devices

                print(f"{parameter}, {device.get('device_name')}: heteroskedasticity {device.get('heteroskedasticity')}"
                      f" proportional bias {device.get('proportional_bias')}")

                # plotting bland-altman plot.
                joint_plot = JointGrid(
                    dropna=True,
                    ratio=joint_plot_ratio,
                    height=joint_plot_height,
                )

                joint_plot.ax_joint.set_xlim(
                    device.get("x_lim_left"),
                    device.get("x_lim_right")
                )

                scatterplot(
                    x=device.get("reference_to_plot"),
                    y=device.get("params_to_bias"),
                    color='Blue',
                    edgecolor='white',
                    ax=joint_plot.ax_joint
                )

                kdeplot(
                    y=device.get("params_to_bias"),
                    color='Blue',
                    ax=joint_plot.ax_marg_y
                )  # kernel density estimation
                # plotted in the ax_marg_y axis
                # (to the right of ax_joint)

                joint_plot.ax_marg_x.remove()
                # no marginal plot depicted along
                # the x-axis

                sns.regplot(
                    x=device.get("reference_to_plot"),
                    y=device.get("params_to_bias"),  # plotting bias
                    x_ci='ci',
                    n_boot=boot_n_resamples,
                    scatter=False,
                    color='red',
                    ax=joint_plot.ax_joint
                )  # plotting the upper limit of agreement

                sns.regplot(
                    x=device.get("reference_to_plot"),
                    y=device.get("params_upper_loa"),  # plotting upper loa
                    x_ci=ci_bootstrapping,
                    n_boot=boot_n_resamples,
                    scatter=False,
                    color='gray',
                    ax=joint_plot.ax_joint
                )

                sns.regplot(
                    x=device.get("reference_to_plot"),
                    y=device.get("params_lower_loa"),  # plotting lower loa
                    x_ci=ci_bootstrapping,
                    n_boot=boot_n_resamples,
                    scatter=False,
                    color='gray',
                    ax=joint_plot.ax_joint
                )

                # setting every other parameter
                # of joint_plot

                joint_plot.ax_joint.set_xticks(
                    joint_plot.ax_joint.get_xticks().round(0),
                    fontsize=axis_ticks_fontsize
                )
                joint_plot.ax_joint.set_yticks(
                    joint_plot.ax_joint.get_yticks().round(0),
                    fontsize=axis_ticks_fontsize
                )

                joint_plot.ax_joint.grid(
                    visible=True,
                    which='major',
                    axis='both'
                )
                joint_plot.ax_joint.set_ylabel(
                    f"Δ({device.get('device_name')} - {self._reference_col}) "
                    f"({unit_of_measurement})",
                    fontsize=axis_label_fontsize
                )
                joint_plot.ax_joint.set_xlabel(
                    f'{self._reference_col} ({unit_of_measurement})',
                    fontsize=axis_label_fontsize
                )

                # if par_name == "W":
                #     par_name_plot = "Wake"
                # elif par_name == "R":
                #     par_name_plot = "REM"
                # else:
                #     par_name_plot = par_name

                joint_plot.ax_joint.set_title(
                    f'{parameter}',
                    fontsize=title_fontsize
                )

                joint_plot.ax_joint.set_ylim(
                    -device.get("y_lim"),
                    device.get("y_lim")
                )  # ylim forced around y=0

                joint_plot.ax_joint.autoscale(
                    enable=True,
                    axis='x',
                    tight=True
                )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self._savepath_bland_altman_plots,
                        f"{device.get('device_name')}_{parameter}.png"
                    ),
                    dpi=plot_dpi
                )
                plt.show()

        return None


    @staticmethod
    def __unit_of_measurment_to_labels(par_name: Text):
        """
        Automatically detects the type of parameter passed.

        it's used to set the correct unit of measurment to be
        displayed between parentheses in x ad y axis' labels.
        Args:
            par_names: Text
            par_names assigned in bland_altman_plot

        Returns:
            unit_of_measurment: Text
            passed later to xlabels and ylabels
            to print the unit of measurment between
            parentheses.

        """
        if par_name == 'TST' or par_name == 'WASO' or par_name == 'SOL':
            unit_of_measurement = 'min'
        elif par_name == 'SE':
            unit_of_measurement = '%'
        else:  # called for sleep stages.
            unit_of_measurement = 'min'

        return unit_of_measurement


