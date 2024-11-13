import os
from math import inf

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
        Generate Bland-Altman plots with confidence intervals, allowing for bias and limits of agreement
        visualization across specified parameters and devices.

        This function produces Bland-Altman plots to visualize agreement between sleep-tracking devices
        and reference data, following the Bland and Altman method (1999). The plots allow for device-by-device
        comparison, confidence interval adjustments, and display settings.

        Parameters
        ----------
        log_transform : bool, optional
            Whether to apply log transformation to data before plotting. Defaults to False.

        parameter_to_plot : list[str], optional
            list of parameters to include in the plots. If None, all available parameters are plotted.

        device_to_plot : list[str], optional
            list of device names to plot. If None, all devices are included.

        ci_level : float, optional
            Confidence level for interval estimation, overriding the class default if specified.

        title_fontsize : int, optional
            Font size for plot titles. Default is 10.

        axis_label_fontsize : int, optional
            Font size for axis labels. Default is 11.

        axis_ticks_fontsize : int, optional
            Font size for axis ticks. Default is 13.

        ci_bootstrapping : str, optional
            Method to calculate confidence intervals using bootstrapping. If None, the default method from
            the class configuration is used.

        boot_n_resamples : int, optional
            Number of bootstrap resamples for confidence interval calculations, if `ci_bootstrapping`
            is specified. If None, uses the class-configured number of resamples.

        joint_plot_ratio : int, optional
            Ratio of joint plot area height to marginal plot height. Default is 6.

        joint_plot_height : int, optional
            Height of the joint plot figure. Default is 10.

        augmentation_factor_ylimits : float, optional
            Factor to expand y-axis limits based on the range of plotted values, calculated as:
            `max_value_y * (1 + augmentation_factor_ylimits)`. Default is 0.3.

        augmentation_factor_xlimits : float, optional
            Factor to expand x-axis limits (currently unsupported). Default is 0.1.

        plot_dpi : int, optional
            DPI for saved plot images. If None, uses the class-defined DPI setting.

        Returns
        -------
        None
            The function saves Bland-Altman plots to a specified directory and displays each plot
            with the defined configuration.

        Notes
        -----
        - Plots confidence intervals based on Bland and Altman’s method, visualizing both proportional bias
          and heteroskedasticity where applicable.
        - If `log_transform` is True, data is transformed logarithmically, particularly useful for data
          with skewed distributions.
        - `augmentation_factor_ylimits` enables flexible expansion of the y-axis, aiding visual comparison
          across devices by increasing the y-limits proportionally.
        - Confidence intervals are drawn based on either standard methods or bootstrapping (if specified),
          depending on `ci_bootstrapping` and `boot_n_resamples`.

        Example
        -------
        >>> iclass.bland_altman_plot()
        """

        print('Generating Bland-Altman plots')

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

        par_plot = self.calculate_bland_altmann_parameters(
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
                    list(map(int, joint_plot.ax_joint.get_xticks())),
                    list(map(lambda x: str(int(x)), joint_plot.ax_joint.get_xticks())),
                    fontsize=axis_ticks_fontsize
            )

                joint_plot.ax_joint.set_yticks(
                    list(map(int, joint_plot.ax_joint.get_yticks())),
                    list(map(lambda y: str(int(y)), joint_plot.ax_joint.get_yticks())),
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
    def __unit_of_measurment_to_labels(par_name: str):
        """
        Automatically detects the type of parameter passed.

        it's used to set the correct unit of measurment to be
        displayed between parentheses in x ad y axis' labels.
        Args:
            par_names:str
            par_names assigned in bland_altman_plot

        Returns:
            unit_of_measurment:str
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


