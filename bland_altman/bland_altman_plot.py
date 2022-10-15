from math import inf
from itertools import chain
from typing import Text, Tuple

import pandas as pd
from numpy import nan, array, nanstd, arange

import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import JointGrid, scatterplot, kdeplot

from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan


class BlandAltmanPlot:

    def bland_altman_plot(
            self,
            log_transformed=False,
            parameter_to_plot=None,
            device_to_plot=None,
            x_axis_mean=None,
            ci_bootstrapping='ci',
            n_bootstrapping=10000,
            linewidth_lines=3,
            joint_plot_ratio=6,
            joint_plot_height=10,
            confidence_level=0.95,
            augmenting_factor_ylimits=0.3
    ):

        device_to_scatter = self.sleep_parameters_difference
        reference_to_scatter = self.sleep_parameters.loc[:, self._reference_col]
        if log_transformed is True:
            bland_parameters = self.bland_bias_loa_log
        else:
            bland_parameters = self.bland_bias_loa

        device_to_scatter = device_to_scatter.replace({-inf: nan})

        y_limits = self._BlandAltmanPlot__y_limits_calculation(
            device_to_scatter,
            bland_parameters,
            augmenting_factor_ylimits
        )
        # used later to force y_limits around the 0 axis.

        if device_to_plot is None:
            pass  # all devices are plotted.
        else:  # specifies those devices that the user would like to plot.
            reference_to_scatter = reference_to_scatter.loc[:, device_to_plot]

        if parameter_to_plot is None:
            pass  # all parameters are plotted.
        else:  # specifies those parameters that the user would like to plot.

            device_to_scatter = device_to_scatter.xs(parameter_to_plot, level='parameter', axis=0)
            reference_to_scatter = reference_to_scatter.xs(parameter_to_plot, level='parameter', axis=0)

        # start plotting each sleep stage for each device.
        # the outer for loop-loop iterates over each
        # column of device_to_scatter, which contains
        # data relative to a single device.
        # the inner for-loop then plots for each
        # stage the relative bland-altman plot.
        # y-metrics of bland-altman plots will be
        # forced to be the same for every stage
        # and for every participant.

        sns.set_style("darkgrid")

        to_append = []
        for i in device_to_scatter.items():
            dev_name = i[0]
            dev_to_plot = i[1]  # gets the series resulting from iteritems

            for j in dev_to_plot.groupby(level=1):  # used to iterate over parameters
                par_name = j[0]

                unit_of_measurement = self._BlandAltmanPlot__unit_of_measurment_to_labels(par_name)
                # parameter processed is automatically detected. it's unit
                # measurement is here assigned to be depicted in x and y labels.

                par_to_plot = j[1].droplevel(level=1, axis=0)
                # parameter to plot.
                ref_to_plot = reference_to_scatter.xs(par_name, level='parameter')

                if x_axis_mean is True:
                    ref_to_plot = pd.concat([par_to_plot, ref_to_plot], axis=1).mean(axis=1)
                else:
                    pass

                proportional_bias, heteroskedasticity = \
                    self._BlandAltmanPlot__proportional_bias_heteroskedasticity_testing(
                    par_to_plot,
                    ref_to_plot,
                    confidence_level
                )

                if proportional_bias is False:
                    pass
                else:
                    constant_propbias, x1_propbias, std_resid_propbias = proportional_bias
                    proportional_bias = True

                if heteroskedasticity is False:
                    pass
                else:
                    constant_hetersked, x1_hetersked, results_hetersked = heteroskedasticity
                    heteroskedasticity = True

                # plotting bland-altman plot.
                joint_plot = JointGrid(
                    dropna=True,
                    ratio=joint_plot_ratio,
                    height=joint_plot_height,
                )

                scatterplot(
                    x=ref_to_plot,
                    y=par_to_plot,
                    color='Blue',
                    edgecolor='white',
                    ax=joint_plot.ax_joint
                )

                kdeplot(
                    y=par_to_plot,
                    color='Blue',
                    ax=joint_plot.ax_marg_y
                )  # kernel density estimation
                # plotted in the ax_marg_y axis
                # (to the right of ax_joint)

                joint_plot.ax_marg_x.remove()
                # no marginal plot depicted along
                # the x-axis

                # bias and its ci intervals.
                bias = [
                    bland_parameters.loc[(dev_name, 'bias'), par_name],
                    bland_parameters.loc[(dev_name, 'bias_ci_upper_ci'), par_name],
                    bland_parameters.loc[(dev_name, 'bias_ci_lower_ci'), par_name]
                ]

                if proportional_bias is False:  # standard bland-altman plot.
                    # bool type is returned only if there is no proportional bias
                    # in difference

                    count = 0
                    for k in bias:
                        if count == 0:
                            ls_plot = '-'  # solid line used to plot bias
                        else:
                            ls_plot = '--'  # dashed line used to plot confidence intervals

                        joint_plot.ax_joint.axhline(
                            k,
                            color='red',
                            ls=ls_plot,
                            linewidth=linewidth_lines
                        )  # plotting the upper limit of agreement
                        count += 1
                    del ls_plot

                else:
                    sns.regplot(
                        x=ref_to_plot,
                        y=par_to_plot,
                        x_ci='ci',
                        n_boot=n_bootstrapping,
                        scatter=False,
                        color='red',
                        ax=joint_plot.ax_joint
                    )  # plotting the upper limit of agreement

                # It follows plotting of loas and their confidence intervals.
                if log_transformed is True:
                    x_loa = ref_to_plot
                    if proportional_bias is True:
                        bias_to_bland = constant_propbias
                    else:
                        bias_to_bland = None

                    loa_ci_to_plot = [
                        self._BlandAltmanPlot__loa_ci_extraction(
                            k,
                            dev_name,
                            par_name,
                            bland_parameters,
                            x_loa,
                            bias_to_bland
                        )
                        for k in ['lower', 'upper']
                    ]
                    loa_ci_to_plot = list(chain.from_iterable(loa_ci_to_plot))
                    # getting the lower limit of agreement,
                    # along with its lower and upper confidence
                    # interval

                    count = 0
                    for k in loa_ci_to_plot:
                        sns.regplot(
                            x=ref_to_plot,
                            y=k,
                            x_ci=ci_bootstrapping,
                            n_boot=n_bootstrapping,
                            scatter=False,
                            color='gray',
                            ax=joint_plot.ax_joint
                        )
                        count += 1
                    del k, loa_ci_to_plot

                else:  # data is not log-transformed. Loas to be plotted
                    # are modified according to the presence of any bias and
                    # heteroskedasticity of in the sample.

                    if proportional_bias is False and heteroskedasticity is False:
                        x_loa = None
                        # it generates an array, spanning from the left to the
                        # right limits of joint_plot.ax_joint. for further details
                        # on this if-statement please check the docstring of loa_ci_extraction.

                        loa_ci_to_plot = [
                            self._BlandAltmanPlot__loa_ci_extraction(k, dev_name, par_name, bland_parameters, x_loa)
                            for k in ['lower', 'upper']
                        ]
                        loa_ci_to_plot = list(chain.from_iterable(loa_ci_to_plot))
                        # getting the lower limit of agreement,
                        # along with its lower and upper confidence
                        # interval

                        count = 0
                        for k in loa_ci_to_plot:
                            if count == 0 or count == 3:
                                ls_plot = '-'
                            else:
                                ls_plot = '--'

                            joint_plot.ax_joint.axhline(
                                k,
                                color='gray',
                                ls=ls_plot,
                                linewidth=linewidth_lines
                            )  # plotting the lower limit of agreement
                            count += 1

                        del ls_plot, count, loa_ci_to_plot

                    else:
                        if proportional_bias is True and heteroskedasticity is False:
                            y_loa_to_plot = [
                                par_to_plot + k * nanstd(par_to_plot)
                                for k in [1.96, -1.96]
                            ]

                        elif proportional_bias is False and heteroskedasticity is True:
                            y_loa_to_plot = [
                                bias + (k * results_hetersked)
                                for k in [2.46, -2.46]
                            ]

                        elif proportional_bias is True and heteroskedasticity is True:
                            y_loa_to_plot = [
                                (par_to_plot + k * nanstd(par_to_plot)) + (k * results_hetersked)
                                for k in [2.46, -2.46]
                            ]

                        for k in y_loa_to_plot:
                            sns.regplot(
                                x=ref_to_plot,
                                y=k,
                                x_ci=ci_bootstrapping,
                                n_boot=n_bootstrapping,
                                scatter=False,
                                color='gray',
                                ax=joint_plot.ax_joint
                            )
                        del k, y_loa_to_plot

                # setting every other parameter
                # of joint_plot

                joint_plot.ax_joint.set_xticks(
                    joint_plot.ax_joint.get_xticks().round(0),
                    fontsize='xx-large'
                )
                joint_plot.ax_joint.grid(
                    visible=True,
                    which='major',
                    axis='both'
                )
                joint_plot.ax_joint.set_ylabel(
                    f'Δ({ref_to_plot.name} - {par_to_plot.name}) ({unit_of_measurement})',
                    fontsize='xx-large'
                )
                joint_plot.ax_joint.set_xlabel(
                    f'{ref_to_plot.name} ({unit_of_measurement})',
                    fontsize='xx-large'
                )
                joint_plot.ax_joint.set_title(
                    f'{par_name}',
                    fontsize='xx-large'
                )

                joint_plot.ax_joint.set_ylim(
                    -y_limits[par_name],
                    y_limits[par_name]
                )  # ylim forced around y=0

                joint_plot.ax_joint.autoscale(
                    enable=True,
                    axis='x',
                    tight=True
                )

                plt.tight_layout()
                plt.show()

        return to_append

    @staticmethod
    def __y_limits_calculation(
            device_to_scatter_in: pd.DataFrame,
            bland_parameters_in:pd.DataFrame,
            augmenting_factor_ylimits: float
    ):
        """
        Calculates the value to be assigned as upper and lower y-limits
        in Bland-Altman plots.

        Called in BlandAltmanPlot.bland_altman_plot
        Args:
            device_to_scatter_in: pd.DataFrame
                device_to_scatter
            bland_parameters_in: pd.DataFrame
                bland_parameters
            augmenting_factor_ylimits:
                used to enlarge the ylimits


        Returns:
            y_limits: int
            y_limits to be applied. Note that
            is returned only one absolute-value
            integer. When setting y-axis' limits,
            the positive and negative values of
            this absolute value will be set. This
            procedure makes the y-axis forced around the 0.
        """

        def y_limits_calculation_device_parameters(device_to_scatter_in):
            par_name = device_to_scatter_in[0]
            to_min_max = device_to_scatter_in[1]
            min_val = to_min_max.min()
            max_val = to_min_max.max()

            try:
                min_val = min_val.min()
                max_val = max_val.max()

            except AttributeError:  # 'float' object has no attribute 'max'? 'max'
                # used to manage the case in which the max value is calculated don
                # bland_parameters_in
                pass

            y_axis_lim = max(abs(min_val), abs(max_val))

            return pd.Series(y_axis_lim, index=[par_name])

        y_limits_device = pd.concat(
            map(
                y_limits_calculation_device_parameters,
                device_to_scatter_in.groupby(level='parameter', axis=0)
            )
        )
        y_limits_parameters = pd.concat(
            map(
                y_limits_calculation_device_parameters,
                bland_parameters_in.items()
            )
        )
        y_limits = pd.concat([y_limits_device, y_limits_parameters], axis=1)
        y_limits = y_limits.max(axis=1).round(0)
        y_limits = y_limits + y_limits * augmenting_factor_ylimits  # adding 10% to allow a better
        # visualization of limits.
        return y_limits

    @staticmethod
    def __proportional_bias_heteroskedasticity_testing(
            par_to_plot_in: pd.Series,
            ref_to_plot_in: pd.Series,
            conf_level: float
    ):
        """
        Tests for proportionality in bias nad heteroskedasticity.
        Args:
            par_to_plot_in: pd.Series
                par_to_plot
            ref_to_plot_in: pd.Series
                ref_to_plot
            conf_level: float
                conf_level

        Returns: Tuple[bool, bool]
            proportional_bias: bool or List
                if proportional_bias is a boolean, it means that
                there is no proportional_bias in the sample under
                study. In this case, proportional_bias equals False.
                If a list is returned, there is proportional bias in the
                difference. The list returned contains the necessary to
                model the bias and loas according to Bland-Altman (1999).
                In particular, the first element of the list is the b0 (intercept),
                the second element is the slope while the third
                is the standard deviation of the residuals.
            heteroskedasticity: bool or List
                if heteroskedasticity is a boolean,
                there is no heteroskedasticity in the sample under
                study. In this case heteroskedasticity equals False.
                If a list is returned, there is heteroskedasticity in the
                difference. The list returned contains the necessary to
                model the bias and loas according to Bland-Altman (1999).
                In particular, the first element of the list is the c0 (intercept),
                the second eleemnt is the slope. The third element is the prediciton
                of the linear regression model fitted on absolute values of residuals.
        """
        par_to_plot_in = par_to_plot_in.dropna()
        ref_to_plot_in = ref_to_plot_in.dropna()

        regmod = OLS(par_to_plot_in, add_constant(ref_to_plot_in))
        # instance of OSL, linear the sum of squared
        # vertical distances is minimized through
        # ordinary least square method.

        results = regmod.fit()
        resid = results.resid
        exog = results.model.exog
        params = results.params

        alpha_to_ci = 1 - conf_level

        lower_confint = results.conf_int(alpha=alpha_to_ci)[0][0]
        higher_confint = results.conf_int(alpha=alpha_to_ci)[1][0]
        if lower_confint > 0 or higher_confint < 0:  # proportional_bias was
            # found. the function will return parameters to plot
            # bias and loas taking into account the proportional_bias.

            summary = results.summary()
            summary = summary.tables[1].data

            constant = float(summary[1][1])  # b0
            x1 = float(summary[2][1])  # b1
            std_resid = nanstd(resid)  # standard deviation of residuals.

            proportional_bias = [constant, x1, std_resid]
        else:
            proportional_bias = False

        heteroskedasticity = het_breuschpagan(resid, exog)
        if heteroskedasticity[3] < 0.05:  # gets the p-value

            regmod_hetersked = OLS(abs(resid), add_constant(ref_to_plot_in))
            # instance of OSL, linear the sum of squared
            # vertical distances is minimized through
            # ordinary least square method.

            results_hetersked = regmod_hetersked.fit()

            summary_hetersked = results_hetersked.summary()
            summary_hetersked = summary_hetersked.tables[1].data

            constant_hetersked = float(summary_hetersked[1][1])  # c0
            x1_hetersked = float(summary_hetersked[2][1])  # c1

            prediction_hetersked = results_hetersked.predict()

            heteroskedasticity = [constant_hetersked, x1_hetersked, prediction_hetersked]

        else:
            heteroskedasticity = False

        return proportional_bias, heteroskedasticity

    @staticmethod
    def __loa_ci_extraction(
            loa_to_extract: Text,
            dev_name_in: Text,
            par_name_in: Text,
            bland_parameters_in: pd.DataFrame,
            x_loa_in: array = None,
            constant_propbias: float = None
    ) -> Tuple[array, array, array]:
        """
        Extracts limits of agreement along with
        their confidence interval from
        BlandAltmanPlot.bland_altman_plot
        bland_parameters local variable.

        Fuction implemented only to improve
        the readability of BlandAltmanPlot.bland_altman_plot
        method.

        Args:
            loa_to_extract: Text
            Either  'upper' or 'lower'.
            dev_name_in: Text
                dev_name in BlandAltmanPlot.bland_altman_plot.
            par_name_in: Text
                par_name in BlandAltmanPlot.bland_altman_plot.
            bland_parameters_in:
                bland_parameters in BlandAltmanPlot.bland_altman_plot.
            x_loa_in:
                x_loa in BlandAltmanPlot.bland_altman_plot
                (if statement that checks if log transformation
                should be applied to data).

        Returns:
            y_loa: np.array
                limits of agreement
            lower_ci: np.array
                lower confidence interval
            upper_ci: np.array
                upper confidence interval
        """
        y_loa = bland_parameters_in.loc[(dev_name_in, f'{loa_to_extract}_loa'), par_name_in]
        lower_ci = bland_parameters_in.loc[(dev_name_in, f'{loa_to_extract}_loa_lower_ci'), par_name_in]
        upper_ci = bland_parameters_in.loc[(dev_name_in, f'{loa_to_extract}_loa_upper_ci'), par_name_in]

        if x_loa_in is None:
            pass
        else:
            if constant_propbias is None:
                bias_in = bland_parameters_in.loc[(dev_name_in, 'bias'), par_name_in]
            else:
                bias_in = constant_propbias

            y_loa = y_loa * x_loa_in + bias_in  # adding up the bias as intercept
            lower_ci = lower_ci * x_loa_in + bias_in  # adding up the bias as intercept
            upper_ci = upper_ci * x_loa_in + bias_in  # adding up the bias as intercept
        return y_loa, lower_ci, upper_ci

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
            unit_of_measurement = 'count'

        return unit_of_measurement
