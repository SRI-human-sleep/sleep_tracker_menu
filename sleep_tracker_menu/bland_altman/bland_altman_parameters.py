from itertools import repeat

import numpy as np
import pandas as pd
import numpy.typing as npt

from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan


class BlandAltmanParameters:
    def __init__(self):
        self._ci_level = None
        self._augmentation_factor_ylimits = None
        self._augmentation_factor_xlimits = None

    def calculate_bland_altmann_parameters(
            self,
            reference: pd.Series,
            device: pd.DataFrame,
            ci_level: float,
            augmentation_factor_ylimits: float,
            augmentation_factor_xlimits: float,
    ):
        """

        Args:
            reference: pd.Series
            device: pd.DataFrame

        Returns:

        """
        self._ci_level: float = ci_level
        self._augmentation_factor_ylimits: float =\
            augmentation_factor_ylimits
        self._augmentation_factor_xlimits: float =\
            augmentation_factor_xlimits

        to_params = pd.concat([reference, device], axis=1)

        parameters = list(
            map(
                self._calculate_parameters,
                to_params.groupby(level=1)
            )
        )
        return parameters

    def _calculate_parameters(self, to_parameter):
        parameter_name = to_parameter[0]
        data = to_parameter[1]
        reference = data.iloc[:, 0]
        device = data.iloc[:, 1:]

        stats = list(
            map(
                self.correct_for_proportional_bias_and_heteroskedasticity,
                device.items(),
                repeat(reference)
            )
        )
        for i in stats:
            i["parameter_name"] = parameter_name

        return parameter_name, stats

    def correct_for_proportional_bias_and_heteroskedasticity(
            self,
            par_to_plot_in: pd.Series,
            ref_to_plot_in: pd.Series
    ):
        """
        Correct parameters for proportionality to the
        size of measurement and heteroskedasticity.

        They will be used to fit regression lines with
        seaborn regplot.
        Args:
            par_to_plot_in: pd.Series
                The series contains values of a single
                parameter (e.g. TST, SE) for each participant,
                estimated by the device
            ref_to_plot_in: pd.Series
                The series contains values of a single
                parameter (e.g. TST, SE) for each participant,
                estimated by the reference
            conf_level: float
                confidence level

        Returns:
            device_name: str
                header of the device processed
            parameters_to_return: tuple[]

            (proportional_bias, heteroskedasticity)

        """

        device_name: str = par_to_plot_in[0]
        par_to_plot_in: pd.Series = par_to_plot_in[1]

        par_to_plot_in = par_to_plot_in.dropna()
        ref_to_plot_in: pd.Series = ref_to_plot_in.dropna()

        regmod = OLS(par_to_plot_in, add_constant(ref_to_plot_in))
        # instance of OSL, linear the sum of squared
        # vertical distances is minimized through
        # ordinary least square method.

        results = regmod.fit()
        resid: pd.Series = results.resid
        exog: npt.NDArray = results.model.exog
        params: pd.Series = results.params

        alpha_to_ci: float = 1 - self._ci_level

        lower_confint: float = results.conf_int(alpha=alpha_to_ci).iloc[0, 0]
        higher_confint: float = results.conf_int(alpha=alpha_to_ci).iloc[1, 0]

        if lower_confint > 0 or higher_confint < 0:  # proportional_bias was
            # found. the function will return parameters to plot
            # bias and loas taking into account the proportional_bias.

            summary: list[list[str]] = results.summary()
            summary = summary.tables[1].data

            constant: float = float(summary[1][1])  # b0
            x1: float = float(summary[2][1])  # b1
            std_resid: np.float64 = np.nanstd(resid)  # standard deviation of residuals.

            proportional_bias: bool = True
        else:
            proportional_bias: bool = False

        heteroskedasticity: tuple = het_breuschpagan(resid, exog)
        if heteroskedasticity[3] < 0.05:  # gets the p-value

            regmod_hetersked = OLS(abs(resid), add_constant(ref_to_plot_in))
            # instance of OSL, linear the sum of squared
            # vertical distances is minimized through
            # ordinary least square method.

            results_hetersked = regmod_hetersked.fit()

            summary_hetersked: str = results_hetersked.summary()
            summary_hetersked: list[list[str]] = summary_hetersked.tables[1].data

            constant_hetersked: float = float(summary_hetersked[1][1])  # c0
            x1_hetersked: float = float(summary_hetersked[2][1])  # c1

            prediction_hetersked: npt.NDArray = results_hetersked.predict()

            heteroskedasticity: bool = True

        else:
            heteroskedasticity: bool = False

        params_to_bias = par_to_plot_in.copy()

        std = par_to_plot_in.std()
        bias = par_to_plot_in.mean()
        if proportional_bias is True and heteroskedasticity is False:
            params_upper_loa: pd.Series = par_to_plot_in + 1.96 * std
            params_lower_loa: pd.Series = par_to_plot_in - 1.96 * std

        elif proportional_bias is True and heteroskedasticity is False:
            params_upper_loa: pd.Series = par_to_plot_in + 1.96 * std
            params_lower_loa: pd.Series = par_to_plot_in + -1.96 * std

        elif proportional_bias is False and heteroskedasticity is True:
            params_upper_loa: pd.Series = bias + (2.46 * prediction_hetersked)
            params_lower_loa: pd.Series = bias + (-2.46 * prediction_hetersked)

        elif proportional_bias is True and heteroskedasticity is True:
            params_upper_loa: pd.Series =\
                (par_to_plot_in + 2.46 * std) + (2.46 * prediction_hetersked)
            params_lower_loa: pd.Series =\
                (par_to_plot_in + -2.46 * std) + (-2.46 * prediction_hetersked)

        y_lim = pd.concat([params_lower_loa, params_upper_loa], axis=0)

        y_lim = y_lim.max() * (1 + self._augmentation_factor_ylimits)

        x_lim_left = ref_to_plot_in.min()
        x_lim_right = ref_to_plot_in.max()

        to_return = dict(
            device_name=device_name,
            reference_to_plot=ref_to_plot_in,
            params_to_bias=params_to_bias,
            params_upper_loa=params_upper_loa,
            params_lower_loa=params_lower_loa,
            proportional_bias=proportional_bias,
            heteroskedasticity=heteroskedasticity,
            y_lim=y_lim,
            x_lim_left=x_lim_left,
            x_lim_right=x_lim_right
        )

        return to_return
