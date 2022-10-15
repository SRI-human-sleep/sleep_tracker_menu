from math import inf
from typing import Text

import pandas as pd
from numpy import nan, nanmean, nanstd

from utils.confidence_interval import confidence_interval_calculation, loa_ci_calculation, bias_ci_calculation


class BlandAltman:
    back_transform_no_log = False
    back_transform_log = True

    @property
    def bland_bias_loa(
            self,
            ci_level: float = None,
            ci_bootstrapping: bool = None,
            boot_method: Text = None,
            boot_n_resamples: int = None
    ) -> pd.DataFrame:
        """
        Calculates the bias and the upper and lower limit of agreement,
        along with their limits of agreement on NON-log transformed differences
        Args:
            self
            ci_level: float
                lambda for confidence interval.
                The default is 0.95.
            ci_bootstrapping: bool
                if True, ci is calculated through bootstrapping,
                otherwise it's calculated on a t distribution.
                The default is None.
            boot_method: Text
                bootstrap's method through which the
                ci should be calculated.
                The default is None.
            boot_n_resamples: int
                number of resamples to bootstrap.
                Ignored if ci_bootstrapping is
                False.
                The default is None.

        Returns:
            bland_altman: pd.DataFrame
            metrics used to plot the Bland-Altman plot.

        """

        if ci_level is None:
            ci_level = self.ci_level
            # if None, alpha is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        if ci_bootstrapping is None:
            ci_bootstrapping = self.ci_bootstrapping
            # if None, alpha is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        if boot_method is None:
            boot_method = self.boot_method
            # if None, boot_method is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        if boot_n_resamples is None:
            boot_n_resamples = self.boot_n_resamples
            # if None, boot_n_resamples is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        bland_altman = pd.concat(
            map(
                lambda x:
                self._BlandAltman__bias_loa_ci_calculation(
                    diff_bland=x[1],
                    parameter_to_bland=x[0],
                    ci_level=ci_level,
                    back_transform=self.back_transform_no_log,
                    ci_bootstrapping=ci_bootstrapping,
                    boot_method=boot_method,
                    boot_n_resamples=boot_n_resamples
                ),
                self.sleep_parameters_difference.groupby(level=1)
            ),
            axis=1
        )
        return bland_altman

    @property
    def bland_bias_loa_log(
            self,
            ci_level: float = None,
            ci_bootstrapping: bool = None,
            boot_method: Text = None,
            boot_n_resamples: int = None
    ) -> pd.DataFrame:
        """
        Calculates the bias and the upper and lower limit of agreement,
        along with their limits of agreement on LOG transformed differences
        Args:
            self
            ci_level: float
                lambda for confidence interval.
                The default is None.
            ci_bootstrapping: bool
                if True, ci is calculated through bootstrapping,
                otherwise it's calculated on a t distribution.
                The default is None.
            boot_method: Text
                bootstrap's method through which the
                ci should be calculated.
                The default is None.
            boot_n_resamples: int
                number of resamples to bootstrap.
                Ignored if ci_bootstrapping is
                False.
                The default is None.

        Returns:
            bland_altman: pd.DataFrame
            metrics used to plot the Bland-Altman plot.

        """
        if ci_level is None:
            ci_level = self.ci_level
            # if None, alpha is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        if ci_bootstrapping is None:
            ci_bootstrapping = self.ci_bootstrapping
            # if None, alpha is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        if boot_method is None:
            boot_method = self.boot_method
            # if None, boot_method is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        if boot_n_resamples is None:
            boot_n_resamples = self.boot_n_resamples
            # if None, boot_n_resamples is set to be the same as specified
            # when the class is constructed.
        else:
            pass

        bland_altman = pd.concat(
            map(
                lambda x:
                self._BlandAltman__bias_loa_ci_calculation(
                    diff_bland=x[1],
                    parameter_to_bland=x[0],
                    ci_level=ci_level,
                    back_transform=self.back_transform_log,
                    ci_bootstrapping=ci_bootstrapping,
                    boot_method=boot_method,
                    boot_n_resamples=boot_n_resamples
                ),
                self.sleep_parameters_difference_log.groupby(level=1)
            ),
            axis=1
        )
        return bland_altman

    @staticmethod
    def __bias_loa_ci_calculation(
            diff_bland: pd.DataFrame,
            parameter_to_bland: Text,
            ci_level: float = 0.95,
            back_transform: bool = False,
            ci_bootstrapping: bool = False,
            boot_method: Text = "basic",
            boot_n_resamples: int = 10000
    ) -> pd.DataFrame:
        """
        Calculates the bias and the upper and lower limit of agreement,
        along with their limits of agreement.
        Args:
            diff_bland: pd.DataFrame
                Each column represents the difference
                between the device under study and the
                reference.
            parameter_to_bland: Text
                name of the parameter processed by the function. Used
                as level 1 of columns' multiindex.
            ci_level: float
                lambda for confidence interval.
                The default is 0.95.
            back_transform: bool
                if True, data are backtransformed to the
                non-log space. The default is False.
            ci_bootstrapping: bool
                if True, confidence interval is estimated
                through bootstrapping.
                The default is False.
            boot_method: Text
                bootstrap's method through which the
                ci should be calculated.
                The default is 'basic'.
            boot_n_resamples: int
                number of resamples to bootstrap.
                Ignored if ci_bootstrapping is
                False. The default is 10,000

        Returns:
            bland_altman: pd.DataFrame
            metrics used to plot the Bland-Altman plot.

        """
        diff_bland = diff_bland.replace({-inf: nan})

        # bias and bias' ci calculation
        bias = diff_bland.mean()
        bias.name = 'bias'
        std = diff_bland.std()

        if ci_bootstrapping is True:
            bias_ci = pd.concat(
                map(
                    lambda x: confidence_interval_calculation(
                        to_ci=pd.DataFrame(x[1]),
                        stage_device_name=x[0],
                        function_to_ci=nanmean,
                        return_annot_df=False,
                        ci_level=ci_level,
                        ci_bootstrapping=ci_bootstrapping,
                        boot_method=boot_method,
                        boot_n_resamples=boot_n_resamples
                    ),
                    diff_bland.iteritems()
                ),
                axis=0
            )
        else:
            bias_ci = bias_ci_calculation(
                bias=bias,
                std=std,
                length_of_array=diff_bland.size,
                ci_level=ci_level
            )

        bias_ci.columns = [f"bias_ci_{i}" for i in bias_ci.columns]

        # calculation of limits of agreement
        upper_loa = bias + 1.96 * std
        upper_loa.name = "upper_loa"
        lower_loa = bias - 1.96 * std
        lower_loa.name = "lower_loa"

        if ci_bootstrapping is True:
            upper_loa_ci = pd.concat(
                map(
                    lambda x: confidence_interval_calculation(
                        to_ci=pd.DataFrame(x[1]),
                        stage_device_name=x[0],
                        function_to_ci=lambda y: nanmean(y) + 1.96 * nanstd(y),
                        ci_level=ci_level,
                        ci_bootstrapping=ci_bootstrapping,
                        boot_method=boot_method,
                        boot_n_resamples=boot_n_resamples
                    ),
                    diff_bland.iteritems()
                )
            )
        else:
            upper_loa_ci = loa_ci_calculation(
                loa=upper_loa,
                bias=bias,
                std=std,
                length_of_array=diff_bland.size,
                ci_level=ci_level
            )
        upper_loa_ci.columns = [f"upper_loa_{i}" for i in upper_loa_ci.columns]

        if ci_bootstrapping is True:
            lower_loa_ci = pd.concat(
                map(
                    lambda x: confidence_interval_calculation(
                        to_ci=pd.DataFrame(x[1]),
                        stage_device_name=x[0],
                        function_to_ci=lambda y: nanmean(y) - 1.96 * nanstd(y),
                        ci_level=ci_level,
                        ci_bootstrapping=ci_bootstrapping,
                        boot_method=boot_method,
                        boot_n_resamples=boot_n_resamples
                    ),
                    diff_bland.iteritems()
                )
            )
        else:
            lower_loa_ci = loa_ci_calculation(
                loa=upper_loa,
                bias=bias,
                std=std,
                length_of_array=diff_bland.size,
                ci_level=ci_level
            )
        lower_loa_ci.columns = [f'lower_loa_{i}' for i in lower_loa_ci.columns]
        # back transformation of log-transformed data.
        # applied only if back_transform is True

        if back_transform is True:
            bias = 2 * (10 ** bias - 1) / (10 ** bias + 1)
            bias_ci = 2 * (10 ** bias_ci - 1) / (10 ** bias_ci + 1)

            lower_loa = 2 * (10 ** lower_loa - 1) / (10 ** lower_loa + 1)
            upper_loa = 2 * (10 ** upper_loa - 1) / (10 ** upper_loa + 1)

            lower_loa_ci = 2 * (10 ** lower_loa_ci - 1) / (10 ** lower_loa_ci + 1)
            upper_loa_ci = 2 * (10 ** upper_loa_ci - 1) / (10 ** upper_loa_ci + 1)
        else:
            pass

        # generating the output
        bland_altman = pd.concat(
            [
                bias,
                lower_loa,
                upper_loa,
                bias_ci,
                lower_loa_ci,
                upper_loa_ci
            ],
            axis=1
        )
        bland_altman.columns = pd.MultiIndex.from_product(
            [[parameter_to_bland], list(bland_altman.columns)],
            names=["parameter", "bland_altman_metric"]
        )
        bland_altman = bland_altman.stack()
        return bland_altman
