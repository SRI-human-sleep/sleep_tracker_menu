from typing import Callable, Text, Tuple

import pandas as pd
from numpy import nanmean
from scipy.stats import bootstrap, sem, t


def calculate_t_ci(
        function_to_ci: Callable,
        data_to_ci: pd.Series,
        ci_level: float
):
    """
    Used to calculate the t distribution based
    confidence interval.
    It's called in confidence_interval_calculation function.
    Args:
        function_to_ci: Callable
        data_to_ci: pd.Series
        ci_level: float

    Returns:
        low_ci: float
        high_ci: float

    """

    data_to_ci = data_to_ci.to_list()
    parameter_to_ci = function_to_ci(data_to_ci)
    standard_error = sem(data_to_ci, nan_policy="omit")
    ci = standard_error * t.ppf((1 + ci_level) / 2., len(data_to_ci) - 1)
    low_ci = parameter_to_ci - ci

    high_ci = parameter_to_ci + ci
    return low_ci, high_ci


def confidence_interval_calculation(
        to_ci: pd.DataFrame,
        stage_device_name: Text,
        function_to_ci: Callable = nanmean,
        return_annot_df: bool = False,
        ci_level: float = 0.95,
        digit: int = 2,
        ci_bootstrapping: bool = False,
        boot_method: Text = "basic",
        boot_n_resamples: int = 100000
) -> Tuple:

    """
    Calculates the confidence interval (ci for short).
    It allows to calculate the ci in different
    methods.

    Args:
        to_ci: pd.DataFrame
            Dataframe to confidence interval
        stage_device_name: Text
            Stage or device on which data the ci is calculated.
            Argument named after the fact that in bland-altman functions,
            the CI is calculated on single device, while in performance
            metrics the function is applied to every single sleep stage.
        function_to_ci: Callable
            callable of moment.
            The default is nanmean.
        return_annot_df: bool
            if true, the function returns a Tuple
            having as first element the ci as float,
            and as second element the ci formatted as
            string. The DataFrame formatted as string
            is passed to annot argument in seaborn heatmap
            functions.
            if false, only the ci interval as float is returned.
            The default is False.
        ci_level: float
            lambda (confidence level) for ci.
            The default is 0.95.
        digit: int
            digit for rounding.
            The default is 2.
        ci_bootstrapping: bool
            if True, ci is calculated through bootstrapping.
            The default is False
        boot_method: Text
            type of bootstrapping applied.
            Supported: 'percentile', 'basic', 'BCa'.
            The default is 'basic'.
        boot_n_resamples: int
            number of resamples for bootstrapping.
            Ignored if ci_boostrapping is false.
            The default is 10,000.

    Returns:
        ci_output: pd.DataFrame
        see return_annot_df in Args for further
        details on output.

    """
    if ci_bootstrapping is False:

        ci = to_ci.apply(
            lambda x: calculate_t_ci(
                function_to_ci=function_to_ci,
                data_to_ci=x,
                ci_level=ci_level
            ),
            axis=0
        ).transpose()
        ci.columns = ['lower_ci', 'upper_ci']

    else:
        ci = list(
            map(
                lambda x:
                (x[0],
                 bootstrap(
                     data=[x[1]],
                     statistic=function_to_ci,
                     vectorized=False,
                     n_resamples=boot_n_resamples,
                     batch=None,
                     axis=0,
                     confidence_level=ci_level,
                     method=boot_method,
                     random_state=None
                 )
                 ),
                to_ci.iteritems()
            )
        )
        index = pd.Series(map(lambda x: x[0], ci))
        low_ci = pd.Series(map(lambda x: x[1].confidence_interval.low, ci))
        high_ci = pd.Series(map(lambda x: x[1].confidence_interval.high, ci))
        ci = pd.concat([low_ci, high_ci], axis=1)
        ci.index = index
        ci.columns = ["lower_ci", "upper_ci"]

    ci = round(ci, digit)

    if return_annot_df is False:
        ci_output = ci
    else:
        ci = ci.astype(str)
        ci_output = '[' + ci["lower_ci"]
        ci_output = ci_output.add(', ')
        ci_output = ci_output.add(ci["upper_ci"])
        ci_output = pd.DataFrame(ci_output.add(']')).transpose()
        ci_output.index = [stage_device_name]

    return ci_output
