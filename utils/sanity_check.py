from warnings import warn
from itertools import chain
from typing import Dict, List, Text
import pandas as pd


def sanity_check(
        file_in_process: pd.DataFrame,
        sleep_scoring: Dict,
        reference_col: Text,
        device_col: List[Text],
        drop_wrong_labels=True
):
    """
    Finds any value that is not in self.sleep_scoring.

    If such a value is found, it means that during any procedure
    that happened before feeding the dataframe to this pipeline,
    an error has been committed.

    To help the researcher, a dataframe contaning
    information about the wrong sleep scores is returned.

    Parameters
    ----------
    file_in_process: pd.DataFrame
    sleep_scoring: Dict
    reference_col: Text
    device_col: List[Text]
    drop_wrong_labels

    Returns
    -------
    file_in_process: pd.DataFrame
    wrong_rows: pd.DataFrame
    """

    sleep_scoring = list(
        chain.from_iterable(map(lambda x: x[1], sleep_scoring.items()))
    )

    ref_col_wrong_labels = set(file_in_process.loc[:, reference_col])
    ref_col_wrong_labels = [i for i in ref_col_wrong_labels if i not in sleep_scoring]

    dev_col_wrong_labels = list(map(lambda x: set(file_in_process.loc[:, x]), device_col))
    dev_col_wrong_labels = [i for i in dev_col_wrong_labels if i not in dev_col_wrong_labels]

    if drop_wrong_labels is True:
        if ref_col_wrong_labels:
            warn(f"The following not-expected labels have been found in {reference_col}: {ref_col_wrong_labels}")

            index_of_wrong_labels = file_in_process.loc[:, reference_col].isin(ref_col_wrong_labels)
            index_of_wrong_labels = index_of_wrong_labels.where(index_of_wrong_labels != False).dropna(how='all')
            index_of_wrong_labels = index_of_wrong_labels.index
            wrong_rows_ref = file_in_process.loc[index_of_wrong_labels, :]
            file_in_process = file_in_process.drop(index=index_of_wrong_labels)
        else:
            wrong_rows_ref = pd.DataFrame()

        if dev_col_wrong_labels:
            warn(f"The following not-expected labels have been found in {device_col}: {dev_col_wrong_labels}")
            wrong_rows_dev = []
            for i in device_col:
                index_of_wrong_labels = file_in_process.loc[:, i].isin(dev_col_wrong_labels)
                index_of_wrong_labels = index_of_wrong_labels.where(index_of_wrong_labels != False).dropna(how='all')
                index_of_wrong_labels = index_of_wrong_labels.index
                wrong_rows_dev.append(file_in_process.loc[index_of_wrong_labels, :])
                file_in_process = file_in_process.drop(index=index_of_wrong_labels)
                del index_of_wrong_labels
                print(wrong_rows_dev)
            wrong_rows_dev = pd.concat(wrong_rows_dev)
            wrong_rows_dev = pd.concat(wrong_rows_dev)
        else:
            wrong_rows_dev = pd.DataFrame()

        wrong_epochs = pd.concat([wrong_rows_ref, wrong_rows_dev], axis=0)

    else:
        wrong_epochs = None

    return file_in_process, wrong_epochs
