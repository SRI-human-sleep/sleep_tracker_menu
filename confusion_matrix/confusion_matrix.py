from itertools import repeat
from typing import List, Text, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from confusion_matrix.confusion_matrix_plot import ConfusionMatrixPlot

from utils.confidence_interval import confidence_interval_calculation


class ConfusionMatrix(ConfusionMatrixPlot):

        @property
        def standard_absolute_confusion_matrix(self) -> List[Tuple[Text, pd.DataFrame]]:
            """
            Calculates the absolute confusion matrix.

            Parameters
            ----------
            self

            Returns
            -------
            List[Tuple[Text, pd.DataFrame]]

            confusion matrix for each device

            """
            labels_to_confusion = [self.sleep_scoring.get("Wake")] + self.sleep_scoring.get("Sleep")
            standard_absolute_confusion_matrix = list(
                map(
                    self._ConfusionMatrix__standard_confusion_matrix_each_device,
                    repeat(self.reference),
                    self.device,
                    repeat(labels_to_confusion),
                    repeat(self._savepath_standard_absolute_confusion_matrix_xlsx),
                    repeat(True)
                )
            )
            return standard_absolute_confusion_matrix

        @property
        def standard_normalized_confusion_matrix(self) -> List[Tuple[Text, pd.DataFrame]]:
            """
            Calculates the absolute confusion matrix.

            Parameters
            ----------
            self

            Returns
            -------
            List[Tuple[Text, pd.DataFrame]]

            confusion matrix for each device

            """
            labels_to_confusion = [self.sleep_scoring.get("Wake")] + self.sleep_scoring.get("Sleep")
            standard_normalized_confusion_matrix = list(
                map(
                    self._ConfusionMatrix__standard_confusion_matrix_each_device,
                    repeat(self.reference),
                    self.device,
                    repeat(labels_to_confusion),
                    repeat(self._savepath_standard_normalized_confusion_matrix_xlsx),
                    repeat(False)
                )
            )

            standard_normalized_confusion_matrix = list(
                map(
                    lambda x: (x[0], x[1].round(self.digit)),
                    standard_normalized_confusion_matrix
                    )
            )
            return standard_normalized_confusion_matrix

        @property
        def proportional_confusion_matrix(self) -> List[Tuple[Text, Tuple[pd.DataFrame, pd.DataFrame]]]:
            """
            Calculates the proportional confusion matrix.

            Parameters
            ----------
            self

            Returns
            -------
                List[Tuple[Text, Tuple[pd.DataFrame, pd.DataFrame]]]

                The first DataFrame in the tuple contains
                the numeric confusion matrix, while the second
                DataFrame contains the confusion matrix
                as string. The latter will be passed to sns.heatmap
                as annot argument in
                ConfusionMatrixPlot.proportional_confusion_matrix_plot

            """
            labels_to_confusion = [self.sleep_scoring.get("Wake")] + self.sleep_scoring.get("Sleep")

            device = self.device
            reference = self.reference

            digit = self.digit
            ci_level = self.ci_level
            id_col = self.id

            output = []
            for i in device:
                dev_name = i.columns[-1]
                dev = i.iloc[:, -1]

                to_proportional_confusion_matrix = pd.concat([reference, dev], axis=1)
                to_proportional_confusion_matrix = to_proportional_confusion_matrix.groupby(id_col)

                individual_level_matrix = list(
                    map(
                        self._ConfusionMatrix__individual_level_matrix_calculation,
                        to_proportional_confusion_matrix,
                        repeat(labels_to_confusion)
                    )
                )

                to_stat = pd.concat(individual_level_matrix)

                mean, annot_plot, annot_excel = self._ConfusionMatrix__confusion_matrix_statistics_calculation(
                    to_stat,
                    ci_level=ci_level,
                    digit=digit,
                    ci_bootstrapping=self.ci_bootstrapping,
                    boot_method=self.boot_method,
                    boot_n_resamples=self.boot_n_resamples
                )

                path_to_xlsx = self._savepath_proportional_confusion_matrix_xlsx + f"_{dev_name}.xlsx"
                with pd.ExcelWriter(path_to_xlsx) as xlsx:
                    mean.to_excel(xlsx, sheet_name="mean")
                    annot_excel.to_excel(xlsx, sheet_name="annot_excel")

                output.append((dev_name, (mean, annot_plot)))

                del (
                    dev_name, dev,
                    to_proportional_confusion_matrix,
                    individual_level_matrix, to_stat,
                    mean, annot_plot, annot_excel#, path_to_xlsx
                )

            return output

        @staticmethod
        def __standard_confusion_matrix_each_device(
                ref: pd.DataFrame,
                dev: pd.DataFrame,
                labels_to_confusion: List[Text],
                save_path: Text,
                absolute: bool = True
        ):
            """
            Calculates confusion matrix for each device

            Parameters
            ----------
            ref : pd.DataFrame
                self.reference
            dev : pd.DataFrame
                one element of self.device
            labels_to_confusion: List[Text]
                list containing self.scoring items.
                Passed to heatmap for labeling.
            savepath: Text
                path for saving the matrix in xlsx

            Returns
            -------
            List[Text, np.ndarray]
                calculated absolute confusion matrix for the overall sample

            """
            ref = ref.iloc[:, -1]

            dev_name = dev.columns[-1]
            dev = dev.iloc[:, -1]

            if absolute is True:
                standard_confusion_matrix = confusion_matrix(
                    y_true=ref,
                    y_pred=dev,
                    labels=labels_to_confusion,
                    normalize=None
                )
            else:
                standard_confusion_matrix = confusion_matrix(
                    y_true=ref,
                    y_pred=dev,
                    labels=labels_to_confusion,
                    normalize='true'
                )

            standard_confusion_matrix = pd.DataFrame(
                standard_confusion_matrix,
                columns=labels_to_confusion,
                index=labels_to_confusion
            )

            save_path = save_path + f'_{dev_name}.xlsx'
            standard_confusion_matrix.to_excel(save_path)

            return dev_name, standard_confusion_matrix

        @staticmethod
        def __individual_level_matrix_calculation(
                to_individual_level_matrix: Tuple[Text, pd.DataFrame],
                labels_to_confusion: List[Text]
        ) -> pd.DataFrame:
            """
            Calculates the individual level matrix.

            Applied to each element of a list,
            resulting from a group by operation.
            The key of group by is: self.id.

            Parameters
            ----------
            to_individual_level_matrix: Tuple[Text, pd.DataFrame]
                first element: id
                second element: reference and device values of a
                single participant.

            labels_to_confusion: List[Text]
                list containing self.scoring items.
                Passed to heatmap for labeling.
            Returns
            -------
            pd.DataFrame
            individual level confusion matrix.

            """

            to_individual_level_matrix = to_individual_level_matrix[1]
            # pd.DataFrame to individual matrix

            conf_matrix = confusion_matrix(
                y_true=to_individual_level_matrix.iloc[:, 1],
                y_pred=to_individual_level_matrix.iloc[:, 2],
                labels=labels_to_confusion
            )

            conf_matrix = pd.DataFrame(
                conf_matrix,
                columns=labels_to_confusion,
                index=labels_to_confusion
            )
            marginal = conf_matrix.sum(axis=1)
            conf_matrix = conf_matrix.div(marginal, axis=0)
            return conf_matrix

        @staticmethod
        def __confusion_matrix_statistics_calculation(
                to_stat: pd.DataFrame,
                ci_level: int,
                digit: int,
                ci_bootstrapping: bool,
                boot_method: Text,
                boot_n_resamples: int
        ):
            """
            Calculates the statistics (mean. std, ci)
            for proportional confusion matrix.


            Parameters
            ----------
            to_stat: pd.DataFrame
                corresponds to to_stat in
                proportional_heatmap_each_device

            ci_level: int
                level of significance for ci

            digit: int
                self.digit

            Returns
            -------
            Tuple[pd.DataFrame, pd.DataFrame]

            mean values (for heatmap)
            and annotations for heatmap.

            """
            columns_to_reindexing = list(to_stat.columns)

            to_stat = to_stat.groupby(level=0)

            idx = list(map(lambda x: x[0], to_stat))

            mean = pd.concat(
                map(
                    lambda x: pd.DataFrame(x[1].apply(
                        lambda y: np.mean(y),
                        axis=0
                    )
                    ).transpose(),
                    to_stat
                )
            )  # mean will be passed to heatmap function
            # to generate the heatmap
            mean.index = idx
            mean = mean.reindex(columns_to_reindexing)
            mean = round(mean * 100, digit)

            std = pd.concat(
                map(
                    lambda x: pd.DataFrame(x[1].apply(
                        lambda y: np.std(y),
                        axis=0
                    )
                    ).transpose(),
                    to_stat
                )
            )
            std.index = idx
            std = round(std * 100, digit)
            std = std.reindex(columns_to_reindexing)


            ci = map(
                lambda x:
                    confidence_interval_calculation(
                        to_ci=x[1],
                        stage_device_name=x[0],
                        return_annot_df=True,
                        ci_level=ci_level,
                        digit=digit,
                        ci_bootstrapping=ci_bootstrapping,
                        boot_method=boot_method,
                        boot_n_resamples=boot_n_resamples
                ),
                to_stat
            )

            ci = pd.concat(ci)

            ci = ci.reindex(columns_to_reindexing)

            # follows the creation of annot dataframe
            # to be passed to heatmap function

            annot_mean_excel = + mean.astype(str) + '\n'

            annot_mean_plot = mean.astype(str) + '\n'
            annot_std = ' (' + std.astype(str) + ')\n'
            annot_heatmap_plot = annot_mean_plot.add(annot_std)
            annot_heatmap_plot = annot_heatmap_plot.add(ci)
            annot_heatmap_excel = annot_mean_excel.add(ci)

            return mean, annot_heatmap_plot, annot_heatmap_excel
