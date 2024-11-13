import os
from itertools import product, repeat
from typing import Dict, List, Text, Tuple

import pandas as pd
import matplotlib.pyplot as plt


class HypnogramPlot:
    def hypnogram_plot(
            self,
            reference: pd.Series | List[pd.Series] = None,
            sleep_scoring: Dict = None,
            sleep_stages: Dict = None,
            colors_for_difference: List = None,
            figsize: List = None,
            plot_dpi: int = None,
            save_path: Text = None
    ):
        """

        Plots the hypnogram.

        Args:
            reference: pd.Series | List[pd.Series]
                Reference(s) to be plotted. if a list
                is passed, each device and reference is
                 plotted. (multi-gold standard
                comparison).
                The default is None.
            sleep_scoring
            sleep_stages
            colors_for_difference: List
                Epoch-by-Epoch discrepancies
                between device and reference(s)
                are highlighted with vertical
                lines. This argument allows to
                specify the list of colors that
                are used highlight the discrepancies.
                If anything is passed, a default
                list is used.
                The default is None.
            figsize: List[int, int]
                specifies the dimension of the plot.
                If anything is passed, figsize is
                set to be the matplotlib.pyplot's default.
                The default is None.
            plot_dpi: int
                specifies the dpi of the plot. If
                nothing is passed, self.plot_dpi
                passed in InterfaceClass.
                The default is None.
            save_path: Text
                specifies the path in which to save
                the hypnograms. If anything is passed,
                save_path is set to equal to
                self._savepath_hypnograms_plot.
                The default is None.

        Returns:
            None

        """
        if reference is None:
            reference = self.reference
        else:
            pass

        if isinstance(reference, list) is True:
            pass
        else:
            reference = [reference]
        # this if statement is implemented to deal with
        # more than 1 gold standard device

        reference = list(map(self.id_to_index, reference, repeat(self.id)))
        list_of_reference = list(map(lambda x: x.columns[0], reference))

        device = list(map(self.id_to_index, self.device, repeat(self.id)))

        if colors_for_difference is None:
            colors_for_difference = [
                'orange', 'lime', 'darkturquoise',
                'coral', 'violet', 'cornflowerblue',
                'lightslategray'
            ]
        else:
            pass

        if len(colors_for_difference) < len(reference):
            raise ValueError ('The number of colors passed with colors_for_difference must be greater than or equal to the number of reference devices used.')

        if figsize is None:
            self._HypnogramPlot__figsize = [6.4, 4.8]  # matplotlib.pyplot default
        else:
            self._HypnogramPlot__figsize = figsize

        if plot_dpi is None:
            self._HypnogramPlot__plot_dpi = self.plot_dpi  # matplotlib.pyplot default
        else:
            self._HypnogramPlot__plot_dpi = plot_dpi

        if save_path is None:
            self._HypnogramPlot__savepath_hypnograms_plot=self._savepath_hypnograms_plot
        else:
            self._HypnogramPlot__savepath_hypnograms_plot = save_path

        if sleep_scoring is None:
            sleep_scoring = self.sleep_scoring
        else:
            pass

        if sleep_stages is None:
            sleep_stages = self.sleep_stages
        else:
            pass


        # if the user specifies the list of
        # colors to highlight the difference
        # between each gold standard and the
        # devices, this list will be used in
        # the plot. If not specified, a default
        # colors_for_difference is assigned
        # Colors in this list was chosen
        # to optimally highlight differences
        # between different gold-standards.
        # Note that as more gold-standards
        # are added to the plot, similar
        # and more similar colors will be
        # progressively added to the list
        # and used, commpromising the
        # visual inspection. Up to now,
        # seven different colors are
        # implemented in the default
        # color_to_difference variable. and
        # they should be different enough to
        # allow a clear distinction between them

        self._HypnogramPlot__colors_for_difference = dict(zip(list_of_reference, colors_for_difference))
        del list_of_reference
        # each gold standard has a unique color assigned. The same color
        # will be used in every plot for the same gold standard sleep scoring.
        # saved as a property of the class because it is called in multiple functions.

        to_hypno = list(product([reference], device))
        # this generates a list, in which each element corresponds to
        # a single gold standard device. In this way, it is possible to
        # automatically generate a subplot having a number of rows that
        # compares each device under study against a gold-standard.

        to_hypno = list(
            map(
                lambda x: pd.concat(
                    [
                        x[1],  # device, which is a pd.DataFrame
                        pd.concat(x[0], axis=1)  # concatenates reference, which has been turned to a list
                        # to allow to apply product before.
                    ],
                    axis=1
                ),
                to_hypno
            )
        )
        # this mapping returns a list, where each element
        # is represented by a pd.DataFrame. Each DataFrame contains
        # a single device under study, and all references.
        # Note: the order (device, gold standard(s)) was chosen
        # to plot the device under study in the upper row of the subplots,
        # while gold standard(s) in the lower row(s).

        wake_stage = [sleep_scoring.get("Wake")]
        if isinstance(self.sleep_stages, dict) is False:
            # if sleep_stages is False, it's assumed that
            # the device under study does not allow any distinction between
            # NREM and REM sleep. Sleep Stages specified in self.sleep_scoring.get("SLeep")
            # are assumed to be ordered in descending order of sleep depth, that is
            # from the lightest (N1) to the deepest sleep stage (N3).

            self.HypnogramPlot__rem_stage = None
            sleep_stage_to_hypno = sleep_scoring.get("Sleep")
        else:
            self.HypnogramPlot__rem_stage = [sleep_stages.get("REM")]
            nrem_stage = sleep_stages.get("NREM")  # the lightest
            # sleep stage is the first element, the last is the deepest.
            if isinstance(nrem_stage, list) is True:
                pass
            else:
                nrem_stage = [nrem_stage]
            sleep_stage_to_hypno = self.HypnogramPlot__rem_stage + nrem_stage  # in this way,
            # sleep stages are ordered from rem to the deepest NREM
            # sleep stage. Used later to replace strings in to_hypno
            # variable with an ordered list of integers used to plot
            # the hypnogram for each participant.

        stages_to_hypno_ordered = wake_stage + sleep_stage_to_hypno

        self._HypnogramPlot__stages_to_hypno_ordered = dict(
            zip(
                stages_to_hypno_ordered,
                list(range(len(stages_to_hypno_ordered), 0, -1))
            )
        )
        del stages_to_hypno_ordered

        to_hypno = list(
            map(
                lambda x: x.replace(self._HypnogramPlot__stages_to_hypno_ordered),
                to_hypno
            )
        )

        hypno_grouped = list(map(lambda x: list(x.groupby(level=0)), to_hypno))
        # each element of this list is represented by
        # a device under study grouped by IDs.
        # It will be used later to generate a plot for
        # each device against all gold standards.

        [self.plot_hypnogam_participant(j) for i in hypno_grouped for j in i]
        return None

    def plot_hypnogam_participant(self, participant_to_hypnogram: Tuple[Text, pd.DataFrame]):
        """
        Plots the hypnogram for one participant.

        Args:
            participant_to_hypnogram: Tuple[Text, pd.DataFrame]

        Returns:

        """

        participant_id = participant_to_hypnogram[0]

        data_to_hypno = participant_to_hypnogram[1]
        dev_name = data_to_hypno.columns[0]

        device_to_diff = data_to_hypno.iloc[:, 0]

        differences_gs_device = pd.concat(
            map(
                self.calculate_discrepancy_scoring,
                repeat(device_to_diff),
                data_to_hypno.iloc[:, 1:].iteritems()
            ),
            axis=1
        )
        differences_gs_device.index = list(range(len(differences_gs_device)))

        rows_to_hypno = len(data_to_hypno.columns)

        fig, ax = plt.subplots(
            nrows=rows_to_hypno,
            ncols=1,
            dpi=self._HypnogramPlot__plot_dpi,
            sharex=True,
            figsize=self._HypnogramPlot__figsize
        )

        count = 0
        for i in data_to_hypno.iteritems():
            participant_name = i[1].index[0]
            to_plot = i[1]
            to_plot.index = list(range(len(i[1])))
            ax[count].plot(
                to_plot.index,
                to_plot,
                c='k'
            )  # plotting the hypnogram

            if self.HypnogramPlot__rem_stage is not None:
                rem_to_plot = to_plot.where(
                    to_plot == self._HypnogramPlot__stages_to_hypno_ordered.get(
                        self.HypnogramPlot__rem_stage[0]
                    )
                ).dropna(how='all')
                rem_to_plot += 0.2  # this float
                # is added to allow to depict the
                # rem points slightly above the
                # hypnogram line

                ax[count].scatter(
                    rem_to_plot.index,
                    rem_to_plot,
                    s=1,
                    c='r',
                    marker='*'
                )
            else:
                pass

            if count == 0:
                for j in differences_gs_device.iteritems():
                    differences_plot = j[1]
                    differences_plot = differences_plot.where(differences_plot != 0).dropna()

                    [ax[count].axvline(
                        k,
                        alpha=1,
                        linewidth=0.2,
                        c=self._HypnogramPlot__colors_for_difference.get(j[0])
                    )
                        for k in differences_plot.index
                    ]

            else:
                differences_plot = differences_gs_device.loc[:, i[0]]
                differences_plot = differences_plot.where(differences_plot != 0).dropna()
                [ax[count].axvline(
                    j,
                    alpha=1,
                    linewidth=0.2,
                    c=self._HypnogramPlot__colors_for_difference.get(i[0])
                )
                    for j in differences_plot.index
                ]

            ax[count].set_ylabel(i[0])
            if count == rows_to_hypno - 1:
                ax[count].set_xlabel("Time (min)")
            else:
                pass
            # the xlabel is set only on the last axis,
            # given that the xaxis is shared in the subplot

            ax[count].set_yticks(
                list(self._HypnogramPlot__stages_to_hypno_ordered.values())
            )
            ax[count].set_yticklabels(
                list(self._HypnogramPlot__stages_to_hypno_ordered.keys())
            )

            ax[count].set_ylim(0.5, len(self._HypnogramPlot__stages_to_hypno_ordered) + 0.5)
            # making some room between the hypno and the limits of y

            fig.suptitle(participant_name)

            count += 1

        save_path = os.path.join(
            self._HypnogramPlot__savepath_hypnograms_plot,
            f'{participant_id}_{dev_name}.png'
        )

        plt.savefig(
            save_path,
            dpi=self._HypnogramPlot__plot_dpi
        )

        plt.show(block=True)
        return None

    @staticmethod
    def id_to_index(df_to_index, id_col):
        """
        Index sleep scores.
        devices by corresponding patient ID.

        Args:
            df_to_index: pd.DataFrame
                dataframe to new index.

            id_col: Text
                self.id

        Returns: pd.DataFrame
                The purpose of indexing a new DataFrame by Patient ID is to coalesce
                the results from the device and the gold standard sleep score by their
                corresponding ID columns , eliminating redundancies from the two separate
                DataFrames.

        """
        df_to_index.index = df_to_index[id_col]
        df_to_index = df_to_index.drop(columns=[id_col])
        return df_to_index

    @staticmethod
    def calculate_discrepancy_scoring(
            device_to_diff: pd.Series,
            gold_standard_to_diff: pd.Series
    ) -> pd.Series:
        """
        Calculates the difference in scoring between the device
        under investigation and gold standard(s). It will be used to generate
        vertical colored lines to highlight in each plot where an epoch
        was misclassified by the device.

        Args:
            device_to_diff: pd.Series
                It corresponds to the sleep scoring that
                has been produced by a device
                for just one participant.
            gold_standard_to_diff: pd.Series
                It corresponds to the sleep scoring that
                has been produced by a gold standard
                for just one participant.

        Returns: pd.Series
            series containing the difference between
            the device and a single gold standard.

        """
        series_name = gold_standard_to_diff[0]
        gold_standard_to_diff = pd.to_numeric(gold_standard_to_diff[1])

        difference = pd.to_numeric(device_to_diff) - gold_standard_to_diff
        difference.name = series_name
        return difference
