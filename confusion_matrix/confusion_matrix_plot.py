from typing import Tuple
import matplotlib.pyplot as plt
from seaborn import heatmap


class ConfusionMatrixPlot:
    def absolute_confusion_matrix_plot(self):
        """
        Plots absoulte confusion matrix.

        Uses self.absolute_confusion_matrix
        property created in confusion_matrix.ConfusionMatrix

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        absolute_confusion_matrix = self.absolute_confusion_matrix
        for i in absolute_confusion_matrix:
            fig, ax_in_plot = plt.subplots(
                nrows=1,
                ncols=1,
                dpi=self.plot_dpi,
                figsize=(10,10)
            )

            dev_name = i[0]
            to_abs_conf_matrix = i[1]

            heatmap(
                to_abs_conf_matrix,
                annot=True,
                ax=ax_in_plot,
                cmap="Blues",
                fmt='g'
            )

            ax_in_plot.set_ylabel(self._reference_col)

            ax_in_plot.set_xlabel(dev_name)

            plt.tight_layout()

            plt.savefig(
                f'{self._savepath_absolute_confusion_matrix_plot}_{dev_name}.png',
                dpi=self.plot_dpi
            )

            plt.show(block=True)

        return None

    def proportional_confusion_matrix_plot(self, annot_fontsize=None, figsize: Tuple[int, int] = None):
        """
        Plot proportional confusion matrix.

        Uses self.proportional_confusion_matrix
        property created in confusion_matrix.ConfusionMatrix
        Parameters
        ----------
        annot_fontsize
        Allows to adjust the dimension of the font in heatmap's annotations.
        figsize : tuple(int, int)
        Allows to modify the dimension of the figure in matplotlib.pyplot.subplots

        Returns
        -------

        """
        if annot_fontsize is None:
            annot_fontsize = 'medium'
        else:
            pass

        if figsize is None:
            figsize = (6.4, 4.8) # matplotlib.pyplot default
        else:
            pass

        proportional_confusion_matrix = self.proportional_confusion_matrix
        for i in proportional_confusion_matrix:
            dev_name = i[0]
            mean = i[1][0]
            annot = i[1][1]

            fig, ax_in_plot = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=figsize,
                dpi=self.plot_dpi
            )

            heatmap(
                mean,
                annot=annot,
                cmap="Blues",
                fmt='',
                annot_kws={"fontsize": annot_fontsize, "in_layout": True},
                square=True,
                linewidths=0.4,
                ax=ax_in_plot
            )

            plt.yticks(fontsize="xx-large")
            ax_in_plot.set_ylabel(self._reference_col, fontsize="xx-large")

            plt.xticks(fontsize="xx-large")
            ax_in_plot.set_xlabel(dev_name, fontsize="xx-large")

            savepath = self._savepath_proportional_confusion_matrix_plot + f"_{dev_name}.png"

            plt.tight_layout()

            plt.savefig(savepath, dpi=self.plot_dpi)

            plt.show(block=True)

        return None
