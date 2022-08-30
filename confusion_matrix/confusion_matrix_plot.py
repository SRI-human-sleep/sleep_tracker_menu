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
                dpi=self.plot_dpi
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

    def proportional_confusion_matrix_plot(self):
        """
        Plot proportional confusion matrix.

        Uses self.proportional_confusion_matrix
        property created in confusion_matrix.ConfusionMatrix

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        proportional_confusion_matrix = self.proportional_confusion_matrix
        for i in proportional_confusion_matrix:
            dev_name = i[0]
            mean = i[1][0]
            annot = i[1][1]

            fig, ax_in_plot = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(11, 9),
                dpi=self.plot_dpi
            )

            heatmap(
                mean,
                annot=annot,
                cmap="Blues",
                fmt='',
                annot_kws={"fontsize": "small", "in_layout": True},
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
