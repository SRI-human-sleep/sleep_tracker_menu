from typing import Tuple
import matplotlib.pyplot as plt
from seaborn import heatmap


class ConfusionMatrixPlot:
    def standard_confusion_matrix_plot(
            self,
            absolute=True,
            annot_fontsize=None,
            figsize: Tuple[int, int]=None,
            title_text: str = '',
            title_fontsize: int = 10,
            cmap_colors:str='Blues',
            axis_label_fontsize: int=11,
            axis_ticks_fontsize: int=13
    ):
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
        if absolute is True:
            confusion_matrix_to_plot = self.standard_absolute_confusion_matrix
        else:
            confusion_matrix_to_plot = self.standard_normalized_confusion_matrix

        for i in confusion_matrix_to_plot:
            fig, ax_in_plot = plt.subplots(
                nrows=1,
                ncols=1,
                dpi=self.plot_dpi,
                figsize=figsize
            )

            dev_name = i[0]
            to_abs_conf_matrix = i[1]

            heatmap(
                to_abs_conf_matrix,
                annot=True,
                fmt='.70g',
                cmap=cmap_colors,
                annot_kws={"fontsize": annot_fontsize, "in_layout": True},
                square=True,
                linewidths=0.4,
                ax=ax_in_plot
            )

            ax_in_plot.set_ylabel(self._reference_col, fontsize=axis_label_fontsize)

            ax_in_plot.set_xlabel(dev_name, fontsize=axis_label_fontsize)

            plt.xticks(fontsize=axis_ticks_fontsize)
            plt.yticks(fontsize=axis_ticks_fontsize)

            plt.title(title_text, fontsize=title_fontsize)


            plt.tight_layout()

            if absolute is True:
                plt.savefig(
                    f'{self._savepath_standard_absolute_confusion_matrix_plot}_{dev_name}.png',
                    dpi=self.plot_dpi
                )
            else:
                plt.savefig(
                    f'{self._savepath_standard_normalized_confusion_matrix_plot}_{dev_name}.png',
                    dpi=self.plot_dpi
                )
            plt.show(block=True)

        return None

    def proportional_confusion_matrix_plot(
            self,
            annot_fontsize=None,
            figsize: Tuple[int, int] = None,
            title_text: str = '',
            title_fontsize: int = 10,
            cmap_colors: str="Blues",
            axis_label_fontsize: int =11,
            axis_ticks_fontsize: int=13
    ):
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
            figsize = (6.4, 4.8)  # matplotlib.pyplot default
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
                cmap=cmap_colors,
                fmt='',
                annot_kws={"fontsize": annot_fontsize, "in_layout": True},
                square=True,
                linewidths=0.4,
                ax=ax_in_plot
            )

            plt.xticks(fontsize=axis_ticks_fontsize)
            ax_in_plot.set_xlabel(dev_name, fontsize=axis_label_fontsize)

            plt.yticks(fontsize=axis_ticks_fontsize)
            ax_in_plot.set_ylabel(self._reference_col, fontsize=axis_label_fontsize)

            savepath = self._savepath_proportional_confusion_matrix_plot + f"_{dev_name}.png"

            plt.title(title_text, fontsize=title_fontsize)

            plt.tight_layout()

            plt.savefig(savepath, dpi=self.plot_dpi)

            plt.show(block=True)

        return None
