import matplotlib.pyplot as plt
from seaborn import heatmap


class ConfusionMatrixPlot:

    def standard_confusion_matrix_plot(
            self,
            absolute=True,
            annot_fontsize=None,
            figsize: tuple[int, int]=None,
            title_text: str = '',
            title_fontsize: int = 10,
            cmap_colors:str='Blues',
            axis_label_fontsize: int = 11,
            axis_ticks_fontsize: int = 13
    ):
        """
        Plots a confusion matrix for the classifier, displaying either absolute or normalized values.

        This method generates a confusion matrix plot for each device or classification category.
        If `absolute` is `True`, it uses the absolute confusion matrix (`self.standard_absolute_confusion_matrix`);
        otherwise, it uses the normalized confusion matrix (`self.standard_normalized_confusion_matrix`).
        Each plot is saved as a PNG file based on the specified `absolute` flag and displayed using Matplotlib.

        Parameters
        ----------
        absolute : bool, optional
            If `True`, plots the absolute confusion matrix; if `False`, plots the normalized confusion matrix.
            Default is `True`.
        annot_fontsize : int, optional
            Font size for annotations within the heatmap cells. If `None`, a default size is used.
        figsize : tuple of int, optional
            Figure size as a tuple (width, height) for each plot. If `None`, the default Matplotlib size is used.
        title_text : str, optional
            Title text for the plot. Default is an empty string.
        title_fontsize : int, optional
            Font size for the plot title. Default is `10`.
        cmap_colors : str, optional
            Color map for the heatmap. Default is 'Blues'.
        axis_label_fontsize : int, optional
            Font size for the X and Y axis labels. Default is `11`.
        axis_ticks_fontsize : int, optional
            Font size for the axis tick labels. Default is `13`.

        Returns
        -------
        None
            This method does not return a value; it generates and displays plots.

        Notes
        -----
        - The method saves each plot to a PNG file with a filename based on the `absolute` flag and the device name.
        - The `absolute` flag controls whether to plot the absolute or normalized values from the confusion matrix.

        Examples
        --------
        >>> iclass.standard_confusion_matrix_plot()
        """

        print('Generating standard confusion matrix plot.')

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
            figsize: tuple[int, int] = None,
            title_text: str = '',
            title_fontsize: int = 10,
            cmap_colors: str="Blues",
            axis_label_fontsize: int =11,
            axis_ticks_fontsize: int=13
    ):
        """
        Plots a proportional confusion matrix.

        This method generates a confusion matrix plot representing the proportion of each classification category
        across devices. It uses `self.proportional_confusion_matrix`, where each entry represents the proportional
        values and corresponding annotations. Each plot is saved as a PNG file and displayed using Matplotlib.

        Parameters
        ----------
        annot_fontsize : int, optional
            Font size for annotations within the heatmap cells. Defaults to 'medium' if `None`.
        figsize : tuple of int, optional
            Figure size as a tuple (width, height) for each plot. If `None`, the default size (6.4, 4.8) is used.
        title_text : str, optional
            Title text for the plot. Default is an empty string.
        title_fontsize : int, optional
            Font size for the plot title. Default is `10`.
        cmap_colors : str, optional
            Color map for the heatmap. Default is 'Blues'.
        axis_label_fontsize : int, optional
            Font size for the X and Y axis labels. Default is `11`.
        axis_ticks_fontsize : int, optional
            Font size for the axis tick labels. Default is `13`.

        Returns
        -------
        None
            This method does not return a value; it generates and displays plots.

        Notes
        -----
        - Each plot is saved to a PNG file with a filename based on the device name.
        - The `annot_fontsize` and `figsize` parameters default to Matplotlib’s sizes if not provided.

        Examples
        --------
        >>> inclass.proportional_confusion_matrix_plot()
        """

        print('Generating proportional confusion matrix plot.')

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
