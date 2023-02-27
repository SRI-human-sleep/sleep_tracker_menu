import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import GLS
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.mixed_linear_model import MixedLM

from file_to_analysis import file_preprocessing

os.chdir(r"C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper")

file = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\2022_04_10.csv'
ncanda_dataset = r"C:\Users\e34476\OneDrive - SRI International\NCANDA\data\spreadsheets\0YFU.xlsx"
to_drop = r"C:\Users\e34476\OneDrive - SRI International\NCANDA\data\spreadsheets\yasa_to_drop.xlsx"

file = file_preprocessing(
    file_name=file,
    ncanda_dataset_name=ncanda_dataset,
    to_drop_name=to_drop,
    night_in_processing="overall"
)

metrics = pd.read_excel("performance.xlsx", index_col=[0, 1])

data = metrics.join(file, how="outer")
data = data.dropna()
data.rename(columns={0: 'mcc'}, inplace=True)

# %%
mcc = data.drop("all", level=0)
mcc = mcc.xs("MCC", level=1)
mcc.rename(columns={0: 'mcc'}, inplace=True)

#mcc["mcc"] = mcc["mcc"].transform(np.log)
# data log transformed to account for left skewness

mcc["sex"] = mcc["sex"].replace({'F': 1, 'M': 0})
mcc["site"] = mcc["site"].replace({'sri': 0, 'pit': 1})
mcc["night"] = mcc["night"].replace({"first_night": 0, "ERP": 1})

mcc["Scorer"] = mcc["Scorer"].replace({
    "justin": 0,
    "lena": 1,
    "sc": 2,
    "max": 3,
    "fiona": 4,
    "sarah": 5
})

mcc["ID"] = mcc["ID"].replace(
    dict(zip(mcc["ID"], list(range(1000, len(mcc["ID"])+1000))))
)
mcc.to_csv("mcc_analysis.csv")
# %%
first_night = mcc.where(mcc.night == 0).dropna(how="all")

dependent = first_night["mcc"]
fixed_effects = first_night.loc[:, ["age", "sex", "Scorer",  "site"]]
groups = first_night["ID"]
first_night_model = MixedLM(
    endog=dependent,
    exog=fixed_effects,
    groups=groups
)

first_night_model_result = first_night_model.fit()
first_night_p_values = multipletests(
    pvals=first_night_model_result.pvalues,
    method="fdr_bh",
    alpha=0.05
)

print(first_night_model_result.summary())
print(first_night_p_values)
print(first_night_model_result.conf_int(0.05).round(2))


# %%

erp_night = mcc.where(mcc.night == 1).dropna(how="all")

dependent = erp_night["mcc"]
fixed_effects = erp_night.loc[:, ["age", "sex", "Scorer", "site"]]

erp_night_model = GLS.from_formula(formula='mcc~age+sex+Scorer+site+Scorer:site', data=erp_night)
# erp_night_model = GLS(
#     endog=dependent,
#     exog=fixed_effects
# )

erp_night_model_result = erp_night_model.fit()
erp_p_values = multipletests(
    pvals=erp_night_model_result.pvalues,
    method="fdr_bh",
    alpha=0.05
)

print(erp_night_model_result.summary())
print(erp_p_values)
print(erp_night_model_result.conf_int(0.05).round(3))

# %%
# path_to_plot =\
#     r"C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\pipeline_run\regression_lines"
# dataset_to_plot = data
# dataset_to_plot["Scorer"] = dataset_to_plot["Scorer"].replace(
#     {
#         'justin': 1,
#         'fiona': 2,
#         'max': 3,
#         'lena': 4,
#         'sc': 5,
#         'sarah': 6
#     }
# )
# dataset_to_plot.rename(
#     columns={"age": "Age (years)", "mcc": "MCC", "site": "Site", "sex": "Sex"},
#     inplace=True
# )
# dataset_to_plot["night"] = dataset_to_plot["night"].replace(
#     {
#         "first_night": "Standard PSG night",
#         "ERP": "ERP/PSG night"
#     }
# )
# dataset_to_plot["Site"] = dataset_to_plot["Site"].replace(
#     {"pit": "University of\n Pittsburgh", "sri": "SRI\n International"}
# )
# #x_to_plot = "Age (years)"
# #x_to_plot = "Site"
# x_to_plot = "Scorer"
# #x_to_plot = "Sex"
# y_to_plot = "MCC"
# hue = "night"
# save_path = os.path.join(path_to_plot, x_to_plot)
#
# if x_to_plot == "Age (years)":
#     sns.lmplot(
#         x=x_to_plot,
#         y=y_to_plot,
#         hue=hue,
#         x_estimator=np.mean,
#         data=dataset_to_plot,
#         legend=False,
#         height=12,
#         palette="Blues"
#     )
#     plt.yticks(list(np.arange(0.7, 0.95, 0.05)), fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.xlabel(x_to_plot, fontsize=30)
#     plt.ylabel(y_to_plot, fontsize=30)
#     plt.ylim(0.7, 0.9)
#     plt.legend(fontsize=25)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=1000)
#     plt.show()
# else:
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
#     sns.barplot(
#         data=dataset_to_plot,
#         x=x_to_plot,
#         y=y_to_plot,
#         hue=hue,
#         ax=ax,
#         palette="Blues"
#     )
#
#     if x_to_plot == "Scorer":
#         pass
#     else:
#         plt.scatter([0.7], [0.863], color="black", marker='*')
#         plt.hlines(y=0.86, xmin=0.20, xmax=1.20, color="black")
#         plt.vlines(x=1.200, ymin=0.8585, ymax=0.8615, color="black")
#         plt.vlines(x=0.20, ymin=0.8585, ymax=0.8615, color="black")
#         if x_to_plot == "Site":
#             pass
#         else:
#             plt.scatter([0.3], [0.853], color="black", marker='*')
#             plt.hlines(y=0.85, xmin=-0.20, xmax=.8, color="black")
#             plt.vlines(x=0.8, ymin=0.8485, ymax=0.8515, color="black")
#             plt.vlines(x=-0.20, ymin=0.8485, ymax=0.8515, color="black")
#
#     plt.yticks(list(np.arange(0.7, 0.95, 0.05)), fontsize=30)
#     plt.xticks(fontsize=35)
#     plt.xlabel(x_to_plot, fontsize=40)
#     plt.ylabel(y_to_plot, fontsize=37)
#     plt.ylim(0.7, 0.9)
#     plt.legend().remove()
#     plt.tight_layout()
#     plt.legend(fontsize=25)
#     plt.savefig(save_path, dpi=1000)
#     plt.show()
