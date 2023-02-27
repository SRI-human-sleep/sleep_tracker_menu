import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

save_path = r"C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\pipeline_run"

wake = (93.0, 2.9, 2.4, 0.3, 1.5)
n1 = (15.1, 45.4, 27.5, 0.3, 11.8)
n2 = (2.1, 3.8, 85.7, 5.4, 3.1)
n3 = (0.4, 0.1, 16.1, 83.2, 0.2)
rem = (3.0, 3.5, 6.7, 0.0, 86.8)

to_heat = [wake, n1, n2, n3, rem]

to_heat = pd.DataFrame(
    to_heat,
    columns=['W', 'N1', "N2", "N3", "R"],
    index=['W', 'N1', "N2", "N3", "R"]
)

del wake, n1, n2, n3, rem

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=1000)
sns.heatmap(
    to_heat,
    annot=True,
    fmt='g',
    cmap="Blues",
    annot_kws={"fontsize": 17, "in_layout": True},
    square=True,
    linewidths=0.4,
    ax=ax
)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel("Human", fontsize=20)
plt.xlabel("YASA", fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "performance_vallat.png"), dpi=1000)
plt.show()
