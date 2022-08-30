# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:05:01 2022

@author: E34476
"""

import os
import pandas as pd

os.chdir(r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data')

file = pd.read_excel(r'scoring.xlsx')

file = file.iloc[:, 1:]

col_to_get = ['Max', 'Justin Greco', 'Fiona', 'Sarah', 'Leo', 'Devika', 'Vanessa',
              'Yun Qi', 'Aimee', 'Lena', 'Dilara', 'Steph', 'LV', 'TD',
              'unknown_scorer', 'BA', 'EA', 'MI'
              ]

scorer = file.loc[:, col_to_get]

del col_to_get

scorer = scorer.dropna(how='all')

scorer = scorer.index

file_clean = file.loc[scorer, :]

del scorer

file_clean = file_clean.where(file_clean.downloaded == 0).dropna(how='all')

to_get = []
for i in file_clean.iterrows():
    if 'B-' in i[1].ID_full :
        to_get.append(i[0])

file_clean = file_clean.loc[to_get, :]

del to_get, i

# confronto di eventi in una notte. 