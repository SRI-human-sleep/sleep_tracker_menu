# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:48:09 2022

@author: e34476
"""
import os
from datetime import datetime
import mne
import numpy as np
import mat73
os.chdir(r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data')

file_test = mat73.loadmat('template_file (1).mat')

file_in = file_test.get('subject')

# =============================================================================
# ch_names = file_in.get('chan_labels')
# 
# if file_in.get('Sex') == 'Female':
#     gender_to_info = 2
# elif file_in.get('Sex') == 'Male':
#     gender_to_info = 1
# else:
#     gender_to_info = 0
# 
# subject_info = {
#     'id': file_in.get('Reference'),
#     'his_id': file_in.get('ReasonForStudy'),
#     'last_name': file_in.get('GivenName'),
#     'sex': gender_to_info
#     }
# 
# info = mne.Info(subject_info)
# =============================================================================
date_in = file_in.get('DOB')
def birthday_management(date_in):
    '''
    used in mat_preprocessing. It converts the birthday object stored in mat
    files into a format viable to be passed to mne.Info constructor.

    Parameters
    ----------
    date_in : Text
        date to convert

    Returns
    -------
    int
        year.
    int
        month.
    int
        day.

    '''
    date_in = datetime.strptime(date_in, '%m/%d/%Y %I:%M:%S %p')

    return (date_in.year, date_in.month, date_in.day)

birthday