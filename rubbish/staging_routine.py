# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:25:42 2022

@author: bened
"""

import os
import glob as glb
from pathlib import Path

import mne
from numpy import savetxt
from yasa import SleepStaging

os.chdir(r"C:\Users\bened\OneDrive\SRI\NCANDA\data\staging_perfomance")

def staging_function(file):
    export_name = Path(file).parts[-1]
    file = mne.io.read_raw_edf(file,
                               preload=True
                               )
    file.resample(100)

    file.filter(0.4, 30,
                n_jobs="cuda")
    staging = SleepStaging(file,
                           eeg_name="C4",
                           emg_name="Chin 1"
                           )

    pred = staging.predict()
    savetxt(os.path.join(savepath, f"{export_name}.csv"),
            pred,
            fmt='%s'
            )
    return None

if __name__ =="__main__":
    savepath = r"C:\Users\bened\OneDrive\SRI\NCANDA\data\staging_perfomance\staging_algorithm"

    edf_path =r"C:\Users\bened\OneDrive\SRI\NCANDA\data\staging_perfomance\edf\*.edf"
    edf_path = glb.glob(edf_path)
    list(map(staging_function, edf_path))

