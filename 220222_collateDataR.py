'''
author: Felix Hol
date: 2022  Feb
content: collate lizard data
'''

# import matplotlib as mpl
# import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import os
from tqdm import tnrange, tqdm
import glob
import math
import pickle
import warnings
# import seaborn as sns

dataH5s = glob.glob('/home/felix/Dropbox/HongKong/[0-9]*/*allDataPerFrame.h5')
print('processing ' + str(len(dataH5s)) + ' files')
print(dataH5s)

def stripStack(file):
    df = pd.read_hdf(file)
    df.columns.set_levels([os.path.basename(file)[:-19]], level='scorer', inplace=True)
    scorer = df.columns.get_level_values(0)[0]
    df.drop(['liz5', 'liz6', 'single'], axis=1, level='individuals', inplace=True)
    df.drop(['x', 'y', 'likelihood'], axis=1, level='coords', inplace=True)
    df.drop(['nose', 'neck', 'tailbase'], axis=1, level='bodyparts', inplace=True)

    for ind in df.columns.get_level_values(1).unique():
        df[scorer, ind, 'back', 'active'] = df[scorer, ind, 'back', 'distance'] > 8
        df[scorer, ind, 'back', 'active'] = df[scorer, ind, 'back', 'active'].astype(int)
        df[scorer, ind, 'head', 'active'] = df[scorer, ind, 'back', 'distance'] > 8
        df[scorer, ind, 'head', 'active'] = df[scorer, ind, 'head', 'active'].astype(int)

        for bp in df.columns.get_level_values(2).unique():
            df[scorer, ind, bp, 'frameNo'] = df.index
            df[scorer, ind, bp, 'time_sec'] = df[scorer, ind, bp, 'frameNo'] * 5

    df = df.reindex(columns=df.columns.get_level_values(1).unique(), level='individuals')
    df = df.reindex(columns=df.columns.get_level_values(2).unique(), level='bodyparts')
    df.rename_axis(columns={'scorer': 'experiment'}, inplace=True)
    df = df.stack(level=['experiment', 'individuals', 'bodyparts'])
    df = df.droplevel(level=0)
    return df



allData = []

for file in dataH5s:
    df = stripStack(file)
    allData.append(df)

allData = pd.concat(allData)

allData.to_csv('/home/felix/Dropbox/HongKong/test/allExperiments220222.csv', na_rep='NA')
