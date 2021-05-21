'''
author: Felix Hol
date: 2021 May 20
content: process DeepLabCut detections for multiple animals that mainly move in one direction
'''

import pandas as pd
import numpy as np
import os
from tqdm import tnrange, tqdm
import glob

baseDir = '/mnt/DATA/geckos/toAnalyze/'
# fullDLCpickle = '/mnt/DATA/geckos/test_b8/021118_38394041shortDLC_resnet50_gekApr30shuffle1_180000_full.pickle'
DLCdf = '/mnt/DATA/geckos/test_b8/021118_38394041shortDLC_resnet50_gekApr30shuffle1_180000_bx.h5'
singleBpts = ['topleftcorner', 'toprightcorner', 'bottomrightcorner', 'bottomleftcorner']
nAnimals = 4
individuals = ['gecko1', 'gecko2', 'gecko3', 'gecko4']
ind = 'single'
bpts = ['head', 'back']

def getSinglePBTS(df, f, scorer, ind, bpts):
    for bpt in tqdm(bpts):
        bptN = header['all_joints_names'].index(bpt)
        frameNames = list(f)

        for i in range(header['nframes']):
            try:
                df[scorer, ind, bpt, 'x'].loc[i] = f[frameNames[i]]['coordinates'][0][bptN][0][0]
                df[scorer, ind, bpt, 'y'].loc[i] = f[frameNames[i]]['coordinates'][0][bptN][0][1]
                df[scorer, ind, bpt, 'likelihood'].loc[i] = f[frameNames[i]]['confidence'][bptN][0][0]
            except:
                pass
    return df

def assignIndBPTS(df, f, scorer, individuals, bpts, trackYs):
    frameNames = list(f)
    for bpt in bpts:
        bptN = header['all_joints_names'].index(bpt)
        for i in tnrange(header['nframes']):
            try:
                j = 0
                for coor in f[frameNames[i]]['coordinates'][0][bptN]:
                    x = coor[0]
                    y = coor[1]
                    c = f[frameNames[i]]['confidence'][0][j][0]
                    j = j +1
                    if y > top + 15:
                        indN = next(x for x, val in enumerate(trackYs) if val > y)
                        ind = individuals[indN - 1]
                        df[scorer, ind, bpt, 'x'].loc[i] = x
                        df[scorer, ind, bpt, 'y'].loc[i] = y
                        df[scorer, ind, bpt, 'likelihood'].loc[i] = c
            except:
                pass
    return df

fullPickles = glob.glob(baseDir + '*full.pickle')

for fullDLCpickle in fullPickles[1:3]:
    print('Processing: ' + fullDLCpickle)

    g = pd.read_hdf(DLCdf)
    f = pd.read_pickle(fullDLCpickle)
    header = f.pop('metadata')
    df = pd.DataFrame(columns=g.columns, index=range(header['nframes']))
    scorer = df.columns.get_level_values(0)[0]

    df = getSinglePBTS(df, f, scorer, 'single', singleBpts)

    top = df[scorer, 'single', 'toprightcorner'].y.mean()
    bottom = df[scorer, 'single', 'bottomrightcorner'].y.mean()
    trackYs = np.linspace(top, bottom, nAnimals + 1)

    df = assignIndBPTS(df, f, scorer, individuals, bpts, trackYs)

    newFile = fullDLCpickle[:-11] + 'sk.h5'

    print('saving: ' + newFile)

    df.to_hdf(newFile, key="df_with_missing", mode="w")
