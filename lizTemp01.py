'''
author: Felix Hol
date: 2021  Nov
content: process DeepLabCut detections for multiple animals that mainly move in one direction.
Use corner coordinates to rotate detections (includes filtering), use rotated detections to assign to
specific Y-track. Use x-coordinate and temperature callibration to calculate corresponding temperature.
'''

#import matplotlib as mpl
#import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import os
from tqdm import tnrange, tqdm
import glob
import math
import deeplabcut
import pickle
import warnings
# import seaborn as sns
import cv2
warnings.simplefilter('ignore')

### Load DLC detections and a dataframe to steal the header for convenience

videos = glob.glob('/home/felix/Dropbox/HongKong/*/*.AVI')

print('processing ' + str(len(videos)) + ' videos')
print(videos)



for videoName in videos:
    try:
        fullDLCpickle = videoName[:-4] + 'DLC_resnet50_lizNov2shuffle1_140000_full.pickle'
        DLCdf = '/home/felix/Dropbox/HongKong/12082019_64656667/12082019_64656667-3DLC_resnet50_lizNov2shuffle1_140000.h5'

        g = pd.read_hdf(DLCdf)
        f = pd.read_pickle(fullDLCpickle)
        header = f.pop('metadata')

        ### create empty dataframe to move data to
        df = pd.DataFrame(columns=g.columns, index=range(header['nframes']))
        scorer = df.columns.get_level_values(0)[0]

        ### first get coordinates for all arena corners to use to assign animals to tracks
        corners = df[scorer, 'single'].columns.get_level_values(0).unique()
        scorer = df.columns.get_level_values(0)[0]
        ind = 'single'

        for bpt in tqdm(corners):
            bptN = header['all_joints_names'].index(bpt)
            frameNames = list(f)

            for i in range(header['nframes']):
                try:
                    df[scorer, ind, bpt, 'x'].loc[i] = f[frameNames[i]]['coordinates'][0][bptN][0][0]
                    df[scorer, ind, bpt, 'y'].loc[i] = f[frameNames[i]]['coordinates'][0][bptN][0][1]
                    df[scorer, ind, bpt, 'likelihood'].loc[i] = f[frameNames[i]]['confidence'][bptN][0][0]
                except:
                    pass

        #### Interpolate (fill NaN), and rolling mean (window) of corners, add _filt columns
        window = 200

        for corner in corners:
            df[scorer, 'single', corner + '_filt', 'x'] = df[scorer, 'single', corner].x.interpolate(limit_area='inside').rolling(window, min_periods=1).mean()
            df[scorer, 'single', corner + '_filt', 'y'] = df[scorer, 'single', corner].y.interpolate(limit_area='inside').rolling(window, min_periods=1).mean()


        ###calculate angles, interpolate, and rolling mean (window)
        window = 2000

        df[scorer, 'single', 'angle', 'angle1'] = \
        np.arctan((df[scorer, 'single', 'bottomrightcorner_filt'].y - df[scorer, 'single', 'bottomleftcorner_filt'].y) / \
        (df[scorer, 'single', 'bottomrightcorner_filt'].x - df[scorer, 'single', 'bottomleftcorner_filt'].x))

        df[scorer, 'single', 'angle', 'angle1'] = df[scorer, 'single', 'angle', 'angle1'].rolling(window, min_periods=1).mean()

        df[scorer, 'single', 'angle', 'angle2'] = \
        np.arctan((df[scorer, 'single', 'toprightcorner_filt'].y - df[scorer, 'single', 'topleftcorner_filt'].y) / \
        (df[scorer, 'single', 'toprightcorner_filt'].x - df[scorer, 'single', 'topleftcorner_filt'].x))

        df[scorer, 'single', 'angle', 'angle2'] = df[scorer, 'single', 'angle', 'angle2'].rolling(window, min_periods=1).mean()

        ### average angle

        df[scorer, 'single', 'angle', 'angleM'] = np.nanmean([df[scorer, 'single', 'angle', 'angle1'], df[scorer, 'single', 'angle', 'angle2']], axis=0)

        ### perform rotation

        for corner in corners:
            df[scorer, 'single', corner + '_rot', 'x'] = \
            (df[scorer, 'single', corner + '_filt'].x * np.cos(df[scorer, 'single', 'angle'].angleM)) + \
            (df[scorer, 'single', corner + '_filt'].y * np.sin(df[scorer, 'single', 'angle'].angleM))

            df[scorer, 'single', corner + '_rot', 'y'] = \
            (-1 * df[scorer, 'single', corner + '_filt'].x * np.sin(df[scorer, 'single', 'angle'].angleM)) + \
            (df[scorer, 'single', corner + '_filt'].y * np.cos(df[scorer, 'single', 'angle'].angleM))


        bpts = df[scorer, 'liz1'].columns.get_level_values(0).unique()

        nAnimals = 4
        nBpts = 5
        individuals = df[scorer].columns.get_level_values(0).unique()[0:nAnimals]
        top = df[scorer, 'single', 'toprightcorner_rot'].y.mean()
        bottom = df[scorer, 'single', 'bottomrightcorner_rot'].y.mean()
        bottomCorrection = 15
        trackYs = np.linspace(top, bottom - bottomCorrection, nAnimals + 1)

        scorer = df.columns.get_level_values(0)[0]

        for bp in tqdm(bpts):
            bpt = bp
            bptN = header['all_joints_names'].index(bpt)
            frameNames = list(f)

            for i in tnrange(header['nframes']):
                try:
                    j = 0
                    for coor in f[frameNames[i]]['coordinates'][0][bptN]:
                        x = coor[0] * np.cos(df.iloc[i][scorer, 'single', 'angle'].angleM) + \
                        coor[1] * np.sin(df.iloc[i][scorer, 'single', 'angle'].angleM)
                        y = -1 * coor[0] * np.sin(df.iloc[i][scorer, 'single', 'angle'].angleM) + \
                        coor[1] * np.cos(df.iloc[i][scorer, 'single', 'angle'].angleM)
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

        bpts = df[scorer, 'liz1'].columns.get_level_values(0).unique()
        x_left = np.mean(df[scorer, 'single', 'bottomleftcorner_rot'].x)
        x_right = np.mean(df[scorer, 'single', 'bottomrightcorner_rot'].x)

        for ind in individuals:
            for bp in bpts:
                df[scorer, ind, bp, 'temp'] = (-0.34 * 65 / (x_right - x_left)) * (df[scorer, ind, bp, 'x'] - x_left) + 39.12


        pixSum = []
        x_tl = int(df[scorer, 'single', 'topleftcorner'].x.mean())
        width = 900
        y_tl = int(df[scorer, 'single', 'topleftcorner'].y.mean())
        heigth = 350

        cap = cv2.VideoCapture(videoName)
        success,image = cap.read()
        count = 0
        while success:
            success,image = cap.read()
            if image is not None:
                count += 1
                pixSum = np.append(pixSum, np.sum(image[y_tl-170:y_tl+heigth, x_tl-70:x_tl+width, 0]))

        pixSum = np.append(pixSum, pixSum[-1])

        LDratio = np.percentile(pixSum, 80)/np.percentile(pixSum, 20)

        if LDratio > 3:
            lightTh = np.percentile(pixSum, 50)
            pixSumBool = pixSum > lightTh
        else:
            pixSumBool = pixSum > 0


        for ind in individuals:
            for bp in bpts:
                df[scorer, ind, bp, 'light'] = pixSumBool

        newFile = videoName[:-4] + '_Temperature.h5'
        print('saving: ' + newFile)
        df.to_hdf(newFile, key="df_with_missing", mode="w")

    except:
        pass
