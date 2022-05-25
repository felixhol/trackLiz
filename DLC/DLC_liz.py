# import deeplabcut
import os
from pathlib import Path
import glob

# os.environ["DLClight"]="True"

import deeplabcut

# video_path = glob.glob('/home/felix/Dropbox/Hong Kong/*/*.AVI')
# config_path = deeplabcut.create_new_project('liz','fe', video_path, copy_videos=False, multianimal=True)

###### CHANGE THE config_path TO WHERE THE config.yaml FILE IS STORED ON YOUR LOCAL COMPUTER.

config_path = '/home/felix/lizards/liz-fe-2020-11-13-pcd_box/config.yaml'

# deeplabcut.add_new_videos(config_path, ['/home/felix/lizards/TLC00002.avi'])

# deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False, crop=False)

# deeplabcut.label_frames(config_path)

# deeplabcut.SkeletonBuilder(config_path)

# deeplabcut.check_labels(config_path)

# deeplabcut.cropimagesandlabels(config_path, userfeedback=False, size=(640,640), cropdata=True, updatevideoentries=False)

# deeplabcut.create_multianimaltraining_dataset(config_path, windows2linux=True)

## deeplabcut.create_training_dataset(config_path, num_shuffles=2)

# deeplabcut.train_network(config_path)

# deeplabcut.evaluate_network(config_path, plotting=True, gputouse=0)
#
# video = ['/home/felix/lizards/HKdata/11062019_60616263-2.AVI','/home/felix/lizards/HKdata/11062019_60616263-3.AVI']

videos = glob.glob('/home/felix/Dropbox/HongKong/*/*trim.AVI') #+ glob.glob('/home/felix/Dropbox/HongKong/2*/*.AVI')

print('analyzing ' + str(len(videos)) + ' videos:')
print(videos)

#
#deeplabcut.analyze_videos(config_path, videos)
# #
#deeplabcut.create_video_with_all_detections(config_path, videos, 'DLC_resnet50_lizNov2shuffle1_140000')
#
#deeplabcut.convert_detections2tracklets(config_path, videos, shuffle=1, videotype='mp4', trainingsetindex=0, track_method='box')
#
trackletPickles = glob.glob('/home/felix/Dropbox/HongKong/29*/*trim*_bx.pickle')

print('analyzing:')
print(trackletPickles)

#
#
for tracklet in trackletPickles:
    deeplabcut.convert_raw_tracks_to_h5(config_path, tracklet)
#    deeplabcut.convert_raw_tracks_to_h5(config_path, tracklet)

deeplabcut.create_labeled_video(config_path, videos, videotype='.mp4', track_method='box', draw_skeleton = False, color_by='individual')
