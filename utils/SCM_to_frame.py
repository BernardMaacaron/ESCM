import cv2
import json
import os
import numpy as np
import sys
import glob

import BrianHF
from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import EROS
from datasets.utils.export import ensure_location, str2bool #, get_movenet_keypoints, get_center
from bimvee.importIitYarp import importIitYarp as import_dvs
from bimvee.importAe import importAe
from bimvee.importIitYarp import importIitYarpBinaryDataLog

def create_ts_list(fps,ts):
    out = dict()
    out['ts'] = list()
    x = np.arange(ts[0],ts[-1],1/fps)
    for i in x:
        out['ts'].append(i)
    return out

def process(data_dvs_file, output_path, skip=None, args=None):

    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1

    print('Importing file...', data_dvs_file)
    data_dvs = importAe(filePathOrName=data_dvs_file)
    print('File imported.')

        
    data_dvs = next(BrianHF.find_keys(data_dvs, 'dvs'))
    data_ts = create_ts_list(args['fps'],data_dvs['ts'])
    
    print(f"{data_dvs_file.split('/')[-3]}: \n start: {(-1)*data_dvs['tsOffset']} \n duration: {data_dvs['ts'][-1]}")
    iterator = batchIterator(data_dvs, data_ts)
    
    frame_width = np.max(data_dvs['x'])+1
    frame_height = np.max(data_dvs['y'])+1
    
    eros = EROS(kernel_size=args['eros_kernel'], frame_width=frame_width, frame_height=frame_height)

    poses_movenet = []
    if args['write_video']:
        output_path_video = os.path.join(output_path,'eros-out.mp4')
        print(output_path_video)
        video_out = cv2.VideoWriter(output_path_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args['fps'],
                                    (frame_width, frame_height))

    for fi, (events, pose, batch_size) in enumerate(iterator):
        sys.stdout.write(f'frame: {fi}/{len(data_ts["ts"])}\r')
        sys.stdout.flush()

        # if args['dev']:
        #     print('frame: ', fi)
        for ei in range(batch_size):
            eros.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))
        if fi % skip != 0:
            continue


        # frame = cv2.GaussianBlur(frame, (args['gauss_kernel'], args['gauss_kernel']), 0)

        filename = os.path.basename(data_dvs_file)
        if args['write_images']:
            cv2.imwrite(os.path.join(output_path, f'{filename}_{fi:04d}.jpg'), frame)
        if args['write_video']:
            video_out.write(frame)
            print('writing')

    if args['write_video']:
        video_out.release()

    return

