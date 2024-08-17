import cv2
import json
import os
import numpy as np
import sys
import glob
import os
from tqdm import tqdm
import bisect

# Find the ERO-SNN folder and add it to the python path
current_dir = os.getcwd()

while os.path.basename(current_dir) != 'ERO-SNN':
    print(os.path.basename(current_dir))
    current_dir = os.path.dirname(current_dir)
    
print(f"Found ERO-SNN folder: {current_dir}")
sys.path.append(current_dir)

import BrianHF
from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import EROS
from datasets.utils.export import ensure_location, str2bool #, get_movenet_keypoints, get_center
from bimvee.importIitYarp import importIitYarp as import_dvs
from bimvee.importAe import importAe
from bimvee.importIitYarp import importIitYarpBinaryDataLog

def create_ts_list(frame_length, frame_interval, ts):
    out = {'ts': []}
    
    # Create a list of timestamps starting from ts[0] and incrementing by 30ms
    x = np.arange(ts[0], ts[-1], frame_interval / 1000.0)
    
    # Convert ts to a NumPy array for faster operations
    ts = np.array(ts)
    
    for start_time in tqdm(x, desc="Processing time windows"):
        # Create a window of frame_length ms
        end_time = start_time + frame_length / 1000.0
        
        # Use binary search to find the indices of the timestamps within the window
        start_idx = bisect.bisect_left(ts, start_time)
        end_idx = bisect.bisect_right(ts, end_time)
        
        # Collect all timestamps within this window
        window_ts = ts[start_idx:end_idx]
        out['ts'].extend(window_ts)
    
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
    data_ts = create_ts_list(args['frame_length'], args['interval_length'], data_dvs['ts'])
    
    print(f"{data_dvs_file.split('/')[-3]}: \n start: {(-1)*data_dvs['tsOffset']} \n duration: {data_dvs['ts'][-1]} \n scaled duration: {data_ts['ts'][-1]}")
    iterator = batchIterator(data_dvs, data_ts)
    
    frame_width = np.max(data_dvs['x'])+1
    frame_height = np.max(data_dvs['y'])+1
    
    # Calculate fps based on the number of frames and total duration
    total_duration = data_dvs['ts'][-1] - data_dvs['ts'][0]
    num_frames = len(data_ts['ts'])

    args['fps'] = int(min(num_frames / total_duration, 1024))
    print(f"FPS: {args['fps']}")

    
    if args['write_video']:
        output_path_video = os.path.join(output_path,'scm-out.mp4')
        print(output_path_video)
        video_out = cv2.VideoWriter(output_path_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args['fps'],
                                    (frame_width, frame_height))

    for fi, (events, pose, batch_size) in enumerate(iterator):
        sys.stdout.write(f'frame: {fi}/{len(data_ts["ts"])}\r')
        sys.stdout.flush()
        # old_vx = None
        # old_vy = None

        frame = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # if args['dev']:
        #     print('frame: ', fi)
        for ei in range(batch_size):                
            vx=int(events['x'][ei])
            vy=int(events['y'][ei])
            
            # if old_vx is None and old_vy is None:
            #     old_vx = vx
            #     old_vy = vy
            # else:
            #     frame[old_vx,old_vy] = 0
            #     old_vx = vx
            #     old_vy = vy
                
            frame[vy,vx] = 255
        
            
        if fi % skip != 0:
            continue

        # frame = cv2.GaussianBlur(frame, (args['gauss_kernel'], args['gauss_kernel']), 0)

        filename = os.path.basename(data_dvs_file)
        
        if args['write_images']:
            cv2.imwrite(os.path.join(output_path, f'{filename}_{fi:04d}.jpg'), frame)
        if args['write_video']:
            framergb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_out.write(frame)

    if args['write_video']:
        video_out.release()

    return

def main():
    # Define the variables directly
    eros_kernel = 8
    gauss_kernel = 7
    skip_image = None
    input_data_dir = 'SimulationResultsFinal'
    output_base_path = 'EROS_like'
    write_images = False
    write_video = True
    frame_length = 1.1 #ms
    interval_length = 30 #ms
    fps = interval_length/frame_length
    dev = False
    ts_scaler = 1.0
    
    # Ensure the base output path exists
    output_base_path = os.path.abspath(output_base_path)
    ensure_location(output_base_path)
    input_data_dir = os.path.abspath(input_data_dir)
    
    datasets = ['EyeTracking', 'h36m_sample', 'MVSEC_short_outdoor']

    # Create a dictionary to hold the arguments
    args = {
        'eros_kernel': eros_kernel,
        'gauss_kernel': gauss_kernel,
        'write_images': write_images,
        'write_video': write_video,
        'frame_length': frame_length,
        'interval_length': interval_length,
        'fps': fps,
        'dev': dev,
        'ts_scaler': ts_scaler
        }
    
    # input_data_dir = os.path.join(input_data_dir, dataset)
    log_files = []
    h36m = glob.glob(os.path.join(input_data_dir, '**/cam2_S1_Directions/ch0dvs/YarpSpikeLog/**/data.log'), recursive=True)
    EyeTracking = glob.glob(os.path.join(input_data_dir, '**/user_5_1/ch0dvs_old/YarpSpikeLog/**/data.log'), recursive=True)
    MVSEC = glob.glob(os.path.join(input_data_dir, '**/MVSEC_short_outdoor/YarpSpikeLog/**/data.log'), recursive=True)
    
    log_files.extend(h36m)
    log_files.extend(EyeTracking)
    log_files.extend(MVSEC)
    
    print(log_files)
    
    for log_file in log_files:
        # Create a corresponding output path
        relative_path = os.path.relpath(log_file, input_data_dir)
        output_path = os.path.join(output_base_path, os.path.dirname(relative_path))
        ensure_location(output_path)

        print("Processing:", log_file)
        print("Output path:", output_path)


        if process(log_file, output_path, skip=skip_image, args=args):
            print(f"Processed {log_file}")
        else:
            print(f"Error processing {log_file}")  
if __name__ == '__main__':
    main()