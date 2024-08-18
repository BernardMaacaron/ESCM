import cv2
import json
import os
import numpy as np
import sys
import glob
import os
from tqdm import tqdm
import bisect

# +
# Find the ERO-SNN folder and add it to the python path
current_dir = os.getcwd()

while os.path.basename(current_dir) != 'ERO-SNN':
    print(os.path.basename(current_dir))
    current_dir = os.path.dirname(current_dir)
    
print(f"Found ERO-SNN folder: {current_dir}")
sys.path.append(current_dir)
# -

import BrianHF
from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import EROS
from datasets.utils.export import ensure_location, str2bool #, get_movenet_keypoints, get_center
from bimvee.importIitYarp import importIitYarp as import_dvs
from bimvee.importAe import importAe
from bimvee.importIitYarp import importIitYarpBinaryDataLog


# +
# def create_ts_list(fps,ts):
#     out = dict()
#     out['ts'] = list()
#     x = np.arange(ts[0],ts[-1],1/fps)
#     for i in x:
#         out['ts'].append(i)
#     return out

# def create_ts_list(frame_length, frame_interval, ts):
#     out = {'ts': []}
    
#     # Create a list of timestamps starting from ts[0] and incrementing by 30ms
#     x = np.arange(ts[0], ts[-1], frame_interval / 1000.0)
    
#     # Convert ts to a NumPy array for faster operations
#     ts = np.array(ts)
    
#     for start_time in tqdm(x, desc="Processing time windows"):
#         # Create a window of frame_length ms
#         end_time = start_time + frame_length / 1000.0
        
#         # Use binary search to find the indices of the timestamps within the window
#         start_idx = bisect.bisect_left(ts, start_time)
#         end_idx = bisect.bisect_right(ts, end_time)
        
#         # Collect all timestamps within this window
#         window_ts = ts[start_idx:end_idx]
#         out['ts'].extend(window_ts)
    
#     return out
# -

def create_ts_list(frame_length, frame_interval, data_dvs):
    out = {'ts': [], 'x': [], 'y': []}
    
    # Extract ts, x, and y from data_dvs
    ts = np.array(data_dvs['ts'])
    x_coords = np.array(data_dvs['x'])
    y_coords = np.array(data_dvs['y'])
    
    # Sort the data by timestamps
    sorted_indices = np.argsort(ts)
    ts = ts[sorted_indices]
    x_coords = x_coords[sorted_indices]
    y_coords = y_coords[sorted_indices]
    
    # Create a list of timestamps starting from ts[0] and incrementing by frame_interval ms
    time_windows = np.arange(ts[0], ts[-1], frame_interval / 1000.0)
    
    for start_time in tqdm(time_windows, desc="Processing time windows"):
        # Create a window of frame_length ms
        end_time = start_time + frame_length / 1000.0
        
        # Use binary search to find the indices of the timestamps within the window
        start_idx = bisect.bisect_left(ts, start_time)
        end_idx = bisect.bisect_right(ts, end_time)
        
        # Collect all timestamps, x, and y within this window
        window_ts = ts[start_idx:end_idx]
        window_x = x_coords[start_idx:end_idx]
        window_y = y_coords[start_idx:end_idx]
        
        out['ts'].append(window_ts)
        out['x'].append(window_x)
        out['y'].append(window_y)
    
    return out

# +
# def process(data_dvs_file, output_path, skip=None, args=None):

#     if skip == None:
#         skip = 1
#     else:
#         skip = int(skip) + 1

#     print('Importing file...', data_dvs_file)
#     data_dvs = importAe(filePathOrName=data_dvs_file)
#     # data_dvs = import_dvs(filePathOrName=data_dvs_file)
#     print('File imported.')
#     # data_dvs = importIitYarpBinaryDataLog(filePathOrName=data_dvs_file)
    
#     # try:
#     #     data_dvs['data']['left']['dvs']['ts'] /= args['ts_scaler']
#     #     side = 'left'
#     # except KeyError:
#     #     data_dvs['data']['right']['dvs']['ts'] /= args['ts_scaler']
#     #     side = 'right'
        
#     data_dvs = next(BrianHF.find_keys(data_dvs, 'dvs'))
#     # data_ts = create_ts_list(args['fps'],data_dvs['ts'])
#     data_ts = create_ts_list(args['frame_length'], args['interval_length'], data_dvs['ts'])

#     # print(f"{data_dvs_file.split('/')[-3:-1]}: \n start: {data_dvs['data'][side]['dvs']['ts'][0]} \n stop: {data_dvs['data'][side]['dvs']['ts'][-1]}")
#     print(f"{data_dvs_file.split('/')[-3]}: \n start: {(-1)*data_dvs['tsOffset']} \n duration: {data_dvs['ts'][-1]}")
#     iterator = batchIterator(data_dvs, data_ts)
    
#     frame_width = np.max(data_dvs['x'])+1
#     frame_height = np.max(data_dvs['y'])+1
    
#     eros = EROS(kernel_size=args['eros_kernel'], frame_width=frame_width, frame_height=frame_height)

#     poses_movenet = []
#     if args['write_video']:
#         output_path_video = os.path.join(output_path,'eros-out.mp4')
#         print(output_path_video)
#         video_out = cv2.VideoWriter(output_path_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args['fps'],
#                                     (frame_width, frame_height))

#     for fi, (events, pose, batch_size) in enumerate(iterator):
#         sys.stdout.write(f'frame: {fi}/{len(data_ts["ts"])}\r')
#         sys.stdout.flush()

#         # if args['dev']:
#         #     print('frame: ', fi)
#         for ei in range(batch_size):
#             eros.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))
#         if fi % skip != 0:
#             continue

#         frame = eros.get_frame()
#         # frame = cv2.GaussianBlur(frame, (args['gauss_kernel'], args['gauss_kernel']), 0)

#         if args['dev']:
#             # keypoints = np.reshape(sample_anno['keypoints'], [-1, 3])
#             # h, w = frame.shape
#             # for i in range(len(keypoints)):
#             #     frame = cv2.circle(frame, [int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)], 1, (255, 0, 0), 2)
#             # frame = cv2.circle(frame, [int(sample_anno['center'][0] * w), int(sample_anno['center'][1] * h)], 1,
#             #                    (255, 0, 0), 4)
#             cv2.imshow('', frame)
#             cv2.waitKey(1)
#             if fi>50:
#                 break
        
        
#         filename = os.path.basename(data_dvs_file)
        
#         if args['write_images']:
#             images_path =  os.path.join(output_path,'Images')
#             ensure_location(images_path)
#             path = os.path.join(images_path, f'frame_{fi:08d}.jpg')
#             sys.stdout.write("Saving image to " + path + "\r")
#             cv2.imwrite(path, frame)
        
#         if args['write_video']:    
#             framergb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#             video_out.write(framergb)

#     if args['write_video']:
#         print('writing')
#         video_out.release()

#     return
# -

def process(data_dvs_file, output_path, skip=None, args=None):

    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1

    print('Importing file...', data_dvs_file)
    data_dvs = importAe(filePathOrName=data_dvs_file)
    print('File imported.')

        
    data_dvs = next(BrianHF.find_keys(data_dvs, 'dvs'))
    
    data_ts = create_ts_list(args['frame_length'], args['interval_length'], data_dvs)
    
    print(f"{data_dvs_file.split('/')[-3]}: \n start: {(-1)*data_dvs['tsOffset']} \n duration: {data_dvs['ts'][-1]}")
    # iterator = batchIterator(data_dvs, data_ts)
    
    frame_width = np.max(data_dvs['x'])+1
    frame_height = np.max(data_dvs['y'])+1
    
    # Calculate fps based on the number of frames and total duration
    total_duration = data_dvs['ts'][-1] - data_dvs['ts'][0]
    num_frames = len(data_ts['ts'])

    args['fps'] = int(num_frames / total_duration)
    print(f"FPS: {args['fps']}")

    
    eros = EROS(kernel_size=args['eros_kernel'], frame_width=frame_width, frame_height=frame_height)

    # poses_movenet = []
    
    # if args['write_video']:
    #     output_path_video = os.path.join(output_path,'eros-out.mp4')
    #     print(output_path_video)
    #     video_out = cv2.VideoWriter(output_path_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args['fps'],
    #                                 (frame_width, frame_height))

    for fi, (events_x, events_y) in enumerate(zip(data_ts['x'], data_ts['y'])):
        sys.stdout.write(f'frame: {fi}/{len(data_ts["ts"])}\r')
        sys.stdout.flush()


        # if args['dev']:
        #     print('frame: ', fi)
        for x, y in zip(events_x, events_y):
            eros.update(vx=x, vy=y)
        if fi % skip != 0:
            continue

        frame = eros.get_frame()
        
        
        filename = os.path.basename(data_dvs_file)
        
        if args['write_images']:
            images_path =  os.path.join(output_path,'Images')
            ensure_location(images_path)
            path = os.path.join(images_path, f'frame_{fi:08d}.jpg')
            sys.stdout.write("Saving image to " + path + "\r")
            cv2.imwrite(path, frame)
        
        # if args['write_video']:    
        #     framergb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #     video_out.write(framergb)

    # if args['write_video']:
    #     print('writing')
    #     video_out.release()

    return True


def setup_testing_list(path):
    if not os.path.exists(path):
        return []
    with open(str(path), 'r+') as f:
        poses = json.load(f)
    files = [sample['original_sample'] for sample in poses]
    files_unique = set(files)
    return files_unique


# +
# Define the variables directly
eros_kernel = 8
skip_image = None
input_data_dir = 'InputData'
output_base_path = 'EROS'
write_images = True
write_video = False
frame_length = 1.1 #ms
interval_length = 30 #ms
fps = interval_length/frame_length
dev = False
ts_scaler = 1.0

# Ensure the base output path exists
output_base_path = os.path.join(current_dir, output_base_path)
input_data_dir = os.path.join(current_dir, input_data_dir)

print('Input data directory: ', input_data_dir)
print('Output base path: ', output_base_path)

datasets = ['h36m_sample', 'EyeTracking', 'MVSEC_short_outdoor']

# Create a dictionary to hold the arguments
args = {
    'eros_kernel': eros_kernel,
    'write_images': write_images,
    'write_video': write_video,
    'frame_length': frame_length,
    'interval_length': interval_length,
    'fps': fps,
    'dev': dev,
    'ts_scaler': ts_scaler
    }

log_files = ['h36m_sample/cam2_S1_Directions/ch0dvs/data.log',
         'EyeTracking/user_5_1/ch0dvs_old/data.log',
         'MVSEC_short_outdoor/leftdvs/data.log']
# -

for i, log_file in enumerate(log_files):
    # Create a corresponding output path
    log_file = os.path.join(input_data_dir, log_file)
    output_path = os.path.join(output_base_path, datasets[i])
    print("Processing:", log_file)
    print("Output path:", output_path)
    
    if process(log_file, output_path, skip=skip_image, args=args):
        print(f"Processed {log_file}")
    else:
        print(f"Error processing {log_file}")   

# +

# for dataset in datasets:
#     input_data_dir = os.path.join(input_data_dir, dataset)
#     # Iterate over .log files in the input data directory
#     log_files = glob.glob(os.path.join(input_data_dir, '**/data.log'), recursive=True)
#     for log_file in log_files:
#         try:
#             # Create a corresponding output path
#             relative_path = os.path.relpath(log_file, input_data_dir)
#             output_path = os.path.join(output_base_path, dataset, os.path.dirname(relative_path))
#             ensure_location(output_path)

#             print("Processing:", log_file)
#             print("Output path:", output_path)

#             # Create a dictionary to hold the arguments
#             args = {
#                 'eros_kernel': eros_kernel,
#                 # 'frame_width': frame_width,
#                 # 'frame_height': frame_height,
#                 'gauss_kernel': gauss_kernel,
#                 'write_images': write_images,
#                 'write_video': write_video,
#                 'fps': fps,
#                 'dev': dev,
#                 'ts_scaler': ts_scaler
#             }
#             process(log_file, output_path, skip=skip_image, args=args)
#         except:
#             print(f"Error processing {log_file}")
