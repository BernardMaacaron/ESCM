import cv2
import json
import os
import numpy as np
import sys
import glob
# sys.path.append('/home/ggoyal/code/hpe-core')
# sys.path.append('/home/ggoyal/code/hpe-core/example/movenet')
#
# from lib import init, MoveNet, Task
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
    # data_dvs = import_dvs(filePathOrName=data_dvs_file)
    print('File imported.')
    # data_dvs = importIitYarpBinaryDataLog(filePathOrName=data_dvs_file)
    
    # try:
    #     data_dvs['data']['left']['dvs']['ts'] /= args['ts_scaler']
    #     side = 'left'
    # except KeyError:
    #     data_dvs['data']['right']['dvs']['ts'] /= args['ts_scaler']
    #     side = 'right'
        
    data_dvs = next(BrianHF.find_keys(data_dvs, 'dvs'))
    data_ts = create_ts_list(args['fps'],data_dvs['ts'])
    # print(f"{data_dvs_file.split('/')[-3:-1]}: \n start: {data_dvs['data'][side]['dvs']['ts'][0]} \n stop: {data_dvs['data'][side]['dvs']['ts'][-1]}")
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

        frame = eros.get_frame()
        # frame = cv2.GaussianBlur(frame, (args['gauss_kernel'], args['gauss_kernel']), 0)

        if args['dev']:
            # keypoints = np.reshape(sample_anno['keypoints'], [-1, 3])
            # h, w = frame.shape
            # for i in range(len(keypoints)):
            #     frame = cv2.circle(frame, [int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)], 1, (255, 0, 0), 2)
            # frame = cv2.circle(frame, [int(sample_anno['center'][0] * w), int(sample_anno['center'][1] * h)], 1,
            #                    (255, 0, 0), 4)
            cv2.imshow('', frame)
            cv2.waitKey(1)
            if fi>50:
                break
        filename = os.path.basename(data_dvs_file)
        if args['write_images']:
            cv2.imwrite(os.path.join(output_path, f'{filename}_{fi:04d}.jpg'), frame)
        if args['write_video']:
            
            framergb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_out.write(framergb)

    if args['write_video']:
        print('writing')
        video_out.release()

    return


def setup_testing_list(path):
    if not os.path.exists(path):
        return []
    with open(str(path), 'r+') as f:
        poses = json.load(f)
    files = [sample['original_sample'] for sample in poses]
    files_unique = set(files)
    return files_unique


# $$ \text{FPS} = \frac{1}{\text{interval in seconds}} $$
#
# Given that the interval is 30 milliseconds, you need to convert this to seconds:
#
# $$ \text{interval in seconds} = \frac{30 \text{ ms}}{1000 \text{ ms/s}} = 0.03 \text{ s} $$
#
# Now, calculate the FPS:
#
# $$ \text{FPS} = \frac{1}{0.03 \text{ s}} \approx 33.33 $$
#
# So, the FPS should be approximately 33.33.

def main():
    # Define the variables directly
    eros_kernel = 8
    # frame_width = 640
    # frame_height = 480
    gauss_kernel = 7
    skip_image = None
    input_data_dir = 'InputData'
    output_base_path = 'EROS'
    write_images = False
    write_video = True
    frame_length = 30 #ms
    interval_length = 1000 #ms
    fps = interval_length/frame_length
    dev = False
    ts_scaler = 1.0
    
    # Ensure the base output path exists
    output_base_path = os.path.abspath(output_base_path)
    ensure_location(output_base_path)
    input_data_dir = os.path.abspath(input_data_dir)
    
    datasets = ['DHP19_Sample', 'EyeTracking', 'h36m_sample', 'MVSEC_short_outdoor']
    datasets = ['EyeTracking', 'h36m_sample','MVSEC_short_outdoor']

    for dataset in datasets:
        input_data_dir = os.path.join(input_data_dir, dataset)
        # Iterate over .log files in the input data directory
        log_files = glob.glob(os.path.join(input_data_dir, '**/data.log'), recursive=True)
        for log_file in log_files:
            try:
                # Create a corresponding output path
                relative_path = os.path.relpath(log_file, input_data_dir)
                output_path = os.path.join(output_base_path, dataset, os.path.dirname(relative_path))
                ensure_location(output_path)

                print("Processing:", log_file)
                print("Output path:", output_path)

                # Create a dictionary to hold the arguments
                args = {
                    'eros_kernel': eros_kernel,
                    # 'frame_width': frame_width,
                    # 'frame_height': frame_height,
                    'gauss_kernel': gauss_kernel,
                    'write_images': write_images,
                    'write_video': write_video,
                    'fps': fps,
                    'dev': dev,
                    'ts_scaler': ts_scaler
                }
                process(log_file, output_path, skip=skip_image, args=args)
            except:
                print(f"Error processing {log_file}")

if __name__ == '__main__':
    main()
