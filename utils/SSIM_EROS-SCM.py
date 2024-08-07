# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: IIT
#     language: python
#     name: python3
# ---

import os
import sys
import glob
import cv2
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim,\
                             mean_squared_error as mse,\
                             normalized_root_mse as nrmse,\
                             normalized_mutual_information as nmi


# +
# Find the ERO-SNN folder and add it to the python path
current_dir = os.getcwd()

while os.path.basename(current_dir) != 'ERO-SNN':
    print(os.path.basename(current_dir))
    current_dir = os.path.dirname(current_dir)
    
print(f"Found ERO-SNN folder: {current_dir}")
sys.path.append(current_dir)
os.chdir(current_dir)

# +
''' For later
Input_files = glob.glob(os.path.join(input_data_dir, '**/input-out.mp4'), recursive=True)
EROS_files = glob.glob(os.path.join(EROS_data_dir, '**/eros-out.mp4'), recursive=True)

'''
Input_files = ['InputData/h36m_sample/cam2_S1_Directions/ch0dvs/input-out.mp4',
               'InputData/EyeTracking/user_5_0/ch0dvs/input-out.mp4',
               'InputData/MVSEC_short_outdoor/leftdvs/input-out.mp4']

EROS_files = ['EROS/h36m_sample/cam2_S1_Directions/ch0dvs/eros-out.mp4',
              'EROS/EyeTracking/user_5_0/ch0dvs/eros-out.mp4',
              'EROS/MVSEC_short_outdoor/leftdvs/eros-out.mp4']

SCM_data_dir = 'EROS_like'
SCM_files_N_4 = glob.glob(os.path.join(SCM_data_dir, '**/SCM_LIF_OUT_NEIGHBORS-tau=200.us-vt=0.1-vr=0.0-P=0-incoming_spikes=0-method_Neuron=exact-Num_Neighbours=4-beta=0.5-Wi=6.0-Wk=-3.0-method_Syn=exact-Sim_Clock=0.5ms-Sample_Perc=1.0-/**/scm-out.mp4'), recursive=True)
SCM_files_N_8 = glob.glob(os.path.join(SCM_data_dir, '**/SCM_LIF_OUT_NEIGHBORS-tau=200.us-vt=0.1-vr=0.0-P=0-incoming_spikes=0-method_Neuron=exact-Num_Neighbours=8-beta=0.5-Wi=6.0-Wk=-3.0-method_Syn=exact-Sim_Clock=0.5ms-Sample_Perc=1.0-/**/scm-out.mp4'), recursive=True)
SCM_files_N_12 = glob.glob(os.path.join(SCM_data_dir, '**/SCM_LIF_OUT_NEIGHBORS-tau=200.us-vt=0.1-vr=0.0-P=0-incoming_spikes=0-method_Neuron=exact-Num_Neighbours=12-beta=0.5-Wi=6.0-Wk=-3.0-method_Syn=exact-Sim_Clock=0.5ms-Sample_Perc=1.0-/**/scm-out.mp4'), recursive=True)
SCM_files_N_20 = glob.glob(os.path.join(SCM_data_dir, '**/SCM_LIF_OUT_NEIGHBORS-tau=200.us-vt=0.1-vr=0.0-P=0-incoming_spikes=0-method_Neuron=exact-Num_Neighbours=20-beta=0.5-Wi=6.0-Wk=-3.0-method_Syn=exact-Sim_Clock=0.5ms-Sample_Perc=1.0-/**/scm-out.mp4'), recursive=True)

SCM_files = [SCM_files_N_4, SCM_files_N_8]
# , SCM_files_N_12, SCM_files_N_20

datasets = ['Human3.6M', 'EyeTracking', 'MVSEC']
Number_of_Neighbours = [4, 8]
# , 12, 20

# +

results = {dataset: {n_neighbors:
    {'ssim_scores': {},'mse_scores': {},
     'nmse_scores': {}, 'nmi_scores': {}} for n_neighbors in Number_of_Neighbours} for dataset in datasets}

for SCM_file, N_Neighbors in zip(SCM_files, Number_of_Neighbours):
    for input_path, eros_path, scm_path, dataset_name in zip(Input_files, EROS_files, SCM_file, datasets):
        print(f"Processing combination: {input_path},\n {eros_path},\n {scm_path}\n")

        # Load videos
        input_video = cv2.VideoCapture(input_path)
        eros_video = cv2.VideoCapture(eros_path)
        scm_video = cv2.VideoCapture(scm_path)

        # Check if the videos are the same length and frame rate
        input_fps = input_video.get(cv2.CAP_PROP_FPS)
        eros_fps = eros_video.get(cv2.CAP_PROP_FPS)
        scm_fps = scm_video.get(cv2.CAP_PROP_FPS)
        
        input_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
        eros_frames = eros_video.get(cv2.CAP_PROP_FRAME_COUNT)
        scm_frames = scm_video.get(cv2.CAP_PROP_FRAME_COUNT)

        if not (eros_fps == scm_fps == input_fps and eros_frames == scm_frames == input_frames):
            print("Error: Videos do not have the same frame rate or frame count.")
            print(f"Input FPS: {input_fps}, EROS FPS: {eros_fps}, SCM FPS: {scm_fps}")
            print(f"Input Frames: {input_frames}, EROS Frames: {eros_frames}, SCM Frames: {scm_frames}")
            inp = input("Do you want to continue? (y/n)")
            if inp.lower() != 'y':
                break

        ssim_scores = {'Input-EROS': [], 'EROS-SCM': [], 'Input-SCM': []}
        mse_scores = {'Input-EROS': [], 'EROS-SCM': [], 'Input-SCM': []}
        nmse_scores = {'Input-EROS': [], 'EROS-SCM': [], 'Input-SCM': []}
        nmi_scores = {'Input-EROS': [], 'EROS-SCM': [], 'Input-SCM': []}

        while True:
            # Read frames from both videos
            ret1, frame1 = input_video.read()
            ret2, frame2 = eros_video.read()
            ret3, frame3 = scm_video.read()

            # Break the loop if any video ends
            if not ret1 or not ret2 or not ret3:
                break

            # Convert frames to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            
            # Apply binary threshold to convert to black and white
            _, bw1 = cv2.threshold(gray1, 254, 255, cv2.THRESH_BINARY)
            _, bw2 = cv2.threshold(gray2, 254, 255, cv2.THRESH_BINARY)
            _, bw3 = cv2.threshold(gray3, 254, 255, cv2.THRESH_BINARY)

            # Compute SSIM between the two frames and append to the list
            ssim12, _ = ssim(bw1, bw2, full=True)
            ssim23, _ = ssim(bw2, bw3, full=True)
            ssim13, _ = ssim(bw1, bw3, full=True)
            ssim_scores['Input-EROS'].append(ssim12)
            ssim_scores['EROS-SCM'].append(ssim23)            
            ssim_scores['Input-SCM'].append(ssim13)
            
            
            # Compute MSE between the frames and append to the list
            mse12 = mse(bw1, bw2)
            mse23 = mse(bw2, bw3)
            mse13 = mse(bw1, bw3)
            mse_scores['Input-EROS'].append(mse12)
            mse_scores['EROS-SCM'].append(mse23)            
            mse_scores['Input-SCM'].append(mse13)

            
            # Compute NMSE between the frames and append to the list
            nmse12 = nrmse(bw1, bw2)
            nmse23 = nrmse(bw2, bw3)
            nmse13 = nrmse(bw1, bw3)
            nmse_scores['Input-EROS'].append(nmse12)
            nmse_scores['EROS-SCM'].append(nmse23)
            nmse_scores['Input-SCM'].append(nmse13)
            
            
            # Compute NMI between the frames and append to the list
            nmi12 = nmi(bw1, bw2)
            nmi23 = nmi(bw2, bw3)
            nmi13 = nmi(bw1, bw3)
            nmi_scores['Input-EROS'].append(nmi12)
            nmi_scores['EROS-SCM'].append(nmi23)
            nmi_scores['Input-SCM'].append(nmi13)
            
        
        # Release video captures
        input_video.release()
        eros_video.release()
        scm_video.release()
        
        # Create dataframes
        results[dataset_name][N_Neighbors]['ssim_scores'] = pd.DataFrame(ssim_scores)
        results[dataset_name][N_Neighbors]['mse_scores'] = pd.DataFrame(mse_scores)
        results[dataset_name][N_Neighbors]['nmse_scores'] = pd.DataFrame(nmse_scores)
        results[dataset_name][N_Neighbors]['nmi_scores'] = pd.DataFrame(nmi_scores)

# +
# Create a directory to save the results
output_dir = 'quantitative_results'
os.makedirs(output_dir, exist_ok=True)

# Iterate through the results dictionary and save each DataFrame to a CSV file
for dataset_name, neighbors_dict in results.items():
    for n_neighbors, scores_dict in neighbors_dict.items():
        for score_type, df in scores_dict.items():
            # Define the filename
            filename = f"{dataset_name}_NN_{n_neighbors}_{score_type}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save the DataFrame to a CSV file
            df.to_csv(filepath, index=False)
            print(f"Saved {filepath}")
