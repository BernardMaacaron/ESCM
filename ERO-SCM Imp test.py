# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
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

# +
from brian2 import *
# set_device('cpp_standalone') # for faster execution - builds a standalone C++ program

import NeuronEquations
import BrianHF
import numpy as np

from bimvee.importAe import importAe
from EvCamDatabase import DAVIS_346B

import os
import gc
import time

# TODO XXX: Turn all lists into numpy arrays for the sake of memory efficiency

# +
# XXX: Extract the event stream using bimvee - needs refactoring to be more general
# grid_width, grid_height= ATIS_GEN3['width'], ATIS_GEN3['height']
# filePath = 'massi'
# events = importAe(filePathOrName=filePath)
# eventStream = events['data']['ch0']['dvs']
# -

grid_width, grid_height= DAVIS_346B['width'], DAVIS_346B['height']
filePath = 'MVSEC_short_outdoor'
events = importAe(filePathOrName=filePath)
try:
    eventStream = events[0]['data']['left']['dvs']
except:
    eventStream = events[0]['data']['right']['dvs']
eventStream.popitem()
print()

# ## Steps for the setting up the network:
# 1. Generate the input spikes
# 2. Set the parameters for the Simulation and Network (Neurons, Synapses, etc.)
# 3. Prepare the directory structure for saving the results
# 4. Create the neuron group(s)
# 5. Create the synapses
# 6. Connect the synapses
# 7. Define the weights of the synapses
# 8. Set up monitors according to need
# 9. Run the simulation
# 10. (Optional) Visualize the results
#

# ### 1. Generate the input spikes from the event stream
# ###### Brian Simulation Scope

# +
# HACK XXX: The input is now not in real time, must be fixed eventually to process real time event data
# IMPORTANT NOTE: Output is float, so we need to convert to Quantities (i.e give them units)

# Simulation Parameters
defaultclock.dt = 0.1*ms
samplePerc = 1.0
SaveNumpyFrames = False
GenerateGIFs = False
GenerateVideos = True

GenerateInputVisuals = False
GenerateOutputVisuals = False

simTime, clockStep, inputSpikesGen = BrianHF.event_to_spike(eventStream, grid_width, grid_height,
                                                            dt = defaultclock.dt, timeScale = 1.0, samplePercentage=samplePerc, interSpikeTiming=None)
# defaultclock.dt = clockStep*ms
print("Input event stream successfully converted to spike trains\n")
# -

# ### 2. Set the parameters for the Simulation and Network (Neurons, Synapses, etc.)
# Parameter values can be tuned per namespace (this is done for clarity purposes).

# +
# NOTE: The values are unitless for now, time constants are in ms simply to be able to tune to the simulation clock

# Neuron Parameters
N_Neurons = grid_width * grid_height    # Number of neurons

Neuron_Params = {'tau': 0.2*ms, 'vt': 0.1, 'vr': 0.0, 'P': 0, 'incoming_spikes': 0, 'method_Neuron': 'exact'}
tau = Neuron_Params['tau']
vt = Neuron_Params['vt']
vr = Neuron_Params['vr']
P = Neuron_Params['P']
incoming_spikes = Neuron_Params['incoming_spikes']

Eqs_Neurons = NeuronEquations.EQ_SCM_IF    # Neurons Equation
 
# Synapse Parameters
'''
Neighborhood Size (num_Neighbors) - Affects the number of neighbors a central neuron based on the L1 Distance
Neighboring Neurons --> (abs(X_pre - X_post) <= Num_Neighbours  and abs(Y_pre - Y_post) <= Num_Neighbours)
'''
Syn_Params = {'Num_Neighbours' : 8, 'beta': 0.4, 'Wi': 0.6, 'Wk': -4.25, 'method_Syn': 'exact'}
Num_Neighbours = Syn_Params['Num_Neighbours']
beta = Syn_Params['beta']
Wi = Syn_Params['Wi']
Wk = Syn_Params['Wk']

# Generate the dictionary of parameters for the network
networkParams = {**Neuron_Params, **Syn_Params, 'Sim_Clock': defaultclock.dt, 'Sample_Perc': samplePerc}
display(networkParams)
# -

# ### 3. Prepare the directory structure for saving the results

# +
resultPath = 'SimulationResults'
inputStr = BrianHF.filePathGenerator('SCM_IF_IN_TEMP', networkParams).replace(" ", "")
outputStr = BrianHF.filePathGenerator('SCM_IF_OUT_TEMP', networkParams).replace(" ", "")

# Create the folder if it doesn't exist
if not os.path.exists(resultPath):
    os.makedirs(resultPath)

logPath = os.path.join('YarpSpikeLog', outputStr)
# Create the subfolders if they don't exist
subfolders = [logPath, 'spikeFrames', 'numpyFrames', 'gifs', 'videos']
for subfolder in subfolders:
    subfolderPath = os.path.join(resultPath, subfolder)
    if not os.path.exists(subfolderPath):
        os.makedirs(subfolderPath)
# -

# ### 4. Create the neuron group(s)

print('Creating Neuron Groups...')

# +
# TODO XXX: The events are running on the clockStep, I should at least fix them to use the (event_driven) setting in Brian2
neuronsGrid = NeuronGroup(N_Neurons, Eqs_Neurons, threshold='v>vt',
                            reset='''
                            v = vr
                            incoming_spikes_post = 0
                            ''',
                            refractory='0.01*ms',
                            events={'P_ON': 'v > vt', 'P_OFF': '(timestep(t - lastspike, dt) > timestep(dt, dt) and v <= vt)'},
                            method= Neuron_Params['method_Neuron'],
                            namespace=Neuron_Params)


# Define the created events, the actions to be taken as well as when they should be evaluated and executed
neuronsGrid.set_event_schedule('P_ON', when = 'after_thresholds')
neuronsGrid.run_on_event('P_ON', 'P = 1' , when = 'after_thresholds')
neuronsGrid.set_event_schedule('P_OFF', when = 'groups')
neuronsGrid.run_on_event('P_OFF', 'P = 0', when = 'groups')

# FIXME: Verify the grid coordinates and assign the X and Y values to the neurons accordingly
# Generate x and y values for each neuron
# x_values = np.repeat(np.arange(grid_width), grid_height)
# y_values = np.tile(np.arange(grid_height), grid_width)
y_values, x_values = divmod(neuronsGrid.i, grid_width)
neuronsGrid.X = x_values
neuronsGrid.Y = y_values
# -

print('Neuron Groups created successfully\n')

# #### 5. Create the synapses

print('Creating Synapse Connections...')

# +
Syn_Input_Neurons = Synapses(inputSpikesGen, neuronsGrid,
                             'beta : 1 (constant)',
                             on_pre='ExtIn_post = beta',
                             method='exact',
                             namespace=Syn_Params)

# NOTE: In hopes of reducing simulation time, I am using _pre keyword to avoid the need for an autapse. Hence only one Synapse group is needed
Syn_Neurons_Neurons = Synapses(neuronsGrid, neuronsGrid,
                               '''
                               Wi : 1
                               Wk : 1
                               ''',
                               on_pre={
                                   'pre':'incoming_spikes_post += 1; Exc_pre = Wi',
                                   'pre_2': 'Inh_post = P_post * Wk * incoming_spikes_post'},
                               method= 'exact',
                               namespace=Syn_Params)
# -

# ### 6. Connect the synapses

# Connect the first synapses from input to G_neurons on a 1 to 1 basis
Syn_Input_Neurons.connect(j='i') 
Syn_Input_Neurons.beta = beta

# +
''' NOTE: The following code implements the connections for the Neurons to Neurons synapses
which as mentioned before implicitly includes the autapses.
The code below is a faster implementation of:
    Syn_Neurons_Neurons.connect(condition='i != j and abs(X_pre - X_post) <= Num_Neighbours and abs(Y_pre - Y_post) <= Num_Neighbours')
'''

indexes_i, indexes_j = BrianHF.calculate_ChebyshevNeighbours(neuronsGrid, Num_Neighbours)
Syn_Neurons_Neurons.connect(i=indexes_i, j=indexes_j)
Syn_Neurons_Neurons.Wi = Wi
Syn_Neurons_Neurons.Wk = Wk
# -

print('Synapse Connections created successfully\n')

# ### 7. Set up monitors

# +
if GenerateInputVisuals:
    SpikeMon_Input = SpikeMonitor(inputSpikesGen)    # Monitor the spikes from the input

# IMPORTANT NOTE: The SpikeMonitor for the active neurons is moved to the simulation block to avoid memory issues.
# The data is now broken down into smaller chunks and saved to disk

# StateMon_Neurons = StateMonitor(neuronsGrid, variables=True, record=True)    # Monitor the state variables - True for all variables. NOTE: WARNING!! Excessive memory usage
# -

# ### 8. Run the simulation

# +
BrianLogger.log_level_error()    # Only log errors to avoid excessive output

warn = ''
N = 10  # Number of runs
run_time = simTime / N  # Duration of each run
spikeTimeStamps = np.array([])
spikeIndices = np.array([], dtype=int)

# SpikeMon_Neurons = SpikeMonitor(neuronsGrid)    # Monitor the spikes from the neuron grid


print("Starting simulation - Total time: ", simTime*ms)
for i in range(N):
    print(f"\nRunning simulation chunk {i+1}/{N} for {run_time} ms")
    SpikeMon_Neurons = SpikeMonitor(neuronsGrid)    # Monitor the spikes from the neuron grid
    
    if 'ipykernel' in sys.modules:
        # Running in a Jupyter notebook
        print("Running in a Jupyter notebook")
        reportVar = 'text'
        report_periodVar = 5*second
    else:
        # Not running in a Jupyter notebook
        print("Not running in a Jupyter notebook")
        reportVar = BrianHF.ProgressBar()
        report_periodVar = 2*second
    
    FailedSim = False
    
    run(run_time*ms, report=reportVar, report_period=report_periodVar, profile=False)

        
    # try:
    #     run(run_time*ms, report=reportVar, report_period=report_periodVar, profile=False)
    #     # print(profiling_summary())
    # except Exception as e:
    #     print("Simulation failed:", str(e), '\n')
    #     print("Trying to re-run by dividing the time into smaller chunks\n")
    #     N_inner = 5
    #     run_time_inner = run_time / N_inner
    #     for j in range(N_inner):
    #         print(f"Running INNER simulation chunk {j+1}/{N_inner} for {run_time_inner} ms")
    #         try:
    #             run(run_time_inner*ms, report=reportVar, report_period=report_periodVar, profile=False)
    #         except Exception as e_inner:
    #             print("Simulation failed:", str(e_inner))
    #             warn = " - WARNING: The results may be incomplete"
    #             FailedSim = True
    #             break
    #     if not FailedSim:
    #         break
        
        
    if SpikeMon_Neurons.num_spikes > 0:
        spikeTimeStamps = np.append(spikeTimeStamps, SpikeMon_Neurons.t[:])
        spikeIndices = np.append(spikeIndices, SpikeMon_Neurons.i[:])
    
    del SpikeMon_Neurons
    gc.collect()
    
print("Simulation complete",warn,"\n")

# -

# #### _9. (Optional) Visualize the results_

# +
# print("Generating Visualizable Outputs:")
# BrianHF.visualise_connectivity(Syn_Input_Neurons)
# BrianHF.visualise_spikes([SpikeMon_Input, SpikeMon_Neurons])
# BrianHF.visualise_spike_difference(SpikeMon_Input, SpikeMon_Neurons)
# -


# Generate the frames for input
if GenerateInputVisuals:
    print("Generating Frames for Input...", end=' ')
    inputFrames = BrianHF.generate_frames(SpikeMon_Input.t, SpikeMon_Input.i, grid_width, grid_height, num_neurons=N_Neurons)
    print("Input Frames Generation Complete.")

    # Save the frames
    if SaveNumpyFrames:
        print("Saving Input Frames as Numpy Arrays...")
        filename = os.path.join(resultPath, 'numpyFrames', inputStr+'.npy')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'numpyFrames', f"{inputStr}_{int(time.time())}.npy")
        np.save(filename, inputFrames)
        print("Input Numpy Array Saved.")


    # Generate the GIFs from the frames
    if GenerateGIFs:
        print("Generating Input GIFs...")
        filename = os.path.join(resultPath, 'gifs', inputStr+'.gif')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'gifs', f"{inputStr}_{int(time.time())}.gif")
        BrianHF.generate_gif(inputFrames, filename, simTime, replicateDuration=True, duration=1e-8)
        print("Input GIF Generation Complete.")


    # Generate the Videos from the frames
    if GenerateVideos:
        print("Generating Videos...")
        filename = os.path.join(resultPath, 'videos', inputStr+'.mp4')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'videos', f"{inputStr}_{int(time.time())}.mp4")
        BrianHF.generate_video(inputFrames, filename, simTime/1000)
        print("Input Video Generation Complete.")

    del inputFrames
    del SpikeMon_Input
    gc.collect()


print("Exporting Yarp Spike Log...")
filename = os.path.join(resultPath, logPath, 'data.log')
if os.path.exists(filename):
    filename = os.path.join(resultPath, logPath, f"data_{int(time.time())}.log")
BrianHF.generate_YarpDvs(spikeTimeStamps, spikeIndices, neuronsGrid, filename)


if GenerateOutputVisuals:
    
    print("Generating Frames for Output...", end=' ')
    #outputFrames = BrianHF.generate_frames(SpikeMon_Neurons.t/ms, SpikeMon_Neurons.i, grid_width, grid_height, num_neurons=N_Neurons)
    outputFrames = BrianHF.generate_frames(spikeTimeStamps, spikeIndices, grid_width, grid_height, num_neurons=N_Neurons)
    print("Output Frames Generation Complete.")
    videoTime = spikeTimeStamps[-1] if len(spikeTimeStamps) > 0 else simTime
    print(f"Video Time: {videoTime}")
    
    del spikeTimeStamps, spikeIndices
    gc.collect()
    
    
    # Save the frames
    if SaveNumpyFrames:
        print("Saving Input Frames as Numpy Arrays...")
        filename = os.path.join(resultPath, 'numpyFrames', outputStr+'.npy')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'numpyFrames', f"{outputStr}_{int(time.time())}.npy")
        np.save(filename, outputFrames)
        print("Output Numpy Array Saved.")


    # Generate the GIFs from the frames
    if GenerateGIFs:
        print("Generating Input GIFs...")
        filename = os.path.join(resultPath, 'gifs', outputStr+'.gif')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'gifs', f"{outputStr}_{int(time.time())}.gif")
        BrianHF.generate_gif(outputFrames, filename, simTime, replicateDuration=True, duration=1e-8)
        print("Output GIF Generation Complete.")


    # Generate the Videos from the frames
    if GenerateVideos:
        print("Generating Videos...")
        filename = os.path.join(resultPath, 'videos', outputStr+'.mp4')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'videos', f"{outputStr}_{int(time.time())}.mp4")
        BrianHF.generate_video(outputFrames, filename, videoTime)
        print("Output Video Generation Complete.")

    del outputFrames

