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

import time

# TODO XXX: Turn all lists into numpy arrays for the sake of memory efficiency
# -

# XXX: Extract the event stream using bimvee - needs refactoring to be more general
grid_width, grid_height= DAVIS_346B['width'], DAVIS_346B['height']
filePath = 'MVSEC_short_outdoor'
events = importAe(filePathOrName=filePath)

try:
    eventStream = events[0]['data']['left']['dvs']
except:
    eventStream = events[0]['data']['right']['dvs']
eventStream.popitem()

# ### Steps for the setting up the network:
# 1. Generate the input spikes
# 2. Set the parameters for the network (neurons and synapses)
# 3. Create the neuron group(s)
# 4. Create the synapses
# 5. Connect the synapses
# 6. Define the weights of the synapses
# 7. Set up monitors according to need
# 8. Run the simulation
# 9. (Optional) Visualize the results
#
# #### _1. Generate the input spikes from the event stream_
# ###### _Brian Simulation Scope_
#

# +
# HACK XXX: The input is now not in real time, must be fixed eventually to process real time event data
# IMPORTANT NOTE: Output is float, so we need to convert to Quantities (i.e give them units)


# Simulation Parameters
defaultclock.dt = 0.05*ms
SaveNumpyFrames = False
GenerateGIFs = False


simTime, clockStep, inputSpikesGen = BrianHF.event_to_spike(eventStream, grid_width, grid_height, dt = defaultclock.dt, timeScale = 1.0, samplePercentage=1.0)
# defaultclock.dt = clockStep*ms
print("Input event stream successfully converted to spike trains\n")
# -

# #### _2. Set the parameters for the network (neurons and synapses)_
# Parameter values can be tuned per namespace(this is done for clarity purposes).

# +
# NOTE: The values are unitless for now, time constants are in ms simply to be able to tune to the simulation clock

# Neuron Parameters
N_Neurons = grid_width * grid_height    # Number of neurons

Neuron_Params = {'tau': 0.1*ms, 'tauSpi': 0.5*ms, 'vt': 1.0, 'vr': 0.0, 'P': 0, 'incoming_spikes': 0, 'method_Neuron': 'rk2'}
tau = Neuron_Params['tau']
tauSpi = Neuron_Params['tauSpi']
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
Syn_Params = {'Num_Neighbours' : 2, 'beta': 1.5, 'Wi': 11.3, 'Wk': -2.0, 'method_Syn': 'rk2'}
Num_Neighbours = Syn_Params['Num_Neighbours']
beta = Syn_Params['beta']
Wi = Syn_Params['Wi']
Wk = Syn_Params['Wk']

# Generate the dictionary of parameters for the network
networkParams = {**Neuron_Params, **Syn_Params, 'Sim_Clock': defaultclock.dt}
# -

# #### _3. Create the neuron group(s)_

print('Creating Neuron Groups...')

# +
# TODO XXX: The events are running on the clockStep, I should at least fix them to use the (event_driven) setting in Brian2
neuronsGrid = NeuronGroup(N_Neurons, Eqs_Neurons, threshold='v>vt',
                            reset='''
                            v = vr
                            incoming_spikes_post = 0
                            ''',
                            refractory='0*ms',
                            events={'P_ON': 'v > vt', 'P_OFF': '(timestep(t - lastspike, dt) > timestep(dt, dt) and v <= vt)'},
                            method= 'euler',
                            namespace=Neuron_Params)


# Define the created events, the actions to be taken as well as when they should be evaluated and executed
neuronsGrid.run_on_event('P_ON', 'P = 1' , when = 'after_thresholds')
neuronsGrid.set_event_schedule('P_OFF', when = 'before_groups')
neuronsGrid.run_on_event('P_OFF', 'P = 0', when = 'before_groups')

# FIXME: Verify the grid coordinates and assign the X and Y values to the neurons accordingly
# Generate x and y values for each neuron
x_values = np.repeat(np.arange(grid_width), grid_height)
y_values = np.tile(np.arange(grid_height), grid_width)
neuronsGrid.X = x_values
neuronsGrid.Y = y_values
# -

print('Neuron Groups created successfully\n')

# #### _4. Create the synapses_

print('Creating Synapse Connections...')

# +
Syn_Input_Neurons = Synapses(inputSpikesGen, neuronsGrid, 'beta : 1 (constant)', on_pre='ExtIn_post = beta')

# NOTE: In hopes of reducing simulation time, I am using _pre keyword to avoid the need for an autapse. Hence only one Synapse group is needed
Syn_Neurons_Neurons = Synapses(neuronsGrid, neuronsGrid,
                               '''
                               Wi : 1 (constant)
                               Wk : 1 (constant)
                               ''',
                               on_pre={
                                   'pre':'incoming_spikes_post += 1; Exc_pre = Wi',
                                   'after_pre': 'Inh_post = clip(Inh_post + Wk * incoming_spikes_post/N_outgoing, Wk, 0)'},
                               method= 'euler',
                               namespace=Syn_Params)
# -

# #### _5. Connect the synapses_

# Connect the first synapses from input to G_neurons on a 1 to 1 basis
Syn_Input_Neurons.connect(j='i') 
Syn_Input_Neurons.beta = beta

# Connect the inhibitory group of synapses from a neuron to its neighbors. Reminder: No need for autapses as we are using _pre
Syn_Neurons_Neurons.connect(condition='i != j and abs(X_pre - X_post) <= Num_Neighbours and abs(Y_pre - Y_post) <= Num_Neighbours')
Syn_Neurons_Neurons.Wi = Wi
Syn_Neurons_Neurons.Wk = Wk

print('Synapse Connections created successfully\n')

# #### _6. Set up monitors_

SpikeMon_Input = SpikeMonitor(inputSpikesGen)    # Monitor the spikes from the input
SpikeMon_Neurons = SpikeMonitor(neuronsGrid)    # Monitor the spikes from the neuron grid
# StateMon_Neurons = StateMonitor(neuronsGrid, variables=True, record=True)    # Monitor the state variables - True for all variables. NOTE: WARNING!! Excessive memory usage

# ### 7. Run the simulation_

print(f"Running the simulation for {simTime} ms")
if 'ipykernel' in sys.modules:
    # Running in a Jupyter notebook
    print("Running in a Jupyter notebook")
    run(simTime*ms, profile=True)
    # profiling_summary(net)
else:
    # Not running in a Jupyter notebook
    print("Not running in a Jupyter notebook")
    run(simTime*ms, report=BrianHF.ProgressBar(), report_period=1*second, profile=True)
    # print(profiling_summary(net))
print("Simulation complete\n")

# #### _9. (Optional) Visualize the results_

# +
print("Generating Visualizable Outputs:")
resultPath = 'SimulationResults'
inputStr = BrianHF.filePathGenerator('tempnetIn', networkParams)
outputStr = BrianHF.filePathGenerator('tempnetOut', networkParams)

# Create the folder if it doesn't exist
if not os.path.exists(resultPath):
    os.makedirs(resultPath)

# Create the subfolders if they don't exist
subfolders = ['numpyFrames', 'gifs', 'videos']
for subfolder in subfolders:
    subfolderPath = os.path.join(resultPath, subfolder)
    if not os.path.exists(subfolderPath):
        os.makedirs(subfolderPath)
# -

# BrianHF.visualise_connectivity(Syn_Input_Neurons)
BrianHF.visualise_spikes([SpikeMon_Input, SpikeMon_Neurons])
# BrianHF.visualise_spike_difference(SpikeMon_Input, SpikeMon_Neurons)

# +
# Generate the frames for input and output
print("Generating Frames for Input...", end=' ')
inputFrames = BrianHF.generate_frames(SpikeMon_Input, grid_width, grid_height, num_neurons=N_Neurons)
print("Generation Complete.")
print("Generating Frames for Output...", end=' ')
outputFrames = BrianHF.generate_frames(SpikeMon_Neurons, grid_width, grid_height, num_neurons=N_Neurons)
print("Generation Complete.")

if SaveNumpyFrames:
    # Save the frames
    for name, frames in [(inputStr, inputFrames), (outputStr, outputFrames)]:
        filename = os.path.join(resultPath, 'numpyFrames', name+'.npy')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'numpyFrames', f"{name}_{int(time.time())}.npy")
        np.save(filename, frames)

if GenerateGIFs:
    # Generate the GIFs from the frames
    for name, frames in [(inputStr, inputFrames), (outputStr, outputFrames)]:
        filename = os.path.join(resultPath, 'gifs', name+'.gif')
        if os.path.exists(filename):
            filename = os.path.join(resultPath, 'gifs', f"{name}_{int(time.time())}.gif")
        BrianHF.generate_gif(frames, filename, simTime, replicateDuration=True, duration=1e-8)

# Generate the Videos from the frames
for name, frames in [(inputStr, inputFrames), (outputStr, outputFrames)]:
    filename = os.path.join(resultPath, 'videos', name+'.mp4')
    if os.path.exists(filename):
        filename = os.path.join(resultPath, 'videos', f"{name}_{int(time.time())}.mp4")
    BrianHF.generate_video(frames, filename, simTime/1000)

# +
# Generate the frames for input and output
print("Generating Frames for Input...", end=' ')
inputFrames = BrianHF.generate_frames(SpikeMon_Input, grid_width, grid_height, num_neurons=N_Neurons)
print("Generation Complete.")
print("Generating Frames for Output...", end=' ')
outputFrames = BrianHF.generate_frames(SpikeMon_Neurons, grid_width, grid_height, num_neurons=N_Neurons)
print("Generation Complete.")

if SaveNumpyFrames:
    # Save the frames
    np.save(os.path.join(resultPath, 'numpyFrames', inputStr+'.npy'), inputFrames)
    np.save(os.path.join(resultPath, 'numpyFrames', outputStr+'.npy'), outputFrames)

GenerateGifs = False
if GenerateGIFs:
    # Generate the GIFs from the frames
    print("Generating Input GIF", end=' ')
    BrianHF.generate_gif(inputFrames, os.path.join(resultPath, 'gifs', inputStr+'.gif'), simTime, replicateDuration=True, duration=1e-8)
    print("Generation Complete.")
    print("Generating Output GIF", end=' ')
    BrianHF.generate_gif(outputFrames, os.path.join(resultPath, 'gifs', outputStr+'.gif'), simTime, replicateDuration=True ,duration=1e-8)
    print("Generation Complete.")

# Generate the Videos from the frames
print("Generating Input Video", end=' ')
BrianHF.generate_video(outputFrames, os.path.join(resultPath, 'videos', inputStr+'.mp4'), simTime/1000)
print("Generation Complete.")
print("Generating Output Video", end=' ')
BrianHF.generate_video(outputFrames, os.path.join(resultPath, 'videos', outputStr+'.mp4'), simTime/1000)
print("Generation Complete.")

# +
# store('SimOut', 'SimulationOutput')   # Store the state of the simulation after running
# -

# """
# I SHOULD REMOVE THIS
# %%
# VISUALISATION
# BrianHF.visualise_spikes([SpikeMon_Input], figSize=(12,3), figTitle='Input Spikes')
# BrianHF.visualise_spikes([SpikeMon_Neurons], figSize=(12,3), figTitle='Output Spikes')
#
# Ploting the interspike interval
# BrianHF.visualise_interSpikeInterval(SpikeMon_Neurons, [5])
# BrianHF.visualise_neurons_states(StateMon_Neurons, [4, 5], 'all')    
#

