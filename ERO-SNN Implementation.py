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
tau = 10
vt = 1
vr = 0
v0 = 0 # Currently unused
gL = 1
K_GABA = 0 # Currently unused
K_AMPA = 0 # Currently unused
taugd = 1
tauad = 4  
taugr = 0.25 # Currently unused
tauar = 0.5  # Currently unused
w_N = 100
w_G = 2
w_A = 0 # Currently unused



defaultclock.dt = 1*us
# HACK XXX: The input is now not in real time, must be fixed eventually to process real time event data
# IMPORTANT NOTE: Output is float, so we need to convert to Quantities (i.e give them units)
simTime, clockStep, inputSpikesGen = BrianHF.event_to_spike(eventStream, grid_width, grid_height, timeScale = 1.0, samplePercentage=1.0)
defaultclock.dt = clockStep*ms
print("Input event stream successfully converted to spike trains\n")
# -

# #### _2. Set the parameters for the network (neurons and synapses)_
# Parameter values can be tuned at the top of the document (this is done for clarity purposes).

# +
# Neuron Parameters
N_Neurons = grid_width * grid_height    # Number of neurons
tau = tau*ms    # Time constant
vt = vt*mV    # Threshold Voltage
vr = vr*mV    # Reset Voltage
v0 = v0*mV    # Resting Voltage - Not necessarily used
gL = gL/ms    # Leak Conductance - Not necessarily used

'''
    Neighborhood Size (num_Neighbors) - Affects the number of neighbors a central neuron based on the L1 Distance
    Neighboring Neurons --> (abs(X_pre - X_post) <= num_Neighbors  and abs(Y_pre - Y_post) <= num_Neighbors)
'''
num_Neighbors = 2    # Number of neighbors

Eqs_Neurons = NeuronEquations.EQ_LIF_N    # Neuron Equations

# Synapse Parameters
K_GABA = K_GABA
K_AMPA = K_AMPA
taugd = taugd * ms
tauad = tauad * ms
taugr = taugr * ms
tauar = tauar * ms


# Generate the dictionary of parameters for the network
networkParams = {'N_N': N_Neurons, 'k': num_Neighbors, 'tau': tau, 'vt': vt, 'vr': vr, 'v0': v0, 'gL': gL,
                 'K_GABA': K_GABA, 'K_AMPA': K_AMPA, 'taugd': taugd, 'tauad': tauad, 'taugr': taugr, 'tauar': tauar, 'w_N': w_N, 'w_G': w_G, 'w_A': w_A}
# -

# #### _3. Create the neuron group(s)_

# +
# TODO XXX: The events are running on the clockStep, I should at least fix them to use the (event_driven) setting in Brian2
neuronsGrid = NeuronGroup(N_Neurons, Eqs_Neurons, threshold='v>vt', reset='v = vr', method='euler', refractory=5*ms)

# FIXME: Verify the grid coordinates and assign the X and Y values to the neurons accordingly
# Generate x and y values for each neuron
x_values = np.repeat(np.arange(grid_width), grid_height)
y_values = np.tile(np.arange(grid_height), grid_width)
neuronsGrid.X = x_values
neuronsGrid.Y = y_values
# -

# #### _4. Create the synapses_

Syn_Input_Neurons = Synapses(inputSpikesGen, neuronsGrid, 'w : volt/second', on_pre='Ia += w') # NOTE: Use Ia1 or Ia depending on the neuron equation used
Syn_Neurons_GABA = Synapses(neuronsGrid, neuronsGrid, 'w : volt/second', on_pre='Ig -= w') # NOTE: Use Ia1 or Ia depending on the neuron equation used
Syn_Neurons_AMPA = Synapses(neuronsGrid, neuronsGrid, 'w : volt/second', on_pre='Ia = Ia') # NOTE: Use Ia1 or Ia depending on the neuron equation used

# #### _5. Connect the synapses_

Syn_Input_Neurons.connect(condition= 'i==j')    # Connect the first synapses from input to G_neurons on a 1 to 1 basis
Syn_Neurons_GABA.connect(condition='i != j and abs(X_pre - X_post) <= num_Neighbors and abs(Y_pre - Y_post) <= num_Neighbors')    # Connect the second group of synapses from a neuron to its neighbors
Syn_Neurons_AMPA.connect(condition='i == j')     # Connect the last set of synapses from a neuron to itself (recurrent)

# #### _6. Define the weights of the synapses_

Syn_Input_Neurons.w = w_N * volt/second
Syn_Neurons_GABA.w = w_G * volt/second
Syn_Neurons_AMPA.w = w_A * volt/second

# #### _7. Set up monitors_

SpikeMon_Input = SpikeMonitor(inputSpikesGen)    # Monitor the spikes from the input
SpikeMon_Neurons = SpikeMonitor(neuronsGrid)    # Monitor the spikes from the neuron grid
# StateMon_Neurons = StateMonitor(neuronsGrid, variables=True, record=True)    # Monitor the state variables - True for all variables

# +
# store('PreSim', 'PreSimulationState')    # Store the state of the simulation before running
# -

# ### _8. Run the simulation_

print(f"Running the simulation for {simTime} ms")
net = Network(collect())
if 'ipykernel' in sys.modules:
    # Running in a Jupyter notebook
    print("Running in a Jupyter notebook")
    net.run(simTime*ms, profile=True)
    profiling_summary(net)
else:
    # Not running in a Jupyter notebook
    print("Not running in a Jupyter notebook")
    net.run(simTime*ms, report=BrianHF.ProgressBar(), report_period=1*second, profile=True)
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
# BrianHF.visualise_spikes([SpikeMon_Input, SpikeMon_Neurons])
# BrianHF.visualise_spike_difference(SpikeMon_Input, SpikeMon_Neurons)

# +

# Generate the frames for input and output
print("Generating Frames for Input...", end=' ')
inputFrames = BrianHF.generate_frames(SpikeMon_Input, grid_width, grid_height, num_neurons=N_Neurons)
print("Generation Complete.")
print("Generating Frames for Output...", end=' ')
outputFrames = BrianHF.generate_frames(SpikeMon_Neurons, grid_width, grid_height, num_neurons=N_Neurons)
print("Generation Complete.")
# Save the frames
np.save(os.path.join(resultPath, 'numpyFrames', inputStr+'.npy'), inputFrames)
np.save(os.path.join(resultPath, 'numpyFrames', outputStr+'.npy'), outputFrames)

# Generate the GIFs from the frames
print("Generating Input GIF", end=' ')
BrianHF.generate_gif(inputFrames, os.path.join(resultPath, 'gifs', inputStr+'.gif'), simTime, replicateDuration=True, duration=1e-8)
print("Generation Complete.")
print("Generating Output GIF", end=' ')
BrianHF.generate_gif(outputFrames, os.path.join(resultPath, 'gifs', outputStr+'.gif'), simTime, replicateDuration=True ,duration=1e-8)
print("Generation Complete.")

# Generate the Videos from the frames
print("Generating Input Video", end=' ')
BrianHF.generate_video(outputFrames, os.path.join(resultPath, 'videos', inputStr+'.mp4'), simTime)
print("Generation Complete.")
print("Generating Output Video", end=' ')
BrianHF.generate_video(outputFrames, os.path.join(resultPath, 'videos', outputStr+'.mp4'), simTime)
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

