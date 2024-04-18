from brian2 import *
import cv2
import matplotlib.pyplot as plt
import time

# Visualize the connectivity
def visualise_connectivity(SynapsesGroup):
    Ns = len(SynapsesGroup.source)
    Nt = len(SynapsesGroup.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(SynapsesGroup.i, SynapsesGroup.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(SynapsesGroup.i, SynapsesGroup.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

# Visualize the chosen states of a list of neurons
def visualise_neurons_states(stateMonitor, neuron_indices, states):
    if states == 'all':
        statesList = stateMonitor.record_variables
        
    num_columns = 2
    num_rows = int(np.ceil(len(statesList)/num_columns))

    for neuron_index in neuron_indices:
        figure(figsize=(15, 10))
        for index, state in enumerate(statesList):
            subplot(num_rows, num_columns, index+1)
            plot(stateMonitor.t/ms, getattr(stateMonitor, state)[neuron_index])
            xlabel('Time (ms)')
            ylabel(state)           
            suptitle(f'Neuron {neuron_index}')
    show()


# Generate image frames from the spike monitors of the input and output layers 
def generate_InOut_frames(inSpikeMon, outSpikeMon, heightIn, widthIn, heightOut, widthOut, num_neurons):
    # Create a list to store the frames
    inFramesList = []
    outFramesList = []
    
    # Get the timestamps of the simulation
    spikeTimes = np.unique(np.concatenate((inSpikeMon.t/ms, outSpikeMon.t/ms)))
    
    # Create a frame for each time step
    for t in spikeTimes:
        # Create an array of length num_neurons to store the spikes
        inArray = np.zeros(num_neurons)
        outArray = np.zeros(num_neurons)
        
        # Get the indices of the neurons that spiked at time t
        inIndices = np.where(inSpikeMon.t/ms == t)[0]
        outIndices = np.where(outSpikeMon.t/ms == t)[0]
        
        # Set the corresponding elements to 1
        inArray[inSpikeMon.i[inIndices]] = 1
        outArray[outSpikeMon.i[outIndices]] = 1
        
        inFramesList.append(inArray.reshape(heightIn, widthIn))
        outFramesList.append(outArray.reshape(heightOut, widthOut))
        
    return inFramesList, outFramesList
        
# Generate a binary video from the frames (normally generated using generate_InOut_frames())
def generate_binary_video(frames, output_path):
    
    height, width = frames[0].shape
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1.0, (height+1, width+1), isColor=False)

    # Write each frame to the video
    for frame in frames:
        # Convert the frame to 8-bit binary image
        binary_frame = np.uint8(frame * 254)
        # Reshape the frame to match the desired width and height
        binary_frame = cv2.resize(binary_frame, (height+1, width+1), interpolation=cv2.INTER_CUBIC)

        # Write the frame to the video
        out.write(binary_frame)

    # Release the VideoWriter object
    out.release()
