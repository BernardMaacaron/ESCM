from brian2 import *
import cv2
import time

''' - EVENT CAMERA HANDLING FUNCTIONS -
List of functions to handle the event camera data and adapt it to the simulation requirements.
Event data is considered to be of the form (x, y, t, p) where x and y are the pixel coordinates, t is the time, and p is the polarity.

    The functions are:
    - event_camera_to_spikes(event_camera_data, threshold, time_window)
    - event_camera_to_spikes_with_time(event_camera_data, threshold, time_window)
'''

def event_to_spike(eventStream, height, width):
    """
    Converts an event to a spike based on the threshold.

    Parameters:
    - eventStream (dictionary): The event stream data. The keys are 'x', 'y', 't', and 'p'.
                                The values are lists of the respective data.
                                'x' and 'y' are the pixel coordinates (integers)
                                't' is the time (float)
                                'p' is the polarity (boolean)
                                
    - width (int): The width of the event camera.

    Returns:
    - spikeGen (SpikeGeneratorGroup): The SpikeGeneratorGroup object respective to the event stream.
    """
    N=height*width
    
    firing_x = eventStream['x'][eventStream['p']]
    firing_y = eventStream['y'][eventStream['p']]
    
    indices = array([firing_y[i] + firing_x[i]*width for i in range(N)])
    times = array(eventStream['t'][eventStream['p']])
    
    return SpikeGeneratorGroup(N, indices, times*ms)


''' - NETWORK VISUALIZATION FUNCTIONS -
List of functions to visualize the network architecture, neuron states, spikes, and inter-spike intervals etc.

    The functions are:
    - visualise_connectivity(SynapsesGroup)
    - visualise_neurons_states(stateMonitor, neuron_indices, states)
    - visualise_spikes(spikeMonitorsList, figTitle='')
    - visualise_interSpikeInterval(spikeMonitor, neuron_indices)
    
'''

# Visualize the connectivity
def visualise_connectivity(SynapsesGroup, figSize=(10, 4)):
    """
    Visualizes the connectivity between source and target neurons.

    Parameters:
    SynapsesGroup (object): The SynapsesGroup object containing the connectivity information.
    figSize (tuple, optional): The size of the figure (width, height). Default is (10, 4).

    Returns:
    None
    """

    Ns = len(SynapsesGroup.source)
    Nt = len(SynapsesGroup.target)
    figure(figsize = figSize)
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
def visualise_neurons_states(stateMonitor, neuron_indices, states ,figSize=(10, 4)):
    """
    Visualizes the states of neurons over time.

    Args:
        stateMonitor (StateMonitor): The StateMonitor object that records the neuron states.
        neuron_indices (list): A list of neuron indices to visualize.
        states (str): The states to visualize. If 'all', all recorded states will be visualized.
        figSize (tuple, optional): The size of the figure (width, height). Defaults to (10, 4).

    Returns:
        None
    """
    if states == 'all':
        statesList = stateMonitor.record_variables
        
    num_columns = 2
    num_rows = int(np.ceil(len(statesList)/num_columns))

    for neuron_index in neuron_indices:
        figure(figsize=figSize)
        for index, state in enumerate(statesList):
            subplot(num_rows, num_columns, index+1)
            plot(stateMonitor.t/ms, getattr(stateMonitor, state)[neuron_index])
            xlabel('Time (ms)')
            ylabel(state)           
            suptitle(f'Neuron {neuron_index} - Variable States')
    show()

# Visualize the spikes of of the input and output layers
def visualise_spikes(spikeMonitorsList, figTitle='', figSize=(15, 3)):
    """
    Presents the raster plots of the spike times from the spike monitors

    Args:
        spikeMonitorsList (list): List of SpikeMonitor objects that record the neuron spikes.
        figTitle (str, optional): Title for the plot. Defaults to an empty string.
        figSize (tuple, optional): The size of the figure (width, height). Defaults to (15, 3).

    Returns:
        None
    """
    for spikeMon in spikeMonitorsList:
        figure(figsize=figSize)
        plot(spikeMon.t/ms, spikeMon.i, '.k')
        xlabel('Time (ms)')
        ylabel('Neuron index')
        title(figTitle)

# Function to visualize the time between spikes and how it varies through time for a set or a single neuron
def visualise_interSpikeInterval(spikeMonitor, neuron_indices ,figSize=(10, 5)):
    """
    Visualizes the inter-spike interval of neurons over time.

    Args:
        spikeMonitor (SpikeMonitor): The SpikeMonitor object that records the neuron spikes.
        neuron_indices (list): A list of neuron indices to visualize.
        figSize (tuple, optional): The size of the figure (width, height). Defaults to (10, 5).

    Returns:
        None
    """
    for neuron_index in neuron_indices:
        # Calculate the inter-spike interval of the neurons
        interSpikeIntervals = np.diff(spikeMonitor.spike_trains()[neuron_index])
        
        # Plot the inter-spike interval of the neurons
        figure(figsize=figSize)
        plot(interSpikeIntervals, '.-')
        xlabel('Spike Number')
        ylabel('Inter-Spike Interval (ms)')
        title(f'Inter-Spike Interval for Neuron {neuron_index}')
        show()



''' - VIDEO AND IMAGE GENERATION FUNCTIONS -
List of functions to generate image frames from the spike monitors of the input and output layers
    and generate a binary video from the frames
    
    The functions are:
    - generate_InOut_frames(inSpikeMon, outSpikeMon, heightIn, widthIn, heightOut, widthOut, num_neurons)
    - generate_binary_video(frames, output_path)
    
'''

# Generate image frames from the spike monitors of the input and output layers 
def generate_InOut_frames(inSpikeMon, outSpikeMon, heightIn, widthIn, heightOut, widthOut, num_neurons):
    """
    Generate input and output frames based on spike monitor data.

    Parameters:
    - inSpikeMon (SpikeMonitor): Spike monitor for input neurons.
    - outSpikeMon (SpikeMonitor): Spike monitor for output neurons.
    - heightIn (int): Height of the input frame.
    - widthIn (int): Width of the input frame.
    - heightOut (int): Height of the output frame.
    - widthOut (int): Width of the output frame.
    - num_neurons (int): Number of neurons.

    Returns:
    - inFramesList (list): List of input frames.
    - outFramesList (list): List of output frames.
    """
    
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
