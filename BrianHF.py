from multiprocessing import Pool
import os
import sys

import cv2
import imageio
from bimvee import exportIitYarp
from brian2 import *

# # Figures mpl style
# axes.linewidth : 1
# xtick.labelsize : 8
# ytick.labelsize : 8
# axes.labelsize : 8
# lines.linewidth : 1
# lines.markersize : 2
# legend.frameon : False
# legend.fontsize : 8
# axes.prop_cycle : cycler(color=['e41a1c', '377eb8', '4daf4a', '984ea3', 'ff7f00', 'ffff33'])


# NOTE: The following comments are longterm TODOs and are not urgent. They are just suggestions for future improvements.
# TODO : Validate the documentation of the functions
# TODO : Turn the conditionals into exceptions


''' - EVENT CAMERA HANDLING FUNCTIONS -
List of functions to handle the event camera data and adapt it to the simulation requirements.
Event data is considered to be of the form (x, y, t, p) where x and y are the pixel coordinates, t is the time, and p is the polarity.

    The functions are:
    - event_camera_to_spikes(event_camera_data, threshold, time_window)
    - event_camera_to_spikes_with_time(event_camera_data, threshold, time_window)
'''
# Make sure the data is valid to be used in the simulation - used in the event_to_spike() function
def validate_indices(indices, firing_x, firing_y, width):
    """
    Reverse the operation of building the indices array and check if the reconstructed values match the original values.

    Parameters:
    - indices (array): The indices array.
    - firing_x (array): The array of x coordinates.
    - firing_y (array): The array of y coordinates.
    - width (int): The width of the event camera.

    Returns:
    None
    """
    # Reverse the operation
    reconstructed_y = indices // width
    reconstructed_x = indices % width

    # Check if the reconstructed values match the original values
    if np.all(reconstructed_x == firing_x) and np.all(reconstructed_y == firing_y):
        print("Indices array was built correctly.")
    else:
        print("Indices array was not built correctly.")

def sort_events(times, indices):
    """
    Sorts the times and indices based on the time values.
    This function is adapted to work with NumPy arrays.

    Parameters:
    times (np.ndarray): A NumPy array of time values.
    indices (np.ndarray): A NumPy array of neuron indices.

    Returns:
    tuple: Two sorted NumPy arrays, one for times and another for indices.
    """
    # Example sorting implementation for NumPy arrays
    sorted_indices = indices[np.argsort(times)]
    sorted_times = np.sort(times)
    
    return sorted_times, sorted_indices

# Check for duplicate firing times and neuron indices
# XXX: Consider making the process by default and just extract the unique (Neuron Index-Time) pairs from the set
# XXX: Consider using np.unique() to extract the unique pairs (not sure if it is more efficient)
def clear_duplicates(times, indices):
    """
    Check for and remove duplicate pairs of times and neuron indices, adapted for NumPy arrays.

    Parameters:
    times (np.ndarray): A NumPy array of time values.
    indices (np.ndarray): A NumPy array of neuron indices.

    Returns:
    tuple: A tuple containing the updated NumPy arrays of times and indices.
    """
    print("Checking for duplicate pairs...")
    # Ensure inputs are NumPy arrays
    times = np.array(times)
    indices = np.array(indices)
    
    # Create a structured array to hold pairs and facilitate easy duplicate removal and sorting
    dtype = [('times', times.dtype), ('indices', indices.dtype)]
    structured_array = np.array(list(zip(times, indices)), dtype=dtype)
    
    # Remove duplicates
    unique_structured_array = np.unique(structured_array)
    
    if len(structured_array) != len(unique_structured_array):
        print(f"Duplicate pairs found. Total Number of duplicates: {len(structured_array) - len(unique_structured_array)}")
    else:
        print("No duplicate pairs found.")
    
    print(f"Total number of pairs/spikes after removing duplicates: {len(unique_structured_array)} pairs.")
    
    # Assuming sort_events is adapted to work with structured arrays or can handle separate arrays
    sorted_times, sorted_indices = sort_events(unique_structured_array['times'], unique_structured_array['indices'])
    
    return sorted_times, sorted_indices

# TODO: Implement the function
def nudge_ts(ts, nudge=1e-6):
    return None

# Convert the event camera data to spikes
# XXX: Make it take the keys as arguments or even consider taking x, y, t, p as arguments
# XXX: make sure the time units make sense. Right now it is arbitrarily in ms. I'm not even sure if it is in ms.
# XXX: Refactor to make dt flag dependent on whether the user wants to use the default clock time step or not.
def event_to_spike(eventStream, width, height, dt=None, val_indices=False, clear_dup=True, timeScale: float = 1.0,
                   samplePercentage: float = 1.0, interSpikeTiming=None, polarity=False):
    """
    Converts an event to a spike based on the threshold. The event data is assumed to be in the form of a dictionary 
    and the spike representation is generated as a SpikeGeneratorGroup object from Brian2.

    Parameters:
    - eventStream (dictionary): The event stream data. The keys are 'x', 'y', 'ts', and 'p'.
                                The values are lists of the respective data.
                                'x' and 'y' are the pixel coordinates (integers)
                                'ts' is the time (float)
                                'pol' is the polarity (boolean)
    - height (int): The height of the event camera.
    - width (int): The width of the event camera.
    - dt (float, optional): The time resolution of the spike generator. Defaults to None.
    - val_indices (bool, optional): Flag to validate indices. Defaults to False.
    - clear_dup (bool, optional): Flag to clear duplicate events. Defaults to True.
    - timeScale (float, optional): The scaling factor for the time. Defaults to 1.0.
    - samplePercentage (float, optional): The percentage of spikes to select at regular intervals. Defaults to 1.0.
    - interSpikeTiming (float, optional): The minimum time difference between spikes. Defaults to None.
    - polarity (bool, optional): Flag to take polarity into consideration. Defaults to False.

    Returns:
    - simTime (float*ms): The recommended simulation time that spans all spike times.
    - clockStep (float*ms): The recommended clock time step based on the minimum time difference between events.
    - spikeGen (SpikeGeneratorGroup): The SpikeGeneratorGroup object respective to the event stream.
    """
    
    print("Extracting the event data...")
    num_neurons = height * width
    
    if polarity:
        mask = eventStream['pol']
        print("Polarity was chosen to be considered. Note that for now this means only positive polarity events are extracted.")
    else:
        mask = np.ones(len(eventStream['pol']), dtype=bool)
        print("Polarity was chosen to be ignored. All events are extracted.")
    
    print("Selecting a percentage of the spikes at regular intervals... Percentage: ", samplePercentage*100, "%")
    interval = max(int(1 / samplePercentage), 1)  # Ensure interval is at least 1
    
    firing_x = np.array(eventStream['x'])[mask][::interval]
    firing_y = np.array(eventStream['y'])[mask][::interval]
    times = np.array(eventStream['ts'])[mask][::interval] * 1000 * timeScale  # Convert to ms and apply timeScale 
    
    if interSpikeTiming is not None:
        print(f'Applying the minimum inter-spike timing constraint of {interSpikeTiming} ms.')
        starting_indices = [0]  # Start with the index of the first element
        current_time = times[0]
        for i, time in enumerate(times[1:], start=1):  # Start enumeration at 1 to match times[1:]
            if time - current_time >= interSpikeTiming:
                starting_indices.append(i)
                current_time = time
        
        # Apply the same selection to times and other arrays
        times = times[starting_indices]
        firing_x = firing_x[starting_indices]
        firing_y = firing_y[starting_indices]
    
    print(f'The maximum x index {np.max(firing_x)} while the width is {width}')
    print(f'The maximum y index {np.max(firing_y)} while the height is {height}')
    
    indices = firing_y * width + firing_x  # Convert to neuron indices directly with NumPy
    
    if val_indices:
        validate_indices(indices, firing_x, firing_y, width)
        
    if clear_dup:
        # Assuming clear_duplicates is optimized for NumPy arrays
        times, indices = clear_duplicates(times, indices)
    else:
        print("Skipping checking for duplicate pairs.")
        # Assuming sort_events is optimized for NumPy arrays
        times, indices = sort_events(np.vstack((times, indices)).T)
   
    maxTime = times[-1]
    print(f'The maximum time stamp (scaled) {maxTime} ms.')
    simTime = np.ceil(maxTime)
    print(f'The recommended simulation time (scaled) is {simTime} ms.')
    
    minTimeStep = np.min(np.abs(np.diff(np.unique(times))))
    print(f'The minimum time step (scaled) is {minTimeStep} ms.')    
    clockStep = round(minTimeStep * 10) / 10  # Assuming decimal_index is a function that returns the number of decimals
    print(f'The recommended clock time step (scaled) is {clockStep} ms.')
    
    if dt is None:
        dt = clockStep
    
    return simTime, clockStep, SpikeGeneratorGroup(num_neurons, indices.astype(int), times*ms, dt, sorted=True)




''' - SYNAPTIC CONNECTIVITY FUNCTIONS -'''
def calculate_ChebyshevNeighbours(neuronsGrid, Num_Neighbours, chunk_size=1000):
    """
    Calculate the Chebychev neighbors for a given neurons grid.

    Parameters:
    neuronsGrid (NeuronsGrid): The grid of neurons.
    Num_Neighbours (int): The maximum distance between two neurons to be considered neighbors.
    chunk_size (int, optional): The size of each chunk to process. Defaults to 1000.

    Returns:
    tuple: A tuple containing two lists - indexes_i and indexes_j. These lists represent the pairs of neuron indexes that are considered neighbors.

    """
    # Create a list of coordinates
    coords = np.array(list(zip(neuronsGrid.X, neuronsGrid.Y)))
    # Initialize an empty list to store the pairs
    pairs = []
    # Calculate the number of chunks
    num_chunks = len(coords) // chunk_size + (len(coords) % chunk_size != 0)

    # Process each chunk
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(coords))
        chunk_coords = coords[start:end]

        # Calculate the Manhattan distance between all pairs of points in the chunk
        distances = np.max(np.abs(chunk_coords[:, None] - chunk_coords), axis=-1)

        # Find the pairs of points in the chunk that are within a distance of Num_Neighbours
        chunk_pairs = np.argwhere(distances <= Num_Neighbours)

        # Add the chunk pairs to the list of pairs, adjusting the indices for the current chunk
        pairs.extend((start + i, start + j) for i, j in chunk_pairs if i != j)

    # Unzip the pairs into two lists
    indexes_i, indexes_j = zip(*pairs)
    
    return indexes_i, indexes_j





''' - NETWORK VISUALIZATION FUNCTIONS -
List of functions to visualize the network architecture, neuron states, spikes, and inter-spike intervals etc.

    The functions are:
    - visualise_connectivity(SynapsesGroup)
    - visualise_neurons_states(stateMonitor, neuron_indices, states)
    - visualise_spikes(spikeMonitorsList, figTitle='')
    - visualise_interSpikeInterval(spikeMonitor, neuron_indices)
'''
# BUG: This function is not correctly implemented. Need to either modify the code or the arguments to make it work.
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
    figure(figsize = figSize)
    plot(SynapsesGroup.i, SynapsesGroup.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    show()

# Visualize the chosen states of a list of neurons
def visualise_neurons_states(stateMonitor, neuron_indices, statesList ,figSize=(10, 4), overlap=False, vt = None):
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
    if statesList == 'all':
        statesList = stateMonitor.record_variables
        
    num_columns = 2
    num_rows = int(np.ceil(len(statesList)/num_columns))

    if overlap:
        figure(figsize=figSize)
    for neuron_index in neuron_indices:
        if not overlap:
            figure(figsize=figSize)
        for index, state in enumerate(statesList):
            subplot(num_rows, num_columns, index+1)
            plot(stateMonitor.t/ms, getattr(stateMonitor, state)[neuron_index])
            if state == 'v' and vt is not None:
                axhline(y=vt, linestyle='--', color='r')
            xlabel('Time (ms)')
            ylabel(state)           
            suptitle(f'Neuron {neuron_index} - Variable States')

    show()

# Visualize the chosen states of a list of neurons
def visualise_states(stateMonitor, neuron_indices, statesList ,figSize=(10, 4), overlap=False, vt = None):
    """
    Visualizes the states of neurons over time. Similar to visualise_neurons_states but plots all states per neuron on the same
    figure and one neuron per figure.

    Args:
        stateMonitor (StateMonitor): The StateMonitor object that records the neuron states.
        neuron_indices (list): A list of neuron indices to visualize.
        states (str): The states to visualize. If 'all', all recorded states will be visualized.
        figSize (tuple, optional): The size of the figure (width, height). Defaults to (10, 4).

    Returns:
        None
    """
    if statesList == 'all':
        statesList = stateMonitor.record_variables

    for neuron_index in neuron_indices:
        figure(figsize=figSize)
        for index, state in enumerate(statesList):
            if state == 'v' and vt is not None:
                axhline(y=vt, linestyle='--', color='r')
                
            plot(stateMonitor.t/ms, getattr(stateMonitor, state)[neuron_index])
            xlabel('Time (ms)')
            ylabel(state)           
            suptitle(f'Neuron {neuron_index} - Variable States')
            legend(statesList)
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
    show()

# TODO: Implement the function
def visualise_spike_difference(spikeMon1, spikeMon2, figTitle='', figSize=(15, 3)):
    """
    Presents the raster plots of the spike times from the spike monitors

    Args:
        spikeMonitorsList (list): List of SpikeMonitor objects that record the neuron spikes.
        figTitle (str, optional): Title for the plot. Defaults to an empty string.
        figSize (tuple, optional): The size of the figure (width, height). Defaults to (15, 3).

    Returns:
        None
    """
    figure(figsize=figSize)
    plot(spikeMon1.t/ms, spikeMon1.i, '.k')
    plot(spikeMon2.t/ms, spikeMon2.i, '.r')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title(figTitle)
    show()
    
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
# Generate image frames from the spike monitors of the input and output layers (for videos)
# XXX: THIS NEEDS A COMPLETE REFACTORING. THE FUNCTION IS NOT EFFICIENT.
def gen_InOut_framesVid(inSpikeMon, outSpikeMon, widthIn, heightIn, heightOut, widthOut, num_neurons):
    """
    Generate input and output frames based on spike monitor data Required function if you are attempting
    to generate videos of same lengths as the spike times between input and output are not the same.
    To generate frames for a GIF use generate_frames() function.

    Parameters:
    - inSpikeMon (SpikeMonitor): Spike monitor for input neurons.
    - outSpikeMon (SpikeMonitor): Spike monitor for output neurons.
    - widthIn (int): Width of the input frame.
    - heightIn (int): Height of the input frame.
    - widthOut (int): Width of the output frame.
    - heightOut (int): Height of the output frame.
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
        
        # Set the corresponding elements to np.uint8(254)
        inArray[inSpikeMon.i[inIndices]] = np.uint8(254)
        outArray[outSpikeMon.i[outIndices]] = np.uint8(254)
        
        inFramesList.append(inArray.reshape(widthIn, heightIn))
        outFramesList.append(outArray.reshape(widthOut, heightOut))
        
    return inFramesList, outFramesList

# Create a function to generate a frame for a given time step
def spikes2frame(num_neurons, neuronIndexes, width, height):
    """
    Convert spike information to a frame array.

    Args:
        num_neurons (int): The total number of neurons.
        neuronIndexes (array-like): The indexes of the neurons that spiked at time t.
        width (int): The width of the frame array.
        height (int): The height of the frame array.

    Returns:
        numpy.ndarray: The frame array with spike information.

    """
    # Create an array of length num_neurons to store the spikes
    frameArray = np.zeros(num_neurons)
    
    # Set the elements corresponding to neurons that spiked at time t to 1
    frameArray[neuronIndexes] = 1
    
    return frameArray.reshape(height, width)

# Generate image frames from a spike monitor (for GIFs)
# TODO XXX: Consider a different way to parallelize the process. The current method is not efficient. See NOTE.
# NOTE: The current bottleneck is not the frame generation per se, but the creation of the arguments in argsList (neuron index, timestamps etc.)
def generate_frames(spikeMon_time, spikeMon_index, width, height, num_neurons, samplePercentage=1.0):
    """
    Generate frames based on spike monitor data.

    Parameters:
    - spikeMon (SpikeMonitor): The SpikeMonitor object that records the neuron spikes.
    - width (int): The width of the frame.
    - height (int): The height of the frame.
    - num_neurons (int): The number of neurons.
    - samplePercentage (float, optional): The percentage of spikes to sample (0.0 - 1.0). Default is 1.0.

    Returns:
    - framesList (list): List of frames.

    This function takes a SpikeMonitor object, which records the spikes of neurons, and generates frames based on the spike data.
    The frames are represented as NumPy arrays with dimensions (num_spikes, width, height).
    The function samples a certain percentage of the spikes at regular intervals and creates frames based on the sampled spikes.
    The frames are generated in parallel using a pool of worker processes.

    Example usage:
    spikeMon = SpikeMonitor(...)
    frames = generate_frames(spikeMon, 32, 32, 1000, samplePercentage=0.5)
    """
    
    # Get the timestamps of the simulation
    spikeTimes, occurenceIndex = np.unique(spikeMon_time, return_index=True)
    occurenceIndex = np.append(occurenceIndex, len(spikeMon_time))
    
    # Select a certain percentage of the spikes at regular intervals
    num_spikes = len(spikeTimes)
    interval = int(num_spikes / (num_spikes * samplePercentage))
    spikeTimes = spikeTimes[::interval]
    
    # Create the arguments list for the pool
    argList = [(num_neurons, spikeMon_index[occurenceIndex[i-1]: occurenceIndex[i]], width, height) for i in range(1, len(occurenceIndex))] 

    # argList = [(num_neurons, t, spikeMon.i[occurenceIndex], width, height) for t in spikeTimes]
    
    # print(f'Generating frames using {num_proc} pooled processes...')         
    with Pool() as pool:
        # Use the pool to generate frames in parallel
        framesList = pool.starmap(spikes2frame, argList)
        
    return framesList


    return framesList

# Generate a GIF from the frames
def generate_gif(frames, output_path, simTime: float, replicateDuration=False, duration=0.1):
    """
    Generate a GIF from a list of frames.

    Args:
        frames (list): A list of frames.
        output_path (str): The path to save the generated GIF.
        simTime (float): The total simulation time.
        replicateDuration (bool, optional): Flag to indicate whether to replicate the duration of each frame in the GIF. 
                                            If True, the duration of each frame will be adjusted based on the total simulation time. 
                                            If False, the duration of each frame will be set to the default value. Defaults to False.
        fps (int, optional): The frames per second for the generated GIF. Defaults to 10.

    Returns:
        None
    """
    # Create a list to store the frames
    gif_frames = []
    
    # Convert the frames to 8-bit binary images
    for frame in frames:
        # Convert the frame to a binary image
        binary_frame = np.uint8(frame * 254)
        # Add the binary frame to the list
        gif_frames.append(binary_frame)
    
    if replicateDuration:
        duration = simTime / len(gif_frames)  # Duration of each frame in the GIF
        
    # Save the frames as a GIF
    imageio.mimsave(output_path, gif_frames, duration=duration)

# Generate a video frame from a frame representation of the spikes (for binary videos)
def generate_videoFrame (frame, width, height):
    # Convert the frame to 8-bit binary image
    binary_frame = np.uint8(frame * 254)
    # Reshape the frame to match the desired width and height
    binary_frame = cv2.resize(binary_frame, (width+1, height+1), interpolation=cv2.INTER_CUBIC)
    return binary_frame
    
# Generate a binary video from the frames (normally generated using generate_InOut_frames())
# XXX: Currently I suppose the videos generated will have different lengths. Evaluate if i need to compare the timestamps for input and output.
def generate_video(frames, output_path, simTime: float):
    """
    Generate a binary video from a list of frames.

    Args:
        frames (list): A list of frames.
        output_path (str): The path to save the generated video.

    Returns:
        None
    """
    height, width = frames[0].shape
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frameSize=(width+1, height+1), isColor=False, fps=np.round(len(frames) / simTime))

    # Create a pool of worker processes
    num_proc = os.cpu_count()
    with Pool(processes=num_proc) as pool:
        # Process each frame in parallel
        processed_frames = pool.starmap(generate_videoFrame, [(frame, width, height) for frame in frames])

    # Write each processed frame to the video
    for binary_frame in processed_frames:
        # Write the frame to the video
        out.write(binary_frame)

    # Release the VideoWriter object
    out.release()

def generate_YarpDvs(spikeMon_time, spikeMon_index, NeuronGroup, path):
    x_list = NeuronGroup.X[spikeMon_index]
    y_list = NeuronGroup.Y[spikeMon_index]
    data = {'ts': spikeMon_time, 'x': x_list, 'y': y_list, 'pol': ones(len(spikeMon_time), dtype=int)}
    with open(path, 'wb') as dataFile:
        exportIitYarp.exportDvs(dataFile, data, bottleNumber=0)
        




''' - MISCELLANEOUS -'''
def decimal_index(num):
    """
    Returns the index of the first decimal place in a given number.
    
    Parameters:
    num (float): The number for which the decimal index needs to be calculated.
    
    Returns:
    int: The index of the first decimal place in the given number.
         Returns 0 if the number is 0.
    """
    if num == 0:
        return 0
    return int(np.abs(np.floor(np.log10(np.abs(num)))))

def filePathGenerator(stemName='UndefinedStem', params: dict={}):
    """
    Generates a path based on the stem name and parameters.

    Args:
        stemName (str): The stem name for the path.
        params (dict): A dictionary of parameters.

    Returns:
        str: The generated path.
    """
    # Build a path based on the stem name and the dictionary of parameters
    path = stemName + '-'
    for key, value in params.items():
        path += f'{key}={value}-'
    return path
    
class ProgressBar(object): 
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        
        elapsed_ = elapsed/second
        hours = int(elapsed_ // 3600)
        minutes = int((elapsed_ % 3600 // 60))
        seconds = int(elapsed_ % 60)
        time = f"\nElapsed (real-time): {hours:02d}:{minutes:02d}:{seconds:02d} - Completed: {complete*100:.2f}%"
        
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("\r[%s]"% (" " * self.toolbar_width) + time)
            # sys.stdout.flush()
            # Return to the start of the line, after '['
            sys.stdout.write("\033[F") # Move cursor up one line
            sys.stdout.write("\b" * (self.toolbar_width+1)) # return to start of line, after '['
            
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks) + time)
                sys.stdout.flush()
                # sys.stdout.write("\033[F") # Move cursor up one line
                # sys.stdout.write("\b" * (ticks_needed-self.ticks+1))
                self.ticks = ticks_needed

        if complete == 1.0:
            sys.stdout.write('\n')

