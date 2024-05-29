from multiprocessing import Pool
import os
import sys

import cv2
import imageio
from brian2 import *
from tqdm.autonotebook import tqdm

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

def sort_events(events: list):
    """
    Sorts a list of events based on their times and indices.

    Args:
        events (list): A list of events, where each event is a tuple containing a
        time and an index in that respective order.

    Returns:
        tuple: A tuple containing two lists - sorted times and sorted indices.
    """
    # Sort the events first by time, then by index
    
    sorted_events = sorted(events, key=lambda x: (x[0], x[1]))
    # Unzip the sorted events back into separate lists
    sorted_times, sorted_indices = zip(*sorted_events)
    return sorted_times, sorted_indices

# Check for duplicate firing times and neuron indices
# XXX: Consider making the process by default and just extract the unique (Neuron Index-Time) pairs from the set
# XXX: Consider using np.unique() to extract the unique pairs (not sure if it is more efficient)
def clear_duplicates(times, indices):
    """
    Check for and remove duplicate pairs of times and neuron indices.

    This function takes in two lists, `times` and `indices`, and checks for duplicate pairs of time values and neuron indices.
    If any duplicate pairs are found, they are removed from the lists.

    Parameters:
    times (list): A list of time values.
    indices (list): A list of neuron indices.

    Returns:
    tuple: A tuple containing the updated lists of times and indices.

    Notes:
    - This function sorts the pairs of times and indices according to the sorting method in the `sort_events` function.
    """
    
    print("Checking for duplicate pairs...")
    pairs = list(zip(times, indices))
    pairs_set = list(set(pairs))

    if len(pairs) != len(pairs_set):
        print("Duplicate pairs found. Total Number of duplicates: ", len(pairs) - len(pairs_set))
        print("Total number of pairs/spikes prior to removing duplicates: ", len(pairs), " pairs.")
        print("Removing duplicate pairs...")
        print("Done. Total number of pairs/spikes: ", len(pairs_set), " pairs.")
        sorted_times, sorted_indices = sort_events(pairs_set)
        return sorted_times, sorted_indices
    else:
        print("No duplicate pairs found.")
        print("Total number of pairs/spikes: ", len(pairs_set), " pairs.")
        sorted_times, sorted_indices = sort_events(pairs_set)
        return sorted_times, sorted_indices

# TODO: Implement the function
def nudge_ts(ts, nudge=1e-6):
    return None

# Convert the event camera data to spikes
# XXX: Make it take the keys as arguments or even consider taking x, y, t, p as arguments
# XXX: make sure the time units make sense. Right now it is arbitrarily in ms. I'm not even sure if it is in ms.
def event_to_spike(eventStream, width, height, dt=None, val_indices=False, clear_dup=True, timeScale: float = 1.0, samplePercentage=1.0, polarity=False):
    """
    Converts an event to a spike based on the threshold. the event data is assumed to be in the form of a dictionary 
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
    - polarity (bool, optional): Flag to take polarity into consideration. Defaults to False.

    Returns:
    - simTime (float*ms): The recommended simulation time that spans all spike times.
    - clockStep (float*ms): The recommended clock time step based on the minimum time difference between events.
    - spikeGen (SpikeGeneratorGroup): The SpikeGeneratorGroup object respective to the event stream.
    """
    print("Extracting the event data...")
    num_neurons = height * width
    
    # Define the mask to extract the event data based on whether polarity is considered or not
    if polarity:
        mask = eventStream['pol']
        print("Polarity was chosen to be considered. Note that for now this means only positive polarity events are extracted.")
    else:
        mask = np.ones(len(eventStream['pol']), dtype=bool)
        print("Polarity was chosen to be ignored. All events are extracted.")
    
    # Select a certain percentage of the spikes at regular intervals
    print("Selecting a percentage of the spikes at regular intervals... Percentage: ", samplePercentage*100, "%")
    num_samples = np.count_nonzero(mask)
    interval = int(num_samples / (num_samples * samplePercentage))
    
    # Retrieve the x, y, time, and polarity data from the event stream
    # NOTE: The time extracted from the event stream is in seconds (Read bimvee library documentation).
    #       It is converted into milliseconds post processing.
    firing_x = eventStream['x'][mask][::interval]
    firing_y = eventStream['y'][mask][::interval]
    times = eventStream['ts'][mask][::interval]
        
    print(f'The maximum x index {np.max(firing_x)} while the width is {width}')
    print(f'The maximum y index {np.max(firing_y)} while the height is {height}')
    
    # Check if the data is correct
    if len(firing_x) == len(firing_y) == len(times):
        print("The x,y and time stamp indices are equal, the data is correct.")
        indices = array([firing_y[i]*width + firing_x[i] for i in range(len(times))])
    else:
        print("The x,y and time stamp indices are not equal, the data is incorrect.")
        return None
    
    if val_indices:
        validate_indices(indices, firing_x, firing_y, width)
        
    if clear_dup:
        times, indices = clear_duplicates(times, indices)
    else:
        print("Skipping checking for duplicate pairs.")
        times, indices = sort_events(list(zip(times, indices)))
        
    print("The selected scale is", timeScale)
    # Convert the time from seconds to milliseconds
    times = np.array(times) * 1000 * timeScale

    # Calculate the simulation time as the ceil of the last spike time
    maxTime = times[-1]
    print(f'The maximum time stamp (scaled) {maxTime} ms.')
    simTime = np.ceil(times[-1])
    print(f'The recommended simulation time (scaled) is {simTime} ms.')
    
    # Calculate the (defaultClock.dt) time step as the floor of the smallest time difference between events
    minTimeStep = min(abs(diff(list(set(times)))))
    print(f'The minimum time step (scaled) is {minTimeStep} ms.')    
    clockStep =  round(minTimeStep*10, decimal_index(minTimeStep))/10
    print(f'The recommended clock time step (scaled) is {clockStep} ms.')
    
    if dt is None:
        dt = clockStep*ms
    
    
    return simTime, clockStep, SpikeGeneratorGroup(num_neurons, indices, times*ms, dt, sorted=True)





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
def generate_frames(spikeMon, width, height, num_neurons, samplePercentage=1.0):
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
    spikeTimes, occurenceIndex = np.unique(spikeMon.t/ms, return_index=True)
    occurenceIndex = np.append(occurenceIndex, len(spikeMon.t))
    
    # Select a certain percentage of the spikes at regular intervals
    num_spikes = len(spikeTimes)
    interval = int(num_spikes / (num_spikes * samplePercentage))
    spikeTimes = spikeTimes[::interval]
    
    # Create the arguments list for the pool
    argList = [(num_neurons, spikeMon.i[occurenceIndex[i-1]: occurenceIndex[i]], width, height) for i in range(1, len(occurenceIndex))] 

    # argList = [(num_neurons, t, spikeMon.i[occurenceIndex], width, height) for t in spikeTimes]
    
    num_proc = os.cpu_count()
    # print(f'Generating frames using {num_proc} pooled processes...')         
    with Pool(processes=num_proc) as pool:
        # Use the pool to generate frames in parallel
        framesList = pool.starmap(spikes2frame, argList, chunksize=int(len(argList)/num_proc))
        
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
    path = stemName + '_'
    for key, value in params.items():
        path += f'{key}={value}_'
    return path
    
class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks))
                # sys.stdout.flush()
                
                sys.stdout.write(f"Elapsed (real-time): {elapsed} s. - Completed: {complete*100:.2f}%\n")
                sys.stdout.write(f"Biological: Start: {start} - Duration: {duration}\n")
                sys.stdout.flush()
                
                self.ticks = ticks_needed

        if complete == 1.0:
            sys.stdout.write("\n")


