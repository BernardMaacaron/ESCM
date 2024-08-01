import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_CenterNeighbours(neuronsGrid, Num_Neighbours):
    """
    Calculate the neighbors of the center neuron within the specified distance.

    Parameters:
    neuronsGrid (NeuronsGrid or np.ndarray): The grid of neurons.
    Num_Neighbours (int): The maximum distance between the center neuron and its neighbors.

    Returns:
    list: A list of coordinates of the neighboring neurons.
    """
    # Create a list of coordinates
    if isinstance(neuronsGrid, np.ndarray):
        coords = neuronsGrid
    else:
        coords = np.array(list(zip(neuronsGrid.X, neuronsGrid.Y)))
    
    # Extract X and Y coordinates
    coords_X = coords[:, 0]
    coords_Y = coords[:, 1]
    
    # Find the central neuron
    central_x, central_y = coords_X.max() // 2, coords_Y.max() // 2

    # Calculate the differences in X and Y coordinates
    diff_X = np.abs(coords[:, 0] - central_x)
    diff_Y = np.abs(coords[:, 1] - central_y)

    # Find the coordinates of points that satisfy the conditions
    condition = (diff_X <= Num_Neighbours) & (diff_Y <= Num_Neighbours) & (diff_X + diff_Y != 0)    
    neighbor_coords = coords[condition]
    
    
    return neighbor_coords

def plot_neighborhood(neuronsGrid, Num_Neighbours):
    neighbor_coords = calculate_CenterNeighbours(neuronsGrid, Num_Neighbours)
    # Extract X and Y coordinates
    coords_X = neuronsGrid[:, 0]
    coords_Y = neuronsGrid[:, 1]
     
    # Find the central neuron
    central_x, central_y = coords_X.max() // 2, coords_Y.max() // 2
    
    fig, ax = plt.subplots()
    
    # Plot neurons as blue squares
    for x, y in zip(coords_X, coords_Y):
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='black',facecolor='None'))
    
    # Highlight the central neuron in green
    ax.add_patch(plt.Rectangle((central_x - 0.5, central_y - 0.5), 1, 1, edgecolor='black', facecolor='tab:red', label='Reference Neuron'))
    
    
    # Highlight the neighbors of the central neuron in red
    for i, j in neighbor_coords:
        square = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor='black', label='Neighboring Neurons')
        ax.add_patch(square)
    
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, coords_X.max() - 0.5)
    ax.set_ylim(-0.5, coords_Y.max() - 0.5)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    title = f'Neuron Neighborhood with {Num_Neighbours} Neighbors'
    plt.title(title)
    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Keep only the first two elements
    handles = handles[:2]
    labels = labels[:2]

    # Create the legend with adjusted width
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), handlelength=2)
    plt.show()
        
if __name__ == '__main__':
    # Create a simple NeuronsGrid object
    NeuronsGrid = np.array([[i, j] for i in range(12) for j in range(12)])
    Num_Neighbours = 4
    
    plot_neighborhood(NeuronsGrid, Num_Neighbours)
    