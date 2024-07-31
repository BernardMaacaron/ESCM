import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_ChebyshevNeighbours(neuronsGrid, Num_Neighbours):
    if isinstance(neuronsGrid, np.ndarray):
        coords = neuronsGrid
    else:
        # Create a list of coordinates
        coords = np.array(list(zip(neuronsGrid.X, neuronsGrid.Y)))
        
    # Extract X and Y coordinates
    coords_X = coords[:, 0]
    coords_Y = coords[:, 1]
    
    # Find the central neuron
    central_x, central_y = coords_X.max() // 2, coords_Y.max() // 2
    
    # Calculate the Chebyshev distance between the central point and all points in the grid
    distances = np.max(np.abs(np.stack((coords_X - central_x, coords_Y - central_y), axis=-1)), axis=-1)
    
    # Find the indexes of the N closest neighbors
    neighbor_indexes = np.argsort(distances)[:Num_Neighbours]
    
    # Get the coordinates of the N closest neighbors
    neighbor_coords = coords[neighbor_indexes]
    return coords, neighbor_coords    



def plot_neighborhood(neuronsGrid, Num_Neighbours):
    coords, neighbor_coords = calculate_ChebyshevNeighbours(neuronsGrid, Num_Neighbours)
    # Extract X and Y coordinates
    coords_X = coords[:, 0]
    coords_Y = coords[:, 1]
     
    # Find the central neuron
    central_x, central_y = coords_X.max() // 2, coords_Y.max() // 2
    
    fig, ax = plt.subplots()
    
    # Plot neurons as blue squares
    for x, y in zip(coords_X, coords_Y):
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='black', facecolor='blue'))
    
    # Highlight the central neuron in green
    ax.add_patch(plt.Rectangle((central_x - 0.5, central_y - 0.5), 1, 1, edgecolor='black', facecolor='green'))
    
    
    # Highlight the neighbors of the central neuron in red
    for i, j in neighbor_coords:
        print(i, j)
        square = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor='red', facecolor='red')
        ax.add_patch(square)
    
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, coords_X.max() - 0.5)
    ax.set_ylim(-0.5, coords_Y.max() - 0.5)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Neuron Neighborhood')
    plt.show()
    
if __name__ == '__main__':
    # Create a simple NeuronsGrid object
    NeuronsGrid = np.array([[i, j] for i in range(12) for j in range(12)])
    Num_Neighbours = 2
    
    plot_neighborhood(NeuronsGrid, Num_Neighbours)
    