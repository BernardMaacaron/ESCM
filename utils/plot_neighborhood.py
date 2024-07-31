import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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
    if isinstance(neuronsGrid, np.ndarray):
        coords = neuronsGrid
    else:
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
    
    return coords, indexes_i, indexes_j

def plot_neighborhood(neuronsGrid, Num_Neighbours, chunk_size=1000):
    coords, indexes_i, indexes_j = calculate_ChebyshevNeighbours(neuronsGrid, Num_Neighbours, chunk_size)
    
    coords_X = coords[:, 0]
    coords_Y = coords[:, 1]
    # Plot neurons
    plt.scatter(coords_X, coords_Y, c='blue', label='Neurons')
    
    # Draw lines between neighbors
    for i, j in zip(indexes_i, indexes_j):
        plt.plot([coords_X[i], coords_X[j]], [coords_Y[i], coords_Y[j]], 'r-', alpha=0.5)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Neuron Neighborhood')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Create a simple NeuronsGrid object
    NeuronsGrid = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    Num_Neighbours = 2
    
    plot_neighborhood(NeuronsGrid, Num_Neighbours)
    