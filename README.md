# Event-Driven Spiking Cortical Model (ESCM)

This repository contains the implementation of the **Event-Driven Spiking Cortical Model (ESCM)** as described in the paper **“Persistent Representation of Event Camera Output Using Spiking Neural Networks”**. The ESCM addresses the static scene blindness of event cameras by generating a persistent, spiking representation of visual data through recurrent connections and tailored synaptic dynamics.  

## Overview

Event cameras capture changes in brightness asynchronously with high temporal resolution, but they often overlook static scenes where little change occurs. Inspired by biological visual persistence (as well as mechanisms such as microsaccades), ESCM uses a spiking neural network (SNN) to retain relevant scene information even in the absence of motion.

Key points of the approach include:
- **Persistent Representation:** ESCM maintains and updates spatial information over time via recurrent connections.
- **Neuronal Dynamics:** The model utilizes Leaky Integrate-and-Fire (LIF) neurons with both excitatory autapses and inhibitory synapses defined over a linking field.
- **Practical Implementation:** The code offers both a Jupyter Notebook (`ESCM_W.ipynb`) and a Python script (`ESCM.py`) that can be executed directly.

Feel free to contact me via email (bernard.maacaron@iit.it) for any questions or suggestions regarding the implementation or the underlying model.

## Repository Contents

- **ESCM_W.ipynb:** Jupyter Notebook implementation for interactive exploration and simulation.
- **ESCM.py:** Standalone Python script equivalent of the notebook.
- **Persistent Representation of Event Camera Output Using Spiking Neural Networks.pdf:** The attached paper detailing the model architecture, experiments, and results.

## Installation

Before running the code, ensure that you have Python 3.7 or later installed. I recommend using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/) for managing virtual environments.
To set up the environment:

```bash
# Clone the repository
git clone https://github.com/BernardMaacaron/Event-SCM
cd Event-SCM

# Create a virtual environment with virtualenvwrapper (replace "event-scm" with your preferred env name)
mkvirtualenv event-scm

# Install required packages; if a requirements.txt exists, run:
pip install -r requirements.txt
```

If no requirements file is provided for any reason, ensure you at least have installed the following packages:
- [Brian2](https://brian2.readthedocs.io)
- NumPy (not recommended - Brian2 provides its own set of NumPy functions adapted to work with the unit system of Brian2)
- Matplotlib
- Jupyter Notebook (if using the notebook)

## Usage

There are two ways to run the simulations:

1. **Using the Jupyter Notebook:**
   - Launch Jupyter Notebook from the repository’s directory:
     ```bash
     jupyter notebook ESCM_W.ipynb
     ```
   - Execute all the cells in the notebook to run the simulations and visualize the persistent spiking output.

2. **Using the Python Script:**
   - Run the Python script directly:
     ```bash
     python ESCM.py
     ```
   - The script will execute the same simulation logic as the notebook, generating the outputs based on the implemented ESCM dynamics.

## Experiments and Results

The implementation is designed to replicate key experimental results from the attached paper. In brief:
- **Datasets:** The model has been evaluated on various event camera datasets (e.g., MVSEC Outdoor Day1, Event-Human3.6m).
- **Metrics:** Evaluations are performed using metrics such as Normalized Mutual Information (NMI) and Structural Similarity Index (SSIM) to compare the persistent representation (ESCM) against standard representations like EROS.
- **Observations:** The ESCM achieves a balance between suppressing noise (typical in high event-rate scenarios) and retaining essential scene information in static or low-motion conditions.

For more details on the methodology and experimental evaluation, please refer to the attached paper.

## Contributing

Contributions to this project are welcome! If you find a bug or have suggestions for improvements, please open an issue or submit a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is distributed under the [MIT License](LICENSE).

## Acknowledgments

- The design and implementation of the ESCM are based on the ideas presented in the paper **“Persistent Representation of Event Camera Output Using Spiking Neural Networks”**.
- Special thanks to the EDPR group at the Italian Institute of Technology for their support and collaboration.