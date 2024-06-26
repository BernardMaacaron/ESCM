from Brian2 import *

# Simulation parameters
networkParams = {'tau': 200. * usecond,
 'vt': 0.1,
 'vr': 0.0,
 'P': 0,
 'incoming_spikes': 0,
 'method_Neuron': 'exact',
 'Num_Neighbours': 8,
 'beta': 0.5,
 'Wi': 6.15,
 'Wk': -10,
 'method_Syn': 'exact',
 'Sim_Clock': 0.5 * msecond,
 'Sample_Perc': 1.0}