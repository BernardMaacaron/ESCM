EQ_LIF = '''
dv/dt = -(v - I/gL) / tau                                                        : volt (event-driven)(unless refractory)
X                                                                                : 1 
Y                                                                                : 1

I = I_GABA + I_AMPA                                                              : volt/second (event-driven) # total input current - GABA, Inhibitory and AMPA, Excitatory
I_GABA = K_GABA * Ig                                                             : volt/second (event-driven)
dIg/dt = (-Ig+Ig1)/taugd                                                         : volt/second (event-driven)
dIg1/dt = -Ig1/taugr                                                             : volt/second (event-driven)

I_AMPA = K_AMPA * Ia                                                             : volt/second (event-driven)
dIa/dt = (-Ia+Ia1)/tauad                                                         : volt/second (event-driven)
dIa1/dt = -Ia1/tauar                                                             : volt/second (event-driven)
'''

GEN_EQ_LIF = '''
dv/dt = (v0 - v - I/gL) / tau                                                    : volt (unless refractory)
X                                                                                : 1 
Y                                                                                : 1
I = K_GABA * Ig + K_AMPA * Ia                                                    : volt/second # total input current - GABA, Inhibitory and AMPA, Excitatory
dIg/dt = (-Ig+Ig1)/taugd                                                         : volt/second
dIa/dt = (-Ia+Ia1)/tauad                                                         : volt/second
dIg1/dt = -Ig1/taugr                                                             : volt/second
dIa1/dt = -Ia1/tauar                                                             : volt/second
'''