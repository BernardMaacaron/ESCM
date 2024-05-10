EQ_LIF_N = '''
dv/dt = -(v - I/gL) / tau                                                        : volt (unless refractory)
X                                                                                : 1 
Y                                                                                : 1

I = Ia + Ig                                                              : volt/second # total input current - GABA, Inhibitory and AMPA, Excitatory
dIg/dt = (-Ig)/taugd                                                             : volt/second
dIa/dt = (-Ia)/tauad                                                             : volt/second
'''

#############################

Bi_EQ_LIF = '''
dv/dt = -(v - I/gL) / tau                                                        : volt (unless refractory)
X                                                                                : 1 
Y                                                                                : 1

I = I_GABA + I_AMPA                                                              : volt/second # total input current - GABA, Inhibitory and AMPA, Excitatory
I_GABA = K_GABA * Ig                                                             : volt/second
dIg/dt = (-Ig+Ig1)/taugd                                                         : volt/second
dIg1/dt = -Ig1/taugr                                                             : volt/second

I_AMPA = K_AMPA * Ia                                                             : volt/second
dIa/dt = (-Ia+Ia1)/tauad                                                         : volt/second
dIa1/dt = -Ia1/tauar                                                             : volt/second
'''

############################

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