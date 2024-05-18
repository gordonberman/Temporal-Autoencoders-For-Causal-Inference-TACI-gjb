import os, sys, time, glob, shutil
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.integrate import RK45

from scipy import signal

################################################################
################## Autoregressive Processes  ###################
################################################################

def generate_ar_processes(method, length_vars, init_cond, discard, coupling):

    # Parameters
    T = length_vars  # Total points
    C = coupling
    variance = 0.1

    # Initialize x and y using normal distribution with zero mean and unit variance
    x = np.random.randn(T)
    y = np.random.randn(T)

    # Generate Gaussian noise for nx and ny
    nx = np.random.normal(0, np.sqrt(variance), T)
    ny = np.random.normal(0, np.sqrt(variance), T)

    # Generate the autoregressive processes
    for t in range(1, T):
        x[t] = 0.5 * x[t-1] + 0.1 * y[t-1] + nx[t]
        y[t] = C * x[t-1] + 0.7 * y[t-1] + ny[t]

    variables = np.column_stack([x,y])  
    variables = variables[discard:]
    
    return variables

def ar_processes_plots():
    method = None
    length_vars = 101000
    discard = 1000
    init_cond = None
    couplings = np.linspace(0, 0.6, 9)
    spacing = 1000

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    for ii, ax in enumerate(axes.flatten()):
        coupling = round(couplings[ii], 2)
        variables = generate_ar_processes(method, length_vars, init_cond, discard, coupling)
                
        ax.plot(variables[::spacing, 0], label='Driver')
        ax.plot(variables[::spacing, 1] - 3, label='Response')
        ax.set_title('Driver and Response with C = ' + str(coupling))
        ax.set_xticks([])
        ax.set_yticks([])

    # Create a single legend outside the plots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

    plt.tight_layout()
    plt.show()

    return None
