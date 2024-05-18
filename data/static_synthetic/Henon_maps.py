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
##################### Henon-Henon Attractors  ##################
################################################################

def generate_henon03_henon03(method, length_vars, init_cond, discard, coupling):
    b1, b2 = 0.3, 0.3
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - variables[ii, 0] ** 2 + b1 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (coupling * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - coupling) * variables[ii, 2] ** 2) + b2 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    variables = variables[discard:]

    return variables


def henon03_henon03_plots():

    method = None
    length_vars = 101000
    discard = 1000
    init_cond = [0.7, 0, 0.7, 0]
    couplings = np.linspace(0.0, 0.8, 8)

    coupling = round(couplings[0], 2)
    paint = False
    variables = generate_henon03_henon03(method, length_vars, init_cond, discard, coupling)

    plt.figure(figsize=(20, 15), facecolor='white')
    plt.subplot(3, 3, 1)
    plt.plot(variables[:, 0], variables[:, 1], '.k', markersize=1)
    plt.title('Driver', fontsize=24)
    plt.xticks([])
    plt.yticks([])
    
    for ii in range(len(couplings)):
        coupling = round(couplings[ii], 2)
        paint = False
        variables = generate_henon03_henon03(method, length_vars, init_cond, discard, coupling)

        plt.subplot(3, 3, 2 + ii)
        plt.plot(variables[:, 2], variables[:, 3], '.k', markersize=1)
        plt.title('Response with C = ' + str(coupling), fontsize=24)
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout()
    plt.show()

    return None


def generate_henon03_henon01(method, length_vars, init_cond, discard, coupling):
    b1, b2 = 0.3, 0.1
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - variables[ii, 0] ** 2 + b1 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (coupling * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - coupling) * variables[ii, 2] ** 2) + b2 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    variables = variables[discard:]

    return variables


def generate_henon01_henon03(method, length_vars, init_cond, discard, coupling):
    b1, b2 = 0.1, 0.3
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - variables[ii, 0] ** 2 + b1 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (coupling * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - coupling) * variables[ii, 2] ** 2) + b2 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    variables = variables[discard:]

    return variables


