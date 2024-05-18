import os, sys, time, glob, shutil
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.integrate import RK45

from scipy import signal

################################################################
##################### Lorenz Attractors  ##################
################################################################

def generate_lorenz(method, length_vars, init_cond, discard, coupling):
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    if init_cond is not None:
        X0 = init_cond
    else:
        X0 = np.random.rand(3).flatten()  # Ensuring it's a flat array of length 3

    if method == 'odeint':
        def lorenz_ode(X, t):
            x, y, z = X
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return [dx, dy, dz]

        # Integrate the Lorenz equations on the time grid t
        time = np.arange(0, length_vars / 100, 0.01)
        result = odeint(lorenz_ode, X0, time)

        variables = result[discard:, :]  # Discard transient dynamics

    return variables
