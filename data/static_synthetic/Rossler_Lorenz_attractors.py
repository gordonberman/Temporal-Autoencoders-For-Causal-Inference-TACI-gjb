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
################## Rossler-Lorenz Attractors  ##################
################################################################

def generate_rossler_lorenz(method, length_vars, init_cond, discard, coupling):
    sigma, beta, rho, alpha_freq, C = 10, 8 / 3, 28, 6, coupling

    np.random.seed(42)

    if init_cond is not None:
        X0 = init_cond
    else:
        X0 = np.random.rand(1, 6).flatten()

    if method == 'odeint':

        def rossler_lorenz_ode(X, t):
            x1, x2, x3, y1, y2, y3 = X

            dx1 = -alpha_freq * (x2 + x3)
            dx2 = alpha_freq * (x1 + 0.2 * x2)
            dx3 = alpha_freq * (0.2 + x3 * (x1 - 5.7))
            dy1 = sigma * (-y1 + y2)
            dy2 = rho * y1 - y2 - y1 * y3 + C * x2 ** 2
            dy3 = -(beta) * y3 + y1 * y2
            return [dx1, dx2, dx3, dy1, dy2, dy3]

        # Integrate the Rossler equations on the time grid t
        time = np.arange(0, length_vars / 100, 0.01)
        result = odeint(rossler_lorenz_ode, X0, time)

        variables = result[discard:, :]

        return variables

    elif method == 'rk45':

        def rossler_lorenz_ode(t, X):
            x1, x2, x3, y1, y2, y3 = X

            dx1 = -alpha_freq * (x2 + x3)
            dx2 = alpha_freq * (x1 + 0.2 * x2)
            dx3 = alpha_freq * (0.2 + x3 * (x1 - 5.7))
            dy1 = sigma * (-y1 + y2)
            dy2 = rho * y1 - y2 - y1 * y3 + C * x2 ** 2
            dy3 = -(beta) * y3 + y1 * y2
            return [dx1, dx2, dx3, dy1, dy2, dy3]

        # Integrate the Rossler equations on the time grid t
        integrator = RK45(rossler_lorenz_ode, 0, X0, length_vars, first_step=0.01)

        results = []
        for ii in range(length_vars):
            integrator.step()
            results.append(integrator.y)

        discard = 1000
        variables = np.array(results)[discard:, :].T

        return variables


def rossler_lorenz_plots():
    method = 'odeint'
    length_vars = 101000
    discard = 1000
    init_cond = [0, 0, 0.4, 0.3, 0.3, 0.3]
    couplings = np.linspace(0.0, 3.0, 8)

    coupling = round(couplings[0], 2)
    variables = generate_rossler_lorenz(method, length_vars, init_cond, discard, coupling)

    plt.figure(figsize=(20, 15), facecolor='white')
    plt.subplot(3, 3, 1)
    plt.plot(variables[:, 0], variables[:, 1], '.k', markersize=1)
    plt.title('Driver', fontsize=24)
    plt.xticks([])
    plt.yticks([])
    
    for ii in range(len(couplings)):
        coupling = round(couplings[ii], 2)
        variables = generate_rossler_lorenz(method, length_vars, init_cond, discard, coupling)

        plt.subplot(3, 3, 2 + ii)
        plt.plot(variables[:, 3], variables[:, 4], '.k', markersize=1)
        plt.title('Response with C = ' + str(coupling), fontsize=24)
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout()
    plt.show()

    return None

