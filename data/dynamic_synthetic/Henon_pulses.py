import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

################################################################
############## Henon-Henon Temporal Attractors  ################
################################################################


def generate_henon03_henon03_temporal_coupling(couplingx, couplingy, length_vars, discard, init_cond):
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - (couplingx[ii] * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - couplingx[ii]) * variables[ii, 0] ** 2) + 0.3 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (couplingy[ii] * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - couplingy[ii]) * variables[ii, 2] ** 2) + 0.3 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    return variables[discard:]

def on_off_alternating(length_vars, discard, coupling_constant, init_cond):

    """
    Generates a variable series with an on-off-on pattern based on temporal coupling in a Henon map.

    Parameters:
    - length_vars (int): Total length of the variables array.
    - discard (int): Number of initial values to discard in the pulse function calculation.
    - coupling_constant (float): Constant factor to scale the pulse function.
    - init_cond (list or array, optional): Initial conditions for the Henon map. If None, random initial conditions are used.

    Returns:
    - numpy.ndarray: Array of generated variables based on the specified Henon map dynamics.
    """

    if init_cond is None:
        init_cond = [0.7, 0, 0.7, 0]

    # Generate Variables
    couplingx = np.zeros((length_vars))
    t = np.linspace(0, 1, length_vars-discard)
    pulse_func = signal.square(2 * np.pi * 1.49 * t)
    pulse_func[pulse_func<=0] = 0
    pulse_func = np.concatenate((np.ones((discard)), pulse_func))    
    couplingy = pulse_func*coupling_constant

    variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)

    np.random.seed(0)
    while np.isnan(variables).any():
        init_cond_ = np.random.random(4)
        variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)
        
    return variables, couplingx, couplingy        

def on_off_widths(length_vars, discard, coupling_constant, init_cond):

    """
    The function generates a coupling signal that alternates between 'on' (set to coupling_constant) and 'off' 
    (set to zero) states across ten equal segments of the total length (minus any discarded initial points). 

    Parameters:
    - length_vars (int): Total length of the variables array.
    - discard (int): Number of initial values to discard in the pulse function calculation.
    - coupling_constant (float): Constant factor to scale the pulse function.
    - init_cond (list or array, optional): Initial conditions for the Henon map. If None, random initial conditions are used.

    Returns:
    - numpy.ndarray: Array of generated variables based on the specified Henon map dynamics.
    """

    if init_cond is None:
        init_cond = [0.7, 0, 0.7, 0]

    # Generate Variables
    couplingx = np.zeros((length_vars))
    step = (length_vars-discard)//10
    couplingy = np.ones((length_vars))

    couplingy[:] = coupling_constant*0
    couplingy[1*step:] = coupling_constant*0
    couplingy[2*step:] = coupling_constant
    couplingy[3*step:] = coupling_constant*0
    couplingy[4*step:] = coupling_constant*0
    couplingy[5*step:] = coupling_constant
    couplingy[6*step:] = coupling_constant*0
    couplingy[7*step:] = coupling_constant*0
    couplingy[8*step:] = coupling_constant
    couplingy[9*step:] = coupling_constant
    couplingy[10*step:] = coupling_constant*0

    variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)

    np.random.seed(0)
    while np.isnan(variables).any():
        init_cond_ = np.random.random(4)
        variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)

    return variables, couplingx, couplingy             

def on_off_flip(length_vars, discard, coupling_constant, init_cond):

    """
    The function generates two coupling arrays (`couplingx` and `couplingy`):
    - 'couplingx' starts 'off' (set to zero) and flips to 'on' (set to coupling_constant) halfway through the simulation.
    - 'couplingy' starts 'on' (set to coupling_constant) and flips to 'off' (set to zero) at the same halfway point.

    Parameters:
    - length_vars (int): Total length of the variables array.
    - discard (int): Number of initial values to discard in the pulse function calculation.
    - coupling_constant (float): Constant factor to scale the pulse function.
    - init_cond (list or array, optional): Initial conditions for the Henon map. If None, random initial conditions are used.

    Returns:
    - numpy.ndarray: Array of generated variables based on the specified Henon map dynamics.
    """

    # Generate Variables
    couplingx = np.ones((length_vars))*coupling_constant
    couplingx[:((length_vars-discard)//2+discard)] = couplingx[:(length_vars-discard)//2+discard]*0
    couplingy = np.ones((length_vars))*coupling_constant
    couplingy[(length_vars-discard)//2+discard:] = couplingy[(length_vars-discard)//2+discard:]*0

    variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)

    np.random.seed(0)
    while np.isnan(variables).any():
        init_cond_ = np.random.random(4)
        variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)

    return variables, couplingx, couplingy            


import matplotlib.pyplot as plt

def plot_henon_pulses(variables_, couplingx, couplingy, var1, var2, discard, skip_plot):
    
    """
    Plots various time series and phase plots from a simulation of dynamical systems with coupling.

    Parameters:
    - variables_ (numpy.ndarray): The variables from the dynamical system simulation.
    - couplingx (numpy.ndarray): Time series data for the coupling from Y to X.
    - couplingy (numpy.ndarray): Time series data for the coupling from X to Y.
    - discard (int): Number of initial data points to discard in the coupling plots to avoid initial transients.
    - skip_plot (int): The interval at which data points are plotted to reduce density and improve clarity.

    Outputs:
    A multi-panel plot including time series of selected variables and coupling, along with phase space plots.
    """

    var1 = 0
    var2 = 2

    plt.figure(figsize=(20, 15))

    # Plotting the first variable
    plt.subplot(5, 1, 1)
    plt.plot(variables_[::skip_plot, var1], 'k')
    plt.grid()

    # Plotting the second variable
    plt.subplot(5, 1, 2)
    plt.plot(variables_[::skip_plot, var2], 'k')
    plt.grid()

    # Plotting coupling information
    plt.subplot(5, 1, 3)
    plt.plot(couplingx[discard:], 'r', label='Y to X')
    plt.plot(couplingy[discard:], 'b', label='X to Y')
    plt.grid()
    plt.legend()
    plt.title('Coupling')

    # Phase plot for the first pair of variables
    plt.subplot(5, 1, 4)
    plt.plot(variables_[:, 0], variables_[:, 1], '.k', markersize=1)

    # Phase plot for the second pair of variables
    plt.subplot(5, 1, 5)
    plt.plot(variables_[:, 2], variables_[:, 3], '.k', markersize=1)

    # Show the full figure
    plt.show()

