import os, re, sys, json, argparse, time, glob, pickle, copy, h5py, getopt
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from tcn import tcn
import tensorflow as tf
from sklearn.metrics import r2_score

try:
    from utilities.utils import create_datasets, generate_fourier_surrogate, load_bidirectional_models, bootstrap_scores
    from model.predictability import predictability_full
except ImportError:
    from .utilities.utils import create_datasets, generate_fourier_surrogate, load_bidirectional_models, bootstrap_scores
    from .model.predictability import predictability_full

def synthetic_model_full_handler(model_parameters):

    """
    It is specifically designed for training the synthetic data in ./Data/static_artificial. 
    Handles multiple configurations and coupling constants to study the system's behavior under various scenarios.

    Parameters:
    - model_parameters (dict): Configuration dictionary containing all necessary parameters to control the model 
      and data generation process, including:
        - model: A callable model function for generating data (ar_processes, henon_maps, rossler_lorenz).
        - directory: Base directory to save outputs.
        - method: Method used for generating synthetic data.
        - length_vars: Number of variables (timesteps) to generate.
        - discard: Number of initial values to discard from the generated data.
        - init_cond: Initial conditions for the synthetic data generation.
        - encoder_type: Type of encoder used in the model (0: Padded, 1: Shuffled, 2: Random).
        - var1, var2: Indices of the variables to be used.
        - couplings: Array of coupling constants to use in experiments.
        - seq_length: Length of the sequence for the autoencoder.
        - shift: Shift parameter for data segmentation.
        - lag: Lag parameter for creating time-lagged series.
        - scaling_method: Method used for scaling the data.
        - batch_size: Batch size for training the model.
        - epochs: Number of epochs to train the model.
        - noise: Noise level applied to the autoencoder input.

    This function iteratively processes each specified coupling constant, generates synthetic data, creates datasets,
    trains models, and saves all results to designated directories structured by encoder type, noise level, epochs,
    sequence length, shift, lag, and coupling constant. If the required number of model files is not found in the
    directory, it will proceed with the generation and training processes.

    Returns:
    None
    """

    model = model_parameters['model']
    directory = model_parameters['directory']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = np.array(model_parameters['couplings'])
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    for coupling_constant in couplings:

        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
            epochs) + '/SeqLen_' + str(seq_length) + \
                        '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/Coupling_' + str(coupling_constant) + '/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

        if len(filelist) != 4:

            # Generate Data
            variables = model(method, length_vars, init_cond, discard, coupling=coupling_constant)

            _, _, _, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                  seq_length=seq_length, shift=shift)
            _, _, _, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                     seq_length=seq_length, shift=shift)
            _, _, _, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                  seq_length=seq_length, shift=shift)
            _, _, _, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                     seq_length=seq_length, shift=shift)
            
            if encoder_type == 1:
                _, _, _, full_dataX_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1], seed=42),
                                                        scaling_method=scaling_method, seq_length=seq_length, shift=shift)
                _, _, _, full_dataY_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2], seed=42),
                                                        scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            else:
                full_dataX_sff = []
                full_dataY_sff = []

            # Save Time Series
            filename = models_folder + 'variables.npy'
            np.save(filename, variables)

            # Train Models
            filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
            filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
            filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
            filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
            plot_nameX = models_folder + 'Y to X training and validation.png'
            plot_nameY = models_folder + 'X to Y training and validation.png'

            ##############  Y to X  ##############
            auto_encoder_model_XYX, auto_encoder_model_XYXP = predictability_full(model_parameters, filenameX, filenameXP,
                                                                             full_dataX, full_dataX_, full_dataX_sff, 
                                                                             full_dataY, full_dataY_, full_dataY_sff,
                                                                             save_plot=plot_nameX)

            # ##############  X to Y  ##############
            auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability_full(model_parameters, filenameY, filenameYP,
                                                                             full_dataY, full_dataY_, full_dataY_sff,
                                                                             full_dataX, full_dataX_, full_dataX_sff, 
                                                                             save_plot=plot_nameY)

def synthetic_model_boot_plot(model_parameters):
    """
    Executes bootstrapped evaluations for synthetic models and generates plots of comparative scores
    across different coupling constants.
    """
    
    directory = model_parameters['directory']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']
    encoder_type_name = {0: 'Padded_Encoder', 1: 'Shuffle_encoder', 2: 'Random_Encoder'}[model_parameters['encoder_type']]

    base_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Check for existing score files
    filelist = glob.glob(os.path.join(base_folder, "score*"))
    if len(filelist) == 0:
        perform_bootstrapping(base_folder, model_parameters)
    else:
        print('Loading Scores ... \n')
        load_and_plot_scores(base_folder, model_parameters)

def prepare_datasets(folder, model_parameters, coupling_constant):
    """ Prepares datasets for bootstrapping. """

    model = model_parameters['model']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    encoder_type = model_parameters['encoder_type']     
    fulldata_testing = model_parameters['fulldata_testing']

    # Generate Data
    # filename = folder + 'variables.npy'
    # variables = np.load(filename)
    variables = model(method, length_vars, init_cond, discard, coupling=coupling_constant)

    _, _, testX, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
    _, _, testX_, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                        seq_length=seq_length, shift=shift)
    _, _, testY, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                        seq_length=seq_length, shift=shift)
    _, _, testY_, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                        seq_length=seq_length, shift=shift)
    if encoder_type == 1:
        _, _, testX_sff, full_dataX_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1], seed=42),
                                                scaling_method=scaling_method, seq_length=seq_length, shift=shift)
        _, _, testY_sff, full_dataY_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2], seed=42),
                                                scaling_method=scaling_method, seq_length=seq_length, shift=shift)
    else:
        testX_sff = testX
        testY_sff = testY
        full_dataX_sff = full_dataX
        full_dataY_sff = full_dataY

    if fulldata_testing:
        testX = full_dataX
        testX_ = full_dataX_
        testY = full_dataY
        testY_ = full_dataY_
        testX_sff = full_dataX_sff
        testY_sff = full_dataY_sff

    return testX, testX_, testY, testY_, testX_sff, testY_sff 
        
def plot_scores(model_parameters, folder, score_boot_mean, score_boot_std):

    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']
    couplings = np.array(model_parameters['couplings'])

    # Save Bootstraps Figure
    fig = plt.figure(figsize=(20, 5))
    plt.errorbar(couplings, score_boot_mean[0], yerr=score_boot_std[0], fmt='o-', color='red',
                    label='Y to X')
    plt.errorbar(couplings, score_boot_mean[1], yerr=score_boot_std[1], fmt='o-', color='blue',
                    label='X to Y')
    plt.plot(couplings, np.zeros((len(couplings))), '-k')
    plt.xlabel('Couplings', fontsize=22)
    plt.ylabel('CGSI', fontsize=22)
    plt.title('Henon Maps', fontsize=26)
    plt.legend()

    # Adjust tick parameters
    plt.tick_params(axis='both', which='major', labelsize=18)  
    plt.tick_params(axis='both', which='minor', labelsize=16) 

    if save_fig:
        save_plot = folder + 'TACI_Bootstraps.png'

        plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                    facecolor="w", edgecolor='w', orientation='landscape')

    if paint_fig:
        plt.show()
    else:
        plt.close(fig)

def perform_bootstrapping(base_folder, model_parameters):
    """ Handles the bootstrapping process. """
    
    directory = model_parameters['directory']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    couplings = np.array(model_parameters['couplings'])
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']
    encoder_type_name = {0: 'Padded_Encoder', 1: 'Shuffle_encoder', 2: 'Random_Encoder'}[model_parameters['encoder_type']]

    
    score_boot_mean = np.zeros((2, len(model_parameters['couplings'])))
    score_boot_std = np.zeros((2, len(model_parameters['couplings'])))

    for idx, coupling_constant in enumerate(couplings):
        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
                epochs) + '/SeqLen_' + str(seq_length) + \
                            '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/Coupling_' + str(coupling_constant) + '/'
        filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))
        
        if len(filelist) != 4:
            return  # Early exit if models are not fully saved
        else:
            model_XYX, model_XYXP, model_YXY, model_YXYP = load_bidirectional_models(models_folder)
            testX, testX_, testY, testY_, testX_sff, testY_sff = prepare_datasets(models_folder, model_parameters, coupling_constant)
            
            ### Bootstraps Y to X ###
            scores = bootstrap_scores(model_parameters, testX, testY_sff, testX_, model_XYX, model_XYXP)
            score_boot_mean[0, idx] = np.mean(scores)
            score_boot_std[0, idx] = np.std(scores)
            
            ### Bootstraps X to Y ###                
            scores = bootstrap_scores(model_parameters, testY, testX_sff, testY_, model_YXY, model_YXYP)
            score_boot_mean[1, idx] = np.mean(scores)
            score_boot_std[1, idx] = np.std(scores)

            # Save Scores
            filename = base_folder + 'score_boot_mean.npy'
            np.save(filename, score_boot_mean)
            filename = base_folder + 'score_boot_std.npy'
            np.save(filename, score_boot_std)

    plot_scores(model_parameters, base_folder, score_boot_mean, score_boot_std)

def load_and_plot_scores(base_folder, model_parameters):
    
    # Load Scores
    filename = base_folder + 'score_boot_mean.npy'
    score_boot_mean = np.load(filename)
    filename = base_folder + 'score_boot_std.npy'
    score_boot_std = np.load(filename)

    plot_scores(model_parameters, base_folder, score_boot_mean, score_boot_std)