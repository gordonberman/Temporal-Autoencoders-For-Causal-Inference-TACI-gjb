import os, re, sys, json, argparse, time, glob, pickle, copy, h5py, getopt
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from tcn import tcn
import tensorflow as tf
from sklearn.metrics import r2_score

import glob
from joblib import Parallel, delayed

try:
    from utilities.utils import create_datasets, generate_fourier_surrogate
    from utilities.utils import load_unidirectional_models, load_bidirectional_models, bootstrap_scores
    from model.predictability import predictability_full
except ImportError:
    from .utilities.utils import create_datasets, generate_fourier_surrogate, load_bidirectional_models, bootstrap_scores
    from .model.predictability import predictability_full

def prepare_datasets(model_parameters, variables):
    """ Prepares datasets for bootstrapping. """

    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    encoder_type = model_parameters['encoder_type']     
    fulldata_testing = model_parameters['fulldata_testing']

    _, _, testX, full_dataX, full_data_trackerX = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
    _, _, testX_, full_dataX_, full_data_trackerX_ = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                        seq_length=seq_length, shift=shift)
    _, _, testY, full_dataY, full_data_trackerY = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                        seq_length=seq_length, shift=shift)
    _, _, testY_, full_dataY_, full_data_trackerY_ = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
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

    return testX, testX_, testY, testY_, testX_sff, testY_sff, full_data_trackerX_

def temporal_unidirectional_handler(model_parameters, variables):

    """
    Trains a temporal autoencoder model for unidirectional causal inference on given variables.

    Parameters:
    - model_parameters (dict): Configuration dictionary containing all necessary parameters to control 
      the model and data generation process, including:
        - directory: Base directory to save outputs.
        - encoder_type: Type of encoder used in the model (0: Padded, 1: Shuffled, 2: Random).
        - var1, var2: Indices of the variables to be used.
        - seq_length: Length of the sequence for the autoencoder.
        - shift: Shift parameter for data segmentation.
        - lag: Lag parameter for creating time-lagged series.
        - scaling_method: Method used for scaling the data.
        - epochs: Number of epochs to train the model.
        - noise: Noise level applied to the autoencoder input.

    - variables (numpy.ndarray): The dataset containing the variables to be used for training.

    This function performs the following steps:
    1. Sets up the directory structure based on encoder type and other parameters.
    2. Checks for existing model files to avoid redundant training.
    3. Prepares datasets for training by creating time-lagged series and scaling the data.
    4. Saves the dataset.
    5. Trains the autoencoder models for unidirectional causal inference from variable X to Y.
    
    Returns:
    None
    """

    directory = model_parameters['directory']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    print('\nChecking for trained models ...')

    if len(filelist) != 2:

        print('No models found')
        print('Training ...')

        _, _, _, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
        _, _, _, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        
        if encoder_type == 1:
            _, _, _, full_dataX_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            _, _, _, full_dataY_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
        else:
            full_dataX_sff = []
            full_dataY_sff = []

        # Train Models
        filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
        filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
        plot_nameY = models_folder + 'X to Y training and validation.png'

        # ##############  X to Y  ##############
        auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability_full(model_parameters, filenameY, filenameYP,
                                                                            full_dataY, full_dataY_, full_dataY_sff,
                                                                            full_dataX, full_dataX_, full_dataX_sff, 
                                                                            save_plot=plot_nameY)
        
        print('Training Done')
    else:
        print('Models found')
        print('Loading ...')


def temporal_unidirectional_model_boot_plot(model_parameters, variables):
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
    tol_default = model_parameters['tol_default']
    window_len = model_parameters['window_len']

    base_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                '/Tolerance_' + str(tol_default) + '/window_length_' + str(window_len) + '/'
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Check for existing score files
    filelist = glob.glob(os.path.join(models_folder, "score*"))
    if len(filelist) == 0:
        perform_unidirectional_bootstrapping(base_folder, models_folder, model_parameters, variables)
    else:
        print('Loading Scores ... \n')
        load_and_plot_unidirectional_scores(models_folder, model_parameters)


#def perform_unidirectional_bootstrapping(base_folder, models_folder, model_parameters, variables):
#    """ Handles the bootstrapping process. """
#
#    window_len = model_parameters['window_len']
#
#    filelist = glob.glob(os.path.join(base_folder, "*.hdf5"))
#
#    if len(filelist) != 2:
#        return  # Early exit if models are not fully saved
#    else:
#        model_YXY, model_YXYP = load_unidirectional_models(base_folder)
#        testX, testX_, testY, testY_, testX_sff, testY_sff, full_data_trackerX_ = prepare_datasets(model_parameters, variables)
#
#        if window_len > len(testX_):
#            window_len = len(testX_)
#
#        intervals = np.zeros((len(testX_) // window_len))
#
#        def process_interval(ii):
#            interval = np.arange(ii * window_len, (ii + 1) * window_len)
#            intervals[ii] = np.squeeze(full_data_trackerX_[interval]).flatten()[-1]
#
#            # Bootstraps X to Y                
#            scores = bootstrap_scores(model_parameters, testY[interval], testX_sff[interval], testY_[interval], model_YXY, model_YXYP)
#            return np.mean(scores), np.std(scores)
#
#        results = Parallel(n_jobs=-1)(delayed(process_interval)(ii) for ii in range(len(testX_) // window_len))
#
#        score_boot_mean, score_boot_std = zip(*results)
#        score_boot_mean = np.array(score_boot_mean)
#        score_boot_std = np.array(score_boot_std)
#
#        # Save Scores
#        filename = os.path.join(models_folder, 'intervals.npy')
#        np.save(filename, intervals)
#        filename = os.path.join(models_folder, 'score_boot_mean.npy')
#        np.save(filename, score_boot_mean)
#        filename = os.path.join(models_folder, 'score_boot_std.npy')
#        np.save(filename, score_boot_std)
#
#    plot_unidirectional_scores(models_folder, model_parameters, intervals, score_boot_mean, score_boot_std)


def perform_unidirectional_bootstrapping(base_folder, models_folder, model_parameters, variables):
    """ Handles the bootstrapping process. """
    
    window_len = model_parameters['window_len']
    
    filelist = glob.glob(os.path.join(base_folder, "*.hdf5"))
    
    if len(filelist) != 2:
        return  # Early exit if models are not fully saved
    else:
        model_YXY, model_YXYP = load_unidirectional_models(base_folder)
        testX, testX_, testY, testY_, testX_sff, testY_sff, full_data_trackerX_ = prepare_datasets(model_parameters, variables)
        
        if window_len > len(testX_):
            window_len = len(testX_)
        
        score_boot_mean = np.zeros((len(testX_) // window_len))
        score_boot_std = np.zeros((len(testX_) // window_len))
        intervals = np.zeros((len(testX_) // window_len))

        for ii in range(len(testX_) // window_len):

            interval = np.arange(ii * window_len, (ii + 1) * window_len)
            intervals[ii] = np.squeeze(full_data_trackerX_[interval]).flatten()[-1]
    
            ### Bootstraps X to Y ###                
            scores = bootstrap_scores(model_parameters, testY[interval], testX_sff[interval], testY_[interval], model_YXY, model_YXYP)
            score_boot_mean[ii] = np.mean(scores)
            score_boot_std[ii] = np.std(scores)

        # Save Scores
        filename = models_folder + 'intervals.npy'
        np.save(filename, intervals)
        filename = models_folder + 'score_boot_mean.npy'
        np.save(filename, score_boot_mean)
        filename = models_folder + 'score_boot_std.npy'
        np.save(filename, score_boot_std)

    plot_unidirectional_scores(models_folder, model_parameters, intervals, score_boot_mean, score_boot_std)

def load_and_plot_unidirectional_scores(models_folder, model_parameters):
    
    # Load Scores
    filename = models_folder + 'intervals.npy'
    intervals = np.load(filename)
    filename = models_folder + 'score_boot_mean.npy'
    score_boot_mean = np.load(filename)
    filename = models_folder + 'score_boot_std.npy'
    score_boot_std = np.load(filename)

    plot_unidirectional_scores(models_folder, model_parameters, intervals, score_boot_mean, score_boot_std)

def plot_unidirectional_scores(folder, model_parameters, intervals, score_boot_mean, score_boot_std):

    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']

    # Save Bootstraps Figure
    fig = plt.figure(figsize=(20, 5))
    plt.errorbar(intervals, score_boot_mean, yerr=score_boot_std, fmt='o-', color='blue',
                    label='X to Y')
    plt.plot(intervals, np.zeros((len(intervals))), '-k')
    plt.xlabel('Time', fontsize=22)
    plt.ylabel('CGSI', fontsize=22)
    plt.title('Causal Interaction', fontsize=26)
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


def temporal_bidirectional_handler(model_parameters, variables):

    """
    Trains a temporal autoencoder model for bidirectional causal inference on given variables.

    Parameters:
    - model_parameters (dict): Configuration dictionary containing all necessary parameters to control 
      the model and data generation process, including:
        - directory: Base directory to save outputs.
        - encoder_type: Type of encoder used in the model (0: Padded, 1: Shuffled, 2: Random).
        - var1, var2: Indices of the variables to be used.
        - seq_length: Length of the sequence for the autoencoder.
        - shift: Shift parameter for data segmentation.
        - lag: Lag parameter for creating time-lagged series.
        - scaling_method: Method used for scaling the data.
        - epochs: Number of epochs to train the model.
        - noise: Noise level applied to the autoencoder input.

    - variables (numpy.ndarray): The dataset containing the variables to be used for training.

    This function performs the following steps:
    1. Sets up the directory structure based on encoder type and other parameters.
    2. Checks for existing model files to avoid redundant training.
    3. Prepares datasets for training by creating time-lagged series and scaling the data.
    4. Saves the dataset.
    5. Trains the autoencoder models for bidirectional causal inference between variables X and Y.
    
    Returns:
    None
    """

    directory = model_parameters['directory']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    print('\nChecking for trained models ...')

    if len(filelist) != 4:

        print('No models found')
        print('Training ...')

        _, _, _, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
        _, _, _, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        
        if encoder_type == 1:
            _, _, _, full_dataX_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            _, _, _, full_dataY_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
        else:
            full_dataX_sff = []
            full_dataY_sff = []

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

        print('Training Done')
    else:
        print('Models found')
        print('Loading ...')
        
def temporal_bidirectional_model_boot_plot(model_parameters, variables):
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
    tol_default = model_parameters['tol_default']
    window_len = model_parameters['window_len']

    base_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                '/Tolerance_' + str(tol_default) + '/window_length_' + str(window_len) + '/'
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Check for existing score files
    filelist = glob.glob(os.path.join(models_folder, "score*"))
    if len(filelist) == 0:
        perform_bidirectional_bootstrapping(base_folder, models_folder, model_parameters, variables)
    else:
        print('Loading Scores ... \n')
        load_and_plot_bidirectional_scores(models_folder, model_parameters)
        
def perform_bidirectional_bootstrapping(base_folder, models_folder, model_parameters, variables):
    # Handles the bootstrapping process
    
    window_len = model_parameters['window_len']
    
    filelist = glob.glob(os.path.join(base_folder, "*.hdf5"))
    
    if len(filelist) != 4:
        return  # Early exit if models are not fully saved
    else:
        model_XYX, model_XYXP, model_YXY, model_YXYP = load_bidirectional_models(base_folder)
        testX, testX_, testY, testY_, testX_sff, testY_sff, full_data_trackerX_ = prepare_datasets(model_parameters, variables)
        
        if window_len > len(testX_):
            window_len = len(testX_)
        
        score_boot_mean = np.zeros((2, len(testX_) // window_len))
        score_boot_std = np.zeros((2, len(testX_) // window_len))
        intervals = np.zeros((len(testX_) // window_len))

        for ii in range(len(testX_) // window_len):

            interval = np.arange(ii * window_len, (ii + 1) * window_len)
            intervals[ii] = np.squeeze(full_data_trackerX_[interval]).flatten()[-1]

            ### Bootstraps Y to X ###
            scores = bootstrap_scores(model_parameters, testX[interval], testY_sff[interval], testX_[interval], model_XYX, model_XYXP)
            score_boot_mean[0, ii] = np.mean(scores)
            score_boot_std[0, ii] = np.std(scores)
    
            ### Bootstraps X to Y ###                
            scores = bootstrap_scores(model_parameters, testY[interval], testX_sff[interval], testY_[interval], model_YXY, model_YXYP)
            score_boot_mean[1, ii] = np.mean(scores)
            score_boot_std[1, ii] = np.std(scores)

        # Save Scores
        filename = models_folder + 'intervals.npy'
        np.save(filename, intervals)
        filename = models_folder + 'score_boot_mean.npy'
        np.save(filename, score_boot_mean)
        filename = models_folder + 'score_boot_std.npy'
        np.save(filename, score_boot_std)

    plot_bidirectional_scores(models_folder, model_parameters, intervals, score_boot_mean, score_boot_std)

def load_and_plot_bidirectional_scores(models_folder, model_parameters):
    
    # Load Scores
    filename = models_folder + 'intervals.npy'
    intervals = np.load(filename)
    filename = models_folder + 'score_boot_mean.npy'
    score_boot_mean = np.load(filename)
    filename = models_folder + 'score_boot_std.npy'
    score_boot_std = np.load(filename)

    plot_bidirectional_scores(models_folder, model_parameters, intervals, score_boot_mean, score_boot_std)

def plot_bidirectional_scores(folder, model_parameters, intervals, score_boot_mean, score_boot_std):

    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']

    # Save Bootstraps Figure
    fig = plt.figure(figsize=(20, 5))
    plt.errorbar(intervals, score_boot_mean[0], yerr=score_boot_std[0], fmt='o-', color='red',
                    label='Y to X')
    plt.errorbar(intervals, score_boot_mean[1], yerr=score_boot_std[1], fmt='o-', color='blue',
                    label='X to Y')
    plt.plot(intervals, np.zeros((len(intervals))), '-k')
    plt.xlabel('Time', fontsize=22)
    plt.ylabel('CGSI', fontsize=22)
    plt.title('Causal Interaction', fontsize=26)
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
