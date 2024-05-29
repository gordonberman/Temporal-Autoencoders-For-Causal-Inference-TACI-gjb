import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tcn import tcn
import tensorflow as tf

def calculate_acf(max_lag=1000, paint=False):
    
    corr = np.empty((variables.shape[1], max_lag))

    print('Autocorrelation Time:')

    for ii in range(variables.shape[1]):
        for jj in range(max_lag):

            if jj == 0:
                corr[ii, jj] = 1
            else:
                corr[ii, jj] = np.corrcoef(variables[jj:, ii], pos_data[:-jj, ii])[0][1]
    
    if paint:
        plt.figure(figsize=(20,5))
        for ii in range(variables.shape[1]):
            plt.plot(corr[ii])
        plt.plot(np.zeros((max_lag)), '-k')
        plt.xlabel('Lags')
        plt.title('Autocorrelation Time')
        plt.show()
    
    return corr

def create_time_lagged_series(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_and_reshape_lagged_series(data, seq_length, shift):
    """Helper function to create a time-lagged series and reshape it."""
    raw_df = create_time_lagged_series(data.reshape(-1, 1), seq_length, 1, dropnan=True)
    reshaped_data = np.array(raw_df)[:, :-1]
    reshaped_data = np.expand_dims(reshaped_data, axis=-1)
    return reshaped_data[::shift]

def create_datasets(dataset, scaling_method='None', seq_length=1, shift=1, subsample=1, test_size=0.2, validation_size=0.1):
    
    """
    Create datasets by scaling, creating a time-lagged series, and splitting into train, validation, and test sets.

    Parameters:
    - dataset: Input dataset.
    - scaling_method: Method for scaling data.
    - seq_length: Sequence length for time-lagged series.
    - shift: Shift value for subsampling.
    - subsample: Rate for subsampling.
    - test_size: Proportion of dataset to include in the test split.
    - validation_size: Proportion of dataset to include in the validation split.

    Returns:
    - trainX, valX, testX: Training, validation, and test datasets.
    """
    
    # Scaling the dataset based on the provided method
    scalers = {
        'MinMax': MinMaxScaler(feature_range=(0, 1)),
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'None': None
    }
    scaler = scalers.get(scaling_method)
    if scaler:
        dataset_scaled = scaler.fit_transform(dataset.reshape(-1, 1))
    else:
        dataset_scaled = dataset.reshape(-1, 1)

    # Create and reshape time-lagged series for dataset and index
    full_data = create_and_reshape_lagged_series(dataset_scaled, seq_length, shift)
    full_data_ind = create_and_reshape_lagged_series(np.arange(len(dataset)).reshape(-1, 1), seq_length, shift)

    dataset_subsampled = full_data.copy()
    indx = full_data_ind.copy()

    # Apply subsampling
    if subsample > 1:
        dataset_subsampled = dataset_subsampled[::subsample]
        indx = indx[::subsample]

    # Splitting dataset into training and test sets
    if len(dataset) <= 100000:
        test_size_ = 0.3
    elif 100000 < len(dataset) <= 1000000:
        test_size_ = 0.2
    else:  # dataset_subsampled > 1000000
        test_size_ = 0.02

    X_train, X_temp, y_train, y_temp = train_test_split(dataset_subsampled, dataset_subsampled, test_size=test_size_, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, X_temp, test_size=0.5, random_state=42, shuffle=True)

    return X_train, X_val, X_test, full_data, indx

def generate_fourier_surrogate(data, seed=None):
    
    if seed is not None:
        np.random.seed(seed)  # Initialize the random number generator for reproducibility

    ts = data.copy()
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, len(ts_fourier)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    surrogate = np.fft.irfft(ts_fourier_new, n=len(ts))

    return surrogate

def relative_directionality(scores_x, scores_y):

    if (scores_x)<0:
        scores_x = 0
    if (scores_y)<0:
        scores_y = 0

    if (scores_x + scores_y) == 0:
        directionality = 0
    else:
        directionality = (scores_y-scores_x)/(np.abs(scores_x+scores_y))

    return directionality

def load_unidirectional_models(folder):
    """ Loads the four models required for bootstrapping. """
    
    # Load Models
    filenameY = folder + 'auto_encoder_model_YXY.hdf5'
    model_YXY = tf.keras.models.load_model(filenameY, custom_objects={'TCN': tcn.TCN})

    filenameYP = folder + 'auto_encoder_model_YXYP.hdf5'
    model_YXYP = tf.keras.models.load_model(filenameYP, custom_objects={'TCN': tcn.TCN})
    
    return model_YXY, model_YXYP


def load_bidirectional_models(folder):
    """ Loads the four models required for bootstrapping. """
    
    # Load Models
    filenameX = folder + 'auto_encoder_model_XYX.hdf5'
    model_XYX = tf.keras.models.load_model(filenameX, custom_objects={'TCN': tcn.TCN})

    filenameXP = folder + 'auto_encoder_model_XYXP.hdf5'
    model_XYXP = tf.keras.models.load_model(filenameXP, custom_objects={'TCN': tcn.TCN})

    filenameY = folder + 'auto_encoder_model_YXY.hdf5'
    model_YXY = tf.keras.models.load_model(filenameY, custom_objects={'TCN': tcn.TCN})

    filenameYP = folder + 'auto_encoder_model_YXYP.hdf5'
    model_YXYP = tf.keras.models.load_model(filenameYP, custom_objects={'TCN': tcn.TCN})
    
    return model_XYX, model_XYXP, model_YXY, model_YXYP

def comparative_index(y_true, y_pred, y_predP, tol_default=1e-4, default=1e-4):

    """
    Calculates a comparative index between two prediction sets compared to the true values using the R2 score.

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predictions from the first model.
    - y_predP (array-like): Predictions from the second model.
    - tol_default (float): Tolerance used to determine if two R2 scores are considered equivalent.
    - default (float): Default value to use for the score when the R2 score is negative or equivalent by tolerance.

    Returns:
    - list: A list containing the comparative scores for each set of predictions.

    The function evaluates the R2 score for each prediction set against the true values. If the R2 scores are
    negative or approximately equal within the specified tolerance, the scores are adjusted to the default value.
    The comparative index is then calculated as the normalized difference between the two scores.
    """
    
    rse1 = r2_score(y_true, y_pred)
    rse2 = r2_score(y_true, y_predP)

    # Adjust scores based on conditions
    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
        rse2 = default
        rse1 = default
    if rse2 < 0:
        rse2 = default

    # Calculate comparative score
    score = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

    return score

def bootstrap_scores(model_parameters, testX, testY, testX_, model, modelP):

    bootstraps_samples = model_parameters['bootstraps_samples']
    batch_size = model_parameters['batch_size']
    tol_default = model_parameters['tol_default']
    default = 1e-4

    scores = []                
    for jj in range(bootstraps_samples):

        indxs = np.random.choice(np.arange(testX.shape[0]), size=testX.shape[0], replace=True)
        
        boot_testX = np.squeeze(testX)[indxs]
        boot_testY = np.squeeze(testY)[indxs]
        boot_testX_ = np.squeeze(testX_)[indxs]

        y_true_boot = boot_testX_.flatten()
        y_pred_boot = model.predict([boot_testX, boot_testY], batch_size=batch_size, verbose=0).flatten()
        y_predP_boot = modelP.predict([boot_testX, np.random.random(boot_testY.shape)], 
                                                        batch_size=batch_size, verbose=0).flatten()
        
        scores.append(comparative_index(y_true_boot, y_pred_boot, y_predP_boot, tol_default, default))

    return scores

import argparse
import numpy as np

def create_model_parameters(nb_filters=32, dilations=[1, 2, 4, 8, 16, 32], kernel_size=32, nb_stacks=1, 
                            ts_dimension=1, dropout_rate_tcn=0.0, dropout_rate_hidden=0.0,
                            L1=0.0, L2=0.0, conv_kernel_init='he_normal', padding='causal',
                            tcn_act_funct='elu', latent_sample_rate=2, act_funct='elu',
                            filters_conv1d=16, kernel_size_conv1d=1, activation_conv1d='elu',
                            layer_nodes=128, n_layers=4, pad_encoder=False, concatenate_layers=False,
                            model_summary=False, encoder_type=2, batch_size=512, epochs=300,
                            noise=0.0, verbose=0, loss_funct='mse', shuffle=True, 
                            fulldata_testing=True, scaling_method='Standard', bootstraps_samples=10,
                            tol_default=0.01, paint_fig=False, save_fig=True):
    """
    Create model parameters with direct input or default values.

    Returns a dictionary of model parameters.
    """

    # Assemble the model parameters dictionary
    model_parameters = {
        'nb_filters': nb_filters,
        'dilations': dilations,
        'kernel_size': kernel_size,
        'nb_stacks': nb_stacks,
        'ts_dimension': ts_dimension,
        'dropout_rate_tcn': dropout_rate_tcn,
        'dropout_rate_hidden': dropout_rate_hidden,
        'kernel_regularizer': {'L1': L1, 'L2': L2},
        'conv_kernel_init': conv_kernel_init,
        'padding': padding,
        'tcn_act_funct': tcn_act_funct,
        'latent_sample_rate': latent_sample_rate,
        'act_funct': act_funct,
        'filters_conv1d': filters_conv1d,
        'kernel_size_conv1d': kernel_size_conv1d,
        'activation_conv1d': activation_conv1d,
        'layer_nodes': layer_nodes,
        'n_layers': n_layers,
        'pad_encoder': pad_encoder,
        'concatenate_layers': concatenate_layers,
        'model_summary': model_summary,
        'encoder_type': encoder_type,
        'batch_size': batch_size,
        'epochs': epochs,
        'noise': noise,
        'verbose': verbose,
        'loss_funct': loss_funct,
        'shuffle': shuffle,
        'fulldata_testing': fulldata_testing,
        'scaling_method': scaling_method,
        'bootstraps_samples': bootstraps_samples,
        'tol_default': tol_default,
        'paint_fig': paint_fig,
        'save_fig': save_fig,
    }

    return model_parameters
