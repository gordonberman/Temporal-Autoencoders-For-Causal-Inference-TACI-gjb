import numpy as np
from tcn import tcn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Conv1D, Reshape, Flatten, GaussianNoise, Lambda, \
    Add, Multiply
from tensorflow.keras.layers import UpSampling1D, AveragePooling1D, Multiply, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

'''
Purpose:
    
    Contains the full architecture of our model.
    The Encoder function constructs a neural network layer specifically designed for encoding time series data.
    It utilizes Temporal Convolutional Networks (TCN) and additional layers to process the input data into a
    compressed representation.
    The Bottleneck function creates a model to combine and process encoded features from two separate inputs.
    It uses multiplication to merge these features, emphasizing interaction effects between the two inputs.
    The Decoder function constructs the decoder part of an autoencoder. It is designed to reconstruct the
    original input data from the encoded representation, reversing the dimensionality reduction done by the
    Encoder and Bottleneck.

Technical Description of Inputs:

    'parameters' (dict): A dictionary containing key model parameters and hyperparameters which configure the
                 Encoder's architecture, including sequence length, time series dimension, number of filters,
                 kernel size, number of stacks, dilations, activation functions, dropout rates, and padding options.
'''

######################################## ENCODER ########################################
def Encoder(parameters):
    
    encoder_inputs = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    
    if parameters['noise'] > 0:
        encoder = GaussianNoise(stddev=parameters['noise'])(encoder_inputs)
    else:
        encoder = encoder_inputs
    
    encoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'], nb_stacks=parameters['nb_stacks'], 
                        dilations= parameters['dilations'], activation=parameters['tcn_act_funct'], padding=parameters['padding'], 
                        use_skip_connections=True, dropout_rate=parameters['dropout_rate_tcn'], return_sequences=True,
                        kernel_initializer=parameters['conv_kernel_init'])(encoder_inputs)
    
    encoder = Conv1D(filters=parameters['filters_conv1d'],
                     kernel_size=parameters['kernel_size_conv1d'], activation=parameters['activation_conv1d'], padding=parameters['padding'])(encoder)

    encoder = AveragePooling1D(pool_size=parameters['latent_sample_rate'], strides=None, padding='valid',
                               data_format='channels_last')(encoder)

    encoder = Activation(parameters['act_funct'])(encoder)

    encoder_ouputs = Flatten()(encoder)

    layer_nodes = parameters['layer_nodes']

    for ii in range(parameters['n_layers'] - 1):

        if layer_nodes < encoder_ouputs.shape[1]:

            encoder_ouputs = Dense(layer_nodes, activation=parameters['act_funct'])(encoder_ouputs)
            
            if parameters['dropout_rate_hidden'] > 0:
                encoder_ouputs = Dropout(parameters['dropout_rate_hidden'])(encoder_ouputs)

        layer_nodes = layer_nodes // 2

    if layer_nodes < encoder_ouputs.shape[1]:

        encoder_ouputs = Dense(layer_nodes, activation=parameters['act_funct'])(encoder_ouputs)

    Encoder_model = Model(inputs=[encoder_inputs], outputs=[encoder_ouputs])
    
    return Encoder_model

######################################## Bottleneck  ########################################
def Bottleneck(parameters):
    
    layer_nodes = parameters['layer_nodes'] // (2 ** (parameters['n_layers'] - 1))

    bottleneck_inputs_X = Input(batch_shape=(None, layer_nodes), name='Input_Bot_X')

    bottleneck_inputs_Y = Input(batch_shape=(None, layer_nodes), name='Input_Bot_Y')

    bottleneck_outputs = Multiply()([bottleneck_inputs_X, bottleneck_inputs_Y])

    Bottleneck_model = Model(inputs=[bottleneck_inputs_X, bottleneck_inputs_Y],
                             outputs=[bottleneck_outputs], name='Bottleneck')

    return Bottleneck_model

######################################## DECODER X ########################################
def Decoder(parameters):
    
    layer_nodes = parameters['layer_nodes'] // (2 ** (parameters['n_layers'] - 1))

    decoder_inputs = Input(batch_shape=(None, layer_nodes), name='Input_Dec')

    decoder = decoder_inputs

    if parameters['seq_length'] // parameters['latent_sample_rate'] != 1:

        for ii in range(parameters['n_layers'] - 1):

            layer_nodes = layer_nodes * 2

            if layer_nodes < (parameters['seq_length'] // parameters['latent_sample_rate'] * (parameters['nb_filters']//2)):

                decoder = Dense(layer_nodes, activation=parameters['act_funct'])(decoder)

                if parameters['dropout_rate_hidden'] > 0:
                    decoder = Dropout(parameters['dropout_rate_hidden'])(decoder)

        decoder = Dense( parameters['seq_length'] // parameters['latent_sample_rate'] * (parameters['nb_filters']//2),
                        activation=parameters['act_funct'])(decoder)

    decoder = Reshape((parameters['seq_length'] // parameters['latent_sample_rate'], parameters['nb_filters']//2))(
        decoder)

    decoder = UpSampling1D(size=parameters['latent_sample_rate'])(decoder)

    decoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'], nb_stacks=parameters['nb_stacks'], 
                      dilations= parameters['dilations'], activation=parameters['tcn_act_funct'], padding=parameters['padding'], 
                      use_skip_connections=True, dropout_rate=parameters['dropout_rate_tcn'], return_sequences=True,
                      kernel_initializer=parameters['conv_kernel_init'])(decoder)
    
    decoder_outputs = Dense(parameters['ts_dimension'], activation=parameters['act_funct'])(decoder)

    Decoder_model = Model(inputs=[decoder_inputs], outputs=[decoder_outputs], name='Decoder')
    
    return Decoder_model

