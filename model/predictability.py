import numpy as np
import matplotlib.pyplot as plt

try:
    from model_structure import Encoder, Bottleneck, Decoder
except ImportError:
    from .model_structure import Encoder, Bottleneck, Decoder
    
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def predictability_full(parameters, filenameX, filenameXP, 
                        full_dataX, full_dataX_, full_dataX_sff, 
                        full_dataY, full_dataY_, full_dataY_sff, save_plot='',
                        bottleneck_XYX_filename,bottleneck_XYXP_filename):

    '''
    Purpose:

        The predictability_full function is designed to train and validate an autoencoder model on an entire dataset 
        without explicitly distinguishing between training and validation datasets. This allows for end-to-end training 
        or full-cycle evaluation scenarios.

    Technical Description of Inputs:

        'parameters' (dict): A dictionary containing key model parameters and hyperparameters which configure the 
                            model's architecture, behavior during training, and performance evaluation criteria. 
                            Entries may include seq_length, ts_dimension, batch_size, epochs, shuffle, verbose, 
                            loss_funct, encoder_type, and other model-specific settings such as dropout rates and padding options.
        'filenameX' (str): The file path where the model checkpoint for the standard configuration will be saved 
                        during training. This supports model persistence by allowing the model's state to be saved 
                        at the best-performing iteration based on the loss metric.
        'filenameXP' (str): The file path for saving the model checkpoint under a modified encoder configuration. 
                            This path is used for the experimental or alternate version of the autoencoder.

    Full Dataset Handling:

        'full_dataX', 'full_dataY' (array-like): Full dataset inputs used as primary and secondary features for 
                                                training the autoencoder. 
        'full_dataX_', 'full_dataY_sff' (array-like): Target outputs and shuffled or modified versions of the 
                                                    secondary features.

    Technical Description of Outputs:

        - auto_encoder_model_XYX is the model trained with standard parameters using the full dataset.
        - auto_encoder_model_XYXP is the model trained under a modified parameter set as determined by the 
        'encoder_type', also using the full dataset.
    '''

    ##################################### XY to X #####################################

    # Use Output Encoder Y
    parameters.update({'pad_encoder': False})

    encoder_XX = Encoder(parameters)
    encoder_YX = Encoder(parameters)
    bottleneck_XYX = Bottleneck(parameters)
    decoder_XYX = Decoder(parameters)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    bottleneck_output_XYX = bottleneck_XYX([encoder_XX([encoder_input_X]), encoder_YX([encoder_input_Y])])
    decoder_output = decoder_XYX(bottleneck_output_XYX)
                            
    # Compile Autoencoder
    auto_encoder_model_XYX = Model(inputs=[encoder_input_X, encoder_input_Y],
                                   outputs=decoder_output,
                                   name='AutoEncoder_XYX')
    auto_encoder_model_XYX.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameX, monitor='loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYX = auto_encoder_model_XYX.fit([full_dataX, full_dataY], full_dataX_, 
                                            batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                            shuffle=parameters['shuffle'], use_multiprocessing=True,
                                            callbacks=[checkpoint, early_stopping, reduce_lr],
                                            verbose=parameters['verbose'])

    bottleneck_model_XYX = Model(inputs=[encoder_input_X, encoder_input_Y], outputs=bottleneck_output_XYX)
    bottleneck_model_XYX.save(bottleneck_XYX_filename)
                           
    ##################################### XY to X updated #####################################

    if parameters['encoder_type'] == 0:
        padded_encoder = True
        shuffle_encoder = False
        random_encoder = False
    elif parameters['encoder_type'] == 1:
        padded_encoder = False
        shuffle_encoder = True
        random_encoder = False
    elif parameters['encoder_type'] == 2:
        padded_encoder = False
        shuffle_encoder = False
        random_encoder = True

    if padded_encoder:
        # Make Output Encoder Y to 0
        parameters.update({'pad_encoder': True})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        # Make Y Weights 0
        old_weigths = encoder_YXP.get_weights()
        new_weigths = old_weigths.copy()

        for ii in range(len(old_weigths[0])):
            new_weigths[ii] = np.zeros_like(old_weigths[ii])

        encoder_YXP.set_weights(new_weigths)

        # Freeze Encoder Layers
        for layer in encoder_YXP.layers[:]:
            layer.trainable = False

    elif shuffle_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        full_dataY = full_dataY_sff.copy()

    elif random_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        full_dataY = np.random.random(full_dataY.shape)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    bottleneck_output_XYXP = bottleneck_XYXP([encoder_XXP([encoder_input_X]), encoder_YXP([encoder_input_Y])])
    decoder_output = decoder_XYXP(bottleneck_output_XYXP)    
                            
    # Compile Autoencoder
    auto_encoder_model_XYXP = Model(inputs=[encoder_input_X, encoder_input_Y],
                                   outputs=decoder_output,
                                   name='AutoEncoder_XYXP')
    auto_encoder_model_XYXP.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    # print("\n> Starting Training XYX ...")
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameXP, monitor='loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYXP = auto_encoder_model_XYXP.fit([full_dataX, full_dataY], full_dataX_,
                                              batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                              shuffle=parameters['shuffle'], use_multiprocessing=True,
                                              callbacks=[checkpoint, early_stopping, reduce_lr],
                                              verbose=parameters['verbose'])

    bottleneck_model_XYXP = Model(inputs=[encoder_input_X, encoder_input_Y], outputs=bottleneck_output_XYXP)
    bottleneck_model_XYXP.save(bottleneck_XYXP_filename)

    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(1, 2, figure=fig)

    ax = fig.add_subplot(gs[:, 0])
    ax.semilogy(np.arange(len(historyXYX.history['loss'])),
                np.sqrt(historyXYX.history['loss']), 'k-', label='Training Loss')
    ax.set_title('XY --> X shifted')

    ax = fig.add_subplot(gs[:, 1])
    ax.semilogy(np.arange(len(historyXYXP.history['loss'])),
                np.sqrt(historyXYXP.history['loss']), 'k-', label='Training Loss')
    ax.set_title('XY padded --> X shifted')

    plt.legend()

    if save_plot != '':
        plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                    facecolor="w", edgecolor='w', orientation='landscape')

    if parameters['paint_fig']:
        plt.show()
    else:
        plt.close(fig)

    return auto_encoder_model_XYX, auto_encoder_model_XYXP, bottleneck_model_XYX, bottleneck_model_XYXP


def predictability_val_split(parameters, filenameX, filenameXP, trainX, valX, trainX_, valX_,
                   trainY, valY, trainY_sff, valY_sff, save_plot='',bottleneck_XYX_filename,bottleneck_XYXP_filename):

    '''
    Purpose:

        The predictability_split function trains and validates an autoencoder model specifically designed 
        for time series data, utilizing distinct training and validation sets. This function is tailored 
        to experiment with various encoder configurations and assess their impact on model performance.

    Technical Description of Inputs:

        'parameters' (dict): A dictionary containing key model parameters and hyperparameters which configure the 
                            model's architecture, behavior during training, and performance evaluation criteria. 
                            Typical entries include seq_length, ts_dimension, batch_size, epochs, shuffle, verbose, 
                            loss_funct, encoder_type, and other model-specific settings like dropout rates and padding options.
        'filenameX' (str): The file path where the model checkpoint for the standard configuration will be saved 
                        during training. 
        'filenameXP' (str): The file path where the model checkpoint for the modified encoder configuration will be saved 
                            during training. 

    Training and Validation Data:

        'trainX', 'valX' (array-like): Input features for training and validation. 
        'trainX_', 'valX_' (array-like): Target outputs corresponding to trainX and valX for training and validation, 
                                        respectively. These are used as ground truth for calculating loss and gradients 
                                        during backpropagation.
        'trainY', 'valY' (array-like): Secondary input features for training and validation.
        'trainY_sff', 'valY_sff' (array-like): Shuffled or modified versions of trainY and valY.

    Technical Description of Outputs:

        - auto_encoder_model_XYX is the model trained with standard parameters.
        - auto_encoder_model_XYXP is the model trained under a modified parameter set as determined by the 'encoder_type'.
    '''

    ##################################### XY to X #####################################

    # Use Output Encoder Y
    parameters.update({'pad_encoder': False})

    encoder_XX = Encoder(parameters)
    encoder_YX = Encoder(parameters)
    bottleneck_XYX = Bottleneck(parameters)
    decoder_XYX = Decoder(parameters)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    bottleneck_output_XYX = bottleneck_XYX([encoder_XX([encoder_input_X]), encoder_YX([encoder_input_Y])])
    decoder_output = decoder_XYX(bottleneck_output_XYX)
                       
    # Compile Autoencoder
    auto_encoder_model_XYX = Model(inputs=[encoder_input_X, encoder_input_Y],
                                   outputs=decoder_output,
                                   name='AutoEncoder_XYX')
                       
    auto_encoder_model_XYX.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameX, monitor='val_loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYX = auto_encoder_model_XYX.fit([trainX, trainY], trainX_, validation_data=([valX, valY], valX_),
                                            batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                            shuffle=parameters['shuffle'], use_multiprocessing=True,
                                            callbacks=[checkpoint, early_stopping, reduce_lr],
                                            verbose=parameters['verbose'])
                       
    bottleneck_model_XYX = Model(inputs=[encoder_input_X, encoder_input_Y], outputs=bottleneck_output_XYX)
    bottleneck_model_XYX.save(bottleneck_XYX_filename)                 
                       
    ##################################### XY to X updated #####################################

    if parameters['encoder_type'] == 0:
        padded_encoder = True
        shuffle_encoder = False
        random_encoder = False
    elif parameters['encoder_type'] == 1:
        padded_encoder = False
        shuffle_encoder = True
        random_encoder = False
    elif parameters['encoder_type'] == 2:
        padded_encoder = False
        shuffle_encoder = False
        random_encoder = True

    if padded_encoder:
        # Make Output Encoder Y to 0
        parameters.update({'pad_encoder': True})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        # Make Y Weights 0
        old_weigths = encoder_YXP.get_weights()
        new_weigths = old_weigths.copy()

        for ii in range(len(old_weigths[0])):
            new_weigths[ii] = np.zeros_like(old_weigths[ii])

        encoder_YXP.set_weights(new_weigths)

        # Freeze Encoder Layers
        for layer in encoder_YXP.layers[:]:
            layer.trainable = False

    elif shuffle_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        trainY = trainY_sff.copy()
        valY = valY_sff.copy()

    elif random_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        trainY = np.random.random(trainY.shape)
        valY = np.random.random(valY.shape)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    bottleneck_output_XYXP = bottleneck_XYXP([encoder_XX([encoder_input_X]), encoder_YX([encoder_input_Y])])
    decoder_output = decoder_XYXP(bottleneck_output_XYXP)
                       
    # Compile Autoencoder
    auto_encoder_model_XYXP = Model(inputs=[encoder_input_X, encoder_input_Y],
                                   outputs=decoder_output,
                                   name='AutoEncoder_XYXP')
                       
    auto_encoder_model_XYXP.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    # print("\n> Starting Training XYX ...")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameXP, monitor='val_loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYXP = auto_encoder_model_XYXP.fit([trainX, trainY], trainX_, validation_data=([valX, valY], valX_),
                                              batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                              shuffle=parameters['shuffle'], use_multiprocessing=True,
                                              callbacks=[checkpoint, early_stopping, reduce_lr],
                                              verbose=parameters['verbose'])

    bottleneck_model_XYXP = Model(inputs=[encoder_input_X, encoder_input_Y], outputs=bottleneck_output_XYXP)
    bottleneck_model_XYXP.save(bottleneck_XYXP_filename)
                       
    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(1, 2, figure=fig)

    ax = fig.add_subplot(gs[:, 0])
    ax.semilogy(np.arange(len(historyXYX.history['loss'])),
                np.sqrt(historyXYX.history['loss']), 'k-', label='Training Loss')
    ax.semilogy(np.arange(len(historyXYX.history['val_loss'])),
                np.sqrt(historyXYX.history['val_loss']), 'k:', label='Validation Loss')
    ax.set_title('XY --> X shifted')

    ax = fig.add_subplot(gs[:, 1])
    ax.semilogy(np.arange(len(historyXYXP.history['loss'])),
                np.sqrt(historyXYXP.history['loss']), 'k-', label='Training Loss')
    ax.semilogy(np.arange(len(historyXYXP.history['val_loss'])),
                np.sqrt(historyXYXP.history['val_loss']), 'k:', label='Validation Loss')
    ax.set_title('XY padded --> X shifted')

    plt.legend()

    if save_plot != '':
        plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                    facecolor="w", edgecolor='w', orientation='landscape')

    if parameters['paint_fig']:
        plt.show()
    else:
        plt.close(fig)

    return auto_encoder_model_XYX, auto_encoder_model_XYXP, bottleneck_model_XYX, bottleneck_model_XYXP
