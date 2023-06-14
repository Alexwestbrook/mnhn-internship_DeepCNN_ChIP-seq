#!/usr/bin/env python

import tensorflow as tf
import keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Conv1D, MaxPool1D, concatenate, Dropout, \
    Dense, Input, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from Modules import tf_utils
from Modules.tf_utils import mae_cor, correlate


def build_model(model_name,
                read_length=101,
                learn_rate=0.001,
                loss='binary_crossentropy',
                metrics=['accuracy'],
                method=0,
                T=1,
                start_reweighting=2000,
                confidence=False):
    model_dict = {
        'inception_dna_v1': inception_dna_v1,
        'inception_dna_v1_confidence': inception_dna_v1_confidence,
        'inception_dna_v2': inception_dna_v2,
        'inception_dna_paired_v1': inception_dna_paired_v1,
        'mnase_model': mnase_model,
        'Yann_original': Yann_original,
        'Yann_with_init': Yann_with_init,
        'Yann_with_init_and_padding': Yann_with_init_and_padding
    }
    model = model_dict[model_name](
        read_length=read_length,
        method=method,
        T=T,
        start_reweighting=start_reweighting
    )
    if confidence:
        model.compile(optimizer=Adam(learning_rate=learn_rate),
                      loss=tf_utils.ConfidenceLoss(),
                      metrics=metrics)
    else:
        model.compile(optimizer=Adam(learning_rate=learn_rate),
                      loss=loss,
                      metrics=metrics)
    return model


def simple_inception_module(x,
                            filters_3,
                            filters_6,
                            filters_9,
                            kernel_init,
                            name=None):
    conv_3 = Conv1D(
        filters=filters_3,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(x)
    conv_6 = Conv1D(
        filters=filters_6,
        kernel_size=6,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(x)
    conv_9 = Conv1D(
        filters=filters_9,
        kernel_size=9,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(x)
    output = concatenate([conv_3, conv_6, conv_9],
                         axis=2,
                         name=name)
    return output


def inception_module(x,
                     filters_1,
                     filters_3_reduce,
                     filters_3,
                     filters_5_reduce,
                     filters_5,
                     filters_pool_proj,
                     kernel_init,
                     name=None):

    conv_1 = Conv1D(
        filters=filters_1,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(x)

    conv_3 = Conv1D(
        filters=filters_3_reduce,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(x)
    conv_3 = Conv1D(
        filters=filters_3,
        kernel_size=3, padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(conv_3)

    conv_5 = Conv1D(
        filters=filters_5_reduce,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(x)
    conv_5 = Conv1D(
        filters=filters_5,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_init)(conv_5)

    pool_proj = MaxPool1D(pool_size=3,
                          strides=1,
                          padding='same')(x)
    pool_proj = Conv1D(filters_pool_proj,
                       kernel_size=1,
                       padding='same',
                       activation='relu',
                       kernel_initializer=kernel_init)(pool_proj)

    output = concatenate([conv_1, conv_3, conv_5, pool_proj],
                         axis=2,
                         name=name)

    return output


def inception_dna_v1(read_length=101,
                     method=0,
                     T=1,
                     start_reweighting=2000):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) read_length: the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    input_layer = Input(shape=(read_length, 4))

    x = simple_inception_module(input_layer,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_1')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_1')(x)
    x = Dropout(0.2)(x)
    x = simple_inception_module(x,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_2')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_2')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=128,
              activation='relu',
              name='dense_1')(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    if method == 0:
        model = tf.keras.Model(input_layer,
                               x,
                               name='inception_dna_v1')
    elif method == 1:
        model = tf_utils.ReweightingModel(input_layer,
                                          x,
                                          T=T,
                                          start_reweighting=start_reweighting,
                                          name='inception_dna_v1')
    return model


def inception_dna_v1_confidence(read_length=101,
                                method=0,
                                T=1,
                                start_reweighting=2000):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) read_length: the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    input_layer = Input(shape=(read_length, 4))

    x = simple_inception_module(input_layer,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_1')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_1')(x)
    x = Dropout(0.2)(x)
    x = simple_inception_module(x,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_2')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_2')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=128,
              activation='relu',
              name='dense_1')(x)
    out = Dense(units=1,
                activation='sigmoid',
                name='out')(x)
    conf = Dense(units=1,
                 activation='sigmoid',
                 name='conf')(x)
    model = tf.keras.Model(input_layer,
                           [out, conf],
                           name='inception_dna_v1_confidence')
    return model


def inception_dna_v2(read_length=101,
                     method=0,
                     T=1,
                     start_reweighting=2000):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) read_length the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    input_layer = Input(shape=(read_length, 4))
    x = Conv1D(filters=64,
               kernel_size=6,
               padding='same',
               kernel_initializer=kernel_init,
               activation='relu',
               name='first_conv')(input_layer)
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_0')(x)
    x = simple_inception_module(x,
                                filters_3=48,
                                filters_6=96,
                                filters_9=24,
                                kernel_init=kernel_init,
                                name='inception_1')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_1')(x)
    x = Dropout(0.2)(x)
    x = simple_inception_module(x,
                                filters_3=64,
                                filters_6=128,
                                filters_9=32,
                                kernel_init=kernel_init,
                                name='inception_2')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_2')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=512,
              activation='relu',
              name='dense_1')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    if method == 0:
        model = tf.keras.Model(input_layer,
                               x,
                               name='inception_dna_v2')
    elif method == 1:
        model = tf_utils.ReweightingModel(input_layer,
                                          x,
                                          T=T,
                                          start_reweighting=start_reweighting,
                                          name='inception_dna_v2')
    return model


def siamese_module(input_shape):
    """
    Builds a siamese module similar to inception_dna_v1

    Arguments
    ---------
    (optional) read_length: the sequence length of reads given as input

    Returns
    -------
    The model
    """
    kernel_init = VarianceScaling()

    # build the CNN model
    inputs = Input(input_shape)

    x = simple_inception_module(inputs,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_1')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_1')(x)
    x = Dropout(0.2)(x)
    x = simple_inception_module(x,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_2')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_2')(x)
    x = Dropout(0.2)(x)
    outputs = Flatten()(x)
    return Model(inputs, outputs)


def inception_dna_paired_v1(read_length=101,
                            method=0,
                            T=1,
                            start_reweighting=2000):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) read_length: the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    read1 = Input(shape=(read_length, 4))
    read2 = Input(shape=(read_length, 4))
    conv_module = siamese_module((read_length, 4))
    x1 = conv_module(read1)
    x2 = conv_module(read2)
    x = concatenate([x1, x2],
                    axis=1,
                    name='concatenate')
    x = Dense(units=128,
              activation='relu',
              name='dense_1')(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    if method == 0:
        model = tf.keras.Model([read1, read2],
                               x,
                               name='inception_dna_paired_v1')
    elif method == 1:
        model = tf_utils.ReweightingModel([read1, read2],
                                          x,
                                          T=T,
                                          start_reweighting=start_reweighting,
                                          name='inception_dna_paired_v1')
    return model


def mnase_model(winsize=2001, **kwargs):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) winsize: the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    input_layer = Input(shape=(winsize, 4))

    x = simple_inception_module(input_layer,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_1')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_1')(x)
    x = Dropout(0.2)(x)
    x = simple_inception_module(x,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_2')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_2')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=128,
              activation='relu',
              name='dense_1')(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    model = tf.keras.Model(input_layer,
                           x,
                           name='mnase_model')
    return model


def mnase_model_batchnorm(winsize=2001, **kwargs):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) winsize: the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    input_layer = Input(shape=(winsize, 4))

    x = simple_inception_module(input_layer,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_1')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_1')(x)
    x = BatchNormalization()(x)
    x = simple_inception_module(x,
                                filters_3=32,
                                filters_6=64,
                                filters_9=16,
                                kernel_init=kernel_init,
                                name='inception_2')
    x = MaxPool1D(pool_size=2,
                  padding='same',
                  strides=2,
                  name='max_pool_2')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(units=128,
              activation='relu',
              name='dense_1')(x)
    x = BatchNormalization()(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    model = tf.keras.Model(input_layer,
                           x,
                           name='mnase_model')
    return model


def mnase_Maxime(winsize=2001, **kwargs):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(winsize, 4)),
        MaxPool1D(2),
        BatchNormalization(),
        Conv1D(32, kernel_size=10, activation='relu'),
        MaxPool1D(2),
        BatchNormalization(),
        Conv1D(32, kernel_size=20, activation='relu'),
        MaxPool1D(2),
        BatchNormalization(),
        Flatten(),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    return model


def mnase_Maxime_decreasing(winsize=2001, **kwargs):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(winsize, 4)),
        MaxPool1D(2),
        BatchNormalization(),
        Conv1D(32, kernel_size=10, activation='relu'),
        MaxPool1D(2),
        BatchNormalization(),
        Conv1D(16, kernel_size=20, activation='relu'),
        MaxPool1D(2),
        BatchNormalization(),
        Flatten(),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    return model


def mnase_Maxime_increasing(winsize=2001, **kwargs):
    model = Sequential([
        Conv1D(16, kernel_size=3, activation='relu', input_shape=(winsize, 4)),
        MaxPool1D(2),
        BatchNormalization(),
        Conv1D(32, kernel_size=10, activation='relu'),
        MaxPool1D(2),
        BatchNormalization(),
        Conv1D(64, kernel_size=20, activation='relu'),
        MaxPool1D(2),
        BatchNormalization(),
        Flatten(),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    return model


def mnase_Etienne(winsize=2001, **kwargs):
    kernel_init = VarianceScaling()
    model = Sequential([
        Conv1D(64, kernel_size=3, padding="same", activation='relu',
               kernel_initializer=kernel_init, input_shape=(winsize, 4)),
        MaxPool1D(2, padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(16, kernel_size=8, padding="same", activation='relu',
               kernel_initializer=kernel_init),
        MaxPool1D(2, padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(8, kernel_size=80, padding="same", activation='relu',
               kernel_initializer=kernel_init),
        MaxPool1D(2, padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(1, activation='relu')
    ])
    return model


def bassenji_Etienne(winsize=32768, **kwargs):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) winsize: the sequence length of reads given as input

    Returns
    -------
    The uncompiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    input_layer = Input(shape=(winsize, 4))

    x = Conv1D(32, kernel_size=12, padding="same", activation='relu',
               kernel_initializer=kernel_init)(input_layer)
    x = MaxPool1D(pool_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, kernel_size=5, padding="same", activation='relu',
               kernel_initializer=kernel_init)(x)
    x = MaxPool1D(pool_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, kernel_size=5, padding="same", activation='relu',
               kernel_initializer=kernel_init)(x)
    x = MaxPool1D(pool_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(16, kernel_size=5, padding="same", activation='relu',
               kernel_initializer=kernel_init, dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x1 = Dropout(0.2)(x)

    x = x1
    x = Conv1D(16, kernel_size=5, padding="same", activation='relu',
               kernel_initializer=kernel_init, dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x2 = Dropout(0.2)(x)

    x = concatenate([x1, x2], axis=2)
    x = Conv1D(16, kernel_size=5, padding="same", activation='relu',
               kernel_initializer=kernel_init, dilation_rate=8)(x)
    x = BatchNormalization()(x)
    x3 = Dropout(0.2)(x)

    x = concatenate([x1, x2, x3], axis=2)
    x = Conv1D(16, kernel_size=5, padding="same", activation='relu',
               kernel_initializer=kernel_init, dilation_rate=16)(x)
    x = BatchNormalization()(x)
    x4 = Dropout(0.2)(x)

    x = concatenate([x1, x2, x3, x4], axis=2)
    x = Conv1D(1, kernel_size=1, padding="same", activation='relu',
               kernel_initializer=kernel_init)(x)
    model = tf.keras.Model(input_layer, x)
    return model


def Yann_original(read_length=101,
                  method=0,
                  T=1,
                  start_reweighting=2000):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) read_length the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    # build the CNN model
    input_layer = Input(shape=(read_length, 4))
    x = Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               name='conv_1_6')(input_layer)
    x = MaxPool1D(pool_size=2,
                  name='maxpool_1')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               name='conv_2_6')(x)
    x = MaxPool1D(pool_size=2,
                  name='maxpool_2')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=128,
              activation='relu',
              name='dense_1')(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    if method == 0:
        model = tf.keras.Model(input_layer,
                               x,
                               name='Yann_original')
    elif method == 1:
        model = tf_utils.ReweightingModel(input_layer,
                                          x,
                                          T=T,
                                          start_reweighting=start_reweighting,
                                          name='Yann_original')
    return model


def Yann_with_init(read_length=101,
                   method=0,
                   T=1,
                   start_reweighting=2000):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) read_length the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()
    # build the CNN model
    input_layer = Input(shape=(read_length, 4))
    x = Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               kernel_initializer=kernel_init,
               name='conv_1_6')(input_layer)
    x = MaxPool1D(pool_size=2,
                  name='maxpool_1')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               kernel_initializer=kernel_init,
               name='conv_2_6')(x)
    x = MaxPool1D(pool_size=2,
                  name='maxpool_2')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=128,
              activation='relu',
              kernel_initializer=kernel_init,
              name='dense_1')(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    if method == 0:
        model = tf.keras.Model(input_layer,
                               x,
                               name='Yann_with_init')
    elif method == 1:
        model = tf_utils.ReweightingModel(input_layer,
                                          x,
                                          T=T,
                                          start_reweighting=start_reweighting,
                                          name='Yann_with_init')
    return model


def Yann_with_init_and_padding(read_length=101,
                               method=0,
                               T=1,
                               start_reweighting=2000):
    """
    Builds a Deep neural network model

    Arguments
    ---------
    (optional) read_length the sequence length of reads given as input

    Returns
    -------
    The compiled model

    """
    kernel_init = VarianceScaling()
    # build the CNN model
    input_layer = Input(shape=(read_length, 4))
    x = Conv1D(filters=64,
               kernel_size=6,
               padding='same',
               activation='relu',
               kernel_initializer=kernel_init,
               name='conv_1_6')(input_layer)
    x = MaxPool1D(pool_size=2,
                  name='maxpool_1')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=64,
               kernel_size=6,
               padding='same',
               activation='relu',
               kernel_initializer=kernel_init,
               name='conv_2_6')(x)
    x = MaxPool1D(pool_size=2,
                  name='maxpool_2')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=128,
              activation='relu',
              kernel_initializer=kernel_init,
              name='dense_1')(x)
    x = Dense(units=1,
              activation='sigmoid',
              name='dense_out')(x)
    if method == 0:
        model = tf.keras.Model(input_layer,
                               x,
                               name='Yann_with_init_and_padding')
    elif method == 1:
        model = tf_utils.ReweightingModel(input_layer,
                                          x,
                                          T=T,
                                          start_reweighting=start_reweighting,
                                          name='Yann_with_init_and_padding')
    return model


if __name__ == "__main__":
    model = mnase_model(2001)
    print(model.summary())
