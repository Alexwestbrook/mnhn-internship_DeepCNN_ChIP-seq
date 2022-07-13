#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, concatenate, Dropout, \
    Dense, Input, Flatten
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam
import Modules.utils as utils


def build_model(model_name,
                read_length=101,
                learn_rate=0.001,
                loss='binary_crossentropy'):
    model_dict = {
        'inception_dna_v1': inception_dna_v1,
        'inception_dna_v2': inception_dna_v2,
        'Yann_original': Yann_original,
        'Yann_with_init': Yann_with_init,
        'Yann_with_init_and_padding': Yann_with_init_and_padding
    }
    model = model_dict[model_name](
        read_length=read_length,
        learn_rate=learn_rate
    )
    model.compile(optimizer=Adam(learning_rate=learn_rate),
                  loss=loss,
                  metrics=['accuracy'])
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


def inception_dna_v1(read_length=101, learn_rate=0.001):
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

    model = tf.keras.Model(input_layer,
                           x,
                           name='inception_dna_v1')
    return model


def inception_dna_v2(read_length=101, learn_rate=0.001):
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

    model = tf.keras.Model(input_layer, x, name='inception_dna_v1')
    return model


def Yann_original(read_length=101):
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
    model = Sequential([
        Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               input_shape=(read_length, 4),
               name='conv_1_6'),
        MaxPool1D(pool_size=2,
                  name='maxpool_1'),
        Dropout(0.2),
        Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               name='conv_2_6'),
        MaxPool1D(pool_size=2,
                  name='maxpool_2'),
        Dropout(0.2),
        Flatten(),
        Dense(units=128,
              activation='relu',
              name='dense_1'),
        Dense(units=1,
              activation='sigmoid',
              name='dense_out')
    ])

    return model


def Yann_with_init(read_length=101):
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
    model = Sequential([
        Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               kernel_initializer=kernel_init,
               input_shape=(read_length, 4),
               name='conv_1_6'),
        MaxPool1D(pool_size=2,
                  name='maxpool_1'),
        Dropout(0.2),
        Conv1D(filters=64,
               kernel_size=6,
               activation='relu',
               kernel_initializer=kernel_init,
               name='conv_2_6'),
        MaxPool1D(pool_size=2,
                  name='maxpool_2'),
        Dropout(0.2),
        Flatten(),
        Dense(units=128,
              activation='relu',
              kernel_initializer=kernel_init,
              name='dense_1'),
        Dense(units=1,
              activation='sigmoid',
              name='dense_out')
    ])

    return model


def Yann_with_init_and_padding(read_length=101):
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
    model = Sequential([
        Conv1D(filters=64,
               kernel_size=6,
               padding='same',
               activation='relu',
               kernel_initializer=kernel_init,
               input_shape=(read_length, 4),
               name='conv_1_6'),
        MaxPool1D(pool_size=2,
                  name='maxpool_1'),
        Dropout(0.2),
        Conv1D(filters=64,
               kernel_size=6,
               padding='same',
               activation='relu',
               kernel_initializer=kernel_init,
               name='conv_2_6'),
        MaxPool1D(pool_size=2,
                  name='maxpool_2'),
        Dropout(0.2),
        Flatten(),
        Dense(units=128,
              activation='relu',
              kernel_initializer=kernel_init,
              name='dense_1'),
        Dense(units=1,
              activation='sigmoid',
              name='dense_out')
    ])

    return model


if __name__ == "__main__":
    model = build_model('inception_dna_v1')
    print(model.summary())
