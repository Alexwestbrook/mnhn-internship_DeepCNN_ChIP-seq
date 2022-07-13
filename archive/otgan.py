import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense
from tensorflow.keras.layers import Flatten, Reshape, UpSampling2D, InputLayer
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import CSVLogger, Callback
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
import os
import time


# GLU activation from "https://github.com/porcelainruler/Gated-Linear-Unit-
# Activation-Implementation-TF/blob/master/GLU.py"
class GLU(Model):
    def __init__(self, filters, kernel_size=3, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim
        self.filter = filters // 2
        self.kernel_size = kernel_size
        self.sig = tf.sigmoid

    # Function to Slice Tensor Equally along Last Dim
    def equal_slice(self, x):
        ndim = len(x.shape)
        slice_idx = x.shape[self.dim] // 2
        if ndim == 3:
            linear_output = x[:, :, :slice_idx]
            gated_output = x[:, :, slice_idx:]
        elif ndim == 4:
            linear_output = x[:, :, :, :slice_idx]
            gated_output = x[:, :, :, slice_idx:]
        elif ndim == 5:
            linear_output = x[:, :, :, :, :slice_idx]
            gated_output = x[:, :, :, :, slice_idx:]
        else:
            raise ValueError(
                "This GLU Activation only support for 1D, 2D, 3D Conv, but "
                "the Input's Dim is={}".format(ndim))

        # Return the 2 slices
        return linear_output, gated_output

    def call(self, inputs, **kwargs):
        assert inputs.shape[self.dim] % 2 == 0

        # Slicing the Tensor in 2 Halfs
        lin_out_slice, gated_out_slice = self.equal_slice(inputs)

        # Applying Sigmoid Activation to 2nd Slice
        siggat_out_slice = self.sig(gated_out_slice)

        # Returning Element-wise Multiply of two Slices
        return lin_out_slice * siggat_out_slice


class Generator(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(128*49, activation=GLU)
        self.reshape = Reshape((128, 7, 7, 1))
        self.upsample = UpSampling2D((2, 2))
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                            padding='same', activation=GLU)
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                            padding='same', activation=GLU)
        self.conv3 = Conv2D(filters=1, kernel_size=(3, 3), strides=1,
                            padding='same', activation='tanh')
        self.dropout = Dropout(0.2)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Critic(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                            padding='same', activation=tf.nn.crelu)
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=2,
                            padding='same', activation=tf.nn.crelu)
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=2,
                            padding='same', activation=tf.nn.crelu)
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = x / tf.norm(x, axis=1, keepdims=True)
        return x


def training_otgan(generator, critic, epochs, ngen):
    generator = Generator()
    critic = Critic()
    for epoch in epochs:
        if epoch % (ngen + 1) == 0:
            train_critic()
        else:
            train_generator()


def train_critic(generated_samples, data_samples):
    return 0


def train_generator(generated_samples, data_samples):
    return 0


def sinkhorn_divergence(x, y, distance, critic, flat_size=1024):
    if critic is not None:
        x, y = critic(x), critic(y)

    if distance == 'euclidean':
        cost = x - y
    return 0


if __name__ == "__main__":
    critic = Critic()
    critic.build(input_shape=(None, 28, 28, 1))
    critic.summary()
