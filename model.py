import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Lambda
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

def yolo_darknet19(input_shape = (416, 416, 3), output_depth = 5):
    X_input = Input(input_shape)
    ALPHA = 0.1

    # Layer 1
    X = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(X_input)
    X = BatchNormalization(name='bn_1')(X)
    X = LeakyReLU(alpha=ALPHA)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # Layer 2
    X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(X)
    X = BatchNormalization(name='bn_2')(X)
    X = LeakyReLU(alpha=ALPHA)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # Layer 3
    X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(X)
    X = BatchNormalization(name='bn_3')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 4
    X = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(X)
    X = BatchNormalization(name='bn_4')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 5
    X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(X)
    X = BatchNormalization(name='bn_5')(X)
    X = LeakyReLU(alpha=ALPHA)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # Layer 6
    X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(X)
    X = BatchNormalization(name='bn_6')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 7
    X = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(X)
    X = BatchNormalization(name='bn_7')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 8
    X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(X)
    X = BatchNormalization(name='bn_8')(X)
    X = LeakyReLU(alpha=ALPHA)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # Layer 9
    X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(X)
    X = BatchNormalization(name='bn_9')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 10
    X = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(X)
    X = BatchNormalization(name='bn_10')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 11
    X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(X)
    X = BatchNormalization(name='bn_11')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 12
    X = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(X)
    X = BatchNormalization(name='bn_12')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 13
    X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(X)
    X = BatchNormalization(name='bn_13')(X)
    X = LeakyReLU(alpha=ALPHA)(X)
    skip_connection = X
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # Layer 14
    X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(X)
    X = BatchNormalization(name='bn_14')(X)
    X = LeakyReLU(alpha=ALPHA)(X)
    
    # Layer 15
    X = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(X)
    X = BatchNormalization(name='bn_15')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 16
    X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(X)
    X = BatchNormalization(name='bn_16')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 17
    X = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(X)
    X = BatchNormalization(name='bn_17')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 18
    X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(X)
    X = BatchNormalization(name='bn_18')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 19
    X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(X)
    X = BatchNormalization(name='bn_19')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 20
    X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(X)
    X = BatchNormalization(name='bn_20')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 21
    skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='bn_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=ALPHA)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    X = concatenate([skip_connection, X])

    # Layer 22
    X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(X)
    X = BatchNormalization(name='bn_22')(X)
    X = LeakyReLU(alpha=ALPHA)(X)

    # Layer 23
    X = yolo_layer(X, output_depth)

    model = Model(inputs=X_input, outputs=X, name='Yolo DarkNet19')

    return model

def yolo_layer(X, depth):
    X = Conv2D(depth, (1, 1), strides=(1, 1), padding='same', name='yolo_tensor_conv')(X)
    return X
