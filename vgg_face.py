# VGG-Face Compatibility for newer tensorflow keras versions

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import get_file

# VGG-Face
WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'

def VGGFace(include_top=True, model='vgg16', weights='vggface', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=2622):
    """
    Parameters:
        include_top: if we should include the 3 fully connected layers.
        model: which model to use.
        input_tensor: image input for the model.
        input_shape: only specify if include top is False.
        pooling: for feature extraction when include top is False.
        classes: optional number of classes to classify images into, only if include top is True.
    """
    
    if model != 'vgg16':
        raise ValueError("Only 'vgg16' is supported in this simplified implementation.")
    
    # The input tensor
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc7')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax', name='fc8')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        else:
            # Flattened features
            x = Flatten(name='flatten')(x)

    model = Model(img_input, x, name='vggface_vgg16')

    # Weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_vgg16.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def preprocess_input(x, data_format=None, version=1):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    # Make a copy of x to avoid modifying the original
    x = np.array(x, dtype='float32')

    if version == 1:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]

    # Zero-center by mean pixel
    if version == 1:
        if data_format == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] -= 93.5940
                x[1, :, :] -= 104.7624
                x[2, :, :] -= 129.1863
            else:
                x[:, 0, :, :] -= 93.5940
                x[:, 1, :, :] -= 104.7624
                x[:, 2, :, :] -= 129.1863
        else:
            if x.ndim == 3:
                x[..., 0] -= 93.5940
                x[..., 1] -= 104.7624
                x[..., 2] -= 129.1863
            else:
                x[..., 0] -= 93.5940
                x[..., 1] -= 104.7624
                x[..., 2] -= 129.1863
    elif version == 2:
        if data_format == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] -= 91.4953
                x[1, :, :] -= 103.8827
                x[2, :, :] -= 131.0912
            else:
                x[:, 0, :, :] -= 91.4953
                x[:, 1, :, :] -= 103.8827
                x[:, 2, :, :] -= 131.0912
        else:
            if x.ndim == 3:
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912
            else:
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912

    return x

def get_vgg_face_model(include_top=True, input_shape=(224, 224, 3), classes=2622, weights='vggface'):
    return VGGFace(include_top=include_top, 
                  input_shape=input_shape, 
                  classes=classes,
                  weights=weights)