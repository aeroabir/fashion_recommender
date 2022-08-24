import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    AveragePooling2D,
    Reshape,
    Flatten,
    Dense,
)


def vgg16_model(pooling='avg'):

    model = tf.keras.models.Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64,
              kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (None, 7, 7, 512)

    if pooling == 'avg':
        model.add(AveragePooling2D(pool_size=(7, 7)))
        model.add(Reshape((512,)))
    elif pooling == 'max':
        model.add(MaxPool2D(pool_size=(7, 7)))
        model.add(Reshape((512,)))
    # model.add(Flatten())
    # model.add(Dense(units=4096, activation="relu"))
    # model.add(Dense(units=dim, activation="relu"))

    return model


def vgg16_model_small(pooling='avg'):

    model = tf.keras.models.Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64,
              kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    # (None, 28, 28, 256)
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    if pooling == 'avg':
        model.add(AveragePooling2D(pool_size=(28, 28)))
        model.add(Reshape((256,)))
    elif pooling == 'max':
        model.add(MaxPool2D(pool_size=(28, 28)))
        model.add(Reshape((256,)))

    return model


def vgg16_model_smaller(pooling='avg'):

    model = tf.keras.models.Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64,
              kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    # (None, 56, 56, 128)
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    if pooling == 'avg':
        model.add(AveragePooling2D(pool_size=(56, 56)))
        model.add(Reshape((128,)))
    elif pooling == 'max':
        model.add(MaxPool2D(pool_size=(56, 56)))
        model.add(Reshape((128,)))

    return model
