from transformer_models_ts import Encoder, FFT
from tensorflow.keras.layers import (
    Conv1D,
    Input,
    Layer,
    concatenate,
    Dropout,
    BatchNormalization,
    Reshape,
)
from tensorflow.keras.layers import (
    Flatten,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Permute,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from numpy.random import seed
from tensorflow.random import set_seed

# from rnn import rnn_layer
import sys

sys.path.insert(0, "/recsys_data/RecSys/fashion/automl/efficientnetv2")
import effnetv2_model


class GLU(keras.layers.Layer):
    def __init__(self, units=32):
        super(GLU, self).__init__()
        self.layer1 = Dense(units=units, activation="sigmoid")
        self.layer2 = Dense(units=units, activation="linear")

    def call(self, inputs):
        y1 = self.layer1(inputs)
        y2 = self.layer2(inputs)
        return tf.keras.layers.Multiply()([y1, y2])


class GRN(keras.layers.Layer):
    def __init__(self, units=None):
        super(GRN, self).__init__()
        self.glu = GLU(units)
        self.layer1 = Dense(units=units, activation="linear")
        self.layer2 = Dense(units=units, activation="linear", use_bias=False)
        self.layer3 = Dense(units=units, activation="linear")
        self.layernorm = tf.keras.layers.LayerNormalization(axis=1)

    def call(self, a, c=None):
        x1 = self.layer1(a)
        if c:
            x1 = x1 + self.layer2(c)
        eta2 = keras.activations.elu(x1, alpha=1.0)
        eta1 = self.layer3(eta2)
        return self.layernorm(a + self.glu(eta1))


def transformer_layer(
    n_timesteps, image_dim, num_layers, d_model, num_heads, dff, pe_input, rate
):
    in1 = Input(
        shape=(
            n_timesteps,
            image_dim[0],
            image_dim[1],
            image_dim[2],
        )
    )
    encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
    encoded = encoder(in1, True, None)
    # flat1 = Flatten()(encoded)
    return in1, encoded


class image_embedding_layer(tf.keras.layers.Layer):
    def __init__(self, dropout, embedding_dim):
        super(image_embedding_layer, self).__init__()
        self.dropout = dropout
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.layer = tf.keras.Sequential(
            [
                # tf.keras.layers.Cropping2D(38),
                effnetv2_model.get_model("efficientnetv2-b0", include_top=False),
                tf.keras.layers.Dropout(rate=self.dropout),
                tf.keras.layers.Dense(self.embedding_dim, activation="tanh"),
            ]
        )

    def call(self, input_tensor):
        return self.layer(input_tensor)


def build_multilevel_transformer(inp_seq_len, inp_shape, **kwargs):
    """
    the input sequence length is same as the output sequence length
    """

    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    num_heads = kwargs.get("num_heads", 8)
    dff = kwargs.get("dff", 128)
    rate = kwargs.get("rate", 0.1)
    components = kwargs.get("components", 2)
    include_fft = kwargs.get("include_fft", False)
    seed_value = kwargs.get("seed", 100)

    #     n_outputs = train_y.shape[1]
    #     _, inp_seq_len, inp_features = train_X[0].shape
    seed(seed_value)
    set_seed(seed_value)

    inputs, flat = [], []
    for ii in range(components):
        t_in, t_flat = transformer_layer(
            inp_seq_len,
            inp_shape,
            num_layers,
            d_model,
            num_heads,
            dff,
            inp_seq_len,
            rate,
        )
        inputs.append(t_in)
        flat.append(t_flat)

    if include_fft:
        fft_layer = FFT(inp_seq_len, 1, 1)
        fft_out = fft_layer(inputs[0])
        flat.append(fft_out)

    if components > 1:
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]

    lstm_out = LSTM(16, activation="relu", return_sequences=True)(merge)
    dense1 = TimeDistributed(Dense(16, activation="relu"))(lstm_out)
    # dropout1 = Dropout(0.2)(dense1)
    output = TimeDistributed(Dense(1))(dense1)

    model = Model(inputs=inputs, outputs=output)
    return model
