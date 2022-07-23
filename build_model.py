from transformer_models_ts import Encoder, FFT, STEncoder
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

# from set_transformer import STEncoder, STDecoder

# from rnn import rnn_layer
import sys

# sys.path.insert(0, "/recsys_data/RecSys/fashion/automl/efficientnetv2")
# import effnetv2_model


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
    n_timesteps,
    image_dim,
    num_layers,
    d_model,
    num_heads,
    dff,
    pe_input,
    rate,
    embedding_activation,
):
    in1 = Input(
        shape=(
            n_timesteps,
            image_dim,
        )
    )
    encoder = Encoder(
        num_layers, d_model, num_heads, dff, pe_input, rate, embedding_activation
    )
    encoded = encoder(in1, True)
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


def build_multilevel_transformer(inp_seq_len, inp_dim, **kwargs):
    """
    the input sequence length is same as the output sequence length
    """

    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    num_heads = kwargs.get("num_heads", 8)
    dff = kwargs.get("dff", 128)
    rate = kwargs.get("rate", 0.1)
    include_fft = kwargs.get("include_fft", False)
    include_text = kwargs.get("include_text", False)
    inp_dim2 = kwargs.get("inp_dim2", None)
    seed_value = kwargs.get("seed", 100)
    num_classes = kwargs.get("num_classes", 2)
    lstm_dim = kwargs.get("lstm_dim", 32)
    embedding_activation = kwargs.get("embedding_activation", "linear")
    lstm_activation = kwargs.get("lstm_activation", "relu")
    final_activation = kwargs.get("final_activation", "sigmoid")

    #     n_outputs = train_y.shape[1]
    #     _, inp_seq_len, inp_features = train_X[0].shape
    seed(seed_value)
    set_seed(seed_value)

    inputs, flat = [], []
    t_in, t_flat = transformer_layer(
        inp_seq_len,
        inp_dim,
        num_layers,
        d_model,
        num_heads,
        dff,
        inp_seq_len,
        rate,
        embedding_activation,
    )
    inputs.append(t_in)
    flat.append(t_flat)

    if include_text:
        t_in2, t_flat2 = transformer_layer(
            inp_seq_len,
            inp_dim2,
            num_layers,
            d_model,
            num_heads,
            dff,
            inp_seq_len,
            rate,
            embedding_activation,
        )
        inputs.append(t_in2)
        flat.append(t_flat2)
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]

    lstm_out = LSTM(lstm_dim, activation=lstm_activation, return_sequences=False)(merge)
    # lstm_out = Dropout(0.2)(lstm_out)
    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation)(lstm_out)
    else:
        dense1 = Dense(num_classes, activation="softmax")(lstm_out)
    output = dense1

    model = Model(inputs=inputs, outputs=output)
    return model


def build_set_transformer(inp_seq_len, inp_dim, **kwargs):
    """
    the input sequence length is same as the output sequence length
    """

    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 120)
    num_heads = kwargs.get("num_heads", 6)
    num_induce = kwargs.get("num_induce", 6)
    dff = kwargs.get("dff", 128)
    rate = kwargs.get("rate", 0.1)
    include_text = kwargs.get("include_text", False)
    inp_dim2 = kwargs.get("inp_dim2", None)
    seed_value = kwargs.get("seed", 100)
    num_classes = kwargs.get("num_classes", 2)
    lstm_dim = kwargs.get("lstm_dim", 32)
    embedding_activation = kwargs.get("embedding_activation", "linear")
    lstm_activation = kwargs.get("lstm_activation", "relu")
    final_activation = kwargs.get("final_activation", "sigmoid")

    #     n_outputs = train_y.shape[1]
    #     _, inp_seq_len, inp_features = train_X[0].shape
    seed(seed_value)
    set_seed(seed_value)

    inp1 = Input(
        shape=(
            inp_seq_len,
            inp_dim,
        )
    )
    x = Dense(d_model, activation="linear")(inp1)
    y = STEncoder(
        n=num_layers,
        d=d_model,
        m=num_induce,
        h=num_heads,
        activation=embedding_activation,
    )(x)
    # output = STDecoder(out_dim=1, d=d_model, h=num_heads, k=1)(y)

    if include_text:
        inp2 = Input(
            shape=(
                inp_seq_len,
                inp_dim2,
            )
        )
        x2 = Dense(d_model, activation="linear")(inp2)
        y2 = STEncoder(
            n=num_layers,
            d=d_model,
            m=num_induce,
            h=num_heads,
            activation=embedding_activation,
        )(x2)
        y = concatenate([y, y2], axis=-1)
        inps = [inp1, inp2]
    else:
        inps = inp1

    lstm_out = LSTM(lstm_dim, activation=lstm_activation, return_sequences=False)(y)
    # lstm_out = Dropout(0.2)(lstm_out)
    if num_classes == 2:
        output = Dense(1, activation=final_activation)(lstm_out)
    else:
        output = Dense(num_classes, activation="softmax")(lstm_out)

    model = Model(inputs=inps, outputs=output, name="set_transformer")
    return model
