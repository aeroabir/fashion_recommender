from tensorflow.keras.layers import (
    Conv1D,
    Input,
    concatenate,
    Dropout,
    BatchNormalization,
    Reshape,
)
from tensorflow.keras.layers import (
    Flatten,
    GRU,
    LSTM,
    TimeDistributed,
    Dense,
    Permute,
    Bidirectional,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from numpy.random import seed
from tensorflow.random import set_seed
import tensorflow as tf


def cnn_layers(n_timesteps, n_features, kernel_size=4):
    in1 = Input(
        shape=(
            n_timesteps,
            n_features,
        )
    )
    conv1 = Conv1D(
        2, kernel_size, strides=1, activation="relu", kernel_initializer="he_normal"
    )(in1)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(
        4, kernel_size, strides=1, activation="relu", kernel_initializer="he_normal"
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(
        8, kernel_size, strides=1, activation="relu", kernel_initializer="he_normal"
    )(conv2)
    # conv3 = BatchNormalization()(conv3)
    flat1 = Flatten()(conv3)

    return in1, flat1


def rnn_layer(layer, n_timesteps, n_features, num_layers, d_model, return_seq=True):
    in1 = Input(
        shape=(
            n_timesteps,
            n_features,
        )
    )

    # Add extra CNN layer to reduce the number of steps
    if n_timesteps > 10000:
        c1 = Conv1D(
            filters=n_features,
            kernel_size=15,
            strides=15,
            activation="relu",
            kernel_initializer="he_normal",
        )(in1)
        c1 = BatchNormalization()(c1)
    else:
        c1 = in1

    # create a mask since not all elements in the sequence are valid
    # seq = tf.reduce_sum(c1, axis=-1)
    # mask = tf.math.equal(seq, 0)  # should be (batch, timestep)
    mask = None

    if layer == "lstm":
        encoder = LSTM(units=d_model, return_sequences=return_seq)
    elif layer == "gru":
        encoder = GRU(units=d_model, return_sequences=return_seq)
    elif layer == "bilstm":
        base_rnn = LSTM(units=d_model, return_sequences=return_seq)
        encoder = Bidirectional(base_rnn, merge_mode="concat")
    elif layer == "bigru":
        base_rnn = GRU(units=d_model, return_sequences=return_seq)
        encoder = Bidirectional(base_rnn, merge_mode="concat")

    encoded = encoder(c1, mask=mask)
    if num_layers > 1:
        encoders = [encoder]
    for ii in range(1, num_layers):
        encoder = LSTM(units=d_model, return_sequences=return_seq)
        encoded = encoder(encoded, mask=mask)
        encoders.append(encoder)
    return in1, encoded


def build_multilevel_rnn_unequal(inp_seq_len, inp_features, **kwargs):
    """
    input sequence length does not have to be same as the target
    sequence length; required for short term forecasting where
    window length is greater than the forecast horizon

    """
    num_classes = kwargs.get("num_classes", 2)
    rnn = kwargs.get("rnn", "lstm")
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    model_name = kwargs.get("model_name", "rnn")
    final_activation = kwargs.get("final_activation", None)
    include_text = kwargs.get("include_text", False)
    text_feature_dim = kwargs.get("text_feature_dim", 768)

    seed(seed_value)
    set_seed(seed_value)

    inputs, flat = [], []
    # for image
    t_in1, t_flat1 = rnn_layer(
        rnn, inp_seq_len, inp_features, num_layers, d_model, rate
    )
    inputs.append(t_in1)
    flat.append(t_flat1)

    if include_text:
        t_in2, t_flat2 = rnn_layer(
            rnn, inp_seq_len, text_feature_dim, num_layers, d_model, rate
        )
        inputs.append(t_in2)
        flat.append(t_flat2)

    components = len(flat)
    if components > 1:
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]
    # merge = BatchNormalization()(merge)

    # convert to the target_sequence_length
    merge = Permute((2, 1), input_shape=(inp_seq_len, components * d_model))(merge)
    merge = Dense(1, activation="relu")(merge)
    merge = tf.squeeze(merge, axis=-1)

    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation)(merge)
    else:
        dense1 = Dense(num_classes, activation="softmax")(merge)

    model = Model(inputs=inputs, outputs=dense1, name=model_name)
    return model


def build_multiscale_rnn(inp_seq_lens, inp_features, tgt_seq_len, **kwargs):
    """
    input sequence length does not have to be same as the target
    sequence length; required for short term forecasting where
    window length is greater than the forecast horizon

    """
    tgt_features = kwargs.get("tgt_features", 1)
    rnn = kwargs.get("rnn", "lstm-lstm")
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    d_model2 = kwargs.get("d_model2", 16)
    d_model3 = kwargs.get("d_model3", 16)
    # num_heads = kwargs.get("num_heads", 8)
    # dff = kwargs.get("dff", 128)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    components = kwargs.get("components", 2)
    future_covariates = kwargs.get("future_covariates", False)
    dim_future_cov = kwargs.get("dim_future_cov", 1)
    quantiles = kwargs.get("quantiles", None)

    seed(seed_value)
    set_seed(seed_value)

    inputs, flat = [], []
    for ii in range(components):
        t_in, t_flat = rnn_layer(
            rnn.split("-")[0], inp_seq_lens[ii], inp_features, num_layers, d_model, rate
        )
        inputs.append(t_in)
        t_flat = Permute((2, 1), input_shape=(inp_seq_lens[ii], d_model))(t_flat)
        t_flat = Dense(tgt_seq_len, activation="relu")(t_flat)
        t_flat = Permute((2, 1), input_shape=(d_model, tgt_seq_len))(t_flat)
        flat.append(t_flat)

    if future_covariates:
        decoder_inputs = Input(
            shape=(
                tgt_seq_len,
                dim_future_cov,
            )
        )
        inputs.append(decoder_inputs)

    if components > 1:
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]
    # merge = BatchNormalization()(merge)

    if rnn.split("-")[1] == "lstm":
        encoder2 = LSTM(units=d_model2, activation="relu", return_sequences=True)
    elif rnn.split("-")[1] == "gru":
        encoder2 = GRU(units=d_model2, activation="relu", return_sequences=True)
    elif rnn.split("-")[1] == "bilstm":
        base_rnn = LSTM(units=d_model2, activation="relu", return_sequences=True)
        encoder2 = Bidirectional(base_rnn, merge_mode="concat")
    elif rnn.split("-")[1] == "bigru":
        base_rnn = GRU(units=d_model2, activation="relu", return_sequences=True)
        encoder2 = Bidirectional(base_rnn, merge_mode="concat")

    if future_covariates:
        # Use the context as initial state for the decoder
        # merge = Flatten()(merge)
        # state_h = Dense(d_model2, activation="relu")(merge)
        # state_c = Dense(d_model2, activation="relu")(merge)
        # encoder_states = [state_h, state_c]
        # lstm_out = encoder2(decoder_inputs, initial_state=encoder_states)

        # Directly add the future covariates to the input to the decoder
        merge = Permute((2, 1), input_shape=(inp_seq_len, components * d_model))(merge)
        merge = Dense(tgt_seq_len, activation="relu")(merge)
        merge = Permute((2, 1), input_shape=(components * d_model, tgt_seq_len))(merge)
        merge = concatenate([merge, decoder_inputs], axis=-1)
        lstm_out = encoder2(merge)

    else:
        # convert to the target_sequence_length
        lstm_out = encoder2(merge)

    dense1 = TimeDistributed(Dense(d_model3, activation="relu"))(lstm_out)
    # dropout1 = Dropout(0.2)(dense1)

    if quantiles:
        outputs = []
        for ii in range(quantiles):
            output = TimeDistributed(
                Dense(tgt_features), name="quantile_" + str(ii + 1)
            )(dense1)
            outputs.append(output)

    else:
        outputs = TimeDistributed(Dense(tgt_features))(dense1)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_multitask_rnn(inp_seq_len, inp_features, tgt_seq_len, **kwargs):
    """
    There are two models - (1) to predict the covariates and (2) to predict
    radiation using the prediction of the first model as future covariates
    """
    tgt_features1 = kwargs.get("tgt_features1", 1)  # targets for the first model
    tgt_features2 = kwargs.get("tgt_features2", 1)  # targets for the second model

    rnn = kwargs.get("rnn", "lstm-lstm")
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    d_model2 = kwargs.get("d_model2", 16)
    d_model3 = kwargs.get("d_model3", 16)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    components = kwargs.get("components", 2)
    future_covariates = kwargs.get("future_covariates", False)
    dim_future_cov = kwargs.get("dim_future_cov", 1)
    quantiles = kwargs.get("quantiles", None)
    model_name1 = kwargs.get("model_name1", "future_output")
    model_name2 = kwargs.get("model_name2", "radiation_output")

    seed(seed_value)
    set_seed(seed_value)

    inputs = []
    for _ in range(components):
        in1 = Input(
            shape=(
                inp_seq_len,
                inp_features,
            )
        )
        inputs.append(in1)

    if future_covariates:
        decoder_inputs1 = Input(
            shape=(
                tgt_seq_len,
                dim_future_cov,
            )
        )
        inputs1 = inputs.copy()
        inputs1.append(decoder_inputs1)

    # First model - without covariates
    model1 = build_multilevel_rnn_unequal(
        inp_seq_len=inp_seq_len,
        inp_features=inp_features,
        tgt_seq_len=tgt_seq_len,
        tgt_features=tgt_features1,
        components=components,
        num_layers=num_layers,
        d_model=d_model,
        d_model2=d_model2,
        d_model3=d_model3,
        rnn=rnn,
        future_covariates=future_covariates,
        dim_future_cov=dim_future_cov,
        model_name=model_name1,
    )

    o1 = model1(inputs1)
    # model = Model(inputs=inputs1, outputs=o1)
    # return model

    # Second model - output of the first model as future covariates
    model2 = build_multilevel_rnn_unequal(
        inp_seq_len=inp_seq_len,
        inp_features=inp_features,
        tgt_seq_len=tgt_seq_len,
        tgt_features=tgt_features2,
        components=components,
        num_layers=num_layers,
        d_model=d_model,
        d_model2=d_model2,
        d_model3=d_model3,
        rnn=rnn,
        future_covariates=True,
        dim_future_cov=dim_future_cov + tgt_features1,
        model_name=model_name2,
    )
    decoder_inputs2 = concatenate([o1, decoder_inputs1], axis=-1)
    inputs.append(decoder_inputs2)
    o2 = model2(inputs)

    model = Model(inputs=inputs1, outputs=[o1, o2])
    return model
