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


def build_multitask_rnn(inp_seq_len, inp_features, **kwargs):
    """
    There are two outputs - (1) to predict the covariates and (2) to predict
    radiation using the prediction of the first model as future covariates
    There is only one model though.
    """
    num_classes = kwargs.get("num_classes", 2)
    rnn = kwargs.get("rnn", "lstm")
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    model_name = kwargs.get("model_name", "rnn")
    merge_activation = kwargs.get("merge_activation", "relu")
    final_activation = kwargs.get("final_activation", None)
    include_text = kwargs.get("include_text", False)
    text_feature_dim = kwargs.get("text_feature_dim", 768)
    model_name1 = kwargs.get("model_name1", "item_classification")
    model_name2 = kwargs.get("model_name2", "compatibility")

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

    classification_layers = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(153, activation="softmax"),
        ]
    )
    class_probs = tf.keras.layers.TimeDistributed(
        classification_layers, name=model_name1
    )(merge)
    # should be batch X sequence_length X #classes

    # convert to the target_sequence_length
    merge = Permute((2, 1), input_shape=(inp_seq_len, components * d_model))(merge)
    merge = Dense(1, activation=merge_activation)(merge)
    merge = tf.squeeze(merge, axis=-1)

    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation, name=model_name2)(merge)
    else:
        dense1 = Dense(num_classes, activation="softmax", name=model_name2)(merge)

    model = Model(inputs=inputs, outputs=[dense1, class_probs], name=model_name)
    return model


# create mask for padding, 0 --> 1 (mask)
def create_padding_mask(seq, add_extra=False):

    # seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.cast(tf.math.greater(seq, 0), tf.float32)

    if add_extra:
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    else:
        return seq


def normalize(x):
    x_ = x / tf.expand_dims(tf.norm(x, axis=-1), axis=-1)
    return x_


class MultiTaskRNN(tf.keras.Model):
    """
    RNN based model with multiple tasks,
    (1) compatibility label prediction, binary
    (2) item category prediction, multi-class
    (3) visual semantic embedding
    """

    def __init__(self, **kwargs):
        super(MultiTaskRNN, self).__init__()
        self.num_layers = kwargs.get("num_layers", 3)
        self.rnn = kwargs.get("rnn", "lstm")
        self.num_layers = kwargs.get("num_layers", 2)
        self.d_model = kwargs.get("d_model", 256)
        self.include_text = kwargs.get("include_text", True)
        self.first_activation = kwargs.get("first_activation", "tanh")
        self.embedding_activation = kwargs.get("embedding_activation", "linear")
        self.lstm_dim = kwargs.get("lstm_dim", 32)
        self.lstm_activation = kwargs.get("lstm_activation", "relu")
        self.final_activation = kwargs.get("final_activation", "sigmoid")
        self.num_classes = kwargs.get("num_classes", 2)
        self.num_categories = kwargs.get("num_categories", 153)
        self.model_name1 = kwargs.get("model_name1", "item_classification")
        self.model_name2 = kwargs.get("model_name2", "compatibility")
        self.margin = kwargs.get("margin", 0.2)
        # tf.keras.backend.set_floatx("float32")

        self.image_embedding = Dense(self.d_model, activation=self.first_activation)
        self.text_embedding = Dense(self.d_model, activation=self.first_activation)

        return_seq = True
        if self.rnn == "lstm":
            unit_encoder = LSTM(
                units=self.d_model, activation="tanh", return_sequences=return_seq
            )
        elif self.rnn == "gru":
            unit_encoder = GRU(units=self.d_model, return_sequences=return_seq)
        elif self.rnn == "bilstm":
            base_rnn = LSTM(
                units=self.d_model, activation="tanh", return_sequences=return_seq
            )
            unit_encoder = Bidirectional(base_rnn, merge_mode="concat")
        elif self.rnn == "bigru":
            base_rnn = GRU(units=self.d_model, return_sequences=return_seq)
            unit_encoder = Bidirectional(base_rnn, merge_mode="concat")

        self.image_encoder = tf.keras.models.Sequential()
        for _ in range(self.num_layers):
            self.image_encoder.add(unit_encoder)

        self.text_encoder = tf.keras.models.Sequential()
        for _ in range(self.num_layers):
            self.text_encoder.add(unit_encoder)

        classification_layers = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(self.num_categories, activation="softmax"),
            ]
        )
        self.time_distributed = tf.keras.layers.TimeDistributed(
            classification_layers, name=self.model_name1
        )
        self.rnn = LSTM(
            self.lstm_dim, activation=self.lstm_activation, return_sequences=False
        )
        if self.num_classes == 2:
            self.output_layer = Dense(
                1, activation=self.final_activation, name=self.model_name2
            )
        else:
            self.output_layer = Dense(
                self.num_classes, activation="softmax", name=self.model_name2
            )

    def call(self, x):
        inp_image, inp_text, neg_image, neg_text = x
        # mask = create_padding_mask(tf.reduce_sum(inp_image, axis=-1))

        # outfit images and texts
        embedded_image = self.image_embedding(inp_image)
        embedded_text = self.text_embedding(inp_text)

        encoded_image = self.image_encoder(embedded_image)
        encoded_text = self.text_encoder(embedded_text)
        combined_image_text = concatenate([encoded_image, encoded_text], axis=-1)
        lstm_out = self.rnn(combined_image_text)
        compatibility_output = self.output_layer(lstm_out)
        class_probs = self.time_distributed(combined_image_text)

        # for contrastive loss
        # process negative image and text samples
        neg_image_embedded = self.image_embedding(neg_image)
        neg_text_embedded = self.text_embedding(neg_text)

        # normalize all the embeddings
        # embedded_image = tf.keras.utils.normalize(embedded_image, axis=-1)
        # embedded_text = tf.keras.utils.normalize(embedded_text, axis=-1)
        # neg_image_embedded = tf.keras.utils.normalize(neg_image_embedded, axis=-1)
        # neg_text_embedded = tf.keras.utils.normalize(neg_text_embedded)
        # embedded_image = normalize(embedded_image)
        # embedded_text = normalize(embedded_text)
        # neg_image_embedded = normalize(neg_image_embedded)
        # neg_text_embedded = normalize(neg_text_embedded)

        # positive image-text
        positive_prod = tf.keras.layers.Multiply()([embedded_image, embedded_text])
        positive_prod = tf.reduce_sum(positive_prod, axis=-1)

        # image vs negative texts, element-wise multiplication
        negative_image_text = tf.keras.layers.Multiply()(
            [embedded_image, neg_text_embedded]
        )  # (?, s, n, 256)
        negative_image_text = tf.reduce_sum(negative_image_text, axis=-1)
        negative_image_text = tf.reduce_sum(negative_image_text, axis=-1)

        # text vs negative images
        negative_text_image = tf.keras.layers.Multiply()(
            [embedded_text, neg_image_embedded]
        )
        negative_text_image = tf.reduce_sum(negative_text_image, axis=-1)
        negative_text_image = tf.reduce_sum(negative_text_image, axis=-1)

        #  positive_prod.shape = (32, 8), negative_text_image.shape = (32, 8)
        contrastive_loss_1 = self.margin - positive_prod + negative_image_text
        condition_1 = tf.less(contrastive_loss_1, 0.0)
        loss_1 = tf.where(
            condition_1, tf.zeros_like(contrastive_loss_1), contrastive_loss_1
        )

        contrastive_loss_2 = self.margin - positive_prod + negative_text_image
        condition_2 = tf.less(contrastive_loss_2, 0.0)
        loss_2 = tf.where(
            condition_2, tf.zeros_like(contrastive_loss_2), contrastive_loss_2
        )
        contrastive_loss = loss_1 + loss_2

        # print(negative_text_image.shape, negative_image_text.shape, positive_prod.shape)
        # print(loss_1.shape, loss_2.shape)

        return compatibility_output, class_probs, contrastive_loss
