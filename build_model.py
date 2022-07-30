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


def build_multitask_set_transformer(inp_seq_len, inp_dim, **kwargs):
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
    first_activation = kwargs.get("first_activation", "linear")
    embedding_activation = kwargs.get("embedding_activation", "linear")
    lstm_activation = kwargs.get("lstm_activation", "relu")
    final_activation = kwargs.get("final_activation", "sigmoid")
    model_name1 = kwargs.get("model_name1", "item_classification")
    model_name2 = kwargs.get("model_name2", "compatibility")
    add_contrastive_loss = kwargs.get("add_contrastive_loss", False)
    num_negative_samples = kwargs.get("num_negative_samples", 8)
    margin = kwargs.get("margin", 0.2)

    #     n_outputs = train_y.shape[1]
    #     _, inp_seq_len, inp_features = train_X[0].shape
    seed(seed_value)
    set_seed(seed_value)

    inp_image = Input(
        shape=(
            inp_seq_len,
            inp_dim,
        )
    )
    image_embedding = Dense(d_model, activation=first_activation)
    embedded_image = image_embedding(inp_image)
    # embedded_image = tf.keras.layers.UnitNormalization(axis=-1)(embedded_image)
    # embedded_image = tf.keras.backend.l2_normalize(axis=-1)(embedded_image)

    encoded_image = STEncoder(
        n=num_layers,
        d=d_model,
        m=num_induce,
        h=num_heads,
        activation=embedding_activation,
    )(embedded_image)

    if include_text:
        inp_text = Input(
            shape=(
                inp_seq_len,
                inp_dim2,
            )
        )
        text_embedding = Dense(d_model, activation=first_activation)
        embedded_text = text_embedding(inp_text)
        # embedded_text = tf.keras.layers.UnitNormalization(axis=-1)(embedded_text)

        encoded_text = STEncoder(
            n=num_layers,
            d=d_model,
            m=num_induce,
            h=num_heads,
            activation=embedding_activation,
        )(embedded_text)
        combined_image_text = concatenate([encoded_image, encoded_text], axis=-1)
        inps = [inp_image, inp_text]
    else:
        inps = inp_image
        combined_image_text = encoded_image

    classification_layers = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(153, activation="softmax"),
        ]
    )
    class_probs = tf.keras.layers.TimeDistributed(
        classification_layers, name=model_name1
    )(combined_image_text)

    lstm_out = LSTM(lstm_dim, activation=lstm_activation, return_sequences=False)(
        combined_image_text
    )
    # lstm_out = Dropout(0.2)(lstm_out)
    if num_classes == 2:
        output = Dense(1, activation=final_activation, name=model_name2)(lstm_out)
    else:
        output = Dense(num_classes, activation="softmax", name=model_name2)(lstm_out)

    outputs = [output, class_probs]

    if add_contrastive_loss:
        embedded_image = embedded_image / tf.expand_dims(
            tf.norm(embedded_image, axis=-1), axis=-1
        )
        embedded_text = embedded_text / tf.expand_dims(
            tf.norm(embedded_text, axis=-1), axis=-1
        )

        neg_image = Input(shape=(inp_seq_len, num_negative_samples, inp_dim))
        neg_text = Input(shape=(inp_seq_len, num_negative_samples, inp_dim2))
        inps.extend([neg_image, neg_text])

        # process negative images
        neg_image_embedded = image_embedding(neg_image)
        neg_image_embedded = neg_image_embedded / tf.expand_dims(
            tf.norm(neg_image_embedded, axis=-1), axis=-1
        )
        # neg_image_embedded = tf.keras.layers.UnitNormalization(axis=-1)(
        #     neg_image_embedded
        # )

        # process negative texts
        neg_text_embedded = text_embedding(neg_text)
        # neg_text_embedded = tf.keras.layers.UnitNormalization(axis=-1)(
        #     neg_text_embedded
        # )
        neg_text_embedded = neg_text_embedded / tf.expand_dims(
            tf.norm(neg_text_embedded, axis=-1), axis=-1
        )

        positive_prod = tf.keras.layers.Multiply()([embedded_image, embedded_text])
        positive_prod = tf.reduce_sum(positive_prod, axis=-1)

        # image vs negative texts
        negative_image_text = tf.keras.layers.Multiply()(
            [embedded_image, neg_text_embedded]
        )
        negative_image_text = tf.reduce_sum(negative_image_text, axis=-1)

        # text vs negative images
        negative_text_image = tf.keras.layers.Multiply()(
            [embedded_text, neg_image_embedded]
        )
        negative_text_image = tf.reduce_sum(negative_text_image, axis=-1)

        contrastive_loss_1 = (
            margin - tf.expand_dims(positive_prod, -1) + negative_image_text
        )
        condition_1 = tf.less(contrastive_loss_1, 0.0)
        loss_1 = tf.where(
            condition_1, tf.zeros_like(contrastive_loss_1), contrastive_loss_1
        )

        contrastive_loss_2 = (
            margin - tf.expand_dims(positive_prod, -1) + negative_text_image
        )
        condition_2 = tf.less(contrastive_loss_2, 0.0)
        loss_2 = tf.where(
            condition_2, tf.zeros_like(contrastive_loss_2), contrastive_loss_2
        )
        contrastive_loss = loss_1 + loss_2
        contrastive_loss = tf.reduce_sum(contrastive_loss, axis=[-1, -2])
        contrastive_loss = tf.expand_dims(contrastive_loss, axis=-1, name="contrastive")
        outputs.append(contrastive_loss)

    model = Model(inputs=inps, outputs=outputs, name="set_transformer")
    return model


class MultiTaskSetTransformer(tf.keras.Model):
    """
    Set Transformer model with multiple tasks,
    (1) compatibility label prediction, binary
    (2) item category prediction, multi-class
    (3) visual semantic embedding
    """

    def __init__(self, **kwargs):
        super(MultiTaskSetTransformer, self).__init__()
        self.num_layers = kwargs.get("num_layers", 3)
        self.d_model = kwargs.get("d_model", 256)
        self.num_heads = kwargs.get("num_heads", 1)
        self.num_induce = kwargs.get("num_induce", 6)
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

        self.image_encoder = STEncoder(
            n=self.num_layers,
            d=self.d_model,
            m=self.num_induce,
            h=self.num_heads,
            activation=self.embedding_activation,
        )
        self.text_encoder = STEncoder(
            n=self.num_layers,
            d=self.d_model,
            m=self.num_induce,
            h=self.num_heads,
            activation=self.embedding_activation,
        )
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

        # outfit images and texts
        embedded_image = self.image_embedding(inp_image)
        embedded_text = self.text_embedding(inp_text)

        # normalize the embeddings
        embedded_image = tf.keras.utils.normalize(embedded_image, axis=-1)
        embedded_text = tf.keras.utils.normalize(embedded_text, axis=-1)

        encoded_image = self.image_encoder(embedded_image)
        encoded_text = self.text_encoder(embedded_text)
        combined_image_text = concatenate([encoded_image, encoded_text], axis=-1)
        lstm_out = self.rnn(combined_image_text)
        compatibility_output = self.output_layer(lstm_out)
        class_probs = self.time_distributed(combined_image_text)

        # process negative images
        neg_image_embedded = self.image_embedding(neg_image)
        neg_image_embedded = tf.keras.utils.normalize(neg_image_embedded, axis=-1)

        # process negative texts
        neg_text_embedded = self.text_embedding(neg_text)
        neg_text_embedded = tf.keras.utils.normalize(neg_text_embedded)

        # positive image-text
        # positive_prod = tf.keras.layers.Dot(axes=2)([embedded_image, embedded_text])
        # positive_prod = tf.matmul(
        #     embedded_image, tf.transpose(embedded_text, perm=[0, 2, 1])
        # )
        positive_prod = tf.keras.layers.Multiply()([embedded_image, embedded_text])
        positive_prod = tf.reduce_sum(positive_prod, axis=-1)

        # image vs negative texts
        negative_image_text = tf.keras.layers.Multiply()(
            [embedded_image, neg_text_embedded]
        )
        negative_image_text = tf.reduce_sum(negative_image_text, axis=-1)

        # text vs negative images
        negative_text_image = tf.keras.layers.Multiply()(
            [embedded_text, neg_image_embedded]
        )
        negative_text_image = tf.reduce_sum(negative_text_image, axis=-1)

        contrastive_loss_1 = (
            self.margin - tf.expand_dims(positive_prod, -1) + negative_image_text
        )
        condition_1 = tf.less(contrastive_loss_1, 0.0)
        loss_1 = tf.where(
            condition_1, tf.zeros_like(contrastive_loss_1), contrastive_loss_1
        )

        contrastive_loss_2 = (
            self.margin - tf.expand_dims(positive_prod, -1) + negative_text_image
        )
        condition_2 = tf.less(contrastive_loss_2, 0.0)
        loss_2 = tf.where(
            condition_2, tf.zeros_like(contrastive_loss_2), contrastive_loss_2
        )
        contrastive_loss = loss_1 + loss_2

        # print(negative_text_image.shape, negative_image_text.shape, positive_prod.shape)
        # print(loss_1.shape, loss_2.shape)

        return compatibility_output, class_probs, contrastive_loss
