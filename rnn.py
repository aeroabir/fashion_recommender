from email.mime import image
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
from tensorflow.keras.applications import resnet, resnet50, inception_v3
from transformer_models_ts import Encoder
from vgg16_models import vgg16_model, vgg16_model_small, vgg16_model_smaller

# resnet152 = resnet.ResNet152(
#     include_top=False,
#     weights='imagenet',
#     pooling='avg',
# )


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # print(scaled_attention_logits.shape, mask.shape)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.seq_len = seq_len

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (-1, self.seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        # batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (-1, self.seq_len, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


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


def create_input(n_timesteps, n_features):
    if type(n_features) is tuple:
        in1 = Input(
            shape=(
                n_timesteps,
                n_features[0],
                n_features[1],
                n_features[2],
            )
        )
        embedded = tf.keras.layers.TimeDistributed(vgg16)(in1)
    else:
        in1 = Input(
            shape=(
                n_timesteps,
                n_features,
            )
        )
    return in1


def get_rnn_model(layer, d_model, return_seq=True):
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
    return encoder


def rnn_layer(layer, n_timesteps, n_features, num_layers, d_model, image_encoder=None, return_seq=True):
    if type(n_features) is tuple:
        in1 = Input(
            shape=(
                n_timesteps,
                n_features[0],
                n_features[1],
                n_features[2],
            )
        )
        assert image_encoder is not None, "image encoder not found !!"
        embedded = tf.keras.layers.TimeDistributed(image_encoder)(in1)
    else:
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
        if type(n_features) is tuple:
            c1 = embedded
        else:
            c1 = in1

    # create a mask since not all elements in the sequence are valid
    # "An individual True entry indicates that the corresponding timestep
    # should be utilized, while a False entry indicates that the
    # corresponding timestep should be ignored."
    seq = tf.reduce_sum(c1, axis=-1)
    mask = tf.math.not_equal(seq, 0.0)  # should be (batch, timestep)
    # mask = tf.keras.layers.Masking(mask_value=0.,
    #                               input_shape=(n_timesteps, 2048))
    # mask = None
    encoder = get_rnn(layer, d_model, return_seq=return_seq)
    encoded = encoder(c1, mask=mask)
    if num_layers > 1:
        encoders = [encoder]

    for _ in range(1, num_layers):
        encoder = get_rnn(layer, d_model, return_seq=return_seq)
        encoded = encoder(encoded, mask=mask)
        encoders.append(encoder)
    return in1, encoded


def get_rnn(layer, d_model, return_seq=True):
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
    return encoder


def build_hybrid_model(inp_seq_len, inp_features, **kwargs):
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
    embedding_activation = kwargs.get("embedding_activation", "linear")
    final_activation = kwargs.get("final_activation", None)
    include_text = kwargs.get("include_text", False)
    text_feature_dim = kwargs.get("text_feature_dim", 768)
    include_item_categories = kwargs.get("include_item_categories", False)
    num_categories = kwargs.get("num_categories", 153)
    mask_zero = kwargs.get("mask_zero", True)
    num_heads = kwargs.get("num_heads", 8)
    dff = kwargs.get("dff", 128)
    model_name = kwargs.get("model_name", "hybrid")

    seed(seed_value)
    set_seed(seed_value)

    inputs, flat = [], []
    # for image
    t_in1 = create_input(inp_seq_len, inp_features)
    o1 = Dense(d_model, activation=embedding_activation)(t_in1)
    inputs.append(t_in1)
    flat.append(o1)

    if include_text:
        t_in2 = create_input(inp_seq_len, text_feature_dim)
        o2 = Dense(d_model, activation=embedding_activation)(t_in2)
        inputs.append(t_in2)
        flat.append(o2)

    if include_item_categories:
        in3 = Input(shape=(inp_seq_len,))  # (?, 8)
        category_embedder = tf.keras.layers.Embedding(
            input_dim=num_categories,
            output_dim=d_model,
            embeddings_initializer="uniform",
            mask_zero=mask_zero,
        )
        category_embedded = category_embedder(in3)
        inputs.append(in3)
        flat.append(category_embedded)

    components = len(flat)
    if components > 1:
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]

    transformer_encoder = Encoder(
        num_layers, d_model, num_heads, dff, inp_seq_len, rate, embedding_activation
    )
    merge = transformer_encoder(merge, True)
    merge = BatchNormalization()(merge)

    rnn_encoder = get_rnn_model(rnn, d_model, return_seq=True)
    mask = tf.math.not_equal(in3, 0)
    merge = rnn_encoder(merge, mask=mask)

    # convert to the target_sequence_length
    merge = Permute((2, 1), input_shape=(
        inp_seq_len, d_model))(merge)
    merge = Dense(1, activation="relu")(merge)
    merge = tf.squeeze(merge, axis=-1)
    merge = tf.keras.layers.Dropout(rate)(merge)

    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation)(merge)
    else:
        dense1 = Dense(num_classes, activation="softmax")(merge)

    model = Model(inputs=inputs, outputs=dense1, name=model_name)
    return model


def build_multilevel_rnn_unequal(inp_seq_len, **kwargs):
    """
    input sequence length does not have to be same as the target
    sequence length; required for short term forecasting where
    window length is greater than the forecast horizon

    """
    num_classes = kwargs.get("num_classes", 2)
    rnn = kwargs.get("rnn", "lstm")
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    num_heads = kwargs.get("num_heads", 2)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    model_name = kwargs.get("model_name", "rnn")
    final_activation = kwargs.get("final_activation", None)
    include_text = kwargs.get("include_text", False)
    image_embedding_dim = kwargs.get("image_embedding_dim", 1280)
    text_feature_dim = kwargs.get("text_feature_dim", 768)
    include_item_categories = kwargs.get("include_item_categories", False)
    image_data_type = kwargs.get("image_data_type", "embedding")
    include_multihead_attention = kwargs.get(
        "include_multihead_attention", False)
    original_image_dim = kwargs.get("original_image_dim", (224, 224, 3))
    num_categories = kwargs.get("num_categories", 153)
    mask_zero = kwargs.get("mask_zero", True)
    image_encoder = kwargs.get("image_encoder", "resnet50")

    seed(seed_value)
    set_seed(seed_value)

    if image_data_type == "original":
        inp_features = original_image_dim
    elif image_data_type == "embedding":
        inp_features = image_embedding_dim
    elif image_data_type == "both":
        inp_features = image_embedding_dim

    if image_data_type in ["original", "both"]:
        if image_encoder == "resnet50":
            i_encoder = resnet50.ResNet50(
                include_top=False,
                weights='imagenet',
                pooling='avg',
            )
        elif image_encoder == "inception":
            i_encoder = inception_v3.InceptionV3(include_top=True,
                                                 weights='imagenet',
                                                 pooling='avg',)
        elif image_encoder == "vgg16":
            i_encoder = tf.keras.applications.vgg16.VGG16(
                include_top=False, weights='imagenet', pooling='avg',)
            # vgg16 = vgg16_model('avg')
            # vgg16 = vgg16_model_small('avg')
            # vgg16 = vgg16_model_smaller('avg')
    else:
        i_encoder = None

    # embedding or image
    inputs, flat = [], []
    # for image
    t_in1, t_flat1 = rnn_layer(
        rnn, inp_seq_len, inp_features, num_layers, d_model, i_encoder,
    )
    inputs.append(t_in1)
    flat.append(t_flat1)

    if image_data_type == "both":
        t_ino, t_flato = rnn_layer(
            rnn, inp_seq_len, original_image_dim, num_layers, d_model, i_encoder,
        )
        inputs.append(t_ino)
        flat.append(t_flato)

    if include_text:
        t_in2, t_flat2 = rnn_layer(
            rnn, inp_seq_len, text_feature_dim, num_layers, d_model, i_encoder,
        )
        inputs.append(t_in2)
        flat.append(t_flat2)

    if include_item_categories:
        in3 = Input(shape=(inp_seq_len,))  # (?, 8)
        # category_embedder = tf.keras.models.Sequential()
        first_embedder = tf.keras.layers.Embedding(
            input_dim=num_categories,
            output_dim=d_model,
            embeddings_initializer="uniform",
            mask_zero=mask_zero,
        )
        # category_embedder.add(first_embedder)
        rnn_encoder = get_rnn(rnn, d_model, return_seq=True)
        # category_embedder.add(rnn_encoder)
        category_embedded = first_embedder(in3)
        category_embedded = rnn_encoder(category_embedded)
        inputs.append(in3)
        flat.append(category_embedded)

    if include_multihead_attention:
        mha = MultiHeadAttention(inp_seq_len, d_model, num_heads)
        # inputs are (V, K, Q)
        # v1, _ = mha(flat[0], flat[1], flat[2], mask=None)
        # v2, _ = mha(flat[1], flat[0], flat[2], mask=None)
        # v3, _ = mha(flat[2], flat[0], flat[1], mask=None)
        # flat = [v1, v2, v3]

    components = len(flat)
    if components > 1:
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]

    if include_multihead_attention:
        merge, _ = mha(merge, merge, merge, mask=None)
    merge = BatchNormalization()(merge)

    # convert to the target_sequence_length
    merge = Permute((2, 1), input_shape=(
        inp_seq_len, components * d_model))(merge)
    merge = Dense(1, activation="relu")(merge)
    merge = tf.squeeze(merge, axis=-1)
    merge = tf.keras.layers.Dropout(rate)(merge)

    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation)(merge)
    else:
        dense1 = Dense(num_classes, activation="softmax")(merge)

    model = Model(inputs=inputs, outputs=dense1, name=model_name)
    return model


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(
        tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def build_multitask_rnn2(inp_seq_len, inp_features, **kwargs):
    """
    There are three outputs:
    (1) to predict outfit compatibility, 
    (2) to predict the item category (for each item in the outfit)
    (3) visual-semantic embedding loss
    There is only one model though.
    """
    num_classes = kwargs.get("num_classes", 2)
    rnn = kwargs.get("rnn", "lstm")
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    model_name = kwargs.get("model_name", "rnn")
    num_negative_samples = kwargs.get("num_negative_samples", 8)
    first_activation = kwargs.get("first_activation", "tanh")
    merge_activation = kwargs.get("merge_activation", "relu")
    final_activation = kwargs.get("final_activation", None)
    text_feature_dim = kwargs.get("text_feature_dim", 768)
    use_contrastive_loss = kwargs.get("use_contrastive_loss", True)
    margin = kwargs.get("margin", 1)
    model_name1 = kwargs.get("model_name1", "item_classification")
    model_name2 = kwargs.get("model_name2", "compatibility")

    seed(seed_value)
    set_seed(seed_value)

    image_embedding = Dense(d_model, activation=first_activation)
    text_embedding = Dense(d_model, activation=first_activation)
    unit_encoder = get_rnn(rnn, d_model)
    image_encoder = tf.keras.models.Sequential()
    for _ in range(num_layers):
        image_encoder.add(unit_encoder)

    text_encoder = tf.keras.models.Sequential()
    for _ in range(num_layers):
        text_encoder.add(unit_encoder)

    classification_layers = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(153, activation="softmax"),
        ]
    )

    inp_images = Input(shape=(inp_seq_len, inp_features,))
    inp_texts = Input(shape=(inp_seq_len, text_feature_dim,))

    embedded_image = image_embedding(inp_images)
    embedded_text = text_embedding(inp_texts)

    embedded_image = tf.keras.layers.BatchNormalization()(embedded_image)
    embedded_text = tf.keras.layers.BatchNormalization()(embedded_text)

    encoded_image = image_encoder(embedded_image)
    encoded_text = text_encoder(embedded_text)
    combined_image_text = concatenate(
        [encoded_image, encoded_text], axis=-1)
    combined_image_text = tf.keras.layers.BatchNormalization()(combined_image_text)

    class_probs = tf.keras.layers.TimeDistributed(
        classification_layers, name=model_name1
    )(combined_image_text)

    # convert to the target_sequence_length
    merge = Permute((2, 1), input_shape=(
        inp_seq_len, d_model))(combined_image_text)
    merge = Dense(1, activation=merge_activation)(merge)
    merge = tf.squeeze(merge, axis=-1)

    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation, name=model_name2)(merge)
    else:
        dense1 = Dense(num_classes, activation="softmax",
                       name=model_name2)(merge)

    if use_contrastive_loss:
        neg_images = Input(
            shape=(inp_seq_len, num_negative_samples, inp_features,))
        neg_texts = Input(
            shape=(inp_seq_len, num_negative_samples, text_feature_dim,))
        # contrastive loss
        neg_image_embedded = image_embedding(neg_images)
        neg_text_embedded = text_embedding(neg_texts)

        positive_label_distance = tf.math.reduce_euclidean_norm(
            embedded_image - embedded_text, axis=-1)  # (?, 8)
        image_negative_text = tf.math.reduce_euclidean_norm(
            neg_text_embedded - tf.expand_dims(embedded_image, -2), axis=-1)  # (?, 8, 8)
        text_negative_image = tf.math.reduce_euclidean_norm(
            neg_image_embedded - tf.expand_dims(embedded_text, -2), axis=-1)  # (?, 8, 8)
        # print(positive_label_distance.shape,
        #       image_negative_text.shape, text_negative_image.shape)
        negative_samples = image_negative_text + text_negative_image
        negative_samples = tf.math.reduce_mean(
            negative_samples, axis=-1)  # (?, 8)

        contrastive_loss = tf.math.square(
            positive_label_distance) + tf.math.square(tf.math.maximum(margin - (negative_samples), 0))
        contrastive_loss = tf.math.reduce_mean(
            contrastive_loss, axis=-1, name="contrastive")  # (?, 8)
        # contrastive_loss = tf.expand_dims(contrastive_loss, -1)
        # print(contrastive_loss.shape)

        # true_images = tf.reduce_sum(inp_images, axis=-1)
        # loss_mask = tf.not_equal(true_images, 0.)
        # contrastive_loss = tf.boolean_mask(
        #     contrastive_loss, loss_mask, name='contrastive')

        inputs = [inp_images, inp_texts, neg_images, neg_texts]
        model = Model(inputs=inputs, outputs=[
            dense1, class_probs, contrastive_loss], name=model_name)
    else:
        inputs = [inp_images, inp_texts]
        model = Model(inputs=inputs, outputs=[
            dense1, class_probs], name=model_name)

    return model


def build_multitask_rnn(inp_seq_len, inp_features, **kwargs):
    """
    There are two outputs - 
    (1) to predict the item category of the next item and 
    (2) to predict the binary compatibility class 
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
    num_classes2 = kwargs.get("num_classes2", 153)
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
            tf.keras.layers.Dense(num_classes2, activation="softmax"),
        ]
    )
    class_probs = tf.keras.layers.TimeDistributed(
        classification_layers, name=model_name1
    )(merge)
    # should be batch X sequence_length X #classes

    # convert to the target_sequence_length
    merge = Permute((2, 1), input_shape=(
        inp_seq_len, components * d_model))(merge)
    merge = Dense(1, activation=merge_activation)(merge)
    merge = tf.squeeze(merge, axis=-1)

    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation, name=model_name2)(merge)
    else:
        dense1 = Dense(num_classes, activation="softmax",
                       name=model_name2)(merge)

    model = Model(inputs=inputs, outputs=[
                  dense1, class_probs], name=model_name)
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


def build_triplet_loss_model(image_features, text_features, **kwargs):
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    image_embedding = kwargs.get("image_embedding", "resnet50")
    model_name = kwargs.get("model_name", "fc")
    first_activation = kwargs.get("first_activation", "tanh")
    merge_activation = kwargs.get("merge_activation", "relu")
    final_activation = kwargs.get("final_activation", None)
    include_text = kwargs.get("include_text", False)
    text_feature_dim = kwargs.get("text_feature_dim", 768)
    margin = kwargs.get("margin", 0.2)

    seed(seed_value)
    set_seed(seed_value)

    inp_seq_len = 3
    inputs, flat = [], []

    if type(image_features) is tuple:
        if image_embedding == "resnet50":
            image_embedder = tf.keras.layers.TimeDistributed(resnet50)
        elif image_embedding == "inceptionv3":
            image_embedder = tf.keras.layers.TimeDistributed(inceptionv3)
        # tf.keras.models.Sequential(
        #         [
        #             tf.keras.layers.Dense(num_classes2, activation="softmax"),
        #         ]
        #     )
    else:
        unit_embedder = Dense(d_model, activation=first_activation)
        image_embedder = tf.keras.layers.TimeDistributed(unit_embedder)

    if type(image_features) is tuple:
        in1 = Input(
            shape=(
                inp_seq_len,
                image_features[0],
                image_features[1],
                image_features[2],
            )
        )
    else:
        in1 = Input(shape=(inp_seq_len, image_features))
    image_embedded = image_embedder(in1)
    image_embedded = BatchNormalization()(image_embedded)
    inputs.append(in1)
    flat.append(image_embedded)

    in2 = Input(shape=(inp_seq_len, text_features))
    text_embedder = tf.keras.layers.TimeDistributed(
        Dense(d_model, activation=first_activation))
    text_embedded = text_embedder(in2)
    inputs.append(in2)
    text_embedded = BatchNormalization()(text_embedded)
    flat.append(text_embedded)
    if len(flat) > 1:
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]

    # i-anchor item, j-positive item, k-negative item
    dij = euclidean_distance(
        [merge[:, 0, :], merge[:, 1, :]])
    dik = euclidean_distance(
        [merge[:, 0, :], merge[:, 2, :]])
    loss = tf.math.maximum(dij - dik + margin, 0)

    model = Model(inputs=inputs, outputs=loss, name=model_name)
    return model


def build_fc_model(inp_seq_len, inp_features, **kwargs):
    """
    Fully connected model with fixed number of inputs
    so that there is no masking, i.e., all items are 
    present for all examples.
    """
    num_classes = kwargs.get("num_classes", 2)
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    image_embedding = kwargs.get("image_embedding", "resnet50")
    model_name = kwargs.get("model_name", "fc")
    first_activation = kwargs.get("first_activation", "tanh")
    merge_activation = kwargs.get("merge_activation", "relu")
    final_activation = kwargs.get("final_activation", None)
    include_text = kwargs.get("include_text", False)
    text_feature_dim = kwargs.get("text_feature_dim", 768)
    include_item_categories = kwargs.get("include_item_categories", False)
    num_categories = kwargs.get("num_categories", 153)
    mask_zero = kwargs.get("mask_zero", True)

    seed(seed_value)
    set_seed(seed_value)

    inputs, flat = [], []
    if type(inp_features) is tuple:
        in1 = Input(
            shape=(
                inp_seq_len,
                inp_features[0],
                inp_features[1],
                inp_features[2],
            )
        )
        if image_embedding == "resnet50":
            image_embedder = tf.keras.layers.TimeDistributed(resnet50)
        elif image_embedding == "inceptionv3":
            image_embedder = tf.keras.layers.TimeDistributed(inceptionv3)
        image_embedded = image_embedder(in1)
    else:
        in1 = Input(shape=(inp_seq_len, inp_features))
        image_embedded = Dense(d_model, activation=first_activation)(in1)
    inputs.append(in1)
    image_embedded = BatchNormalization()(image_embedded)
    flat.append(image_embedded)

    # text
    if include_text:
        in2 = Input(shape=(inp_seq_len, text_feature_dim))
        text_embedded = Dense(d_model, activation=first_activation)(in2)
        inputs.append(in2)
        text_embedded = BatchNormalization()(text_embedded)
        flat.append(text_embedded)

    if include_item_categories:
        in3 = Input(shape=(inp_seq_len,))  # (?, 8)
        category_embedder = tf.keras.layers.Embedding(
            input_dim=num_categories,
            output_dim=d_model,
            embeddings_initializer="uniform",
            mask_zero=mask_zero,
        )
        category_embedded = category_embedder(in3)
        inputs.append(in3)
        flat.append(category_embedded)

    if len(flat) > 1:
        merge = concatenate(flat, axis=-1)  # (b, inp_seq_len, h)
    else:
        merge = flat[0]
    merge = BatchNormalization()(merge)
    merge = Flatten()(merge)
    if num_classes == 2:
        dense1 = Dense(1, activation=final_activation)(merge)
    else:
        dense1 = Dense(num_classes, activation="softmax")(merge)

    model = Model(inputs=inputs, outputs=dense1, name=model_name)
    return model


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
        self.d_model = kwargs.get("d_model", 256)
        self.include_text = kwargs.get("include_text", True)
        self.first_activation = kwargs.get("first_activation", "tanh")
        self.embedding_activation = kwargs.get(
            "embedding_activation", "linear")
        self.lstm_dim = kwargs.get("lstm_dim", 32)
        self.lstm_activation = kwargs.get("lstm_activation", "relu")
        self.final_activation = kwargs.get("final_activation", "sigmoid")
        self.num_classes = kwargs.get("num_classes", 2)
        self.num_categories = kwargs.get("num_categories", 153)
        self.model_name1 = kwargs.get("model_name1", "item_classification")
        self.model_name2 = kwargs.get("model_name2", "compatibility")
        self.return_negative_samples = kwargs.get(
            "return_negative_samples", False)
        self.margin = kwargs.get("margin", 0.2)
        # tf.keras.backend.set_floatx("float32")

        self.image_embedding = Dense(
            self.d_model, activation=self.first_activation)
        self.text_embedding = Dense(
            self.d_model, activation=self.first_activation)

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
                tf.keras.layers.Dense(
                    self.num_categories, activation="softmax"),
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

        if self.return_negative_samples:
            inp_image, inp_text, neg_image, neg_text = x
        else:
            inp_image, inp_text = x
        # mask = create_padding_mask(tf.reduce_sum(inp_image, axis=-1))

        # outfit images and texts
        embedded_image = self.image_embedding(inp_image)
        embedded_text = self.text_embedding(inp_text)

        encoded_image = self.image_encoder(embedded_image)
        encoded_text = self.text_encoder(embedded_text)
        combined_image_text = concatenate(
            [encoded_image, encoded_text], axis=-1)
        lstm_out = self.rnn(combined_image_text)
        compatibility_output = self.output_layer(lstm_out)
        class_probs = self.time_distributed(combined_image_text)

        if self.return_negative_samples:
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
            positive_prod = tf.keras.layers.Multiply()(
                [embedded_image, embedded_text])
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
                condition_1, tf.zeros_like(
                    contrastive_loss_1), contrastive_loss_1
            )

            contrastive_loss_2 = self.margin - positive_prod + negative_text_image
            condition_2 = tf.less(contrastive_loss_2, 0.0)
            loss_2 = tf.where(
                condition_2, tf.zeros_like(
                    contrastive_loss_2), contrastive_loss_2
            )
            contrastive_loss = loss_1 + loss_2

            return compatibility_output, class_probs, contrastive_loss

        return compatibility_output, class_probs
