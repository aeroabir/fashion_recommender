from turtle import dot
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50
# from tensorflow.keras.applications import resnet
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Input,
    concatenate,
    Dropout,
    BatchNormalization,
    Reshape,
    Dense,
    RepeatVector,
    GlobalAveragePooling2D,
    Add,
    LSTM,
    GRU,
    Bidirectional,
)
from tensorflow.keras.models import Model


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
        tf.math.square(x - y), axis=-1, keepdims=False)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


class SubspaceAttn(tf.keras.Model):
    """
    Computes subspace attention between two categories
    Inputs are (1) image, (2) corresponding category and (3) another category
    """

    def __init__(self, **kwargs):
        super(SubspaceAttn, self).__init__()
        self.num_categories = kwargs.get("num_categories", 153)
        self.embedding_dim = kwargs.get("embedding_dim", 32)
        self.num_subspace = kwargs.get("num_subspace", 6)
        self.input_dim = kwargs.get("input_dim", 64)
        self.seq_len = kwargs.get("seq_len", 8)
        self.mask_zero = kwargs.get("mask_zero", True)

        # self.category_embedder = tf.keras.layers.CategoryEncoding(
        #     num_tokens=self.num_categories, output_mode="one_hot")
        self.category_embedder = tf.keras.layers.Embedding(
            input_dim=self.num_categories,
            output_dim=32,
            embeddings_initializer="uniform",
            mask_zero=self.mask_zero,
        )
        self.hidden = Dense(32, activation="tanh")
        self.final = Dense(self.num_subspace, activation="softmax")
        self.masks = Dense(self.num_subspace *
                           self.input_dim, activation="tanh")

    def call(self, x):
        image, cat1, cat2 = x
        cat1 = self.category_embedder(cat1)  # (None, s, 32)
        cat2 = self.category_embedder(cat2)  # (None, 32)
        cat2 = RepeatVector(self.seq_len)(cat2)  # (None, s, 32)
        joined = concatenate([image, cat1, cat2], axis=-1)
        out = self.hidden(joined)
        weights = self.final(out)  # (None, s, 6)
        learnt_masks = self.masks(joined)
        learnt_masks = Reshape(
            (self.seq_len, self.num_subspace, self.input_dim))(learnt_masks)
        x = tf.expand_dims(image, axis=2)  # (None, 8, 1, 64)
        xm = tf.math.multiply(x, learnt_masks)  # (None, 8, 6, 64)
        # xm = tf.keras.layers.UnitNormalization(axis=-1)(xm)
        weights = tf.expand_dims(weights, axis=-1)  # (None, 8, 6, 1)
        f = tf.math.multiply(xm, weights)  # (None, 8, 6, 64)
        f = tf.math.reduce_sum(f, axis=-2)  # (None, 8, 64)
        return f


def build_csanet(inp_seq_len, image_dim, **kwargs):
    """
    CSA-Net triplet processing model using pretrained keras models
    """
    model_name = kwargs.get("model_name", "cse-net")
    image_embedding_dim = kwargs.get("image_embedding_dim", 64)
    mask_zero = kwargs.get("mask_zero", True)
    margin = kwargs.get("margin", 0.4)
    num_categories = kwargs.get("num_categories", 153)
    num_subspace = kwargs.get("num_subspace", 6)

    image_embedder = tf.keras.models.Sequential()
    if type(image_dim) is tuple:
        outfit_images = Input(
            shape=(
                inp_seq_len,
                image_dim[0],
                image_dim[1],
                image_dim[2],
            )
        )
        positive_images = Input(
            shape=(
                image_dim[0],
                image_dim[1],
                image_dim[2],
            )
        )
        negative_images = Input(
            shape=(
                inp_seq_len,
                image_dim[0],
                image_dim[1],
                image_dim[2],
            )
        )
        image_embedder.add(ResNet50(
            include_top=True,
            weights="imagenet",
            pooling='avg',
        ))
    else:
        outfit_images = Input(shape=(inp_seq_len, image_dim))
        positive_images = Input(shape=(image_dim))
        negative_images = Input(shape=(inp_seq_len, image_dim))

    image_embedder.add(Dense(image_embedding_dim, activation="relu"))
    subspace_learner = SubspaceAttn(
        input_dim=image_embedding_dim,
        mask_zero=mask_zero,
        seq_len=inp_seq_len,
        num_categories=num_categories,
        num_subspace=num_subspace)

    outfit_categories = Input(shape=(inp_seq_len))
    positive_categories = Input(shape=())
    negative_categories = Input(shape=(inp_seq_len))

    all_inputs = [outfit_images, positive_images, negative_images,
                  outfit_categories, positive_categories, negative_categories]

    outfit_embedded = tf.keras.layers.TimeDistributed(
        image_embedder)(outfit_images)
    positive_embedded = image_embedder(positive_images)
    negative_embedded = tf.keras.layers.TimeDistributed(
        image_embedder)(negative_images)

    # output images, output categories & positive category
    # (None, 8, 64)
    f_output = subspace_learner(
        [outfit_embedded, outfit_categories, positive_categories])

    pos_repeated = RepeatVector(inp_seq_len)(positive_embedded)
    # positive images, output categories & positive category
    # (None, 8, 64)
    f_positive = subspace_learner(
        [pos_repeated, outfit_categories, positive_categories])

    Dp = euclidean_distance([f_output, f_positive])
    # Dp = tf.math.reduce_euclidean_norm(
    #     f_output - f_positive, axis=-1)  # (None, 8)
    Dp = tf.math.reduce_sum(Dp, axis=-1)

    D_n = []
    for ii in range(inp_seq_len):
        neg_repeated = RepeatVector(inp_seq_len)(negative_embedded[:, ii, :])
        f_neg_ii = subspace_learner(
            [neg_repeated, outfit_categories, negative_categories[:, ii]])
        # D_ni = tf.math.reduce_euclidean_norm(f_output - f_neg_ii, axis=-1)
        D_ni = euclidean_distance([f_output, f_neg_ii])
        D_n.append(D_ni)
    D_n = concatenate(D_n, axis=-1)
    D_n = tf.math.reduce_sum(D_n, axis=-1)

    loss = tf.math.maximum(Dp - D_n + margin, 0)

    model = Model(inputs=all_inputs, outputs=loss, name=model_name)
    return model


def build_csanet2(inp_seq_len, image_dim, **kwargs):
    """
    CSA-Net-II triplet processing model using pretrained keras models
    Inputs: 
        1) an outfit of fixed length (images and categories)
        2) an item image and category
        3) a third item (negative) if modeling with triplet loss
    """
    model_name = kwargs.get("model_name", "CSA-net-II")
    image_embedding_dim = kwargs.get("image_embedding_dim", 64)
    category_embedding_dim = kwargs.get("category_embedding_dim", 64)
    loss_type = kwargs.get("loss_type", "triplet")
    embedding_activation = kwargs.get("embedding_activation", "tanh")
    final_activation = kwargs.get("final_activation", None)
    margin = kwargs.get("margin", 0.4)
    num_categories = kwargs.get("num_categories", 153)
    num_subspace = kwargs.get("num_subspace", 6)
    d_model = kwargs.get("d_model", 128)
    rnn = kwargs.get("rnn", "bilstm")

    # define various encoders
    outfit_encoder = get_rnn(rnn, d_model, return_seq=True)
    if rnn == "bilstm":
        out_dim = 2 * d_model
    else:
        out_dim = d_model
    item_encoder = Dense(out_dim, activation=embedding_activation)

    category_embedder = tf.keras.layers.Embedding(
        input_dim=num_categories,
        output_dim=category_embedding_dim,
        embeddings_initializer="uniform",
        mask_zero=False,
    )
    image_embedder = tf.keras.models.Sequential()
    if type(image_dim) is tuple:
        outfit_images = Input(
            shape=(
                inp_seq_len,
                image_dim[0],
                image_dim[1],
                image_dim[2],
            )
        )
        positive_image = Input(
            shape=(
                image_dim[0],
                image_dim[1],
                image_dim[2],
            )
        )
        if "triplet" in loss_type.lower():
            negative_image = Input(
                shape=(
                    image_dim[0],
                    image_dim[1],
                    image_dim[2],
                )
            )
        image_embedder.add(ResNet50(
            include_top=True,
            weights="imagenet",
            pooling='avg',
        ))
    else:
        outfit_images = Input(shape=(inp_seq_len, image_dim))
        positive_image = Input(shape=(image_dim))
        if "triplet" in loss_type.lower():
            negative_image = Input(shape=(image_dim))

    all_inputs = [outfit_images, positive_image]
    if "triplet" in loss_type.lower():
        all_inputs.append(negative_image)

    image_embedder.add(
        Dense(image_embedding_dim, activation=embedding_activation))

    outfit_categories = Input(shape=(inp_seq_len))
    positive_category = Input(shape=())
    all_inputs += [outfit_categories, positive_category]

    outfit_embedded = tf.keras.layers.TimeDistributed(
        image_embedder)(outfit_images)
    positive_embedded = image_embedder(positive_image)

    if "triplet" in loss_type.lower():
        negative_category = Input(shape=())
        all_inputs.append(negative_category)
        negative_embedded = image_embedder(negative_image)

    outfit_cats = category_embedder(outfit_categories)  # (None, s, 32)
    positive_category = category_embedder(positive_category)  # (None, 32)

    # combine image and category
    outfit = concatenate([outfit_embedded, outfit_cats], axis=-1)
    outfit = outfit_encoder(outfit)

    positive = concatenate([positive_embedded, positive_category])
    positive = item_encoder(positive)

    # get attentional weight
    # attention = tf.keras.layers.Multiply()(
    #     [outfit, RepeatVector(inp_seq_len)(positive)])
    attention = tf.keras.layers.Dot(axes=(2))(
        [outfit, RepeatVector(inp_seq_len)(positive)])
    attention = tf.math.reduce_sum(attention, axis=-1)
    attention = tf.keras.layers.Softmax()(attention)

    # take weighted sum
    outfit = tf.math.multiply(outfit, tf.expand_dims(attention, axis=-1))
    outfit = tf.reduce_sum(outfit, axis=-2)

    if "triplet" in loss_type.lower():
        negative_category = category_embedder(negative_category)  # (None, 32)
        negative = concatenate([negative_embedded, negative_category])
        negative = item_encoder(negative)
        Dp = euclidean_distance([outfit, positive])
        Dp = tf.math.reduce_sum(Dp, axis=-1)
        Dn = euclidean_distance([outfit, negative])
        Dn = tf.math.reduce_sum(Dn, axis=-1)
        loss = tf.math.maximum(Dp - Dn + margin, 0)
    else:
        merge = concatenate([outfit, positive])
        loss = Dense(1, activation=final_activation)(merge)

    model = Model(inputs=all_inputs, outputs=loss, name=model_name)
    return model


def get_rnn(rnn_name, d_model, return_seq=True):
    if rnn_name == "lstm":
        encoder = LSTM(units=d_model, return_sequences=return_seq)
    elif rnn_name == "gru":
        encoder = GRU(units=d_model, return_sequences=return_seq)
    elif rnn_name == "bilstm":
        base_rnn = LSTM(units=d_model, return_sequences=return_seq)
        encoder = Bidirectional(base_rnn, merge_mode="concat")
    elif rnn_name == "bigru":
        base_rnn = GRU(units=d_model, return_sequences=return_seq)
        encoder = Bidirectional(base_rnn, merge_mode="concat")
    return encoder


class ResnetBlock(tf.keras.Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out
