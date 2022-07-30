from numpy.random import seed
from tensorflow.random import set_seed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np


class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs  # (None, 8, 800) (28, 2)

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)
        # (None, 8, 100)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges, batch_dims=1)
        print("2.", node_states_expanded.shape)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        print("3.", node_states_expanded.shape)
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        print("4.", attention_scores.shape)
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1], batch_dims=1)
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)


class GraphAttentionNetwork(tf.keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs


def get_edge_matrix(n):
    """
    Create an edge matrix for a fully connected graph
    n: number of nodes in the graph
    """
    num_rows = int(n * (n - 1) / 2.0)
    edge = np.zeros((num_rows, 2))
    count = 0
    for ii in range(n - 1):
        for jj in range(ii + 1, n):
            edge[count][0] = ii
            edge[count][1] = jj
            count += 1
    assert count == num_rows
    return edge.astype(int)


def build_multitask_gat(inp_seq_len, inp_features, **kwargs):
    """
    There are two outputs - (1) to predict the covariates and (2) to predict
    radiation using the prediction of the first model as future covariates
    There is only one model though.
    """
    num_classes = kwargs.get("num_classes", 2)
    num_heads = kwargs.get("num_heads", 8)
    rnn = kwargs.get("rnn", "lstm")
    num_layers = kwargs.get("num_layers", 2)
    hidden_units = kwargs.get("hidden_units", 128)
    seed_value = kwargs.get("seed", 100)
    rate = kwargs.get("rate", 0.1)
    model_name = kwargs.get("model_name", "GAT")
    merge_activation = kwargs.get("merge_activation", "relu")
    final_activation = kwargs.get("final_activation", None)
    include_text = kwargs.get("include_text", False)
    text_feature_dim = kwargs.get("text_feature_dim", 768)
    model_name1 = kwargs.get("model_name1", "item_classification")
    model_name2 = kwargs.get("model_name2", "compatibility")

    seed(seed_value)
    set_seed(seed_value)

    inputs, flat = [], []

    in1 = layers.Input(shape=(inp_seq_len, inp_features))
    inputs.append(in1)
    if include_text:
        in2 = layers.Input(shape=(inp_seq_len, text_feature_dim))
        inputs.append(in2)
        merge = layers.concatenate(inputs, axis=-1)
    else:
        merge = inputs
    x = layers.Dense(hidden_units * num_heads, activation="relu")(merge)
    attention_layers = [
        MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
    ]
    edges = get_edge_matrix(inp_seq_len)
    for attention_layer in attention_layers:
        x = attention_layer([x, edges]) + x
    # x = (b, inp_seq_len, #head * hidden_units)

    classification_layers = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(153, activation="softmax"),
        ]
    )
    class_probs = tf.keras.layers.TimeDistributed(
        classification_layers, name=model_name1
    )(x)
    # should be batch X sequence_length X #classes

    # convert to the target_sequence_length
    x = layers.Permute((2, 1), input_shape=(inp_seq_len, num_heads * hidden_units))(x)
    x = layers.Dense(1, activation=merge_activation)(x)
    x = tf.squeeze(x, axis=-1)

    if num_classes == 2:
        dense1 = layers.Dense(1, activation=final_activation, name=model_name2)(x)
    else:
        dense1 = layers.Dense(num_classes, activation="softmax", name=model_name2)(x)

    model = Model(inputs=inputs, outputs=[dense1, class_probs], name=model_name)
    return model
