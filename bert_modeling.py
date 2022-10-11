import tensorflow as tf
import pandas as pd
import time
from numpy.random import seed

from tensorflow.random import set_seed
from transformer_models_ts import Encoder
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
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from focal_loss import BinaryFocalLoss
from rnn import build_multilevel_rnn_unequal


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class BertModel(object):
    """BERT model ("Bidirectional Encoder Representations from Transformers").
    Example usage:
        # Already been converted into token ids
        input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
        input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
        token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
        config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
            num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
        model = modeling.BertModel(config=config, is_training=True,
            input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
        label_embeddings = tf.get_variable(...)
        pooled_output = model.get_pooled_output()
        logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
  """

    def __init__(self, config):
        """
        single Transformer attending to all the features
        """
        self.config = config
        seed(config.seed_value)
        set_seed(config.seed_value)

        if config.model_name == "transformer":
            # image data - must
            in1 = Input(shape=(config.inp_seq_len, config.inp_dim))
            image_embedding = Dense(
                config.d_model, activation=config.embedding_activation)
            image_projected = image_embedding(in1)
            inputs, flat = in1, image_projected
            d_model_trfmr = config.d_model

            pooled_dense = Dense(1, activation=config.final_activation,
                                 kernel_initializer=create_initializer(
                                     config.initializer_range)
                                 )

            encoder = Encoder(
                config.num_hidden_layers,
                d_model_trfmr,
                config.num_attention_heads,
                config.dff,
                config.inp_seq_len,
                config.rate,
                config.embedding_activation
            )
            sequence_output = encoder(flat, True)
            first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
            pooled_output = pooled_dense(first_token_tensor)

            self.model = Model(inputs=inputs, outputs=pooled_output)

        elif config.model_name == "rnn":
            self.model = build_multilevel_rnn_unequal(config.inp_seq_len,
                                                      num_classes=2,
                                                      num_layers=2,
                                                      d_model=config.d_model,
                                                      rnn="bilstm",
                                                      final_activation="sigmoid",
                                                      include_text=config.include_text,
                                                      image_embedding_dim=config.image_embedding_dim,
                                                      text_feature_dim=config.text_feature_dim,
                                                      include_item_categories=config.include_item_categories,
                                                      num_categories=154,
                                                      image_data_type=config.image_data_type,
                                                      include_multihead_attention=False,)

    def train(self, train_data, valid_data=None):
        learning_rate = self.config.learning_rate
        epochs = self.config.epochs
        batch_size = self.config.batch_size
        patience = self.config.patience

        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        # loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='binary_crossentropy')
        # loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, name='binary_focal_crossentropy')
        loss = BinaryFocalLoss(gamma=2)

        self.model.compile(loss=loss, optimizer=opt, metrics=[
                           tf.keras.metrics.AUC()])  # "accuracy"
        callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=patience,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        # checkpoint_filepath = base_dir + '/checkpoint'
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=True,
        #     monitor='val_loss',
        #     mode='min',
        #     save_best_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=2, min_lr=1e-07)

        history = self.model.fit(train_data,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=valid_data,
                                 validation_batch_size=32,
                                 validation_freq=1,
                                 callbacks=[callback, reduce_lr],
                                 verbose=1)
        return history
