import os
import json
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from tqdm import tqdm
import time
import pickle
import sys
from data_process import CustomDataGen
from build_model import MultiTaskSetTransformer
from rnn import MultiTaskRNN

BASE_DIR = "/recsys_data/RecSys/fashion/polyvore-dataset/polyvore_outfits"
DATA_TYPE = "disjoint"  # "nondisjoint"
train_dir = os.path.join(BASE_DIR, DATA_TYPE)
image_dir = os.path.join(BASE_DIR, "images")
train_json = "train.json"
valid_json = "valid.json"
test_json = "test.json"

train_file = "compatibility_train.txt"
valid_file = "compatibility_valid.txt"
test_file = "compatibility_test.txt"
item_file = "polyvore_item_metadata.json"
outfit_file = "polyvore_outfit_titles.json"

model_type = "rnn"  # "set-transformer", "rnn"
include_text = True
use_graphsage = False
batch_size = 32
MAX_SEQ_LEN = 8
learning_rate = 1.0e-03
LR_PATIENCE = 3
LR_DECAY = 0.5
EPOCHS = 100
max_patience = 5
VALID_FREQ = 2

if use_graphsage:
    IMAGE_EMBEDDING_DIM = 50
    image_embedding_file = "graphsage_dict2_polyvore.pkl"
else:
    IMAGE_EMBEDDING_DIM = 1280
    image_embedding_file = "effnet_tuned_polyvore.pkl"

TEXT_EMBEDDING_DIM = 768
NUM_NEGATIVES = 8
INCLUDE_CONTRASTIVE_LOSS = False


def evaluate(current_model, data_gen):
    """Evaluate a model on a given data"""
    m = tf.keras.metrics.BinaryAccuracy()
    m2 = tf.keras.metrics.AUC()
    acc_list = []
    pbar = tqdm(range(len(data_gen)))
    ys, yhats = [], []
    for ii in pbar:
        x, targs = data_gen[ii]  # batch size
        preds = current_model(x)
        compatibility_output = preds[0]
        compatibility_label = targs[0]
        m.update_state(compatibility_label, compatibility_output)
        batch_acc = m.result().numpy()
        acc_list.append(batch_acc)
        pbar.set_description("Batch accuracy %g" % batch_acc)
        ys.append(compatibility_label)
        yhats.append(compatibility_output)
    avg_acc = np.mean(acc_list)
    big_y = np.concatenate(ys, axis=0)
    big_yh = np.concatenate(yhats, axis=0)
    m2.update_state(big_y, big_yh)
    auc = m2.result().numpy()
    return auc, avg_acc


def sparse_crossentropy_masked(y_true, y_pred):
    # Cross-entropy loss with mask in the label
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    return K.mean(K.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))


if INCLUDE_CONTRASTIVE_LOSS:
    train_step_signature = [
        (
            tf.TensorSpec(
                shape=(None, MAX_SEQ_LEN, IMAGE_EMBEDDING_DIM), dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None, MAX_SEQ_LEN, TEXT_EMBEDDING_DIM), dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None, MAX_SEQ_LEN, NUM_NEGATIVES, IMAGE_EMBEDDING_DIM),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(None, MAX_SEQ_LEN, NUM_NEGATIVES, TEXT_EMBEDDING_DIM),
                dtype=tf.float32,
            ),
        ),
        (
            tf.TensorSpec(shape=(None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        ),
    ]
else:
    train_step_signature = [
        (
            tf.TensorSpec(
                shape=(None, MAX_SEQ_LEN, IMAGE_EMBEDDING_DIM), dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None, MAX_SEQ_LEN, TEXT_EMBEDDING_DIM), dtype=tf.float32
            ),
        ),
        (
            tf.TensorSpec(shape=(None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int64),
        ),
    ]


def loss_function(yhats, labels):
    """loss function calculation at each step"""

    # compatibility_output, class_probs, contrastive_loss = yhats
    # compatibility_label, item_categories, _ = labels

    item_categories = labels[1]
    mask = tf.cast(tf.math.not_equal(item_categories, -1), tf.float32)

    # binary loss
    loss_1 = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, name="binary_crossentropy"
    )(labels[0], yhats[0])

    # multi-class loss
    loss_2 = sparse_crossentropy_masked(item_categories, yhats[1])
    total_loss = loss_1 + loss_2

    if INCLUDE_CONTRASTIVE_LOSS:
        # contrastive loss
        loss_3 = tf.reduce_sum(yhats[2] * mask) / tf.reduce_sum(mask)
        total_loss += loss_3

    reg_loss = tf.compat.v1.losses.get_regularization_loss()
    # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    # loss += sum(reg_losses)
    total_loss += reg_loss
    return total_loss


train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")


@tf.function(input_signature=train_step_signature)
def train_step(inp, tgt):
    with tf.GradientTape() as tape:
        outputs = model(inp)
        step_loss = loss_function(outputs, tgt)

    gradients = tape.gradient(step_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(step_loss)
    # train_accuracy(accuracy_function(tar, predictions))
    return step_loss


if __name__ == "__main__":

    with open(os.path.join(train_dir, train_json), "r") as fr:
        train_pos = json.load(fr)

    with open(os.path.join(train_dir, valid_json), "r") as fr:
        valid_pos = json.load(fr)

    with open(os.path.join(train_dir, test_json), "r") as fr:
        test_pos = json.load(fr)

    with open(os.path.join(BASE_DIR, item_file), "r") as fr:
        pv_items = json.load(fr)

    with open(os.path.join(BASE_DIR, outfit_file), "r") as fr:
        pv_outfits = json.load(fr)

    with open(os.path.join(train_dir, train_file), "r") as fr:
        train_X, train_y = [], []
        for line in fr:
            elems = line.strip().split()
            train_y.append(elems[0])
            train_X.append(elems[1:])

    with open(os.path.join(train_dir, valid_file), "r") as fr:
        valid_X, valid_y = [], []
        for line in fr:
            elems = line.strip().split()
            valid_y.append(elems[0])
            valid_X.append(elems[1:])

    with open(os.path.join(train_dir, test_file), "r") as fr:
        test_X, test_y = [], []
        for line in fr:
            elems = line.strip().split()
            test_y.append(elems[0])
            test_X.append(elems[1:])

    item_dict = {}
    for ii, outfit in enumerate(train_pos):
        items = outfit["items"]
        mapped = train_X[ii]
        item_dict.update({jj: kk["item_id"] for jj, kk in zip(mapped, items)})
    print(len(item_dict))

    for ii, outfit in enumerate(valid_pos):
        items = outfit["items"]
        mapped = valid_X[ii]
        item_dict.update({jj: kk["item_id"] for jj, kk in zip(mapped, items)})
    print(len(item_dict))

    for ii, outfit in enumerate(test_pos):
        items = outfit["items"]
        mapped = test_X[ii]
        item_dict.update({jj: kk["item_id"] for jj, kk in zip(mapped, items)})
    print(len(item_dict))

    all_item_categories = set([pv_items[item]["category_id"] for item in pv_items])
    len(all_item_categories)
    label_renum_dict = {}
    for ii, k in enumerate(all_item_categories):
        label_renum_dict[k] = ii

    train_gen = CustomDataGen(
        train_X,
        train_y,
        item_dict,
        pv_items,
        image_dir,
        batch_size=batch_size,
        max_len=MAX_SEQ_LEN,
        only_image=not include_text,
        image_embedding=True,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        image_embedding_file=image_embedding_file,
        text_embedding_file="bert_polyvore.pkl",
        return_item_categories=True,
        return_negative_samples=INCLUDE_CONTRASTIVE_LOSS,
        number_negative_samples=8,
        label_dict=label_renum_dict,
    )
    valid_gen = CustomDataGen(
        valid_X,
        valid_y,
        item_dict,
        pv_items,
        image_dir,
        batch_size=batch_size,
        max_len=MAX_SEQ_LEN,
        only_image=not include_text,
        image_embedding=True,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        image_embedding_file=image_embedding_file,
        text_embedding_file="bert_polyvore.pkl",
        return_item_categories=True,
        return_negative_samples=INCLUDE_CONTRASTIVE_LOSS,
        number_negative_samples=8,
        label_dict=label_renum_dict,
    )
    test_gen = CustomDataGen(
        test_X,
        test_y,
        item_dict,
        pv_items,
        image_dir,
        batch_size=batch_size,
        max_len=MAX_SEQ_LEN,
        only_image=not include_text,
        image_embedding=True,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        image_embedding_file=image_embedding_file,
        text_embedding_file="bert_polyvore.pkl",
        return_item_categories=True,
        return_negative_samples=INCLUDE_CONTRASTIVE_LOSS,
        number_negative_samples=8,
        label_dict=label_renum_dict,
    )
    print("\n Sample Input Output shapes:")
    for ii in range(10):
        inps, targs = train_gen[ii]
        print([x.shape for x in inps], [y.shape for y in targs])

    if model_type == "rnn":
        model = MultiTaskRNN(
            num_layers=1,
            d_model=512,
            rnn="bilstm",
            final_activation="sigmoid",
            merge_activation="tanh",
            include_text=include_text,
            return_negative_samples=INCLUDE_CONTRASTIVE_LOSS,
        )
    else:
        model = MultiTaskSetTransformer(
            num_layers=1,
            d_model=64,
            num_heads=1,
            dff=32,
            rate=0.0,
            include_text=include_text,
            inp_dim2=768,
            num_classes=2,
            lstm_dim=32,
            embedding_activation="linear",
            lstm_activation="linear",
            final_activation="sigmoid",
        )
    # print(model.summary())

    best_auc = 0
    best_model = None
    patience = 0
    num_steps = len(train_gen)
    T = 0.0
    t0 = time.time()
    t_valid = evaluate(model, valid_gen)
    print(f"Before Training: valid (AUC: {t_valid[0]:.4f}, Accuracy: {t_valid[1]:.4f})")
    current_learning_rate = learning_rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )

    for epoch in range(1, EPOCHS + 1):

        step_losses = []
        # train_loss.reset_states()
        for step in tqdm(
            range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
        ):
            inputs, targets = train_gen[step]
            loss = train_step(inputs, targets)
            step_losses.append(loss)
        print(
            f"Epoch: {epoch}, Train Loss: {np.mean(step_losses):.3f}, {train_loss.result():.3f}"
        )

        if epoch % VALID_FREQ == 0:
            t1 = time.time() - t0
            T += t1
            print("Evaluating...")
            # t_test = evaluate(model, dataset, args)
            t_valid = evaluate(model, valid_gen)
            print(
                f"epoch: {epoch}, time: {T}, valid (AUC: {t_valid[0]:.4f}, Accuracy: {t_valid[1]:.4f})"
            )
            if t_valid[0] > best_auc:
                print("Performance improved ... updated the model.")
                best_auc = t_valid[0]
                best_model = model
                patience = 0
            else:
                patience += 1
                if patience == max_patience:
                    print(f"Maximum patience {patience} reached ... exiting!")
                    t_test = evaluate(model, test_gen)
                    print(f"Test AUC: {t_valid[0]:.4f}, Accuracy: {t_valid[1]:.4f})")
                    break

                if patience % LR_PATIENCE == 0:
                    current_learning_rate *= LR_DECAY
                    optimizer.lr.assign(current_learning_rate)
                    print(f"LR patience reached, current LR {current_learning_rate}")
