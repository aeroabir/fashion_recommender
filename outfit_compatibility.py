import argparse
from inspect import ArgSpec
import os
import json
from collections import Counter
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import math
import numpy as np
import time
from tqdm import tqdm
import pickle

import sys
# sys.path.insert(0, "/recsys_data/RecSys/fashion/automl/efficientnetv2")
# import effnetv2_model

# %pylab inline
# import matplotlib.pyplot as plt
from build_model import build_multilevel_transformer
from build_model import build_set_transformer
from rnn import build_multilevel_rnn_unequal, build_fc_model, build_hybrid_model
from data_process import CustomDataGen


def get_accuracy_auc(data_gen, model):
    m = tf.keras.metrics.BinaryAccuracy()
    m2 = tf.keras.metrics.AUC()
    acc_list = []
    pbar = tqdm(range(len(data_gen)))
    ys, yhats = [], []
    for ii in pbar:
        x, y = data_gen[ii]  # batch size
        yhat = model(x)
        m.update_state(y, yhat)
        batch_acc = m.result().numpy()
        acc_list.append(batch_acc)
        pbar.set_description("Batch accuracy %g" % batch_acc)
        ys.append(y)
        yhats.append(yhat)
    print(f"Average Accuracy: {np.mean(acc_list)}")
    big_y = np.concatenate(ys, axis=0)
    big_yh = np.concatenate(yhats, axis=0)
    m2.update_state(big_y, big_yh)
    auc = m2.result().numpy()
    print(f"AUC: {auc}")
    return auc

parser = argparse.ArgumentParser(description='Fashion Compatibility Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Type_Specific_Fashion_Compatibility', type=str,
                    help='name of experiment')
parser.add_argument('--polyvore_split', default='disjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--base_dir', default='/recsys_data/RecSys/fashion/polyvore-dataset/polyvore_outfits', type=str,
                    help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--l2_embed', dest='l2_embed', action='store_true', default=False,
                    help='L2 normalize the output of the type specific embeddings')
parser.add_argument('--image_embedding_dim', type=int, default=1280, metavar='M',
                    help='image embedding dimension')
parser.add_argument('--text_embedding_dim', type=int, default=768, metavar='M',
                    help='text embedding dimension (BERT)')
###
parser.add_argument('--model_name', default='rnn', type=str,
                    help='model name')
parser.add_argument('--transformer_name', default='pytorch', type=str,
                    help='Transformer name (different implementation)')
parser.add_argument('--max_seq_len', type=int, default=8, metavar='M',
                    help='maximum number of items in an outfit')
parser.add_argument('--image_data_type', default='embedding', type=str,
                    help='input type of images, one of embedding, original or both')
parser.add_argument('--d_model', type=int, default=64, metavar='M',
                    help='transformer embedding dimension')
parser.add_argument('--patience', type=int, default=5, metavar='M',
                    help='patience for model training')
parser.add_argument('--n_heads', type=int, default=2, metavar='M',
                    help='transformer number of heads')
parser.add_argument('--n_layers', type=int, default=1, metavar='M',
                    help='transformer number of layers')
parser.add_argument('--dropout_rate', type=float, default=0.1, metavar='LR',
                    help='dropout rate (default: 0.1)')
parser.add_argument('--loss_name', default='focal', type=str,
                    help='Loss type')
parser.add_argument('--freeze_layers', type=int, default=0, metavar='M',
                    help='number of layers to freeze (in ResNet models)')
parser.add_argument('--clip_norm', type=float, default=0.5, metavar='CN',
                    help='gradient clipping norm')
parser.add_argument('--use_rnn', default=False, help='include RNN layer or not')


def main():
    global args
    args = parser.parse_args()

    base_dir = args.base_dir
    data_type = args.polyvore_split  # "nondisjoint", "disjoint"
    train_dir = os.path.join(base_dir, data_type)
    image_dir = os.path.join(base_dir, "images")
    embed_dir = "/recsys_data/RecSys/fashion/polyvore-dataset/precomputed"
    train_json = "train.json"
    valid_json = "valid.json"
    test_json = "test.json"

    train_file = "compatibility_train.txt"
    valid_file = "compatibility_valid.txt"
    test_file = "compatibility_test.txt"
    item_file = "polyvore_item_metadata.json"
    outfit_file = "polyvore_outfit_titles.json"

    model_type = args.model_name  # "set-transformer", "rnn"
    include_text = False
    use_image_embedding = True  # False # True
    image_data_type = args.image_data_type  # "original", "embedding", "both"
    image_encoder = "vgg16"  # "resnet50", "vgg16", "inception"
    include_item_categories = False
    use_graphsage = False
    if use_graphsage:
        image_embedding_dim, image_embedding_file = (
            50, os.path.join(embed_dir, "graphsage_dict2_polyvore.pkl"))
#         image_embedding_dim, image_embedding_file = (256, os.path.join(embed_dir, "graphsage_dict2_polyvore_nondisjoint.pkl"))
    else:
        image_embedding_dim, image_embedding_file = (
            1280, os.path.join(embed_dir, "effnet_tuned_polyvore.pkl"))
#         image_embedding_dim, image_embedding_file = (256, os.path.join(embed_dir, "triplet_polyvore_image.pkl"))

    text_embedding_dim, text_embedding_file = (
        768, os.path.join(embed_dir, "bert_polyvore.pkl"))
    # text_embedding_dim, text_embedding_file = (256, os.path.join(embed_dir, "triplet_polyvore_text.pkl"))

    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    learning_rate = args.lr
    epochs = args.epochs
    patience = args.patience
    reload_model = False
    rnn_d_model = 512
    rnn_dropout_rate = 0.5
    monitor = "val_auc"
    optimizer = "adam"

    with open(os.path.join(train_dir, train_json), 'r') as fr:
        train_pos = json.load(fr)

    with open(os.path.join(train_dir, valid_json), 'r') as fr:
        valid_pos = json.load(fr)

    with open(os.path.join(train_dir, test_json), 'r') as fr:
        test_pos = json.load(fr)

    with open(os.path.join(base_dir, item_file), 'r') as fr:
        pv_items = json.load(fr)

    with open(os.path.join(base_dir, outfit_file), 'r') as fr:
        pv_outfits = json.load(fr)
    print(f"Total {len(train_pos)}, {len(valid_pos)}, {len(test_pos)} outfits in train, validation and test split, respectively")

    with open(os.path.join(train_dir, train_file), 'r') as fr:
        train_X, train_y = [], []
        for line in fr:
            elems = line.strip().split()
            train_y.append(elems[0])
            train_X.append(elems[1:])

    with open(os.path.join(train_dir, valid_file), 'r') as fr:
        valid_X, valid_y = [], []
        for line in fr:
            elems = line.strip().split()
            valid_y.append(elems[0])
            valid_X.append(elems[1:])

    with open(os.path.join(train_dir, test_file), 'r') as fr:
        test_X, test_y = [], []
        for line in fr:
            elems = line.strip().split()
            test_y.append(elems[0])
            test_X.append(elems[1:])

    print(f"Total {len(train_X)}, {len(valid_X)}, {len(test_X)} examples in train, validation and test split, respectively")

    item_dict = {}
    for ii, outfit in enumerate(train_pos):
        items = outfit['items']
        mapped = train_X[ii]
        item_dict.update({jj: kk['item_id'] for jj, kk in zip(mapped, items)})
    print(len(item_dict))

    for ii, outfit in enumerate(valid_pos):
        items = outfit['items']
        mapped = valid_X[ii]
        item_dict.update({jj: kk['item_id'] for jj, kk in zip(mapped, items)})
    print(len(item_dict))

    for ii, outfit in enumerate(test_pos):
        items = outfit['items']
        mapped = test_X[ii]
        item_dict.update({jj: kk['item_id'] for jj, kk in zip(mapped, items)})
    print(len(item_dict))

    if model_type == "fc":
        model = build_fc_model(max_seq_len,
                               image_embedding_dim,
                               num_classes=2,
                               num_layers=2,
                               d_model=512,
                               rnn="bilstm",
                               final_activation="sigmoid",
                               include_text=include_text
                               )
    elif model_type == "rnn":
        model = build_multilevel_rnn_unequal(max_seq_len,
                                             num_classes=2,
                                             num_layers=2,
                                             d_model=rnn_d_model,
                                             rnn="bilstm",
                                             final_activation="sigmoid",
                                             include_text=include_text,
                                             image_embedding_dim=image_embedding_dim,
                                             text_feature_dim=text_embedding_dim,
                                             include_item_categories=include_item_categories,
                                             num_categories=154,
                                             image_data_type=image_data_type,
                                             include_multihead_attention=False,
                                             image_encoder=image_encoder,
                                             rate=rnn_dropout_rate,
                                             )
    elif model_type == "transformer":
        model = build_multilevel_transformer(max_seq_len,
                                             image_embedding_dim,
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
                                             final_activation="sigmoid"
                                             )
    elif model_type == "set-transformer":
        model = build_set_transformer(max_seq_len,
                                      image_embedding_dim,
                                      num_layers=3,
                                      d_model=256,
                                      num_heads=2,
                                      dff=256,
                                      num_classes=2,
                                      lstm_dim=256,
                                      include_text=include_text,
                                      inp_dim2=768,
                                      embedding_activation="relu",
                                      lstm_activation="relu",
                                      final_activation="sigmoid"
                                      )
    elif model_type == "hybrid":
        model = build_hybrid_model(max_seq_len,
                                   image_embedding_dim,
                                   rnn="bilstm",
                                   num_layers=2,
                                   d_model=512,
                                   num_heads=2,
                                   dff=32,
                                   rate=0.0,
                                   include_text=include_text,
                                   text_feature_dim=text_embedding_dim,
                                   num_classes=2,
                                   embedding_activation="tanh",
                                   lstm_activation="linear",
                                   final_activation="sigmoid",
                                   include_item_categories=include_item_categories,
                                   num_categories=154,
                                   )

    print(model.summary())

    train_gen = CustomDataGen(train_X, train_y,
                              item_dict,
                              pv_items,
                              image_dir=image_dir,
                              batch_size=batch_size,
                              max_len=max_seq_len,
                              only_image=not include_text,
                              image_embedding=use_image_embedding,
                              image_embedding_dim=image_embedding_dim,
                              image_embedding_file=image_embedding_file,
                              text_embedding_file=text_embedding_file,
                              number_items_in_batch=150,
                              variable_length_input=True,
                              text_embedding_dim=text_embedding_dim,
                              include_item_categories=include_item_categories,
                              image_data=image_data_type,
                              )
    valid_gen = CustomDataGen(valid_X, valid_y,
                              item_dict,
                              pv_items,
                              image_dir=image_dir,
                              batch_size=batch_size,
                              max_len=max_seq_len,
                              only_image=not include_text,
                              image_embedding=use_image_embedding,
                              image_embedding_dim=image_embedding_dim,
                              image_embedding_file=image_embedding_file,
                              text_embedding_file=text_embedding_file,
                              number_items_in_batch=150,
                              variable_length_input=True,
                              text_embedding_dim=text_embedding_dim,
                              include_item_categories=include_item_categories,
                              image_data=image_data_type,
                              )

    test_gen = CustomDataGen(test_X, test_y,
                             item_dict,
                             pv_items,
                             image_dir=image_dir,
                             batch_size=batch_size,
                             max_len=max_seq_len,
                             only_image=not include_text,
                             image_embedding=use_image_embedding,
                             image_embedding_dim=image_embedding_dim,
                             image_embedding_file=image_embedding_file,
                             text_embedding_file=text_embedding_file,
                             number_items_in_batch=150,
                             variable_length_input=True,
                             text_embedding_dim=text_embedding_dim,
                             include_item_categories=include_item_categories,
                             image_data=image_data_type,
                             )

    print(len(train_gen), len(valid_gen), len(test_gen))

    # num_train = len(train_X)
    checkpoint_filepath = base_dir + '/checkpoint'

    if reload_model:
        # Loads the weights
        model.load_weights(checkpoint_filepath)
        train_auc = get_accuracy_auc(train_gen, model)
        valid_auc = get_accuracy_auc(valid_gen, model)
        test_auc = get_accuracy_auc(test_gen, model)
        print(f"AUC: Train {train_auc}, Valid {valid_auc} and Test {test_auc}")

    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "nadam":
        opt = keras.optimizers.Nadam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, name='binary_crossentropy')
    model.compile(loss=loss, optimizer=opt, metrics=[
                  tf.keras.metrics.AUC()])  # "accuracy"
    es_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=patience,
        verbose=0,
        mode="max",  # for AUC it should be 'max'
        baseline=None,
        restore_best_weights=True,
    )

    # wandb_callback = WandbCallback(
    #                         monitor="val_loss",
    #                         verbose=0,
    #                         save_model=(False),
    #                         mode="auto")

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor,
        mode='min',
        save_best_only=True)

    # if monitor = 'val_auc' then mode should be 'max'
    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=0.5, mode='max',
                                  patience=1, min_lr=1e-07)

    # run = wandb.init(reinit=True)
    history = model.fit(train_gen,
                        epochs=epochs,
                        batch_size=batch_size,
                        # steps_per_epoch=math.ceil(num_train/batch_size),
                        validation_data=valid_gen,
                        validation_batch_size=32,
                        validation_freq=1,
                        callbacks=[reduce_lr, es_callback],  # es_callback
                        verbose=1)
    # run.finish()
    max_trn_auc = max(history.history['auc'])
    max_val_auc = max(history.history['val_auc'])
    print(
        f"Training completed, maximum training & validation AUC: {max_trn_auc:.4f} and {max_val_auc:.4f}")

    # train_auc = get_accuracy_auc(train_gen, model)
    valid_auc = get_accuracy_auc(valid_gen, model)
    test_auc = get_accuracy_auc(test_gen, model)

    print(f"AUC: Valid {valid_auc:.4f} and Test {test_auc:.4f}")

    model_path = f"compatibility_{data_type}_{model_type}_model_{max_seq_len}_only_image"
    model.save(model_path)

    print("DONE!")

if __name__ == "__main__":
    main()
