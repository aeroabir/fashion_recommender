import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import math
import numpy as np
import pickle
import random
import time
from tqdm import tqdm
from datetime import datetime
import torch
import torch.utils.data as torch_data
import torch.nn as nn
from numpy.random import seed

from stgnn_transformer import BaseTransformer, SimpleModel
from utils_torch import CustomDataset, train


if __name__ == "__main__":

    base_dir = "/recsys_data/RecSys/fashion/polyvore-dataset/polyvore_outfits"
    data_type = "nondisjoint"  # "nondisjoint", "disjoint"
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

    model_type = "transformer"  # "set-transformer", "rnn"
    include_text = True
    use_graphsage = False
    max_seq_len = 12
    image_data_type = "embedding"  # "original", "embedding", "both"
    include_item_categories = True
    image_encoder = "resnet18"  # "resnet50", "vgg16", "inception"

    model_params = {
        "max_seq_len": max_seq_len,
        "image_data_type": image_data_type,
        'd_model': 64,
        'n_heads': 16,
        'n_layers': 6,
        'rate': 0.1,
        'batch_size': 128,
        'lr': 1e-03,
        'epochs': 100,
        'scheduler': None,
        'small_epochs': 1,
        'output_dir': './',
        'norm_method': 'min_max_m1',
        'exponential_decay_step': 5,
        'validate_freq': 1,
        'early_stop': True,
        'device': 'cuda',
    }

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

    # Read all the required files
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

    # Create a dict that maps encoded item-id to actual item-id
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

    device = model_params['device']

    # model = BaseTransformer(num_layers=model_params['n_layers'],
    #                         d_model=model_params['d_model'],
    #                         num_heads=model_params['n_heads'],
    #                         dff=32,
    #                         rate=model_params["rate"],
    #                         num_classes=2,
    #                         lstm_dim=32,
    #                         device=device,
    #                         image_data_type=image_data_type,
    #                         include_text=include_text,
    #                         include_item_categories=include_item_categories,
    #                         num_categories=154,
    #                         embedding_activation="linear",
    #                         encoder_activation="relu",
    #                         lstm_activation="linear",
    #                         final_activation="sigmoid")

    model = SimpleModel(d_model=model_params['d_model'], 
                        image_embedding_dim=image_embedding_dim,
                        max_seq_len=max_seq_len,
                        device=device,)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param

    train_set = CustomDataset(train_X,
                              train_y,
                              item_dict,
                              pv_items,
                              image_dir=image_dir,
                              max_len=max_seq_len,
                              only_image=not include_text,
                              image_embedding_dim=image_embedding_dim,
                              image_embedding_file=image_embedding_file,
                              text_embedding_file=text_embedding_file,
                              number_items_in_batch=150,
                              variable_length_input=True,
                              text_embedding_dim=text_embedding_dim,
                              include_item_categories=include_item_categories,
                              image_data=image_data_type,
                              input_size=(3, 224, 224),
                              )
    valid_set = CustomDataset(valid_X,
                              valid_y,
                              item_dict,
                              pv_items,
                              image_dir=image_dir,
                              max_len=max_seq_len,
                              only_image=not include_text,
                              image_embedding_dim=image_embedding_dim,
                              image_embedding_file=image_embedding_file,
                              text_embedding_file=text_embedding_file,
                              number_items_in_batch=150,
                              variable_length_input=True,
                              text_embedding_dim=text_embedding_dim,
                              include_item_categories=include_item_categories,
                              image_data=image_data_type,
                              input_size=(3, 224, 224),
                              )

    print(model_params)
    print(f"Total Trainable Params: {total_params}")
    train(model, train_set, valid_set, device='cuda',
          epochs=model_params['epochs'], batch_size=model_params['batch_size'], learning_rate=model_params['lr'])
