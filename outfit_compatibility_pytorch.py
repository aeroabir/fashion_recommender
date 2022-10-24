import argparse
import pandas as pd
import logging
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
from prettytable import PrettyTable

from stgnn_transformer import BaseTransformer, SimpleModel
from set_transformer_pytorch import BaseSetTransformer
from utils_torch import CustomDataset, train

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
parser.add_argument('--model_name', default='transformer', type=str,
                    help='model name')
parser.add_argument('--transformer_name', default='pytorch', type=str,
                    help='Transformer name (different implementation)')
parser.add_argument('--max_seq_len', type=int, default=12, metavar='M',
                    help='maximum number of items in an outfit')
parser.add_argument('--image_data_type', default='embedding', type=str,
                    help='input type of images, one of embedding, original or both')
parser.add_argument('--d_model', type=int, default=64, metavar='M',
                    help='transformer embedding dimension')
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
parser.add_argument('--use_rnn', default=False,
                    help='include RNN layer or not')


def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # create a logger
    logger = logging.getLogger()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    log_filename = f"logs/pytorch_{dt_string}.log"
    fhandler = logging.FileHandler(filename=log_filename, mode='w')
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.INFO)
    logger.addHandler(fhandler)

    args_dict = vars(args)
    for k, v in args_dict.items():
        logging.info(f"{k}: {v}")

    base_dir = args.base_dir
    data_type = args.polyvore_split
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

    include_text = True
    use_graphsage = False
    include_item_categories = True
    image_encoder = "resnet18"  # "resnet50", "vgg16", "inception"

    model_params = {
        "max_seq_len": args.max_seq_len,
        "image_data_type": args.image_data_type,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'rate': args.dropout_rate,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'scheduler': None,
        'small_epochs': 1,
        'output_dir': './',
        'norm_method': 'min_max_m1',
        'exponential_decay_step': 5,
        'validate_freq': 1,
        'early_stop': True,
        'device': 'cuda',
        'name': args.model_name,
        'transformer_name': args.transformer_name,
        'loss': args.loss_name,
        'freeze_layers': args.freeze_layers,
        'clip_norm': args.clip_norm,
        'use_rnn': args.use_rnn,
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
    logging.info(
        f"Total {len(train_pos)}, {len(valid_pos)}, {len(test_pos)} positive examples in train, validation and test split, respectively")

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
    logging.info(
        f"Total {len(train_X)}, {len(valid_X)}, {len(test_X)} examples in train, validation and test split, respectively")

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

    if model_params['name'] == 'transformer':
        model = BaseTransformer(num_layers=model_params['n_layers'],
                                d_model=model_params['d_model'],
                                num_heads=model_params['n_heads'],
                                dff=64,  # 32
                                rate=model_params["rate"],
                                max_seq_len=args.max_seq_len,
                                num_classes=2,
                                lstm_dim=model_params['d_model'],
                                device=device,
                                image_data_type=args.image_data_type,
                                include_text=include_text,
                                include_item_categories=include_item_categories,
                                num_categories=154,
                                embedding_activation="linear",
                                encoder_activation="relu",
                                lstm_activation="linear",
                                transformer_name=model_params["transformer_name"],
                                freeze_layers=model_params["freeze_layers"],
                                use_rnn=model_params["use_rnn"],
                                final_activation="sigmoid")

    elif model_params['name'] == 'set-transformer':
        model = BaseSetTransformer(num_layers=model_params['n_layers'],
                                   d_model=model_params['d_model'],
                                   num_heads=model_params['n_heads'],
                                   dff=32,
                                   rate=model_params["rate"],
                                   max_seq_len=args.max_seq_len,
                                   num_classes=2,
                                   lstm_dim=32,
                                   device=device,
                                   image_data_type=args.image_data_type,
                                   include_text=include_text,
                                   include_item_categories=include_item_categories,
                                   num_categories=154,
                                   embedding_activation="linear",
                                   encoder_activation="relu",
                                   lstm_activation="linear",
                                   final_activation="sigmoid")

    elif model_params['name'] == 'simple':
        model = SimpleModel(d_model=model_params['d_model'],
                            image_embedding_dim=image_embedding_dim,
                            max_seq_len=args.max_seq_len,
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
                              max_len=args.max_seq_len,
                              only_image=not include_text,
                              image_embedding_dim=image_embedding_dim,
                              image_embedding_file=image_embedding_file,
                              text_embedding_file=text_embedding_file,
                              number_items_in_batch=150,
                              variable_length_input=True,
                              text_embedding_dim=text_embedding_dim,
                              include_item_categories=include_item_categories,
                              image_data=args.image_data_type,
                              input_size=(3, 224, 224),
                              )

    valid_set = CustomDataset(valid_X,
                              valid_y,
                              item_dict,
                              pv_items,
                              image_dir=image_dir,
                              max_len=args.max_seq_len,
                              only_image=not include_text,
                              image_embedding_dim=image_embedding_dim,
                              image_embedding_file=image_embedding_file,
                              text_embedding_file=text_embedding_file,
                              number_items_in_batch=150,
                              variable_length_input=True,
                              text_embedding_dim=text_embedding_dim,
                              include_item_categories=include_item_categories,
                              image_data=args.image_data_type,
                              input_size=(3, 224, 224),
                              )

    test_set = CustomDataset(test_X,
                             test_y,
                             item_dict,
                             pv_items,
                             image_dir=image_dir,
                             max_len=args.max_seq_len,
                             only_image=not include_text,
                             image_embedding_dim=image_embedding_dim,
                             image_embedding_file=image_embedding_file,
                             text_embedding_file=text_embedding_file,
                             number_items_in_batch=150,
                             variable_length_input=True,
                             text_embedding_dim=text_embedding_dim,
                             include_item_categories=include_item_categories,
                             image_data=args.image_data_type,
                             input_size=(3, 224, 224),
                             )

    param_table = PrettyTable(["Parameter", "Value"])
    logging.info("Model Parameters:")
    for k, v in model_params.items():
        param_table.add_row([k, v])
        logging.info(f"{k}: {v}")

    print(param_table)
    print(f"Total Trainable Params: {total_params//1e06} M")
    logging.info(f"Total Trainable Params: {total_params//1e06} M")

    train(model, train_set, valid_set, device='cuda',
          epochs=model_params['epochs'],
          batch_size=model_params['batch_size'],
          learning_rate=model_params['lr'],
          loss_name=model_params['loss'],
          clip_norm=model_params["clip_norm"],
          logging=logging,
          test_set=test_set,
          )


if __name__ == "__main__":
    main()
