from PIL import Image
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import auc
from torchvision import transforms
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
import time
from tqdm import tqdm
import inspect
from typing import List, Optional, Tuple, Union, Set, Callable
import glob


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class ImageDataGen(torch_data.Dataset):
    def __init__(
        self,
        item_description,
        input_size=(3, 224, 224),
        label_dict=None,
        loader=default_image_loader,
        shuffle=True,
        valid=False,
        valid_sample=0,
        **kwargs,
    ):
        # item_description is a list of tuples
        # first element of the tuple is the image path
        # second element of the tuple if the category
        self.item_description = item_description
        X, y = [], []
        for item_id, cat in self.item_description:
            X.append(item_id)
            y.append(cat)

        self.X_col = "X"
        self.y_col = "y"
        self.df = pd.DataFrame({self.X_col: X, self.y_col: y})
        self.label_dict = label_dict
        self.num_classes = len(self.label_dict)
        max_example = kwargs.get("max_example", None)
        if max_example:
            self.df = self.df.sample(n=max_example)
        if valid:
            self.df = self.df.sample(n=valid_sample)

        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)
        print(f"Total {self.n} images with {self.num_classes} classes")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize(input_size[1]),
            transforms.CenterCrop(input_size[1]),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform = transform
        self.loader = loader
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_image(self, image_path):
        image = self.loader(image_path)
        # image = T.Resize(size=[224, 224])(image)
        # image_arr = np.array(image, dtype=float)
        # image_arr /= 255.0
        # image_arr = torch.from_numpy(image_arr)
        if self.transform is not None:
            image_arr = self.transform(image)

        return image_arr

    def __get_input(self, example):
        return self.get_image(example)

    def __getitem__(self, index):
        """ Returns a single sample as opposed to Tensorflow syntax
            that returns a batch of samples
        """
        x = self.df.iloc[index]["X"]
        y = torch.tensor(
            int(self.label_dict[self.df.iloc[index]["y"]])).type(torch.long)
        # X = torch.tensor(self.__get_input(x)).type(torch.float)
        X = self.__get_input(x).type(torch.float)
        return X, y

    def __len__(self):
        return self.n


class CustomDataset(torch_data.Dataset):
    def __init__(
        self,
        X,
        y,
        item_dict,
        item_description,
        loader=default_image_loader,
        **kwargs,
    ):
        self.df = pd.DataFrame({"X": X, "y": y})
        # self.X = self.df["X"].tolist()
        # self.y = self.df["y"].tolist()
        self.n = len(self.df)
        self.X_col = "X"
        self.y_col = "y"

        self.image_dir = kwargs.get("image_dir", None)
        # self.batch_size = kwargs.get("batch_size", 32)
        self.input_size = kwargs.get("input_size", (224, 224, 3))
        self.shuffle = kwargs.get("shuffle", True)
        self.max_len = kwargs.get("max_len", 16)
        self.item_dict = item_dict
        self.item_description = item_description
        self.image_embedding_dim = kwargs.get("image_embedding_dim", 1280)
        self.text_embedding_dim = kwargs.get("text_embedding_dim", 768)
        self.image_embedding_file = kwargs.get("image_embedding_file", None)
        self.return_item_categories = kwargs.get(
            "return_item_categories", False)
        self.return_negative_samples = kwargs.get(
            "return_negative_samples", False)
        self.number_negative_samples = kwargs.get("number_negative_samples", 0)
        self.label_dict = kwargs.get("label_dict", None)
        self.number_items_in_batch = kwargs.get("number_items_in_batch", None)
        self.variable_length_input = kwargs.get("variable_length_input", True)
        self.text_embedding_file = kwargs.get("text_embedding_file", None)
        self.include_item_categories = kwargs.get(
            "include_item_categories", False)
        self.category_mask_zero = kwargs.get("category_mask_zero", True)

        # image data can be one of "embedding", "original", "both"
        # both - means both image and embeddings will be returned
        self.image_data = kwargs.get("image_data", "embedding")
        self.only_image = kwargs.get("only_image", True)
        # self.get_image_embedding = kwargs.get("image_embedding", True)
        # self.include_original_images = kwargs.get(
        #     "include_original_images", False)

        # taken from https://github.com/mvasil/fashion-compatibility
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform = transform
        self.loader = loader
        self.zero_image_size = (3, 112, 112)

        if self.include_item_categories:
            if not self.label_dict:
                if self.category_mask_zero:
                    padding = 1
                else:
                    padding = 0
                all_item_categories = set(
                    [item_description[item]['category_id'] for item in item_description])
                self.label_dict = {}
                for ii, k in enumerate(all_item_categories):
                    self.label_dict[k] = ii + padding

        if not self.variable_length_input:
            # filter examples where the number of items in a
            # sequence is lesser than the max_len
            self.df["seq_len"] = self.df[self.X_col].apply(lambda x: len(x))
            self.df = self.df[self.df["seq_len"] == self.max_len]
            self.n = len(self.df)

        if self.return_negative_samples:
            self.items_by_category, self.item2cat = {}, {}
            for item, desc in self.item_description.items():
                cat = desc["category_id"]
                self.item2cat[item] = cat
                if cat in self.items_by_category:
                    self.items_by_category[cat].append(item)
                else:
                    self.items_by_category[cat] = [item]

        if self.image_data in ["embedding", "both"]:
            # "effnet2_polyvore.pkl" - 1280 dimensional vector
            # "effnet_tuned_polyvore.pkl" - 1280 dimensional vector
            # "graphsage_dict_polyvore.pkl" - 50 dimension
            # "graphsage_dict2_polyvore.pkl" - 50 dimension
            with open(self.image_embedding_file, "rb") as fr:
                self.embedding_dict = pickle.load(fr)

        if not self.only_image:
            # "bert_polyvore.pkl" - 768 dimensional vector
            with open(self.text_embedding_file, "rb") as fr:
                self.text_embedding_dict = pickle.load(fr)

        # original data has all positives followed by all negatives
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_texts(self, item_id):
        return torch.from_numpy(self.text_embedding_dict[item_id])

    def get_image(self, item_id):
        if self.image_data in ["embedding", "both"]:
            image_vector = self.embedding_dict[item_id].numpy()
            image_vector = torch.tensor(image_vector)
            if self.image_data == "embedding":
                return image_vector

        image_path = os.path.join(self.image_dir, item_id + ".jpg")
        image = self.loader(image_path)
        # image = T.Resize(size=[224, 224])(image)
        # image_arr = np.array(image, dtype=float)
        # image_arr /= 255.0
        # image_arr = torch.from_numpy(image_arr)
        if self.transform is not None:
            image_arr = self.transform(image)

        return image_arr

    def __get_input(self, example):
        """ same as that of Tensorflow data generator"""
        data, nimage_data, ntext_data = [], [], []
        items = [self.item_dict[x] for x in example[: self.max_len]]
        seq_len = len(items)
        for item in items:
            image = self.get_image(item)
            if self.only_image:
                data.append(image)
            else:
                text = self.get_texts(item)
                data.append((text, image))

            if self.return_negative_samples:
                neg_images, neg_texts = [], []
                neg_items = self.get_negative_samples(
                    item, self.number_negative_samples
                )
                for item_jj in neg_items:
                    image = self.get_image(item_jj)
                    text = self.get_texts(item_jj)
                    neg_images.append(image)
                    neg_texts.append(text)
                neg_images = np.array(neg_images)
                neg_texts = np.array(neg_texts)
                nimage_data.append(neg_images)
                ntext_data.append(neg_texts)

        if self.return_negative_samples:
            zero_elem_image = np.zeros(
                (self.number_negative_samples, self.image_embedding_dim)
            )
            zero_elem_text = np.zeros(
                (self.number_negative_samples, self.text_embedding_dim)
            )
            zeros_image = [
                zero_elem_image for _ in range(self.max_len - len(nimage_data))
            ]
            zeros_text = [zero_elem_text for _ in range(
                self.max_len - len(data))]

            nimage_data = zeros_image + nimage_data
            ntext_data = zeros_text + ntext_data

            # nimage_data = np.array(nimage_data)
            # ntext_data = np.array(ntext_data)
            # print(nimage_data.shape, ntext_data.shape)
            # sys.exit()

        if self.include_item_categories:
            item_cat_data = self.__get_label1(example)

        if self.image_data == "original":
            zero_elem_image = torch.zeros(self.zero_image_size)
            zeros_image = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        elif self.image_data == "embedding":
            zero_elem_image = torch.zeros(
                self.image_embedding_dim)
            zeros_image = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        elif self.image_data == "both":
            zero_elem_image = np.zeros(self.zero_image_size)
            zero_image_vector = np.zeros(self.image_embedding_dim)
            zeros_image_vec = [zero_image_vector for _ in range(
                self.max_len - len(data))]
            zeros_image_arr = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        if self.only_image:
            if self.image_data in ["original", "embedding"]:
                return zeros_image + data

            else:
                img_vecs = [x[0] for x in data]
                img_arrs = [x[1] for x in data]
                return (zeros_image_vec + img_vecs, zeros_image_arr + img_arrs)
        else:
            text_data = [x[0] for x in data]
            zero_elem_text = torch.zeros(self.text_embedding_dim)
            zeros_text = [zero_elem_text for _ in range(
                self.max_len - len(data))]

            if self.image_data in ["original", "embedding"]:
                image_data = [x[1] for x in data]

            elif self.image_data == "both":
                image_data = [x[1][0] for x in data]
                image_array_data = [x[1][1] for x in data]

            if self.return_negative_samples:
                return (
                    zeros_image + image_data,
                    zeros_text + text_data,
                    nimage_data,
                    ntext_data,
                )
            else:
                # only image embeddings
                if self.image_data in ["original", "embedding"]:
                    x = zeros_image + image_data
                    # print("Image", [y.shape for y in zeros_image + image_data])
                    # print("Text", [y.shape for y in zeros_text + text_data])
                    if self.include_item_categories:
                        return (zeros_image + image_data, zeros_text + text_data, item_cat_data, seq_len)
                    else:
                        return (zeros_image + image_data, zeros_text + text_data, seq_len)
                elif self.image_data == "both":
                    if self.include_item_categories:
                        return (zeros_image_vec + image_data, zeros_image_arr + image_array_data, zeros_text + text_data, item_cat_data)
                    else:
                        return (zeros_image_vec + image_data, zeros_image_arr + image_array_data, zeros_text + text_data)

    def __get_label1(self, example, pad=0):
        # creates labels for item classification
        # padded witn -1 to maintain the same sequence length
        items = [self.item_dict[x] for x in example[: self.max_len]]
        data = []
        for item in items:
            data.append(
                self.label_dict[self.item_description[item]["category_id"]])
        if len(data) < self.max_len:
            data = [pad] * (self.max_len - len(data)) + data
        return data

    def __get_label2(self, example, batch_dict):
        # creates labels for *next* item classification
        # padded witn -1 to maintain the same sequence length
        items = [self.item_dict[x] for x in example[: self.max_len]]
        targets = items[1:] + ['eos']
        data = []
        for t in targets:
            if t in batch_dict:
                data.append(batch_dict[t])
            else:
                data.append(batch_dict['unk'])
        if len(data) < self.max_len:
            data = [-1] * (self.max_len - len(data)) + data
        return data

    def __getitem__(self, index):
        """ Returns a single sample as opposed to Tensorflow syntax
            that returns a batch of samples
        """
        x = self.df.iloc[index]["X"]
        y = torch.tensor(int(self.df.iloc[index]["y"])).type(torch.float)
        # y = torch.from_numpy(self.df.iloc[index]["y"]).type(torch.int)
        if self.only_image:
            X = torch.from_numpy(self.__get_input(x)).type(torch.float)
        else:
            combined = self.__get_input(x)
            if self.return_negative_samples:
                X = (torch.tensor(combined[0]).type(torch.float),
                     torch.tensor(combined[1]).type(torch.float),
                     torch.tensor(combined[2]).type(torch.float),
                     torch.tensor(combined[3]).type(torch.float))
            else:
                if self.image_data == "both":
                    if self.include_item_categories:
                        X = (torch.tensor(combined[0]).type(torch.float),
                             torch.tensor(combined[1]).type(torch.float),
                             torch.tensor(combined[2]).type(torch.float),
                             torch.tensor(combined[3]).type(torch.int))
                    else:
                        X = (torch.tensor(combined[0]).type(torch.float),
                             torch.tensor(combined[1]).type(torch.float),
                             torch.tensor(combined[2]).type(torch.float))

                elif self.image_data in ["embedding", "original"]:
                    if self.include_item_categories:
                        # print([x.shape for x in combined[0]])
                        # print([x.shape for x in combined[1]])
                        X = (
                            torch.stack(combined[0]),
                            torch.stack(combined[1]),
                            torch.tensor(combined[2]).type(torch.int),
                            combined[3]
                        )
                    else:
                        X = (
                            torch.tensor(combined[0]).type(torch.float),
                            torch.tensor(combined[1]).type(torch.float),
                            combined[2]
                        )

        return X, y

    def __len__(self):
        return self.n


class FocalLoss(nn.BCEWithLogitsLoss):
    """
    Focal loss for binary classification tasks on imbalanced datasets
    https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
    https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    """

    def __init__(self, gamma, alpha=None, reduction='none'):
        super().__init__(weight=alpha, reduction='none')
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input_, target):
        p = torch.sigmoid(input_)
        cross_entropy = super().forward(input_, target)
        p_t = p * target + (1 - p) * (1 - target)
        loss = torch.pow(1 - p_t, self.gamma) * cross_entropy
        if self.alpha and self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def save_model(model, path):
    torch.save(model.state_dict(), path)


def train(model, train_set, valid_set, device, **kwargs):

    epochs = kwargs.get("epochs", 100)
    batch_size = kwargs.get("batch_size", 32)
    learning_rate = kwargs.get("learning_rate", 1e-05)
    decay_rate = kwargs.get("decay_rate", 0.5)
    exponential_decay_step = kwargs.get("exponential_decay_step", 5)
    validate_freq = kwargs.get("validate_freq", 1)
    early_stop = kwargs.get("early_stop", True)
    early_stop_step = kwargs.get("patience", 10)
    observe = kwargs.get("observe", "auc")
    optimizer = kwargs.get("optimizer", "adam")
    loss_name = kwargs.get("loss_name", "focal")
    clip_norm = kwargs.get("clip_norm", 0.5)

    test_set = kwargs.get("test_set", None)
    model_path = kwargs.get("model_path", None)

    prob_type = None
    if loss_name == "bce":
        criterion = nn.BCELoss(reduction="mean").to(device)
        prob_type = "binary"
    elif loss_name == "bcewithlogit":
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean").to(device)
        prob_type = "binary"
    elif loss_name == "focal":
        criterion = FocalLoss(2.0, reduction="mean").to(device)
        prob_type = "binary"
    elif loss_name == "nll":
        criterion = torch.nn.NLLLoss().to(device)
        prob_type = "multi"
    elif loss_name == "xent":
        criterion = torch.nn.CrossEntropyLoss().to(device)
        prob_type = "multi"
    else:
        print("Unknown loss!")
        return

    if optimizer == "adam":
        my_optim = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )
    elif optimizer == "rmsprop":
        my_optim = torch.optim.RMSprop(
            params=model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        my_optim = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=my_optim, gamma=decay_rate
    )

    train_loader = torch_data.DataLoader(
        train_set,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = torch_data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    num_train_steps = len(train_set)//batch_size
    num_valid_steps = len(valid_set)//batch_size

    dir_name = "indv"
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + dir_name + dt_string
    writer = SummaryWriter(log_dir)
    log_path = "logs/" + dt_string + ".log"
    fw = open(log_path, "w")
    fw.write("\t".join(["epoch", "train-loss",
             "val-loss", "val-a[c,u]c"])+"\n")

    best_validate_auc = 0
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        pbar = tqdm(enumerate(train_loader),
                    total=num_train_steps, leave=False)
        for i, (inputs, target) in pbar:
            if type(inputs) is list:
                inputs = [inp.to(device) for inp in inputs]
            else:
                inputs = inputs.to(device)

            target = target.to(device)
            target = torch.unsqueeze(target, -1)
            # model.zero_grad()

            my_optim.zero_grad()

            pred = model(inputs)
            if loss_name in ("nll", "xent"):
                target = torch.squeeze(target)

            loss = criterion(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            my_optim.step()

            pbar.set_description("loss %g" % float(loss))
            loss_total += float(loss)
            cnt += 1

        # if (epoch + 1) % exponential_decay_step == 0:
        #     my_lr_scheduler.step()
        if (epoch + 1) % validate_freq == 0:
            is_best_for_now = False
            performance_metrics = validate(
                model,
                valid_loader,
                device,
                loss_name=loss_name,
                num_steps=num_valid_steps,
            )
            writer.add_scalar("epoch_loss", loss_total / cnt, epoch)
            if "acc" in performance_metrics:
                writer.add_scalar(
                    "epoch_acc", performance_metrics["acc"], epoch)
            if "auc" in performance_metrics:
                writer.add_scalar(
                    "epoch_auc", performance_metrics["auc"], epoch)
            if "loss" in performance_metrics:
                writer.add_scalar("val_epoch_loss",
                                  performance_metrics["loss"], epoch)
            if performance_metrics[observe] > best_validate_auc:
                best_validate_auc = performance_metrics[observe]
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
                if validate_score_non_decrease_count == 4:
                    my_lr_scheduler.step()
            # save model
            if is_best_for_now:
                save_model(model, model_path)

        # early stop
        if prob_type == "binary":
            print(
                "| Epoch {:3d} | time: {:5.2f}s | train-loss {:5.4f} | val-ACC {:5.4f} | val-AUC {:5.4f} ({:2d})".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    loss_total / cnt,
                    performance_metrics["acc"],
                    performance_metrics["auc"],
                    validate_score_non_decrease_count,
                )
            )
            fw.write(
                f"{epoch}\t{loss_total / cnt:.4f}\t{performance_metrics['acc']:.4f}\t{performance_metrics['auc']:.4f}\t{validate_score_non_decrease_count} \n")
        elif prob_type == "multi":
            print(
                "| Epoch {:3d} | time: {:5.2f}s | train-loss {:5.4f} | val-loss {:5.4f} | val-ACC {:5.4f} ({:2d})".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    loss_total / cnt,
                    performance_metrics["loss"],
                    performance_metrics["acc"],
                    validate_score_non_decrease_count,
                )
            )
            fw.write(
                f"{epoch}\t{loss_total / cnt:.4f}\t{performance_metrics['loss']:.4f}\t{performance_metrics['acc']:.4f}\t{validate_score_non_decrease_count} \n")

        if early_stop and validate_score_non_decrease_count > early_stop_step:
            print(
                f"Exiting as the number of non-decrease count is greater than {early_stop_step}")
            break
    if prob_type == "binary":
        print(f"Best valid AUC: {best_validate_auc:.4f}")
        fw.write(f"Best valid AUC: {best_validate_auc:.4f}\n")
    elif prob_type == "multi":
        print(f"Best valid Accuracy: {best_validate_auc:.4f}")
        fw.write(f"Best valid Accuracy: {best_validate_auc:.4f}\n")

    if test_set is not None:
        test_loader = torch_data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        num_test_steps = len(test_set)//batch_size
        test_metrics = validate(
            model,
            test_loader,
            device,
            loss_name=loss_name,
            num_steps=num_test_steps,
        )
        print("Test results:", test_metrics)
        fw.write(
            f"TEST loss-{test_metrics['loss']:.4f}, acc-{test_metrics['acc']:.4f}\n")

    fw.close()
    return performance_metrics, test_metrics


def evaluate(y, y_hat):
    """
    :param y: array in shape of [count].
    :param y_hat: in same shape with y.
    """
    # print(y, y_hat)
    y_pred_class = np.where(y_hat > 0.5, 1, 0)
    acc = accuracy_score(y, y_pred_class)
    auc = roc_auc_score(y, y_hat)
    return acc, auc


def cross_entropy_loss(y, yhat):
    # numpy implementation of cross-entropy loss
    # probabilities
    phat = np.exp(yhat)/np.sum(np.exp(yhat), axis=0)
    logphat = -np.log(phat)
    loss = np.take_along_axis(logphat, np.expand_dims(y, axis=-1), 1)
    return np.mean(loss)


def evaluate_multiclass(y, y_hat, loss_name):
    y_pred_class = np.argmax(y_hat, axis=1)
    acc = accuracy_score(y, y_pred_class)
    if loss_name == "nll":
        criterion = torch.nn.NLLLoss()
    elif loss_name == "xent":
        loss = cross_entropy_loss(y, y_hat)
    return acc, loss


def inference(model, dataloader, device, num_steps):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in tqdm(enumerate(dataloader), total=num_steps, leave=False):
            if type(inputs) is list:
                inputs = [inp.to(device) for inp in inputs]
            else:
                inputs = inputs.to(device)
            target = target.to(device)
            forecast_steps = model(inputs)
            forecast_set.append(forecast_steps.detach().cpu().numpy())
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, dataloader, device, loss_name, num_steps):
    # start = datetime.now()
    forecast, target = inference(model, dataloader, device, num_steps)
    if loss_name in ("bce", "bcewithlogit", "focal"):
        # Binary classification problem
        scores = evaluate(target, forecast)
        return {"acc": scores[0], "auc": scores[1]}

    elif loss_name in ("nll", "xent"):
        # Multi-class classification problem
        scores = evaluate_multiclass(target, forecast, loss_name)
        return {"acc": scores[0], "loss": scores[1]}


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.
    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(
        new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.
    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.
    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    # Convert to set and remove already pruned heads
    heads = set(heads) - already_pruned_heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.
    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked
    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
    Examples:
    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(
        input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(
        inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(
            num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk)
                              for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)
