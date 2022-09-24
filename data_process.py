# import effnetv2_model
import glob
from itertools import combinations
import json
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import pandas as pd
import pickle
import math
import numpy as np
import random
import sys
from tqdm import tqdm
import uuid

# from transformers import BertTokenizer, BertModel

# import torch

sys.path.insert(0, "/recsys_data/RecSys/fashion/automl/efficientnetv2")

base_dir = "/recsys_data/RecSys/fashion/polyvore-dataset/polyvore_outfits"
train_dir = os.path.join(base_dir, "disjoint")
image_dir = os.path.join(base_dir, "images")
train_json = "train.json"
item_file = "polyvore_item_metadata.json"
outfit_file = "polyvore_outfit_titles.json"

train_file = "compatibility_train.txt"
valid_file = "compatibility_valid.txt"
test_file = "compatibility_test.txt"


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        X,
        y,
        item_dict,
        item_description,
        **kwargs,
    ):
        self.df = pd.DataFrame({"X": X, "y": y})
        # self.X = self.df["X"].tolist()
        # self.y = self.df["y"].tolist()
        self.n = len(self.df)
        self.X_col = "X"
        self.y_col = "y"
        self.image_dir = kwargs.get("image_dir", None)
        self.batch_size = kwargs.get("batch_size", 32)
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

        if self.include_item_categories:
            if not self.label_dict:
                if self.category_mask_zero:
                    padding = 1
                else:
                    padding = 0
                all_item_categories = set(
                    [item_description[item]["category_id"]
                        for item in item_description]
                )
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

            # self.model = tf.keras.models.Sequential(
            #     [
            #         tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
            #         effnetv2_model.get_model("efficientnetv2-b0", include_top=False),
            #     ]
            # )

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
        return self.text_embedding_dict[item_id]
        # item = self.item_description[item_id]  # item attributes
        # text = " ".join(
        #     [
        #         item["url_name"],
        #         item["description"],
        #         item["title"],
        #         item["semantic_category"],
        #     ]
        # )
        # inputs = self.tokenizer(text, return_tensors="pt")
        # outputs = self.text_model(**inputs)
        # pooled_output = outputs.pooler_output.detach().numpy()[0, :]
        # return pooled_output

    def get_image(self, item_id):
        if self.image_data in ["embedding", "both"]:
            image_vector = self.embedding_dict[item_id]
            if self.image_data == "embedding":
                return image_vector
        image_path = os.path.join(self.image_dir, item_id + ".jpg")
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])
        ).numpy()
        image_arr /= 255.0
        # if self.get_image_embedding:
        # return tf.squeeze(self.model(tf.expand_dims(image_arr, 0)))
        if self.image_data == "both":
            return (image_vector, image_arr)
        return image_arr

    def get_negative_samples(self, item_id, num_negative):
        # get items from the same category
        item_cat = self.item2cat[item_id]
        item_pool = self.items_by_category[item_cat].copy()
        try:
            item_pool.remove(item_id)
        except:
            print("cannot find!!")
            print(item_id, item_cat)
            print(item_pool)
        neg_items = np.random.randint(
            low=0, high=len(item_pool), size=num_negative)
        neg_items = [item_pool[ii] for ii in neg_items]
        return neg_items

    def __get_input(self, example):
        data, nimage_data, ntext_data = [], [], []
        items = [self.item_dict[x] for x in example[: self.max_len]]
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
            zero_elem_image = np.zeros(self.input_size)
            zeros_image = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        elif self.image_data == "embedding":
            zero_elem_image = np.zeros(
                self.image_embedding_dim)  # np.zeros((1, 1280))
            zeros_image = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        elif self.image_data == "both":
            zero_elem_image = np.zeros(self.input_size)
            zero_image_vector = np.zeros(self.image_embedding_dim)
            zeros_image_vec = [
                zero_image_vector for _ in range(self.max_len - len(data))
            ]
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
            zero_elem_text = np.zeros(self.text_embedding_dim)
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
                    if self.include_item_categories:
                        return (
                            zeros_image + image_data,
                            zeros_text + text_data,
                            item_cat_data,
                        )
                    else:
                        return (zeros_image + image_data, zeros_text + text_data)
                elif self.image_data == "both":
                    if self.include_item_categories:
                        return (
                            zeros_image_vec + image_data,
                            zeros_image_arr + image_array_data,
                            zeros_text + text_data,
                            item_cat_data,
                        )
                    else:
                        return (
                            zeros_image_vec + image_data,
                            zeros_image_arr + image_array_data,
                            zeros_text + text_data,
                        )

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
        targets = items[1:] + ["eos"]
        data = []
        for t in targets:
            if t in batch_dict:
                data.append(batch_dict[t])
            else:
                data.append(batch_dict["unk"])
        if len(data) < self.max_len:
            data = [-1] * (self.max_len - len(data)) + data
        return data

    def __get_mask(self, example):
        # creates masks for sequential input
        # 1 for padded items and 0 for true items
        mask = [0.0 for x in example[: self.max_len]]
        if len(mask) < self.max_len:
            mask = [1.0] * (self.max_len - len(mask)) + mask
        return mask

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()

        # get all the items for this batch and create a
        # a label dict (dynamic)
        batch_items = set()
        for example in x_batch:
            batch_items |= set([self.item_dict[x]
                               for x in example[: self.max_len]])
        batch_items = list(batch_items)[: self.number_items_in_batch]
        batch_dict = {jj: ii for ii, jj in enumerate(batch_items)}
        batch_dict["unk"] = len(batch_items)
        batch_dict["eos"] = len(batch_items) + 1

        # mask = np.asarray([self.__get_mask(x) for x in x_batch])
        if self.only_image:
            X_batch = np.asarray([self.__get_input(x) for x in x_batch])
        else:
            combined = [self.__get_input(x) for x in x_batch]
            if self.return_negative_samples:
                X_batch = (
                    np.asarray([x[0] for x in combined]),
                    np.asarray([x[1] for x in combined]),
                    np.asarray([x[2] for x in combined]),
                    np.asarray([x[3] for x in combined]),
                )
            else:
                if self.image_data == "both":
                    if self.include_item_categories:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            np.asarray([x[2] for x in combined]),
                            np.asarray([x[3] for x in combined]),
                            # mask,
                        )
                    else:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            np.asarray([x[2] for x in combined]),
                            # mask,
                        )

                elif self.image_data in ["embedding", "original"]:
                    if self.include_item_categories:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            np.asarray([x[2] for x in combined]),
                            # mask,
                        )
                    else:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            # mask,
                        )

        y_batch = np.asarray([int(y) for y in y_batch])
        if not self.return_item_categories and not self.return_negative_samples:
            y_total = y_batch

        if self.return_item_categories:
            y2_batch = np.asarray(
                [self.__get_label2(x, batch_dict) for x in x_batch])
            # y2_batch = np.asarray([self.__get_label1(x) for x in x_batch])
            y_total = [y_batch, y2_batch]

        if self.return_negative_samples:
            y3_batch = np.zeros((len(x_batch), 1))
            # y3_batch = y2_batch
            y_total.append(y3_batch)

        return X_batch, y_total

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)
        # return self.n // self.batch_size


class ZalandoDataGen(tf.keras.utils.Sequence):
    """
    For Zalando dataset the outfit style is also included as
    additional input to the model. This requires a separate
    treatment.
    """

    def __init__(
        self,
        X,
        y,
        item_description,
        **kwargs,
    ):
        self.df = pd.DataFrame({"X": X, "y": y})
        self.n = len(self.df)
        self.X_col = "X"
        self.y_col = "y"
        self.image_dir = kwargs.get("image_dir", None)
        self.batch_size = kwargs.get("batch_size", 32)
        self.input_size = kwargs.get("input_size", (224, 224, 3))
        self.shuffle = kwargs.get("shuffle", True)
        self.max_len = kwargs.get("max_len", 16)
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
        self.include_text = kwargs.get("include_text", False)
        self.category_mask_zero = kwargs.get("category_mask_zero", True)

        # image data can be one of "embedding", "original", "both"
        # both - means both image and embeddings will be returned
        self.image_data = kwargs.get("image_data", "embedding")
        # self.only_image = kwargs.get("only_image", True)
        # self.get_image_embedding = kwargs.get("image_embedding", True)
        # self.include_original_images = kwargs.get(
        #     "include_original_images", False)

        if self.include_item_categories:
            if not self.label_dict:
                if self.category_mask_zero:
                    padding = 1
                else:
                    padding = 0
                all_item_categories = set(
                    [item_description[item]["category_id"]
                        for item in item_description]
                )
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
            # "effnet2_zalando.pkl" - 1280 dimensional vector
            with open(self.image_embedding_file, "rb") as fr:
                self.embedding_dict = pickle.load(fr)

        if self.include_text:
            # "bert_polyvore.pkl" - 768 dimensional vector
            with open(self.text_embedding_file, "rb") as fr:
                self.text_embedding_dict = pickle.load(fr)

        # original data has all positives followed by all negatives
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

    def get_image(self, item_id):
        if self.image_data in ["embedding", "both"]:
            # TODO: rerun image embedding with the full filename as key
            current_name = item_id.split('.')[0]
            image_vector = self.embedding_dict[current_name]
            if self.image_data == "embedding":
                return image_vector
        # item_id should have the right extension jpg or png)
        image_path = os.path.join(self.image_dir, item_id)
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])
        ).numpy()
        image_arr /= 255.0
        # if self.get_image_embedding:
        # return tf.squeeze(self.model(tf.expand_dims(image_arr, 0)))
        if self.image_data == "both":
            return (image_vector, image_arr)
        return image_arr

    def get_negative_samples(self, item_id, num_negative):
        # get items from the same category
        item_cat = self.item2cat[item_id]
        item_pool = self.items_by_category[item_cat].copy()
        try:
            item_pool.remove(item_id)
        except:
            print("cannot find!!")
            print(item_id, item_cat)
            print(item_pool)
        neg_items = np.random.randint(
            low=0, high=len(item_pool), size=num_negative)
        neg_items = [item_pool[ii] for ii in neg_items]
        return neg_items

    def __get_input(self, example):
        data, nimage_data, ntext_data = [], [], []
        style = int(example[0])
        items = [x for x in example[1: self.max_len]]
        for item in items:
            image = self.get_image(item)
            if self.include_text:
                text = self.get_texts(item)
                data.append((text, image))
            else:
                data.append(image)

            if self.return_negative_samples:
                neg_images, neg_texts = [], []
                neg_items = self.get_negative_samples(
                    item, self.number_negative_samples
                )
                for item_jj in neg_items:
                    image = self.get_image(item_jj)
                    neg_images.append(image)
                    if self.include_text:
                        text = self.get_texts(item_jj)
                        neg_texts.append(text)
                neg_images = np.array(neg_images)
                nimage_data.append(neg_images)
                if self.include_text:
                    neg_texts = np.array(neg_texts)
                    ntext_data.append(neg_texts)

        if self.return_negative_samples:
            zero_elem_image = np.zeros(
                (self.number_negative_samples, self.image_embedding_dim)
            )
            zeros_image = [
                zero_elem_image for _ in range(self.max_len - len(nimage_data))
            ]
            nimage_data = zeros_image + nimage_data
            if self.include_text:
                zero_elem_text = np.zeros(
                    (self.number_negative_samples, self.text_embedding_dim)
                )
                zeros_text = [zero_elem_text for _ in range(
                    self.max_len - len(data))]
                ntext_data = zeros_text + ntext_data

        if self.include_item_categories:
            item_cat_data = self.__get_label1(items)

        if self.image_data == "original":
            zero_elem_image = np.zeros(self.input_size)
            zeros_image = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        elif self.image_data == "embedding":
            zero_elem_image = np.zeros(
                self.image_embedding_dim)  # np.zeros((1, 1280))
            zeros_image = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        elif self.image_data == "both":
            zero_elem_image = np.zeros(self.input_size)
            zero_image_vector = np.zeros(self.image_embedding_dim)
            zeros_image_vec = [
                zero_image_vector for _ in range(self.max_len - len(data))
            ]
            zeros_image_arr = [zero_elem_image for _ in range(
                self.max_len - len(data))]

        if not self.include_text and not self.include_item_categories:
            if self.image_data in ["original", "embedding"]:
                return zeros_image + data, style
            else:
                img_vecs = [x[0] for x in data]
                img_arrs = [x[1] for x in data]
                return zeros_image_vec + img_vecs, zeros_image_arr + img_arrs, style

        if not self.include_text and self.include_item_categories:
            if self.image_data in ["original", "embedding"]:
                return zeros_image + data, item_cat_data, style
            else:
                img_vecs = [x[0] for x in data]
                img_arrs = [x[1] for x in data]
                return (zeros_image_vec + img_vecs,
                        zeros_image_arr + img_arrs,
                        item_cat_data,
                        style)

        if self.include_text and self.include_item_categories:
            text_data = [x[0] for x in data]
            zero_elem_text = np.zeros(self.text_embedding_dim)
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
                    item_cat_data,
                    style
                )
            else:
                # only image embeddings
                if self.image_data in ["original", "embedding"]:
                    return (
                        zeros_image + image_data,
                        zeros_text + text_data,
                        item_cat_data,
                        style,
                    )
                elif self.image_data == "both":
                    return (
                        zeros_image_vec + image_data,
                        zeros_image_arr + image_array_data,
                        zeros_text + text_data,
                        item_cat_data,
                        style,
                    )

    def __get_label1(self, items, pad=0):
        # creates labels for item classification
        # padded witn -1 to maintain the same sequence length
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
        items = [x for x in example[1: self.max_len]]
        targets = items[1:] + ["eos"]
        data = []
        for t in targets:
            if t in batch_dict:
                data.append(batch_dict[t])
            else:
                data.append(batch_dict["unk"])
        if len(data) < self.max_len:
            data = [-1] * (self.max_len - len(data)) + data
        return data

    def __get_mask(self, example):
        # creates masks for sequential input
        # 1 for padded items and 0 for true items
        mask = [0.0 for x in example[: self.max_len]]
        if len(mask) < self.max_len:
            mask = [1.0] * (self.max_len - len(mask)) + mask
        return mask

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()

        # get all the items for this batch and create a
        # a label dict (dynamic)
        batch_items = set()
        for example in x_batch:
            batch_items |= set([x for x in example[1: self.max_len]])
        batch_items = list(batch_items)[: self.number_items_in_batch]
        batch_dict = {jj: ii for ii, jj in enumerate(batch_items)}
        batch_dict["unk"] = len(batch_items)
        batch_dict["eos"] = len(batch_items) + 1

        # mask = np.asarray([self.__get_mask(x) for x in x_batch])
        if not self.include_text and not self.include_item_categories:
            if self.image_data in ["original", "embedding"]:
                combined = [self.__get_input(x) for x in x_batch]
                X_batch = (np.asarray([x[0] for x in combined]),
                           np.asarray([x[1] for x in combined]),
                           )
            else:
                combined = [self.__get_input(x) for x in x_batch]
                X_batch = (np.asarray([x[0] for x in combined]),
                           np.asarray([x[1] for x in combined]),
                           np.asarray([x[2] for x in combined]),
                           )

        if not self.include_text and self.include_item_categories:
            if self.image_data in ["original", "embedding"]:
                combined = [self.__get_input(x) for x in x_batch]
                X_batch = (np.asarray([x[0] for x in combined]),
                           np.asarray([x[1] for x in combined]),
                           np.asarray([x[2] for x in combined]),
                           )
            else:
                combined = [self.__get_input(x) for x in x_batch]
                X_batch = (np.asarray([x[0] for x in combined]),
                           np.asarray([x[1] for x in combined]),
                           np.asarray([x[2] for x in combined]),
                           np.asarray([x[3] for x in combined]),
                           )

        if self.include_text and self.include_item_categories:
            # TODO: correct the number of returned values
            combined = [self.__get_input(x) for x in x_batch]
            if self.return_negative_samples:
                X_batch = (
                    np.asarray([x[0] for x in combined]),
                    np.asarray([x[1] for x in combined]),
                    np.asarray([x[2] for x in combined]),
                    np.asarray([x[3] for x in combined]),
                )
            else:
                if self.image_data == "both":
                    if self.include_item_categories:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            np.asarray([x[2] for x in combined]),
                            np.asarray([x[3] for x in combined]),
                            # mask,
                        )
                    else:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            np.asarray([x[2] for x in combined]),
                            # mask,
                        )

                elif self.image_data in ["embedding", "original"]:
                    if self.include_item_categories:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            np.asarray([x[2] for x in combined]),
                            # mask,
                        )
                    else:
                        X_batch = (
                            np.asarray([x[0] for x in combined]),
                            np.asarray([x[1] for x in combined]),
                            # mask,
                        )

        y_batch = np.asarray([int(y) for y in y_batch])
        if not self.return_item_categories and not self.return_negative_samples:
            y_total = y_batch

        if self.return_item_categories:
            y2_batch = np.asarray(
                [self.__get_label2(x, batch_dict) for x in x_batch])
            # y2_batch = np.asarray([self.__get_label1(x) for x in x_batch])
            y_total = [y_batch, y2_batch]

        if self.return_negative_samples:
            y3_batch = np.zeros((len(x_batch), 1))
            # y3_batch = y2_batch
            y_total.append(y3_batch)

        return X_batch, y_total

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)
        # return self.n // self.batch_size


class CSANetGen(tf.keras.utils.Sequence):
    """
    Generates triplets according to CSA-Net model
    1. an outfit with N items
    2. a positive (image, category) that can go well with the outfit
    3. a set of negative (image, category) that will not go well with the outfit

    For each outfit, we follow these steps:
    1> take the first item as the positive image
    2> take the rest of the items as an outfit
    3> for each item in the outfit sample an image from the same category -
        this constitutes the negative set
    """

    def __init__(
        self,
        positive_samples,
        item_description,
        **kwargs,
    ):
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.get_image_embedding = kwargs.get("get_image_embedding", True)
        self.image_embedding_file = kwargs.get("image_embedding_file", None)
        self.text_embedding_file = kwargs.get("text_embedding_file", None)
        self.max_example = kwargs.get("max_example", 100000)
        self.item_description = item_description
        self.image_embedding_dim = kwargs.get("image_embedding_dim", 1280)
        self.input_size = kwargs.get("input_size", (224, 224, 3))
        self.max_len = kwargs.get("max_items", 8)
        self.image_dir = kwargs.get("image_dir", None)
        self.mask_zero = kwargs.get("mask_zero", True)

        # build item->category dict
        # collect all items by categories
        self.items_by_category, self.item2cat = {}, {}
        self.category2id = {}
        if self.mask_zero:
            category_count = 1
            self.category_zero = 0
        else:
            category_count = 0
            self.category_zero = -1

        for item, desc in self.item_description.items():
            cat = desc["category_id"]

            # create a dict of category numbers
            if cat not in self.category2id:
                self.category2id[cat] = category_count
                category_count += 1

            self.item2cat[item] = cat
            if cat in self.items_by_category:
                self.items_by_category[cat].append(item)
            else:
                self.items_by_category[cat] = [item]
        print(f"Total {category_count} item categories")
        # print(self.category2id)

        if self.get_image_embedding:
            self.zero_elem_image = np.zeros(
                self.image_embedding_dim
            )  # np.zeros((1, 1280))
        else:
            self.zero_elem_image = np.zeros(self.input_size)

        # create all the training data
        X, y = [], []
        Z = []
        for outfit in tqdm(positive_samples):
            items = [o["item_id"] for o in outfit["items"]]
            pos_item = items[0]
            outfit_i = items[1:]
            neg_set, neg_cat, outfit_cat = [], [], []
            for item in outfit_i:
                neg = self.get_negative_samples(item)[0]
                neg_set.append(neg)
                outfit_cat.append(self.category2id[self.item2cat[item]])
                neg_cat.append(self.category2id[self.item2cat[neg]])
            X.append([outfit_i, pos_item, neg_set])
            y.append(0)
            Z.append(
                [outfit_cat, self.category2id[self.item2cat[pos_item]], neg_cat])
            if self.max_example and len(y) > self.max_example:
                break
        self.df = pd.DataFrame({"X": X, "y": y, "Z": Z})
        self.X_col = "X"
        self.y_col = "y"
        self.z_col = "Z"
        self.n = len(self.df)
        print(f"Total {self.n} examples")

        if self.get_image_embedding:
            # "effnet2_polyvore.pkl" - 1280 dimensional vector
            # "effnet_tuned_polyvore.pkl" - 1280 dimensional vector
            # "graphsage_dict_polyvore.pkl" - 50 dimension
            # "graphsage_dict2_polyvore.pkl" - 50 dimension
            with open(self.image_embedding_file, "rb") as fr:
                self.embedding_dict = pickle.load(fr)

        # "bert_polyvore.pkl" - 768 dimensional vector
        with open(self.text_embedding_file, "rb") as fr:
            self.text_embedding_dict = pickle.load(fr)

        # original data has all positives followed by all negatives
        self.on_epoch_end()

    def get_negative_samples(self, item_id, num_negative=1):
        # for a given item get another item from the same category
        item_cat = self.item2cat[item_id]
        item_pool = self.items_by_category[item_cat].copy()
        try:
            item_pool.remove(item_id)
        except:
            print("cannot find!!")
            print(item_id, item_cat)
            print(item_pool)
        neg_items = np.random.randint(
            low=0, high=len(item_pool), size=num_negative)
        neg_items = [item_pool[ii] for ii in neg_items]
        return neg_items

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_image(self, item_id):
        if self.get_image_embedding:
            return self.embedding_dict[item_id]
        image_path = os.path.join(self.image_dir, item_id + ".jpg")
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])
        ).numpy()
        image_arr /= 255.0
        # if self.get_image_embedding:
        # return tf.squeeze(self.model(tf.expand_dims(image_arr, 0)))
        return image_arr

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

    def __get_input(self, example, example_cat):
        outfit = example[0][: self.max_len]
        positive = example[1]
        negatives = example[2][: self.max_len]

        outfit_cat = example_cat[0][: self.max_len]
        positive_cat = example_cat[1]
        negatives_cat = example_cat[2][: self.max_len]

        outfit_data, negative_data = [], []
        for ii, jj in zip(outfit, negatives):
            outfit_data.append(self.get_image(ii))
            negative_data.append(self.get_image(jj))
        pos_data = self.get_image(positive)

        if len(outfit) < self.max_len:
            padding = [self.zero_elem_image for _ in range(
                self.max_len - len(outfit))]
            outfit_data = padding + outfit_data
            negative_data = padding + negative_data

            padding_2 = [self.category_zero] * (self.max_len - len(outfit))
            outfit_cat = padding_2 + outfit_cat
            negatives_cat = padding_2 + negatives_cat

        return (
            outfit_data,
            pos_data,
            negative_data,
            outfit_cat,
            positive_cat,
            negatives_cat,
        )

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()
        z_batch = batches["Z"].tolist()
        combined = [self.__get_input(x, z) for x, z in zip(x_batch, z_batch)]
        X_batch = (
            np.asarray([x[0] for x in combined]),
            np.asarray([x[1] for x in combined]),
            np.asarray([x[2] for x in combined]),
            np.asarray([x[3] for x in combined]),
            np.asarray([x[4] for x in combined]),
            np.asarray([x[5] for x in combined]),
        )

        y_batch = np.asarray([int(y) for y in y_batch])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


class TripletGen(tf.keras.utils.Sequence):
    """
    Generates a set of (anchor, positive, negative) items
    for training under triplet loss.

    Ref: Mariya I. Vasileva, Bryan A. Plummer, Krishna Dusad, Shreya Rajpal,
    Ranjitha Kumar, and David Forsyth. Learning type-aware embeddings for
    fashion copatibility. In ECCV, 2018.
    """

    def __init__(
        self,
        positive_samples,
        item_description,
        **kwargs,
    ):
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.get_image_embedding = kwargs.get("get_image_embedding", True)
        self.image_embedding_file = kwargs.get("image_embedding_file", None)
        self.text_embedding_file = kwargs.get("text_embedding_file", None)
        self.max_example = kwargs.get("max_example", 100000)
        self.item_description = item_description
        # build item->category dict
        # collect all items by categories
        self.items_by_category, self.item2cat = {}, {}
        for item, desc in self.item_description.items():
            cat = desc["category_id"]
            self.item2cat[item] = cat
            if cat in self.items_by_category:
                self.items_by_category[cat].append(item)
            else:
                self.items_by_category[cat] = [item]

        # for every item find the items that appeared together
        # in some outfit, to be used subsequently for negative sampling
        self.item_together_dict = {}
        for outfit in positive_samples:
            items = [o["item_id"] for o in outfit["items"]]
            for item in items:
                if item not in self.item_together_dict:
                    self.item_together_dict[item] = set()
                self.item_together_dict[item] |= set(items)

        # create all the training data
        X, y = [], []
        for outfit in positive_samples:
            items = [o["item_id"] for o in outfit["items"]]
            # combs = combinations(items, 2)  # too many items
            combs = [(items[ii], items[ii + 1])
                     for ii in range(len(items) - 1)]
            for itempair in combs:
                anchor = itempair[0]
                pos = itempair[1]
                neg = self.get_negative_samples(pos)[0]
                X.append([anchor, pos, neg])
                y.append(0)
            if self.max_example and len(y) > self.max_example:
                break
        self.df = pd.DataFrame({"X": X, "y": y})
        self.X_col = "X"
        self.y_col = "y"
        self.n = len(self.df)
        print(f"Total {self.n} examples")

        if self.get_image_embedding:
            # "effnet2_polyvore.pkl" - 1280 dimensional vector
            # "effnet_tuned_polyvore.pkl" - 1280 dimensional vector
            # "graphsage_dict_polyvore.pkl" - 50 dimension
            # "graphsage_dict2_polyvore.pkl" - 50 dimension
            with open(self.image_embedding_file, "rb") as fr:
                self.embedding_dict = pickle.load(fr)

        # "bert_polyvore.pkl" - 768 dimensional vector
        with open(self.text_embedding_file, "rb") as fr:
            self.text_embedding_dict = pickle.load(fr)

        # original data has all positives followed by all negatives
        self.on_epoch_end()

    def get_negative_samples(self, item_id, num_negative=1):
        # for a given item get another item from the same category
        item_cat = self.item2cat[item_id]
        item_pool = self.items_by_category[item_cat].copy()
        # make sure to sample items that never appeared together
        # in any outfit
        if self.item_together_dict:
            item_together = self.item_together_dict[item_id].copy()
            item_pool = set(item_pool) - set(item_together)
            item_pool = list(item_pool)

        if item_id in item_pool:
            item_pool.remove(item_id)

        assert len(
            item_pool) >= 1, f"too few items for Item-{item_id}({item_pool})"

        neg_items = np.random.randint(
            low=0, high=len(item_pool), size=num_negative)
        neg_items = [item_pool[ii] for ii in neg_items]
        return neg_items

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_image(self, item_id):
        if self.get_image_embedding:
            return self.embedding_dict[item_id]
        image_path = os.path.join(self.image_dir, item_id + ".jpg")
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])
        ).numpy()
        image_arr /= 255.0
        # if self.get_image_embedding:
        # return tf.squeeze(self.model(tf.expand_dims(image_arr, 0)))
        return image_arr

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

    def __get_input(self, example):
        text_data, image_data = [], []
        for item in example:
            image = self.get_image(item)
            text = self.get_texts(item)
            image_data.append(image)
            text_data.append(text)

        return image_data, text_data

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()
        combined = [self.__get_input(x) for x in x_batch]
        X_batch = (
            np.asarray([x[0] for x in combined]),
            np.asarray([x[1] for x in combined]),
        )

        y_batch = np.asarray([int(y) for y in y_batch])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


class KTupleGen(tf.keras.utils.Sequence):
    """
    Generates triplets according to a variation of the CSA-Net model
    1. an outfit with N items
    2. a positive (image, category) that can go well with the outfit
    3. a negative (image, category) that will not go well with the outfit

    Difference with CSA-Net
    4. either triplet loss or binary cross-entropy loss.
    5. each outfit has a fixed length
    6. there is only one negative image
    """

    def __init__(
        self,
        positive_samples,
        item_description,
        **kwargs,
    ):
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.get_image_embedding = kwargs.get("get_image_embedding", True)
        self.image_embedding_file = kwargs.get("image_embedding_file", None)
        self.text_embedding_file = kwargs.get("text_embedding_file", None)
        self.max_example = kwargs.get("max_example", 100000)
        self.item_description = item_description
        self.image_embedding_dim = kwargs.get("image_embedding_dim", 1280)
        self.input_size = kwargs.get("input_size", (224, 224, 3))
        self.image_dir = kwargs.get("image_dir", None)
        self.mask_zero = kwargs.get("mask_zero", False)
        self.max_outfit_length = kwargs.get("max_outfit_length", 8)
        self.loss_type = kwargs.get("loss_type", "triplet")
        self.hard_negative = kwargs.get("hard_negative", False)
        self.fraction = kwargs.get("fraction", 0.5)
        self.get_text_embedding = kwargs.get("get_text_embedding", True)

        # build item->category dict
        # collect all items by categories
        self.items_by_category, self.item2cat = {}, {}
        self.category2id = {}
        if self.mask_zero:
            category_count = 1
            self.category_zero = 0
        else:
            category_count = 0
            self.category_zero = -1

        for item, desc in self.item_description.items():
            cat = desc["category_id"]

            # create a dict of category numbers
            if cat not in self.category2id:
                self.category2id[cat] = category_count
                category_count += 1

            self.item2cat[item] = cat
            if cat in self.items_by_category:
                self.items_by_category[cat].append(item)
            else:
                self.items_by_category[cat] = [item]
        print(f"Total {category_count} item categories")
        # print(self.category2id)

        if self.hard_negative:
            # for every item find the items that appeared together
            # in some outfit, to be used subsequently for negative sampling
            self.item_together_dict = {}
            for outfit in positive_samples:
                items = [o["item_id"] for o in outfit["items"]]
                for item in items:
                    if item not in self.item_together_dict:
                        self.item_together_dict[item] = set()
                    self.item_together_dict[item] |= set(items)

        # create all the training data
        X, y = [], []
        Z = []
        for outfit in tqdm(positive_samples):
            items = [o["item_id"] for o in outfit["items"]]
            if len(items) >= self.max_outfit_length + 1:
                x_, y_, z_ = self.generate_samples_basic(items)
                X += x_.copy()
                y += y_.copy()
                Z += z_.copy()
                x_, y_, z_ = self.generate_samples_combinations(
                    items, self.fraction)
                X += x_.copy()
                y += y_.copy()
                Z += z_.copy()
            # for ii in range(self.max_outfit_length, len(items)):
            #     pos_item = items[ii]
            #     outfit_i = items[:self.max_outfit_length]
            #     neg_item = self.get_negative_samples(pos_item)[0]
            #     pos_cat = self.category2id[self.item2cat[pos_item]]
            #     neg_cat = self.category2id[self.item2cat[neg_item]]
            #     outfit_cat = [self.category2id[self.item2cat[item]]
            #                   for item in outfit_i]
            #     if "triplet" in self.loss_type.lower():
            #         X.append([outfit_i, pos_item, neg_item])
            #         y.append(0)
            #         Z.append([outfit_cat, pos_cat, neg_cat])
            #     elif "binary" in self.loss_type.lower():
            #         X.append([outfit_i, pos_item])
            #         y.append(1)
            #         Z.append([outfit_cat, pos_cat])
            #         X.append([outfit_i, neg_item])
            #         y.append(0)
            #         Z.append([outfit_cat, neg_cat])

            if self.max_example and len(y) > self.max_example:
                break
        self.df = pd.DataFrame({"X": X, "y": y, "Z": Z})
        self.X_col = "X"
        self.y_col = "y"
        self.z_col = "Z"
        self.n = len(self.df)
        print(f"Total {self.n} examples")

        if self.get_image_embedding:
            # "effnet2_polyvore.pkl" - 1280 dimensional vector
            # "effnet_tuned_polyvore.pkl" - 1280 dimensional vector
            # "graphsage_dict_polyvore.pkl" - 50 dimension
            # "graphsage_dict2_polyvore.pkl" - 50 dimension
            with open(self.image_embedding_file, "rb") as fr:
                self.embedding_dict = pickle.load(fr)

        # "bert_polyvore.pkl" - 768 dimensional vector
        with open(self.text_embedding_file, "rb") as fr:
            self.text_embedding_dict = pickle.load(fr)

        # original data has all positives followed by all negatives
        self.on_epoch_end()

    def generate_samples_basic(self, items):
        X, y, Z = [], [], []
        # strategy-1
        for ii in range(self.max_outfit_length, len(items)):
            pos_item = items[ii]
            outfit_i = items[: self.max_outfit_length]
            neg_item = self.get_negative_samples(pos_item)[0]
            pos_cat = self.category2id[self.item2cat[pos_item]]
            neg_cat = self.category2id[self.item2cat[neg_item]]
            outfit_cat = [self.category2id[self.item2cat[item]]
                          for item in outfit_i]
            if "triplet" in self.loss_type.lower():
                X.append([outfit_i, pos_item, neg_item])
                y.append(0)
                Z.append([outfit_cat, pos_cat, neg_cat])
            elif "binary" in self.loss_type.lower():
                X.append([outfit_i, pos_item])
                y.append(1)
                Z.append([outfit_cat, pos_cat])
                X.append([outfit_i, neg_item])
                y.append(0)
                Z.append([outfit_cat, neg_cat])
        return X, y, Z

    def generate_samples_combinations(self, items, ratio=0.5):
        X, y, Z = [], [], []
        # strategy-2 (generate multiple samples based on combinations)
        all_samples = list(combinations(items, self.max_outfit_length))
        sampled = random.sample(
            all_samples, k=math.floor(0.5 * len(all_samples)))
        for outfit_i in sampled:
            rest = [item for item in items if item not in outfit_i]
            if len(rest) < 1:
                continue
                # print(items)
                # print(all_samples)
                # print(sampled)
                # print(outfit_i)
                # print(rest)
                # sys.exit()
            pos_item = random.sample(rest, 1)[0]
            neg_item = self.get_negative_samples(pos_item)[0]
            pos_cat = self.category2id[self.item2cat[pos_item]]
            neg_cat = self.category2id[self.item2cat[neg_item]]
            outfit_cat = [self.category2id[self.item2cat[item]]
                          for item in outfit_i]
            if "triplet" in self.loss_type.lower():
                X.append([outfit_i, pos_item, neg_item])
                y.append(0)
                Z.append([outfit_cat, pos_cat, neg_cat])
            elif "binary" in self.loss_type.lower():
                X.append([outfit_i, pos_item])
                y.append(1)
                Z.append([outfit_cat, pos_cat])
                X.append([outfit_i, neg_item])
                y.append(0)
                Z.append([outfit_cat, neg_cat])
        return X, y, Z

    def get_negative_samples(self, item_id, num_negative=1):
        # for a given item get another item from the same category
        item_cat = self.item2cat[item_id]
        item_pool = self.items_by_category[item_cat].copy()

        # make sure to sample items that never appeared together
        # in any outfit
        if self.hard_negative:
            item_together = self.item_together_dict[item_id].copy()
            item_pool = set(item_pool) - set(item_together)
            item_pool = list(item_pool)

        if item_id in item_pool:
            item_pool.remove(item_id)

        assert len(
            item_pool) >= 1, f"too few items for Item-{item_id}({item_pool})"

        neg_items = np.random.randint(
            low=0, high=len(item_pool), size=num_negative)
        neg_items = [item_pool[ii] for ii in neg_items]
        return neg_items

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_image(self, item_id):
        if self.get_image_embedding:
            return self.embedding_dict[item_id]
        image_path = os.path.join(self.image_dir, item_id + ".jpg")
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])
        ).numpy()
        image_arr /= 255.0
        # if self.get_image_embedding:
        # return tf.squeeze(self.model(tf.expand_dims(image_arr, 0)))
        return image_arr

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

    def __get_input(self, example, example_cat):

        if len(example) == 2:
            # binary cross-entropy loss
            outfit = example[0]
            item = example[1]
            outfit_cat = example_cat[0]
            item_cat = example_cat[1]
            outfit_data = []
            for ii in outfit:
                image_ii = self.get_image(ii)
                if self.get_text_embedding:
                    text_ii = self.get_texts(ii)
                    outfit_data.append((image_ii, text_ii))
                else:
                    outfit_data.append(image_ii)

            item_image = self.get_image(item)
            if self.get_image_embedding:
                item_text = self.get_texts(item)
                item_data = (item_image, item_text)
            else:
                item_data = item_image
            return outfit_data, item_data, outfit_cat, item_cat

        # triplet loss
        outfit = example[0]
        positive = example[1]
        negative = example[2]

        outfit_cat = example_cat[0]
        positive_cat = example_cat[1]
        negative_cat = example_cat[2]

        outfit_data = []
        for ii in outfit:
            image_ii = self.get_image(ii)
            if self.get_text_embedding:
                text_ii = self.get_texts(ii)
                outfit_data.append((image_ii, text_ii))
            outfit_data.append(image_ii)

        neg_image = self.get_image(negative)
        pos_image = self.get_image(positive)
        if self.get_text_embedding:
            neg_text = self.get_texts(negative)
            pos_text = self.get_texts(positive)
            pos_data = (pos_image, pos_text)
            neg_data = (neg_image, neg_text)
        else:
            pos_data = pos_image
            neg_data = neg_image
        return outfit_data, pos_data, neg_data, outfit_cat, positive_cat, negative_cat

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()
        z_batch = batches["Z"].tolist()
        combined = [self.__get_input(x, z) for x, z in zip(x_batch, z_batch)]

        if "triplet" in self.loss_type.lower():
            X_batch = (
                np.asarray([x[0] for x in combined]),
                np.asarray([x[1] for x in combined]),
                np.asarray([x[2] for x in combined]),
                np.asarray([x[3] for x in combined]),
                np.asarray([x[4] for x in combined]),
                np.asarray([x[5] for x in combined]),
            )
        elif "binary" in self.loss_type.lower():
            if self.get_text_embedding:
                X_batch = (
                    np.asarray([x[0][0] for x in combined]),
                    np.asarray([x[0][1] for x in combined]),
                    np.asarray([x[1][0] for x in combined]),
                    np.asarray([x[1][1] for x in combined]),
                    np.asarray([x[2] for x in combined]),
                    np.asarray([x[3] for x in combined]),
                )
            else:
                X_batch = (
                    np.asarray([x[0] for x in combined]),
                    np.asarray([x[1] for x in combined]),
                    np.asarray([x[2] for x in combined]),
                    np.asarray([x[3] for x in combined]),
                )
        y_batch = np.asarray([int(y) for y in y_batch])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


class ImageDataGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        item_description,
        image_dir,
        batch_size,
        input_size=(224, 224, 3),
        shuffle=True,
        valid=False,
        valid_sample=0,
    ):

        self.image_dir = image_dir
        self.item_description = item_description
        X, y = [], []

        # there are some images that are not present in the json
        # for image_path in glob.glob(image_dir + "/*.jpg"):
        #     item_id = image_path.split("/")[-1].split(".")[0]
        #     X.append(item_id)
        #     y.append(self.item_description[item_id]["category_id"])

        for item_id in self.item_description:
            X.append(item_id)
            y.append(self.item_description[item_id]["category_id"])

        self.X_col = "X"
        self.y_col = "y"
        self.df = pd.DataFrame({self.X_col: X, self.y_col: y})
        if valid:
            self.df = self.df.sample(n=valid_sample)

        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.num_classes = self.df[self.y_col].nunique()
        categories = self.df[self.y_col].unique()
        self.label_dict = {jj: ii for ii, jj in enumerate(categories)}
        print(f"Total {self.n} images with {self.num_classes} classes")

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_texts(self, item_id):
        item = self.item_description[item_id]  # item attributes
        return " ".join(
            [
                item["url_name"],
                item["description"],
                item["title"],
                item["semantic_category"],
            ]
        )

    def get_image(self, item_id):
        image_path = os.path.join(self.image_dir, item_id + ".jpg")
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])
        ).numpy()
        image_arr /= 255.0
        return image_arr

    def __get_input(self, item):
        return self.get_image(item)

    def __get_output(self, label):
        return self.label_dict[label]

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = np.asarray(
            [self.__get_input(x) for x in batches[self.X_col].tolist()]
        )
        y_batch = np.asarray(
            [self.__get_output(y) for y in batches[self.y_col].tolist()]
        )

        return x_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)
        # return self.n // self.batch_size


class OutfitGen(tf.keras.utils.Sequence):
    """
    This generator creates sample outfits using a query item (list)
    and adding one item at a time from a list of common items. The
    outfit thus created is evaluated by an existing model and the
    outfit score is utilized further for creating the best outfit.
    """

    def __init__(
        self,
        embed_dir,
        image_embed_file,
        text_embed_file,
        batch_size,
        max_len,
        image_embedding_dim,
        query_item,
        shuffle=False,
    ):

        image_embedding_file = os.path.join(embed_dir, image_embed_file)
        with open(image_embedding_file, "rb") as fr:
            self.image_embedding_dict = pickle.load(fr)

        text_embedding_file = os.path.join(embed_dir, text_embed_file)
        with open(text_embedding_file, "rb") as fr:
            self.text_embedding_dict = pickle.load(fr)

        common_items = set(self.image_embedding_dict.keys()).intersection(
            self.text_embedding_dict.keys()
        )

        if type(query_item) is not list:
            query_item = [query_item]

        X, y = [], []
        for item in common_items:
            if item not in query_item:
                X.append([item] + query_item)
                y.append(item)

        self.X_col = "X"
        self.y_col = "y"
        self.df = pd.DataFrame({self.X_col: X, self.y_col: y})
        self.batch_size = batch_size
        self.max_len = max_len
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = 768
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

    def get_image(self, item_id):
        return self.image_embedding_dict[item_id]

    def __get_input(self, example):
        data = []
        items = [x for x in example[: self.max_len]]
        for item in items:
            image = self.get_image(item)
            text = self.get_texts(item)
            data.append((text, image))

        text_data = [x[0] for x in data]
        image_data = [x[1] for x in data]
        zero_elem_image = np.zeros(
            self.image_embedding_dim)  # np.zeros((1, 1280))
        zero_elem_text = np.zeros(self.text_embedding_dim)
        zeros_image = [zero_elem_image for _ in range(
            self.max_len - len(data))]
        zeros_text = [zero_elem_text for _ in range(self.max_len - len(data))]

        return (zeros_image + image_data, zeros_text + text_data)

    def __get_output(self, label):
        return self.label_dict[label]

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()

        combined = [self.__get_input(x) for x in x_batch]
        X_batch = (
            np.asarray([x[0] for x in combined]),
            np.asarray([x[1] for x in combined]),
            # mask,
        )
        y_batch = np.asarray([int(y) for y in y_batch])

        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)
        # return self.n // self.batch_size


class OutfitGenWithImage(tf.keras.utils.Sequence):
    """
    This generator creates sample outfits using a query item (list)
    and adding one item at a time from a list of common items. The
    outfit thus created is evaluated by an existing model and the
    outfit score is utilized further for creating the best outfit.

    The difference with OutfitGen is here the query item is an image/numpy array.
    Also, 
        1. we can pass an image embedding dict
        2. item description dict for category information
        3. segmentation of items based on category for reduced search space
        4. categories that can be ignored for reduced search space 
    """

    def __init__(self, query_item, **kwargs):
        embed_dir = kwargs.get("embed_dir", None)
        image_embed_file = kwargs.get("image_embed_file", None)
        text_embed_file = kwargs.get("text_embed_file", None)
        batch_size = kwargs.get("batch_size", 32)
        max_len = kwargs.get("max_len", 8)
        image_embedding_dim = kwargs.get("image_embedding_dim", 1280)
        shuffle = kwargs.get("shuffle", False)
        include_text = kwargs.get("include_text", False)
        include_category = kwargs.get("include_category", False)
        image_embedding_dict = kwargs.get("image_embedding_dict", None)
        item_description_dict = kwargs.get("item_description", None)
        item_category_dict = kwargs.get("item_category_dict", None)
        ignore_categories = kwargs.get("ignore_categories", None)
        search_only_categories = kwargs.get("search_only_categories", None)

        if image_embedding_dict:
            self.image_embedding_dict = image_embedding_dict

        elif image_embedding_file:
            image_embedding_file = os.path.join(embed_dir, image_embed_file)
            with open(image_embedding_file, "rb") as fr:
                self.image_embedding_dict = pickle.load(fr)
        common_items = set(self.image_embedding_dict.keys())
        n0 = len(common_items)

        if include_text:
            text_embedding_file = os.path.join(embed_dir, text_embed_file)
            with open(text_embedding_file, "rb") as fr:
                self.text_embedding_dict = pickle.load(fr)

            common_items = common_items.intersection(
                self.text_embedding_dict.keys())

        # since we filter by item category we need to ensure every item
        # has a category
        assert (
            item_description_dict is not None
        ), "Item-description-dict must be supplied!"
        s_cat = set(item_description_dict.keys())
        common_items = common_items.intersection(s_cat)

        # remove items to be searched based on categories
        if ignore_categories is not None:
            for cat in ignore_categories:
                common_items = common_items.difference(item_category_dict[cat])
        elif search_only_categories is not None:
            common_items = set()
            print(f"Searching limited to {search_only_categories}")
            for cat in search_only_categories:
                common_items |= set(item_category_dict[cat])

        print(f"Original {n0} items, reduced to {len(common_items)} items")

        if type(query_item) is not list:
            query_item = [query_item]

        # Rewrite the query as one of them could be a TF Tensor/Numpy array
        new_ids = []
        for item in query_item:
            if type(item) is not str:
                img_id = uuid.uuid4()
                self.image_embedding_dict[img_id] = item
                new_ids.append(img_id)
            else:
                new_ids.append(item)

        X, y = [], []
        for item in common_items:
            if item not in new_ids:
                X.append([item] + new_ids)
                y.append(item)

        self.X_col = "X"
        self.y_col = "y"
        self.df = pd.DataFrame({self.X_col: X, self.y_col: y})
        self.batch_size = batch_size
        self.max_len = max_len
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = 768
        self.shuffle = shuffle
        self.n = len(self.df)
        self.include_text = include_text
        self.include_category = include_category

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

    def get_image(self, item_id):
        return self.image_embedding_dict[item_id]

    def __get_input(self, example):
        data = []
        items = [x for x in example[: self.max_len]]
        for item in items:
            image = self.get_image(item)
            if self.include_text:
                text = self.get_texts(item)
                data.append((text, image))
            else:
                data.append(image)

        zero_elem_image = np.zeros(
            self.image_embedding_dim)  # np.zeros((1, 1280))
        zeros_image = [zero_elem_image for _ in range(
            self.max_len - len(data))]
        if self.include_text:
            text_data = [x[0] for x in data]
            image_data = [x[1] for x in data]
            zero_elem_text = np.zeros(self.text_embedding_dim)
            zeros_text = [zero_elem_text for _ in range(
                self.max_len - len(data))]
            return (zeros_image + image_data, zeros_text + text_data)
        else:
            return zeros_image + data

    def __get_output(self, label):
        return self.label_dict[label]

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()

        combined = [self.__get_input(x) for x in x_batch]
        if self.include_text:
            X_batch = (
                np.asarray([x[0] for x in combined]),
                np.asarray([x[1] for x in combined]),
                # mask,
            )
        else:
            X_batch = np.asarray(combined)

        y_batch = np.asarray([int(y) for y in y_batch])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)
        # return self.n // self.batch_size


class ZalandoOutfitGenWithImage(tf.keras.utils.Sequence):
    """
    This generator creates sample outfits using a query item (list)
    and adding one item at a time from a list of common items. The
    outfit thus created is evaluated by an existing model and the
    outfit score is utilized further for creating the best outfit.

    Specifically for Zalando data which is slightly different from
    Polyvore.
    """

    def __init__(self, query_item, **kwargs):
        embed_dir = kwargs.get("embed_dir", None)
        image_embed_file = kwargs.get("image_embed_file", None)
        text_embed_file = kwargs.get("text_embed_file", None)
        batch_size = kwargs.get("batch_size", 32)
        max_len = kwargs.get("max_len", 8)
        image_embedding_dim = kwargs.get("image_embedding_dim", 1280)
        shuffle = kwargs.get("shuffle", False)
        include_text = kwargs.get("include_text", False)
        include_category = kwargs.get("include_category", False)
        image_embedding_dict = kwargs.get("image_embedding_dict", None)
        item_description_dict = kwargs.get("item_description", None)
        item_category_dict = kwargs.get("item_category_dict", None)
        ignore_categories = kwargs.get("ignore_categories", None)
        search_only_categories = kwargs.get("search_only_categories", None)
        outfit_style = kwargs.get("outfit_style", None)
        self.include_item_categories = kwargs.get(
            "include_item_categories", False)
        self.label_dict = kwargs.get("label_dict", None)

        if image_embedding_dict:
            self.image_embedding_dict = image_embedding_dict

        elif image_embedding_file:
            image_embedding_file = os.path.join(embed_dir, image_embed_file)
            with open(image_embedding_file, "rb") as fr:
                self.image_embedding_dict = pickle.load(fr)
        common_items = set(self.image_embedding_dict.keys())
        n0 = len(common_items)

        if include_text:
            text_embedding_file = os.path.join(embed_dir, text_embed_file)
            with open(text_embedding_file, "rb") as fr:
                self.text_embedding_dict = pickle.load(fr)

            common_items = common_items.intersection(
                self.text_embedding_dict.keys())

        # since we filter by item category we need to ensure every item
        # has a category
        assert (
            item_description_dict is not None
        ), "Item-description-dict must be supplied!"
        s_cat = set(item_description_dict.keys())
        common_items = common_items.intersection(s_cat)
        self.item_description = item_description_dict

        # remove items to be searched based on categories
        if ignore_categories is not None:
            for cat in ignore_categories:
                common_items = common_items.difference(item_category_dict[cat])
        elif search_only_categories is not None:
            common_items = set()
            print(f"Searching limited to {search_only_categories}")
            for cat in search_only_categories:
                common_items |= set(item_category_dict[cat])

        print(f"Original {n0} items, reduced to {len(common_items)} items")

        if type(query_item) is not list:
            query_item = [query_item]

        # if type(query_cats) is not list:
        #     query_cats = [query_cats]

        # Rewrite the query as one of them could be a TF Tensor/Numpy array
        new_ids = []
        for item in query_item:
            if type(item) is not str:
                img_id = uuid.uuid4()
                self.image_embedding_dict[img_id] = item
                new_ids.append(img_id)
                # self.item_description[img_id] = {"category_id": cat}
            else:
                new_ids.append(item)

        X, y = [], []
        for item in common_items:
            if item not in new_ids:
                X.append([int(outfit_style)] + [item] + new_ids)
                y.append(item)

        self.X_col = "X"
        self.y_col = "y"
        self.df = pd.DataFrame({self.X_col: X, self.y_col: y})
        self.batch_size = batch_size
        self.max_len = max_len
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = 768
        self.shuffle = shuffle
        self.n = len(self.df)
        self.include_text = include_text
        self.include_category = include_category

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

    def get_image(self, item_id):
        # TODO: rerun image embedding with the full filename as key
        # return self.image_embedding_dict[item_id]
        current_name = item_id.split('.')[0]
        return self.image_embedding_dict[current_name]

    def __get_input(self, example):
        data = []
        style = int(example[0])
        items = [x for x in example[1: self.max_len]]
        for item in items:
            image = self.get_image(item)
            if self.include_text:
                text = self.get_texts(item)
                data.append((text, image))
            else:
                data.append(image)

        if self.include_item_categories:
            item_cat_data = self.__get_label1(items)

        zero_elem_image = np.zeros(
            self.image_embedding_dim)  # np.zeros((1, 1280))
        zeros_image = [zero_elem_image for _ in range(
            self.max_len - len(data))]
        if self.include_text:
            text_data = [x[0] for x in data]
            image_data = [x[1] for x in data]
            zero_elem_text = np.zeros(self.text_embedding_dim)
            zeros_text = [zero_elem_text for _ in range(
                self.max_len - len(data))]
            return (zeros_image + image_data, zeros_text + text_data)
        else:
            if self.include_item_categories:
                return zeros_image + data, item_cat_data, style
            else:
                return zeros_image + data, style

    def __get_label1(self, items, pad=0):
        # creates labels for item classification
        # padded witn -1 to maintain the same sequence length
        data = []
        for item in items:
            data.append(
                self.label_dict[self.item_description[item]["category_id"]])
        if len(data) < self.max_len:
            data = [pad] * (self.max_len - len(data)) + data
        return data

    def __get_output(self, label):
        return self.label_dict[label]

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()

        combined = [self.__get_input(x) for x in x_batch]
        if self.include_text:
            # TODO: incomplete
            X_batch = (
                np.asarray([x[0] for x in combined]),
                np.asarray([x[1] for x in combined]),
            )
        else:
            if self.include_item_categories:
                X_batch = (
                    np.asarray([x[0] for x in combined]),
                    np.asarray([x[1] for x in combined]),
                    np.asarray([x[2] for x in combined]),
                )
            else:
                X_batch = (
                    np.asarray([x[0] for x in combined]),
                    np.asarray([x[1] for x in combined]),
                )

        y_batch = np.asarray([y for y in y_batch])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index *
                          self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return math.ceil(self.n / self.batch_size)
        # return self.n // self.batch_size
