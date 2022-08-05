# import effnetv2_model
import glob
import json
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import pandas as pd
import pickle
import math
import numpy as np
import sys
from transformers import BertTokenizer, BertModel

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
        image_dir,
        batch_size,
        max_len=16,
        input_size=(224, 224, 3),
        only_image=True,
        image_embedding=True,
        image_embedding_dim=1280,
        image_embedding_file=None,
        text_embedding_file=None,
        return_item_categories=False,
        return_negative_samples=False,
        number_negative_samples=0,
        number_items_in_batch=None,
        variable_length_input=True,
        label_dict=None,
        shuffle=True,
    ):
        self.df = pd.DataFrame({"X": X, "y": y})
        # self.X = self.df["X"].tolist()
        # self.y = self.df["y"].tolist()
        self.X_col = "X"
        self.y_col = "y"
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.max_len = max_len
        self.only_image = only_image
        self.item_dict = item_dict
        self.item_description = item_description
        self.image_dir = image_dir
        self.get_image_embedding = image_embedding
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = 768
        self.image_embedding_file = image_embedding_file
        self.return_item_categories = return_item_categories
        self.return_negative_samples = return_negative_samples
        self.number_negative_samples = number_negative_samples
        self.label_dict = label_dict
        self.number_items_in_batch = number_items_in_batch
        self.variable_length_input = variable_length_input

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

        if self.get_image_embedding:
            # "effnet2_polyvore.pkl" - 1280 dimensional vector
            # "effnet_tuned_polyvore.pkl" - 1280 dimensional vector
            # "graphsage_dict_polyvore.pkl" - 50 dimension
            # "graphsage_dict2_polyvore.pkl" - 50 dimension
            with open(image_embedding_file, "rb") as fr:
                self.embedding_dict = pickle.load(fr)

            # self.model = tf.keras.models.Sequential(
            #     [
            #         tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
            #         effnetv2_model.get_model("efficientnetv2-b0", include_top=False),
            #     ]
            # )

        if not only_image:
            # "bert_polyvore.pkl" - 768 dimensional vector
            with open(text_embedding_file, "rb") as fr:
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

        if self.get_image_embedding:
            zero_elem_image = np.zeros(
                self.image_embedding_dim)  # np.zeros((1, 1280))
        else:
            zero_elem_image = np.zeros(self.input_size)

        zeros_image = [zero_elem_image for _ in range(
            self.max_len - len(data))]
        if self.only_image:
            return zeros_image + data
        else:
            text_data = [x[0] for x in data]
            image_data = [x[1] for x in data]
            zero_elem_text = np.zeros(self.text_embedding_dim)
            zeros_text = [zero_elem_text for _ in range(
                self.max_len - len(data))]

            if not self.return_negative_samples:
                return (zeros_image + image_data, zeros_text + text_data)
            else:
                return (
                    zeros_image + image_data,
                    zeros_text + text_data,
                    nimage_data,
                    ntext_data,
                )

    def __get_label1(self, example):
        # creates labels for item classification
        # padded witn -1 to maintain the same sequence length
        items = [self.item_dict[x] for x in example[: self.max_len]]
        data = []
        for item in items:
            data.append(
                self.label_dict[self.item_description[item]["category_id"]])
        if len(data) < self.max_len:
            data = [-1] * (self.max_len - len(data)) + data
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
        batch_items = list(batch_items)[:self.number_items_in_batch]
        batch_dict = {jj: ii for ii, jj in enumerate(batch_items)}
        batch_dict['unk'] = len(batch_items)
        batch_dict['eos'] = len(batch_items) + 1

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
            self.text_embedding_dict.keys())

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
        zeros_text = [zero_elem_text for _ in range(
            self.max_len - len(data))]

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


# if __name__ == "__main__":
#     with open(os.path.join(train_dir, train_json), "r") as fr:
#         train_pos = json.load(fr)

#     with open(os.path.join(base_dir, item_file), "r") as fr:
#         item_desc = json.load(fr)

#     with open(os.path.join(base_dir, outfit_file), "r") as fr:
#         outfit_desc = json.load(fr)

#     with open(os.path.join(train_dir, train_file), "r") as fr:
#         train_X, train_y = [], []
#         for line in fr:
#             elems = line.strip().split()
#             train_y.append(elems[0])
#             train_X.append(elems[1:])
#     min_len, max_len = min([len(x) for x in train_X]), max([len(x) for x in train_X])

#     print(
#         f"Read {len(train_pos)} positive samples and {len(train_X)} training examples"
#     )
#     print(f"Minimum {min_len} and maximum {max_len} number of items in outfits")

#     # Create a dict to map train item_ids to their original ids
#     item_dict = {}
#     for ii, outfit in enumerate(train_pos):
#         items = outfit["items"]
#         mapped = train_X[ii]
#         item_dict.update({jj: kk["item_id"] for jj, kk in zip(mapped, items)})

#     x = get_examples(train_X[0], item_dict, item_desc, max_len)
#     print(x)
