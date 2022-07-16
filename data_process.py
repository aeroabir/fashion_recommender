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
import effnetv2_model

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
        shuffle=True,
    ):
        self.df = pd.DataFrame({"X": X, "y": y})
        self.X = self.df["X"].tolist()
        self.y = self.df["y"].tolist()
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
        self.image_embedding_dim = 1280
        self.text_embedding_dim = 768

        if self.get_image_embedding:
            with open("effnet2_polyvore.pkl", "rb") as fr:
                self.embedding_dict = pickle.load(fr)

            # self.model = tf.keras.models.Sequential(
            #     [
            #         tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
            #         effnetv2_model.get_model("efficientnetv2-b0", include_top=False),
            #     ]
            # )

        if not only_image:
            with open("bert_polyvore.pkl", "rb") as fr:
                self.text_embedding_dict = pickle.load(fr)

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

    def __get_input(self, example):
        data = []
        items = [self.item_dict[x] for x in example]
        for item in items:
            text = self.get_texts(item)
            image = self.get_image(item)
            if self.only_image:
                data.append(image)
            else:
                data.append((text, image))

        if self.get_image_embedding:
            zero_elem_image = np.zeros(self.image_embedding_dim)  # np.zeros((1, 1280))
        else:
            zero_elem_image = np.zeros(self.input_size)

        zeros_image = [zero_elem_image for _ in range(self.max_len - len(data))]
        if self.only_image:
            return zeros_image + data
        else:
            text_data = [x[0] for x in data]
            image_data = [x[1] for x in data]
            zero_elem_text = np.zeros(self.text_embedding_dim)
            zeros_text = [zero_elem_text for _ in range(self.max_len - len(data))]
            return (zeros_image + image_data, zeros_text + text_data)

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        x_batch = batches["X"].tolist()
        y_batch = batches["y"].tolist()

        #         batch_X, batch_Y = [], []
        #         for x, y in zip(x_batch, y_batch):
        #             x_num = self.__get_input(x)
        #             batch_X.append(np.vstack(x_num))
        #             batch_Y.append(int(y))

        #         print(batch_X, batch_Y)
        if self.only_image:
            X_batch = np.asarray([self.__get_input(x) for x in x_batch])
        else:
            x1x2 = [self.__get_input(x) for x in x_batch]
            X_batch = (
                np.asarray([x[0] for x in x1x2]),
                np.asarray([x[1] for x in x1x2]),
            )
        y_batch = np.asarray([int(y) for y in y_batch])

        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size : (index + 1) * self.batch_size]
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
        batches = self.df[index * self.batch_size : (index + 1) * self.batch_size]
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
