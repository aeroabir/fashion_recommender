import os
import json
from collections import Counter
import glob

from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import time
from tqdm import tqdm
import pickle
import numpy as np
import wandb
from wandb.keras import WandbCallback

wandb.init(project="finetune-efficientnet")
import sys

sys.path.insert(0, "/recsys_data/RecSys/fashion/automl/efficientnetv2")
import effnetv2_model
from data_process import ImageDataGen

if __name__ == "__main__":

    base_dir = "/recsys_data/RecSys/fashion/polyvore-dataset/polyvore_outfits"
    train_dir = os.path.join(base_dir, "disjoint")
    image_dir = os.path.join(base_dir, "images")
    train_json = "train.json"
    train_file = "compatibility_train.txt"
    item_file = "polyvore_item_metadata.json"
    outfit_file = "polyvore_outfit_titles.json"
    batch_size = 128
    learning_rate = 1.0e-03
    if len(sys.argv) > 1:
        task_type = sys.argv[1]
    else:
        task_type = "inference"

    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    else:
        epochs = 10
    patience = 10
    monitor = "val_loss"
    num_train = 251008
    model_name = "efficientnet"
    checkpoint_filepath = base_dir + "/checkpoint"

    with open(os.path.join(base_dir, item_file), "r") as fr:
        pv_items = json.load(fr)
    print(f"There are {len(pv_items)} items")

    train_gen = ImageDataGen(pv_items, image_dir, batch_size=batch_size)
    valid_gen = ImageDataGen(
        pv_items, image_dir, batch_size=batch_size, valid=True, valid_sample=10000
    )

    if task_type == "train":

        if model_name == "efficientnet":
            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
                    effnetv2_model.get_model("efficientnetv2-b0", include_top=False),
                    tf.keras.layers.Dropout(rate=0.2),
                    tf.keras.layers.Dense(153, activation="softmax"),
                ]
            )

        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"
        )
        model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
        callback = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,
            patience=patience,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        wandb_callback = WandbCallback(
            monitor=monitor, verbose=0, save_model=(False), mode="auto"
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        tic = time.time()
        history = model.fit(
            train_gen,
            epochs=epochs,
            batch_size=batch_size,
            #                     steps_per_epoch=math.ceil(251008./batch_size),
            # validation_data=valid_gen,
            # callbacks=[callback, model_checkpoint_callback, wandb_callback],
            verbose=1,
        )
        print("Time taken to train:", time.time() - tic)
        model.save(f"finetuned_{model_name}")

    elif task_type == "retrain":
        model = tf.keras.models.load_model(f"finetuned_{model_name}")
        tic = time.time()
        history = model.fit(
            train_gen,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        print("Time taken to re-train:", time.time() - tic)
        model.save(f"finetuned_{model_name}")

    elif task_type == "inference":
        model = tf.keras.models.load_model(f"finetuned_{model_name}")
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        acc_list = []
        pbar = tqdm(range(len(train_gen)))
        for ii in pbar:
            x, y = train_gen[ii]  # batch size
            yhat = model(x)
            m.update_state(y, yhat)
            batch_acc = m.result().numpy()
            acc_list.append(batch_acc)
            pbar.set_description("Batch accuracy %g" % batch_acc)
        print(f"Average accuracy: {np.mean(acc_list)}")

    elif task_type == "generate-embeddings":
        model = tf.keras.models.load_model(f"finetuned_{model_name}")
        new_model = tf.keras.models.Sequential(
            [tf.keras.layers.InputLayer(input_shape=[224, 224, 3]), model.layers[0]]
        )
        print(new_model.summary())

        efficient_net_dict = {}
        for image_path in tqdm(glob.glob(image_dir + "/*.jpg")):
            item_id = image_path.split("/")[-1].split(".")[0]
            image = tf.keras.preprocessing.image.load_img(image_path)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            image_arr = tf.image.resize(image_arr, (224, 224)).numpy()
            image_arr /= 255.0
            image_embed = tf.squeeze(new_model(tf.expand_dims(image_arr, 0)))
            efficient_net_dict[item_id] = image_embed

        with open("effnet_tuned_polyvore.pkl", "wb") as output_file:
            pickle.dump(efficient_net_dict, output_file)
