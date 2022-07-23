from datetime import datetime
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import auc
from sklearn.metrics import roc_auc_score, accuracy_score
import time

class CustomDataset(torch_data.Dataset):
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

        if not only_image:
            with open("bert_polyvore.pkl", "rb") as fr:
                self.text_embedding_dict = pickle.load(fr)

    def get_texts(self, item_id):
        return self.text_embedding_dict[item_id]

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
            return np.asarray(zeros_image + data)
        else:
            text_data = [x[0] for x in data]
            image_data = [x[1] for x in data]
            zero_elem_text = np.zeros(self.text_embedding_dim)
            zeros_text = [zero_elem_text for _ in range(self.max_len - len(data))]
            return np.asarray(zeros_image + image_data), np.asarray(zeros_text + text_data)

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
            x1x2 = self.__get_input(x)
            X = (torch.tensor(x1x2[0]).type(torch.float),
                 torch.tensor(x1x2[1]).type(torch.float))

        return X, y

    def __len__(self):
        return self.n

def train(model, train_set, valid_set, device, **kwargs):

    epochs = kwargs.get("epoch", 100)
    batch_size = kwargs.get("batch_size", 32)
    learning_rate = kwargs.get("learning_rate", 1e-03)
    decay_rate = kwargs.get("decay_rate", 0.5)
    exponential_decay_step = kwargs.get("exponential_decay_step", 5)
    validate_freq = kwargs.get("validate_freq", 1)
    early_stop = kwargs.get("early_stop", True)
    early_stop_step = kwargs.get("patience", 10)
    observe = kwargs.get("observe", "auc")
    optimizer = kwargs.get("optimizer", "adam")

    forecast_loss = nn.BCELoss(reduction="mean").to(device)

    if optimizer == "adam":
        my_optim = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )
    elif optimizer == "rmsprop":
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
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

    dir_name = "indv"
    log_dir = "logs/fit/" + dir_name + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            if type(inputs) is list:
                inputs = [inp.to(device) for inp in inputs]
            else:
                inputs = inputs.to(device)

            target = target.to(device)
            target = torch.unsqueeze(target, -1)
            model.zero_grad()
            forecast, _ = model(inputs)
            loss = forecast_loss(forecast, target)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)

        # save_model(model, result_file, epoch)
        # if (epoch + 1) % exponential_decay_step == 0:
        #     my_lr_scheduler.step()
        if (epoch + 1) % validate_freq == 0:
            is_best_for_now = False
            performance_metrics = validate(
                model,
                valid_loader,
                device,
                result_file=None,
            )
            writer.add_scalar("epoch_loss", loss_total / cnt, epoch)
            writer.add_scalar("epoch_acc", performance_metrics["acc"], epoch)
            writer.add_scalar("epoch_auc", performance_metrics["auc"], epoch)
            if best_validate_mae > performance_metrics[observe]:
                best_validate_mae = performance_metrics[observe]
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
                if validate_score_non_decrease_count == 5:
                    my_lr_scheduler.step()
            # save model
        #         if is_best_for_now:
        #             save_model(model, result_file)
        # early stop
        print(
            "| Epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | val-ACC {:5.4f} | val-AUC {:5.4f} ({:2d})".format(
                epoch,
                (time.time() - epoch_start_time),
                loss_total / cnt,
                performance_metrics["acc"],
                performance_metrics["auc"],
                validate_score_non_decrease_count,
            )
        )
        if early_stop and validate_score_non_decrease_count >= early_stop_step:
            break
    return performance_metrics


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


def inference(model, dataloader, device):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            if type(inputs) is list:
                inputs = [inp.to(device) for inp in inputs]
            else:
                inputs = inputs.to(device)
            target = target.to(device)
            # step = 0
            forecast_steps, _ = model(inputs)
            forecast_set.append(forecast_steps.detach().cpu().numpy())
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, dataloader, device, result_file=None):
    # start = datetime.now()
    forecast, target = inference(model, dataloader, device)
    scores = evaluate(target, forecast)
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f"{result_file}/target.csv", forcasting_2d_target, delimiter=",")
        np.savetxt(f"{result_file}/predict.csv", forcasting_2d, delimiter=",")
        np.savetxt(
            f"{result_file}/predict_abs_error.csv",
            np.abs(forcasting_2d - forcasting_2d_target),
            delimiter=",",
        )
        np.savetxt(
            f"{result_file}/predict_ape.csv",
            np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target),
            delimiter=",",
        )

    return {"acc": scores[0], "auc": scores[1]}

