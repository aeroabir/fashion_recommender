import torch
import torch.nn as nn
import numpy as np
import Resnet_18
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score)

def freeze_model_layers(model, freeze_layers=1):
    """https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch"""
    child_counter = 0
    for child in model.children():
        if child_counter < freeze_layers:
            print("child ",child_counter," was frozen")
            for param in child.parameters():
                param.requires_grad = False
        elif child_counter == freeze_layers:
            children_of_child_counter = 0
            for children_of_child in child.children():
                if children_of_child_counter < 1:
                    for param in children_of_child.parameters():
                        param.requires_grad = False
                        print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
                else:
                    print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
                children_of_child_counter += 1
        else:
            print("child ",child_counter," was not frozen")
        child_counter += 1
    return model


class SimpleProtoTypeModel(nn.Module):
    """
    This model returns only the embedding. The loss function
    would take care of the creation of prototypes and distance
    computation.
    """

    def __init__(self, **kwargs):
        super(SimpleProtoTypeModel, self).__init__()

        d_model = kwargs.get("d_model", 128)
        image_encoder = kwargs.get("image_encoder", "resnet18")
        self.freeze_layers = kwargs.get("freeze_layers", 0)
        image_embed_dim = kwargs.get("image_embed_dim", 1280)
        text_embed_dim = kwargs.get("text_embed_dim", 768)
        num_categories = kwargs.get("num_categories", 154)
        num_layers = kwargs.get("num_layers", 2)
        num_heads = kwargs.get("num_heads", 8)
        dff = kwargs.get("dff", 128)
        rate = kwargs.get("rate", 0.1)
        encoder_activation = kwargs.get("encoder_activation", "relu")
        self.image_data_type = kwargs.get("image_data_type", "embedding")
        self.include_text = kwargs.get("include_text", False)
        self.include_item_categories = kwargs.get("include_item_categories", False)
        self.use_rnn = kwargs.get("use_rnn", False)

        embedding_dim = kwargs.get("embedding_dim", 64)
        max_seq_len = kwargs.get("max_seq_len", 12)
        device = kwargs.get("device", "cpu")

        self.seq_len = max_seq_len
        d_model_trfmr = d_model
        if self.image_data_type in ["original", "both"]:
            if image_encoder == "resnet18":
                self.image_embedder = Resnet_18.resnet18(
                    pretrained=True, embedding_size=d_model)
                if self.freeze_layers > 0:
                    self.image_embedder = freeze_model_layers(self.image_embedder,
                        freeze_layers=self.freeze_layers)

        if self.image_data_type in ["embedding", "both"]:
            self.image_projector = nn.Sequential(
                nn.Linear(image_embed_dim, d_model),
                nn.Tanh())

        if self.include_text:
            self.text_projector = nn.Sequential(
                nn.Linear(text_embed_dim, d_model),
                nn.Tanh())
            d_model_trfmr += d_model

        if self.include_item_categories:
            self.category_embedder = torch.nn.Embedding(
                num_embeddings=num_categories, embedding_dim=d_model, padding_idx=0)
            d_model_trfmr += d_model

        self.d_model_trfmr = d_model_trfmr

        if self.use_rnn:
            self.encoder = nn.LSTM(
                input_size=d_model_trfmr,
                hidden_size=d_model_trfmr,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )

            self.proto_embedding = nn.Sequential(
                nn.Linear(2 * d_model_trfmr, embedding_dim),
                nn.Tanh())

        else:
            # output dimension = (?, s, d_model)
            # Pytorch native Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model_trfmr,
                nhead=num_heads,
                dim_feedforward=dff,
                dropout=rate,
                batch_first=True,
                activation=encoder_activation,
            )
            layer_norm = nn.LayerNorm(d_model_trfmr)
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, norm=layer_norm,
            )

            self.proto_embedding = nn.Sequential(
                nn.Linear(max_seq_len * d_model_trfmr, embedding_dim),
                nn.Tanh())

        self.to(device)

    def get_prototype_embedding(self, x):
        flat = []
        counter = 0
        x_image = x[counter]
        seq_len = x_image.shape[1]
        # print("Image:", x_image.shape)
        if self.image_data_type == "original":
            y_image = []
            # since there is no TimeDistributed()
            for jj in range(seq_len):
                img_ii = x_image[:, jj, :, :, :]  # (?, 1, 3, 112, 112)
                img_ii = torch.squeeze(img_ii, dim=1)
                # img_ii = torch.transpose(img_ii, 1, 3)
                h = self.image_embedder(img_ii)  # (?, 64)
                y_image.append(h)
            y_image = torch.stack(y_image, dim=1)

        elif self.image_data_type == "embedding":
            y_image = self.image_projector(x_image)
        # y_image = self.image_transformer(y_image)

        flat.append(y_image)

        if self.include_text:
            counter += 1
            x_text = x[counter]
            y_text = self.text_projector(x_text)
            # y_text = self.text_transformer(y_text)
            flat.append(y_text)
            # print("Text:", x_text.shape)

        if self.include_item_categories:
            counter += 1
            x_cat = x[counter]
            y_cat = self.category_embedder(x_cat)
            flat.append(y_cat)
            src_key_mask = x_cat.eq(0)
            # print("Category:", x_cat)

        # last input, sequence length for each example in a batch
        counter += 1
        seq_lens = x[counter].to('cpu')

        # 1. apply the transformer/RNN encoder
        y = torch.cat(flat, dim=-1)

        if self.use_rnn:
            y_padded = torch.nn.utils.rnn.pack_padded_sequence(
                y, seq_lens, batch_first=True, enforce_sorted=False)
            output, (h_N, c_N) = self.encoder(y_padded)
            # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            # unpacked shape = [?, len*, 384], sequence length len* would be the maximum nonzero length
            # h_N, c_N = (2, ?, h')
            encoded = h_N.transpose(0, 1)
            encoded = torch.flatten(encoded, 1, 2)
            # print(encoded.shape, h_N.shape, c_N.shape)
        else:
            # for Transformer
            encoded = self.encoder(y, src_key_padding_mask=src_key_mask)  # (?, s, h=192)
            encoded = encoded.view(-1, self.seq_len * self.d_model_trfmr)

        # 2. project to the prototype embedding space
        final = self.proto_embedding(encoded)
        return final

    def forward(self, x):
        """
        input contains:
        input_ids, input_mask, segment_ids, label_ids, bbounding_boxes
        """
        final = self.get_prototype_embedding(x)

        return final


def proto_loss(preds, target, n_support):
    """
    Based on https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py
    The idea is to construct prototypes (from the support set)
    inside the loss function and then evaluate on the query set.

    preds: output of the model, (n_support + n_query, seq_len, embedding dimension)
    target: entity labels, (n_support + n_query, seq_len)
    n_support: integer, how many of the samples are part of the support set
        the rest will be part of the query set.
    """
    input_S = preds[:n_support]
    target_S = target[:n_support]

    input_Q = preds[n_support:]
    target_Q = target[n_support:]

    classes = torch.unique(target_S)
    classes = classes[classes.ne(-100)]
    n_classes = len(classes)

    # print(classes)

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_S.eq(c).nonzero()

    support_idxs = list(map(supp_idxs, classes))
    # prototypes = torch.stack([torch.stack([input_S[ind] for ind in indices]).mean(0) for indices in support_idxs])
    prototypes = torch.cat([torch.stack([input_S[ind] for ind in indices]).mean(0) for indices in support_idxs], dim=0)
    # prototypes = [n_class, embedding_dim]
    # input_Q = [n_query, embedding_dim]

    # print(prototypes.shape, input_Q.shape)

    # p_b = torch.stack([prototypes] * input_Q.shape[0], axis=0)
    distance = torch.cdist(input_Q, prototypes)
    # distance = [n_query, n_class]
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean") # .to(device)

    # assuming the classes will always be [0, 1]
    loss = criterion(-distance[:,1], target_Q)

    return loss


def get_prototypes(preds, target, n_support):
    """
    preds: embedded output, shape (?, embedding_dim)
    target: entity classes, shape (?, 1)
    n_support: integer, number of support examples
    """
    input_S = preds[:n_support]
    target_S = target[:n_support]
    classes = torch.unique(target_S)
    classes = classes[classes.ne(-100)]
    n_classes = len(classes)

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_S.eq(c).nonzero()

    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.cat([torch.stack([input_S[ind] for ind in indices]).mean(0) for indices in support_idxs], dim=0)
    return prototypes, classes


def get_class_distance(preds, prototypes):
    """
    preds: shape (?, seq_len, embedding_dim)
    prototypes: (num_classes, embedding_dim)

    returns: (?, seq_len, num_classes)
    """
    distance = torch.cdist(preds, prototypes)
    return distance


def evaluate(model, eval_dataloader, prototypes, device):
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # put model in evaluation mode
    model.eval()

    for batch_x, batch_y in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            batch_device = [x.to(device) for x in batch_x]

            # forward pass
            embeds = model(batch_device)
            distances = get_class_distance(embeds, prototypes)
            logits = -distances
            outputs = logits[:,1]
            labels = batch_y.to(device)

            # get the loss and logits
            tmp_eval_loss = criterion(outputs, labels)
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    pred_labels = np.argmax(preds, axis=1)

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_ids, pred_labels),
        "recall": recall_score(out_label_ids, pred_labels),
        "f1": f1_score(out_label_ids, pred_labels),
        "auc": roc_auc_score(out_label_ids, preds[:,1])
    }
    return results