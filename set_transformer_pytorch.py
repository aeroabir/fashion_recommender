import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import Resnet_18
import sys


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class BaseSetTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(BaseSetTransformer, self).__init__()
        num_layers = kwargs.get("num_layers", 2)
        d_model = kwargs.get("d_model", 128)
        num_heads = kwargs.get("num_heads", 8)
        dff = kwargs.get("dff", 128)
        rate = kwargs.get("rate", 0.1)
        seed_value = kwargs.get("seed", 100)
        num_classes = kwargs.get("num_classes", 2)
        lstm_dim = kwargs.get("lstm_dim", 32)
        device = kwargs.get("device", "cpu")
        embedding_activation = kwargs.get("embedding_activation", "linear")
        encoder_activation = kwargs.get("encoder_activation", "relu")
        lstm_activation = kwargs.get("lstm_activation", "relu")
        final_activation = kwargs.get("final_activation", "sigmoid")
        image_data_type = kwargs.get("image_data_type", "embedding")
        include_text = kwargs.get("include_text", False)
        image_embed_dim = kwargs.get("image_embed_dim", 1280)
        text_embed_dim = kwargs.get("text_embed_dim", 768)
        image_encoder = kwargs.get("image_encoder", "resnet18")
        include_item_categories = kwargs.get("include_item_categories", False)
        num_categories = kwargs.get("num_categories", 154)
        self.max_seq_len = kwargs.get("max_seq_len", 12)

        self.lstm_dim = lstm_dim
        self.image_data_type = image_data_type
        self.include_text = include_text
        self.include_item_categories = include_item_categories
        self.d_model = d_model
        d_model_trfmr = d_model

        if self.image_data_type in ["original", "both"]:
            if image_encoder == "resnet18":
                self.image_embedder = Resnet_18.resnet18(
                    pretrained=True, embedding_size=d_model)

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

        # single set transformer for all features
        self.transformer = SetTransformer(
            dim_input=d_model_trfmr,
            num_outputs=self.max_seq_len,
            dim_output=d_model,
            num_inds=32,
            dim_hidden=128,
            num_heads=num_heads,
        )

        self.d_model_trfmr = d_model_trfmr

        # self.rnn = nn.LSTM(
        #     input_size=d_model_trfmr,
        #     hidden_size=lstm_dim,
        #     num_layers=1,
        #     bidirectional=True,
        #     batch_first=True,
        # )

        self.final = nn.Sequential(
            # only HF-transformer
            nn.Linear(self.d_model * self.max_seq_len, 1),
            # nn.Linear(self.d_model_trfmr * self.max_seq_len, 1),  # only transformer
            # nn.Linear(2 * lstm_dim * self.max_seq_len, 1),
            # nn.Linear(2 * lstm_dim, 1),
            # nn.Sigmoid(),  # for binary classification
        )

        self.to(device)

    def forward(self, x):
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
            # print("Category:", x_cat.shape)

        # last input, sequence length for each example in a batch
        counter += 1
        seq_lens = x[counter].to('cpu')

        y = torch.cat(flat, dim=-1)

        # print("Before:", y.shape)
        ys = self.transformer(y)
        # print("After:", ys.shape)
        # sys.exit()
        # y = ys.last_hidden_state
        rnn_out = ys.view(-1, self.max_seq_len * self.d_model)

        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        # y_padded = torch.nn.utils.rnn.pack_padded_sequence(
        #     y, seq_lens, batch_first=True, enforce_sorted=False)
        # output, (h_N, c_N) = self.rnn(y_padded)
        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # unpacked gets [?, 10, 64], sequence length 2 less than the original
        # output = (?, s, 2*h')
        # h_N, c_N = (2, ?, h')
        # batch_dim = x_image.shape[0]
        # print(h_N.shape, c_N.shape)
        # print(unpacked.shape)
        # rnn_out = unpacked.view(-1, 2 * self.lstm_dim * seq_len)
        # rnn_out = h_N.view(-1, 2 * self.lstm_dim)  # (?, 2*h')
        out = self.final(rnn_out)
        return out
