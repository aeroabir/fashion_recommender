import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from SelfAttention_Family import FullAttention, AttentionLayer

# from layers.Embed import DataEmbedding
import sys
import Resnet_18


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, order, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.order = order
        self.weight = nn.Parameter(
            torch.Tensor(
                1,
                self.order,
                1,
                self.time_step * self.multi,
                self.multi * self.time_step,
            )
        )  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(
            self.time_step * self.multi, self.time_step * self.multi
        )
        self.forecast_result = nn.Linear(
            self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(
                self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = self.order * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.order,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.order,
                        self.time_step * self.output_channel,
                    )
                )
            elif i == 1:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
            else:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        # print("fftin:", input.shape)
        ffted = torch.rfft(input, 1, onesided=False)
        # print("ffted:", ffted.shape)
        real = (
            ffted[..., 0]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, node_cnt, -1)
        )
        img = (
            ffted[..., 1]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, node_cnt, -1)
        )
        # print(" real:", real.shape)
        # print(" imag:", img.shape)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)

        # print(" real:", real.shape)
        # print(" imag:", img.shape)
        real = (
            real.reshape(batch_size, node_cnt, self.order, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        img = (
            img.reshape(batch_size, node_cnt, self.order, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        # print(" real:", real.shape)
        # print(" imag:", img.shape)
        time_step_as_inner = torch.cat(
            [real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        # print(" join:", time_step_as_inner.shape)
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        # print(" ifft:", iffted.shape)
        return iffted

    def forward(self, x, mul_L):
        # self.time_step = input sequence length
        mul_L = mul_L.unsqueeze(1)  # [5, 1, 6, 6]
        x = x.unsqueeze(1)  # [32, 1, 1, 6, 96]

        gfted = torch.matmul(mul_L, x)  # [32, 5, 1, 6, 96]
        gconv_input = self.spe_seq_cell(
            gfted).unsqueeze(2)  # [32, 5, 1, 6, 192]
        # weight shape: [1, 5, 1, 192, 192]

        igfted = torch.matmul(gconv_input, self.weight)  # [32, 5, 1, 6, 192]
        # [32, 1, 6, inp_seq_len * multi = 192]
        igfted = torch.sum(igfted, dim=1)

        forecast_source = torch.sigmoid(
            self.forecast(igfted).squeeze(1)
        )  # [32, 6, 192]
        forecast = self.forecast_result(forecast_source)  # [32, 6, 96]

        # if self.stack_cnt == 0:
        #     backcast_short = self.backcast_short_cut(x).squeeze(1)
        #     backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        # else:
        #     backcast_source = None
        return forecast


class SimpleModel(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleModel, self).__init__()
        image_embed_dim = kwargs.get("image_embed_dim", 1280)
        d_model = kwargs.get("d_model", 128)
        max_seq_len = kwargs.get("max_seq_len", 8)
        device = kwargs.get("device", "cpu")

        self.image_projector = nn.Sequential(
            nn.Linear(image_embed_dim, d_model),
            nn.Tanh())

        self.final = nn.Sequential(
            nn.Linear(max_seq_len * d_model, 1),
            # nn.Sigmoid(),  # for binary classification
        )
        self.to(device)

    def forward(self, x):
        x_image = x[0]
        y_image = self.image_projector(x_image)
        y_image = torch.flatten(y_image, start_dim=1)
        out = self.final(y_image)
        return out

class BaseTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(BaseTransformer, self).__init__()
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

        self.lstm_dim = lstm_dim
        self.image_data_type = image_data_type
        self.include_text = include_text
        self.include_item_categories = include_item_categories
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

        # single transformer for all features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_trfmr,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=rate,
            batch_first=True,
            activation=encoder_activation,
        )
        layer_norm = nn.LayerNorm(d_model_trfmr)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=layer_norm,
        )

        self.rnn = nn.LSTM(
            input_size=d_model_trfmr,
            hidden_size=lstm_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.final = nn.Sequential(
            nn.Linear(2 * lstm_dim, 1),
            # nn.Sigmoid(),  # for binary classification
        )

        self.to(device)

    def forward(self, x):
        flat = []
        counter = 0
        x_image = x[counter]
        if self.image_data_type == "original":
            seq_len = x_image.shape[1]
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

        if self.include_item_categories:
            counter += 1
            x_cat = x[counter]
            y_cat = self.category_embedder(x_cat)
            flat.append(y_cat)
            src_key_mask = x_cat.eq(0)

        # last input, sequence length for each example in a batch
        counter += 1
        seq_lens = x[counter].to('cpu')

        y = torch.cat(flat, dim=-1)

        # https://stackoverflow.com/questions/62399243/transformerencoder-with-a-padding-mask
        # y = self.transformer(y, src_key_padding_mask=src_key_mask)  # (?, s, h)

        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        y_padded = torch.nn.utils.rnn.pack_padded_sequence(
            y, seq_lens, batch_first=True, enforce_sorted=False)
        output, (h_N, c_N) = self.rnn(y_padded)
        # output = (?, s, 2*h')
        # h_N, c_N = (2, ?, h')
        # batch_dim = x_image.shape[0]
        h_N = h_N.view(-1, 2 * self.lstm_dim)  # (?, 2*h')
        out = self.final(h_N)
        return out


class Model(nn.Module):
    def __init__(
        self,
        units,
        stack_cnt,
        time_step,
        multi_layer,
        out_feature=1,
        horizon=1,
        dropout_rate=0.5,
        leaky_rate=0.2,
        device="cpu",
        encoder="bilstm",
        encoder2="fc",
        order=5,
        factor=3,
        dropout=0.05,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=32,
        activation="gelu",
        output_attention="store_true",
        only_transformer=False,
        use_fft_transformer=False,
    ):
        super(Model, self).__init__()
        self.unit = units  # number of input features
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.time_step = time_step
        self.out_feature = out_feature
        self.horizon = horizon
        self.encoder = encoder
        self.encoder2 = encoder2
        self.order = order
        self.only_transformer = only_transformer
        self.use_fft_transformer = use_fft_transformer
        self.d_model = d_model

        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        if self.encoder == "gru":
            self.encoder_model = nn.GRU(
                input_size=self.time_step, hidden_size=self.unit
            )
        elif self.encoder == "lstm":
            self.encoder_model = nn.LSTM(
                self.time_step, self.unit, bidirectional=False)
        elif self.encoder == "bilstm":
            self.encoder_model = nn.LSTM(
                self.time_step, self.unit, bidirectional=True)
            self.encoder_proj = nn.Linear(int(2 * self.unit), int(self.unit))
        self.multi_layer = multi_layer
        if not only_transformer:
            self.stock_block = nn.ModuleList()
            self.stock_block.extend(
                [
                    StockBlockLayer(
                        self.time_step,
                        self.unit,
                        self.multi_layer,
                        self.order,
                        stack_cnt=i,
                    )
                    for i in range(self.stack_cnt)
                ]
            )

        # Pre-projector
        self.projector = nn.Linear(
            self.unit, d_model
        )  # Transformer requires d_model input
        # Encoder
        self.transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # subsequent projection
        if only_transformer:
            if self.use_fft_transformer:
                inner_dim = d_model * order * 2
            else:
                inner_dim = d_model * order
        else:
            if self.use_fft_transformer:
                inner_dim = d_model * order * 2 + self.unit
            else:
                inner_dim = d_model * order + self.unit
        self.projector2 = nn.Linear(inner_dim, self.unit)
        # just to keep the self.fc2 intact, otherwise need to change self.fc2

        # Transformer in the FFT domain
        if self.use_fft_transformer:
            self.fft_transformer = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                False,
                                factor,
                                attention_dropout=dropout,
                                output_attention=output_attention,
                            ),
                            d_model,
                            n_heads,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model),
            )

        if self.encoder2 == "gru":
            self.encoder_model2 = nn.GRU(
                input_size=self.unit, hidden_size=self.unit, batch_first=True
            )
        #             self.encoder_model2 = nn.GRU(input_size=self.time_step, hidden_size=self.unit, batch_first=True)
        elif self.encoder2 == "lstm":
            self.encoder_model2 = nn.LSTM(
                input_size=self.unit,
                hidden_size=self.unit,
                bidirectional=False,
                batch_first=True,
            )
        elif self.encoder2 == "bilstm":
            self.encoder_model2 = nn.LSTM(
                input_size=self.unit,
                hidden_size=self.unit,
                bidirectional=True,
                batch_first=True,
            )
            self.encoder_proj2 = nn.Linear(int(2 * self.unit), int(self.unit))
        else:
            self.encoder2 = nn.Sequential(
                nn.Linear(int(self.time_step), int(self.time_step)),
                nn.LeakyReLU(),
                nn.Linear(int(self.time_step), self.horizon),
            )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.unit), int(self.unit)),
            nn.LeakyReLU(),
            nn.Linear(int(self.unit), self.out_feature),
            nn.Sigmoid(),  # for binary classification
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(
                graph.size(0), device=graph.device, dtype=graph.dtype
            ) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )  # this shoule be one
        second_laplacian = laplacian
        cheb_poly = [0] * self.order
        cheb_poly[0] = first_laplacian
        cheb_poly[1] = second_laplacian
        for ik in range(2, self.order):
            cheb_poly[ik] = (
                2 * torch.matmul(laplacian,
                                 cheb_poly[ik - 1]) - cheb_poly[ik - 2]
            )

        #         third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        #         forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        #         fifth_laplacian = 2 * torch.matmul(laplacian, forth_laplacian) - third_laplacian
        #         multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian, fifth_laplacian], dim=0)
        multi_order_laplacian = torch.cat(cheb_poly, dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        # print("Input: ", x.shape)
        inp, _ = self.encoder_model(x.permute(2, 0, 1).contiguous())
        if self.encoder == "bilstm":
            inp = self.encoder_proj(inp)
        # print("ENCout: ", inp.shape)
        inp = inp.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(inp)  # (b X N X N)
        attention = torch.mean(attention, dim=0)  # (N X N)
        degree = torch.sum(attention, dim=1)  # (N)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat, torch.matmul(
                degree_l - attention, diagonal_degree_hat)
        )  # (N X N)
        # print("    L: ", laplacian.shape)
        mul_L = self.cheb_polynomial(laplacian)
        # print(" Cheb: ", mul_L.shape)
        return mul_L, attention

    def self_graph_attention(self, input):
        # print("Graph Attention:", input.shape)
        input = input.permute(0, 2, 1).contiguous()
        # print("Graph Attention:", input.shape)
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)

        # print("Graph Attention: (key)", key.shape)
        # print("Graph Attention: (query)", query.shape)
        # why not use this?
        # data = torch.matmul(query, torch.transpose(key, 1, 2))
        # data = data / math.sqrt(N)

        # How is this doing QK^T
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        # print("Graph Attention: (data)", data.shape)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def apply_fft_transformer(self, x):
        x_g2 = x.permute(0, 2, 1)
        x_g_hat = torch.rfft(x_g2, 1, onesided=False)
        real = (
            x_g_hat[..., 0]
            .permute(0, 2, 1)
            .contiguous()
            .reshape(-1, self.time_step, self.d_model)
        )
        imag = (
            x_g_hat[..., 1]
            .permute(0, 2, 1)
            .contiguous()
            .reshape(-1, self.time_step, self.d_model)
        )
        real_ii, _ = self.fft_transformer(real)
        imag_ii, _ = self.fft_transformer(imag)
        real = real_ii.permute(0, 2, 1).contiguous()
        img = imag_ii.permute(0, 2, 1).contiguous()
        time_step_as_inner = torch.cat(
            [real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        iffted = iffted.permute(0, 2, 1)  # [32, 96, 32]
        return iffted

    def forward(self, x):
        x = x[0]
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()

        if not self.only_transformer:
            forecast1 = self.stock_block[0](X, mul_L)  # [b, 6, s]

        # Transform to the Graph Fourier Domain
        mul_L = mul_L.unsqueeze(1)  # [3, 1, 6, 6]
        X = X.unsqueeze(1)  # [32, 1, 6, 96]
        gfted = torch.matmul(mul_L, X)  # [32, 3, 1, 6, 96]
        gfted = gfted.squeeze(2)
        result = []
        if self.use_fft_transformer:
            result_fft = []

        # For each of the order component apply Transformer
        for ii in range(self.order):
            x_g = gfted[:, ii, :, :]
            x_g = x_g.permute(0, 2, 1)
            x_g = self.projector(x_g)

            if self.use_fft_transformer:
                iffted = self.apply_fft_transformer(x_g)
                result_fft.append(iffted)

            y_g, _ = self.transformer(x_g)
            result.append(y_g)
        forecast = torch.cat(result, dim=-1)  # [b, s, order * d_model]
        forecast = forecast.permute(0, 2, 1)  # [b, order * d_model, s]

        if not self.only_transformer:
            forecast = torch.cat([forecast, forecast1], dim=-2)

        if self.use_fft_transformer:
            # [b, s, order * d_model]
            forecast_fft = torch.cat(result_fft, dim=-1)
            forecast_fft = forecast_fft.permute(
                0, 2, 1)  # [b, order * d_model, s]
            forecast = torch.cat([forecast, forecast_fft], dim=-2)

        forecast = self.encoder2(forecast)  # [b, order * d_model, H]
        forecast = forecast.permute(0, 2, 1)  # [b, H, order * d_model]
        forecast = self.projector2(forecast)  # [b, H, units]

        forecast = self.fc2(forecast)
        if self.out_feature == 1:
            forecast = torch.squeeze(forecast, dim=-1)

        if self.horizon == 1:
            forecast = torch.squeeze(forecast, dim=-1)

        return forecast, attention
