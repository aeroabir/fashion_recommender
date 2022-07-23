import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
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
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        # print(" join:", time_step_as_inner.shape)
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        # print(" ifft:", iffted.shape)
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)  # 4, 1, 228, 228]
        x = x.unsqueeze(1)  # [32, 1, 1, 228, 12]

        gfted = torch.matmul(mul_L, x)  # [32, 4, 1, 228, 12]
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        # print("gconv_input:", gconv_input.shape)
        # print("weight:", self.weight.shape)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


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

        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        if self.encoder == "gru":
            self.encoder_model = nn.GRU(
                input_size=self.time_step, hidden_size=self.unit
            )
        elif self.encoder == "lstm":
            self.encoder_model = nn.LSTM(self.time_step, self.unit, bidirectional=False)
        elif self.encoder == "bilstm":
            self.encoder_model = nn.LSTM(self.time_step, self.unit, bidirectional=True)
            self.encoder_proj = nn.Linear(int(2 * self.unit), int(self.unit))
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [
                StockBlockLayer(
                    self.time_step, self.unit, self.multi_layer, self.order, stack_cnt=i
                )
                for i in range(self.stack_cnt)
            ]
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
                2 * torch.matmul(laplacian, cheb_poly[ik - 1]) - cheb_poly[ik - 2]
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
            diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat)
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

    def forward(self, x):
        # print(x.shape)
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        # print("    X: ", X.shape)
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            # print("focst-i: ", forecast.shape)
            #             print("    X: ", X.shape)
            result.append(forecast)

        forecast = result[0]
        for ii in range(1, self.stack_cnt):
            forecast += result[ii]
        # print("forcast (before fc):", forecast.shape)  # (b, 6, 96)

        forecast = self.encoder2(forecast)
        # print("focst: ", forecast.shape)  # (b, 6, 6)
        # sys.exit()
        # make sentence length as sequence length
        forecast = forecast.permute(0, 2, 1)  # (b, 96, 6)

        forecast = self.fc2(forecast)
        # print(forecast.shape)
        return forecast, attention
