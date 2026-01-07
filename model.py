import torch.nn as nn
import torch
import torch.nn.functional as F

from operations import operation_dict_all


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ChannelAggregation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = max(in_channels // reduction_ratio, 1)  # 确保不小于1

        # 通道注意力结构
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化（压缩特征维度到1）
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.reduction),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(self.reduction, in_channels),  # 恢复维度
            nn.Sigmoid()  # 归一化通道权重
        )

        # 可学习的聚合参数
        self.aggregate_weight = nn.Parameter(torch.ones(1, in_channels, 1))  # (1, C, 1)

    def forward(self, x):
        batch_size, channels, features = x.size()

        # Squeeze: 全局平均池化 (B, C, F) → (B, C, 1)
        channel_avg = self.avg_pool(x).squeeze(-1)  # (B, C)

        # Excitation: 生成通道权重 (B, C) → (B, C)
        channel_weights = self.fc(channel_avg)  # (B, C)

        # 加权特征通道 (B, C, F) = (B, C, F) * (B, C, 1)
        weighted_x = x * channel_weights.unsqueeze(-1)

        # 聚合通道维度（加权求和）
        aggregated = torch.sum(weighted_x * self.aggregate_weight, dim=1)  # (B, F)

        return aggregated

class MixedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, operation_dict):
        super(MixedLayer, self).__init__()

        self.layers = nn.ModuleList()
        for operation in operation_dict.keys():
            # Create corresponding layer
            layer = operation_dict[operation](input_dim, output_dim)
            self.layers.append(layer)

    def forward(self, x, weights):
        res = [w * layer(x) for w, layer in zip(weights, self.layers)]
        res = sum(res)

        return res

class Network(nn.Module):
    def __init__(self, num_nodes, initial_input_dims, hidden_dim, opt):
        super(Network, self).__init__()
        self.opt = opt
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        self.operation_name_list = []
        self.initial_input_dims = initial_input_dims
        self.num_initial_input = len(self.initial_input_dims)

        # Generate all the mixed layer
        for i in range(self.num_nodes):
            # All previous outputs and additional inputs
            for j in range(i + self.num_initial_input):
                if j < self.num_initial_input:  # Input layer
                    layer = MixedLayer(self.initial_input_dims[j], self.hidden_dim[i], operation_dict_all)
                    self.layers.append(layer)
                    self.operation_name_list.append(list(operation_dict_all.keys()))
                else:  # Middle layers
                    layer = MixedLayer(self.hidden_dim[j-self.num_initial_input], self.hidden_dim[i], operation_dict_all)
                    self.layers.append(layer)
                    self.operation_name_list.append(list(operation_dict_all.keys()))
            if i < self.num_nodes-1 and opt.SelfAttention:
                layer = ChannelAggregation(i + self.num_initial_input)
                self.layers.append(layer)
                self.operation_name_list.append('ChannelAggregation')
        print("")

    def forward(self, s_0, s_1, genotype):
        states = [s_0, s_1, torch.cat((s_0, s_1), dim=-1)]
        offset = 0

        # Input from all previous layers
        for i in range(self.num_nodes):
            if self.opt.SelfAttention:
                s = torch.stack([
                    self.layers[offset + j](cur_state, genotype[offset + j - i])
                    for j, cur_state in enumerate(states)], dim=1)
                if i < self.num_nodes-1:
                    s = self.layers[offset + len(states)](s)
                else:
                    s = torch.sum(s, dim=1)
                offset += len(states) + 1
            else:
                s = sum(
                    self.layers[offset + j](cur_state, genotype[offset + j]) for j, cur_state
                    in enumerate(states))
                offset += len(states)
            states.append(s)

        # Keep last layer output
        return states[-1]

    def get_operation_name_list(self):
        return self.operation_name_list

class MLP_search(nn.Module):
    def __init__(self, opt, flag, num_nodes = 4):
        super(MLP_search, self).__init__()
        self.num_nodes = opt.num_nodes
        self.att_size = opt.attSize
        self.nz = opt.nz
        self.res_size = opt.resSize
        if flag == 'g':
            if self.num_nodes == 4:
                self.hidden_dim = [512, 2048, 4096, 2048]
            else:
                self.hidden_dim = [512, 1024, 2048, 4096, 2048]
            self.initial_input_dims = [
                self.att_size,
                self.nz,
                self.att_size + self.nz
            ]
        else:  # 鉴别器
            if self.num_nodes == 4:
                self.hidden_dim = [4096, 2048, 1024, 1]
            else:
                self.hidden_dim = [4096, 2048, 1024, 512, 1]
            self.initial_input_dims = [
                self.att_size,
                self.res_size,
                self.att_size + self.res_size
            ]
        print('self.hidden_dim', self.hidden_dim)

        self.num_initial_input = len(self.initial_input_dims)
        self.network = Network(self.num_nodes, self.initial_input_dims, self.hidden_dim, opt)
        # Get operation list
        self.operation_name_list = self.network.get_operation_name_list()


    def forward(self, noise, att, genotype):
        h = self.network(att, noise, genotype)
        return h

class Discriminator(nn.Module):

    def __init__(self, res_size,att_size,layer_sizes):
        super().__init__()
        self.MLP = nn.Sequential()
        layer = layer_sizes + [1]
        for i, (in_size, out_size) in enumerate(zip([res_size+att_size] + layer[:-1], layer)):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer):
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU(0.2, True))
        self.apply(weights_init)

    def forward(self, x, s, g=None):
        h = torch.cat((x, s), dim=-1)
        h = self.MLP(h)
        return h

class Generator(nn.Module):

    def __init__(self, layer_sizes, latent_size, attSize):
        super().__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size + attSize

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU(0.2, True))
            else:
                self.MLP.add_module(name="ReLU", module=nn.ReLU(inplace=True))
        self.apply(weights_init)

    def forward(self, z, s, g=None):
        h = torch.cat((z, s), dim=-1)
        x = self.MLP(h)
        return x


class Mapping_net(nn.Module):

    def __init__(self, layer_sizes, att_size):
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([att_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU(0.2, True))
        self.apply(weights_init)

    def forward(self, s):
        w = self.MLP(s)
        return w


class MMD_loss(nn.Module):
    def __init__(self, bu=4, bl=1 / 4):
        super(MMD_loss, self).__init__()
        self.fix_sigma = 1
        self.bl = bl
        self.bu = bu
        return

    def phi(self, x, y):
        total0 = x.unsqueeze(0).expand(int(x.size(0)), int(x.size(0)), int(x.size(1)))
        total1 = y.unsqueeze(1).expand(int(y.size(0)), int(y.size(0)), int(y.size(1)))
        return (((total0 - total1) ** 2).sum(2))

    def forward(self, source, target, type):
        M = source.size(dim=0)
        N = target.size(dim=0)
        if M != N:
            target = target[:M, :]

        L2_XX = self.phi(source, source)
        L2_YY = self.phi(target, target)
        L2_XY = self.phi(source, target)

        bu = self.bu * torch.ones(L2_XX.size()).type(torch.cuda.FloatTensor)
        bl = self.bl * torch.ones(L2_YY.size()).type(torch.cuda.FloatTensor)
        alpha = (1 / (2 * self.fix_sigma)) * torch.ones(1).type(torch.cuda.FloatTensor)
        m = M * torch.ones(1).type(torch.cuda.FloatTensor)

        if type == "critic":
            XX_u = torch.exp(-alpha * torch.min(L2_XX, bu))
            YY_l = torch.exp(-alpha * torch.max(L2_YY, bl))
            XX = (1 / (m * (m - 1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
            YY = (1 / (m * (m - 1))) * (torch.sum(YY_l) - torch.sum(torch.diagonal(YY_l, 0)))
            lossD = XX - YY
            return lossD
        elif type == "gen":
            XX_u = torch.exp(-alpha * L2_XX)
            YY_u = torch.exp(-alpha * L2_YY)
            XY_l = torch.exp(-alpha * L2_XY)
            XX = (1 / (m * (m - 1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
            YY = (1 / (m * (m - 1))) * (torch.sum(YY_u) - torch.sum(torch.diagonal(YY_u, 0)))
            XY = torch.mean(XY_l)
            lossmmd = XX + YY - 2 * XY
            return lossmmd