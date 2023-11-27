import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_


class CNN_Transformer_Fusion(nn.Module):
    pass


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, in_chan_1, in_chan_2, ratio, hidden_dim,drop=0.0 ,drop_path=0.0):
        super(BiFusion_block, self).__init__()

        # channel attn for F_g
        self.fc1 = nn.Linear(in_chan_2, in_chan_2 // ratio)
        self.act_layer1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_chan_2 // ratio, in_chan_2)
        self.sigmoid = nn.Sigmoid()

        # spatial attn
        self.compress = ChannelPool()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7)
        self.bn = nn.BatchNorm2d(1)
        self.act_layer2 = nn.ReLU()

        # bi-linear
        self.w_g = nn.Sequential(
            nn.Conv2d(in_chan_1, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(in_chan_2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim)
        )

        self.w = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        self.drop_path = DropPath(drop_path)
        self.drop = nn.Dropout(drop)

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g*W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse



'''
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True)
'''

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_dim, out_channels=out_dim // 2, kernel_size=1)
        ) #Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_dim // 2), out_channels=int(out_dim // 2), kernel_size=3)
        ) #Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))


        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_dim // 2), out_channels=out_dim, kernel_size=1)
        )#Conv(int(out_dim / 2), out_dim, 1, relu=False)

        self.skip_layer = nn.Sequential(
            nn.Conv2d(in_channels=inp_dim, out_channels=out_dim, kernel_size=1)
        ) #Conv(inp_dim, out_dim, 1, relu=False)

        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out