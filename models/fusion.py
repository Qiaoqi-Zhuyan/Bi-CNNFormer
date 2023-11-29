import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_

import einops

class CNN_Transformer_Fusion(nn.Module):
    pass


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, in_chan_1, in_chan_2, ratio, hidden_dim, drop=0.0, drop_path=0.0):
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


class DwConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, padding=0, bias=True):
        super(DwConv, self).__init__()
        self.dw = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,stride=stride, padding=padding, groups=in_chans ,bias=bias)
        self.pw = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=bias)

    def forward(self, x):

        x = self.dw(x)
        x = self.pw(x)

        return x


class FusionModule(nn.Module):
    def __init__(self, in_chans, drop=0.0):
        super(FusionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_chans * 2, in_chans),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(in_chans, in_chans),
            nn.Dropout(drop),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_t, x_c):

        x = torch.cat([x_c, x_t], dim=1)
        x = einops.rearrange(x, "b c h w -> b h w c")
        wights = self.conv(x)
        wights = einops.rearrange(wights, "b h w c -> b c h w")
        x = x_c * wights + x_t * (1 - wights)

        return x


class Attn_fusion_Block(nn.Module):

    def __init__(self, in_chans, num_heads, fuse_type="add", attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True):
        super(Attn_fusion_Block, self).__init__()

        assert fuse_type == "add" or fuse_type == "cat"
        self.fuse_type = fuse_type
        # attn fusion patch
        self.num_heads = num_heads
        head_dim = in_chans // num_heads
        self.scale = head_dim ** -0.5
        if fuse_type == "add":
            self.in_chans = in_chans
        elif fuse_type == "cat":
            self.in_chans = in_chans * 2

        #self.to_q = DwConv(in_chans=self.in_chans, out_chans=self.in_chans, kernel_size=3, bias=bias)
        #self.to_k = DwConv(in_chans=self.in_chans, out_chans=self.in_chans, kernel_size=3, bias=bias)
        #self.to_v = DwConv(in_chans=self.in_chans, out_chans=self.in_chans, kernel_size=3, bias=bias)

        self.to_q = nn.Linear(in_chans, in_chans)
        self.to_k = nn.Linear(in_chans, in_chans)
        self.to_v = nn.Linear(in_chans, in_chans)

        self.softmax = nn.Softmax(dim=-1)

        self.project_out = nn.Linear(in_features=self.in_chans, out_features=in_chans)

        self.attn_drop = nn.Dropout(attn_drop)


        # cnn feat patch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)



        #transformer feat patch
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _fuse_forward(self, x_t, x_c):
        '''

        :param x_t: [b c h w]
        :param x_c: [b c h w]

            two fusion strategies: add, cat
        :return:      out [b c h w]
        '''
        if self.fuse_type == "cat":
            x = torch.cat([x_c, x_t], dim=1)
        elif self.fuse_type == "add":
            x = torch.add(x_c, x_t)

        B, C, H, W = x.shape

        x = einops.rearrange(x, "b c h w -> b h w c")

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x) # [b h w c ]

        # [b c h w] -> [b c1 num_heads, h w]

        #q = einops.rearrange(q, "b (c head) h w -> b c head (h w)", head=self.num_heads)
        #k = einops.rearrange(k, "b (c head) h w -> b c head (h w)", head=self.num_heads)
        #v = einops.rearrange(v, "b (c head) h w -> b c head (h w)", head=self.num_heads)


        q = einops.rearrange(q, "b h w (c head) -> b (h w) head c", head=self.num_heads)
        k = einops.rearrange(k, "b h w (c head) -> b (h w) head c", head=self.num_heads)
        v = einops.rearrange(v, "b h w (c head) -> b (h w) head c", head=self.num_heads)


        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(dot)
        attn = attn @ v
        # [b c1 num_heads, h w]

        attn = einops.rearrange(attn, "b (h w) head c  -> b h w (head c)", h=H, w=W)

        out = self.project_out(attn)
        self.attn_drop(out)

        out = einops.rearrange(out, "b h w c -> b c h w ")

        return out

    def _transformer_forward(self, x_t):
        x = self.avg_pool(x_t) # [b c h w] -> [b c 1 1]
        print(f"max_pool shape: {x.shape}")

    def _cnn_forward(self, x_c):
        x = self.max_pool(x_c) # [b c h w] -> [b c 1 1]

    def forward(self, x_c, x_t):
        attn_fuse = self._fuse_forward(x_t, x_c)
        return attn_fuse


if __name__ == "__main__":

    x_l = torch.randn(1, 64, 224, 224)
    x_h = torch.randn(1, 64, 224, 224)

    afb = Attn_fusion_Block(in_chans=64, num_heads=8, fuse_type="add")

    y = afb._transformer_forward(x_h)
    print(f"_transformer_forward : ")