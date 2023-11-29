import einops

from models.fusion import Attn_fusion_Block
from models.transformer.swin_transformer_fork import *
from models.cnn.convnet_fork import ConvNextBasicLayer, LayerNorm

from config.config import get_config

from torch import nn
import torch

basic_config = get_config()


class SwinTransformer_Branch(nn.Module):
    def __init__(self, dims=None, layers=None, num_heads=None, img_size=224,
                 patch_size=4, in_chans=3, window_size=7, mlp_ratio=4, embed_dim=64,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, fused_window_process=False,
                 ):
        super(SwinTransformer_Branch, self).__init__()

        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if dims is None:
            dims = [96, 192, 384, 768]
        if layers is None:
            layers = [2, 2, 6, 2]

        self.img_size = to_2tuple(img_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule

        self.trans_stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=dims[0], norm_layer=norm_layer
        )

        self.trans_stage1 = BasicLayer(
            dim=dims[0],
            input_resolution=(self.img_size[0] // 4, self.img_size[1] // 4),
            depth=layers[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:0]):sum(layers[:0 + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging,
            fused_window_process=fused_window_process,
        )

        self.trans_stage2 = BasicLayer(
            dim=dims[1],
            input_resolution=(self.img_size[0] // 8, self.img_size[1] // 8),
            depth=layers[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:1]):sum(layers[:1 + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging,
            fused_window_process=fused_window_process,
        )

        self.trans_stage3 = BasicLayer(
            dim=dims[2],
            input_resolution=(self.img_size[0] // 16, self.img_size[1] // 16),
            depth=layers[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:2]):sum(layers[:2 + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging,
            fused_window_process=fused_window_process,
        )

        self.trans_stage4 = BasicLayer(
            dim=dims[3],
            input_resolution=(self.img_size[0] // 32, self.img_size[1] // 32),
            depth=layers[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:3]):sum(layers[:3 + 1])],
            norm_layer=norm_layer,
            downsample=None,
            fused_window_process=fused_window_process,
        )

    def get_stem(self):
        return self.trans_stem

    def get_stage1(self):
        return self.trans_stage1

    def get_stage2(self):
        return self.trans_stage2

    def get_stage3(self):
        return self.trans_stage3

    def get_stage4(self):
        return self.trans_stage4

    def forward_embed(self, x):

        return self.trans_stem(x)

    def forward_stage1(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''
        return self.trans_stage1(x)

    def forward_stage2(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''
        return self.trans_stage2(x)

    def forward_stage3(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''
        return self.trans_stage3(x)

    def forward_stage4(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''

        return self.trans_stage4(x)

    def forward(self, x):
        print(f'transformer x shape {x.shape}')

        x = self.trans_stem(x)
        print(f'transformer x stem shape {x.shape}')

        x = self.trans_stage1(x)
        print(f'transformer stage1 {x.shape}')

        x = self.trans_stage2(x)
        print(f'transformer stage2 {x.shape}')

        x = self.trans_stage3(x)
        print(f'transformer stage3 {x.shape}')

        x = self.trans_stage4(x)
        print(f'transformer stage4 {x.shape}')

        return x




class ConvNext_branch(nn.Module):
    def __init__(self, in_chans=3,
                 layers=None, dims=None, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super(ConvNext_branch, self).__init__()
        if dims is None:
            dims = [96, 192, 384, 768]
        if layers is None:
            layers = [3, 3, 9, 3]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.cnn_stage1 = ConvNextBasicLayer(
            in_chans=dims[0], out_chans=dims[1], layer=layers[0], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=True,
            drop_path_rate=dpr[sum(layers[:0]):sum(layers[:0 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

        self.cnn_stage2 = ConvNextBasicLayer(
            in_chans=dims[1], out_chans=dims[2], layer=layers[1], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=True,
            drop_path_rate=dpr[sum(layers[:1]):sum(layers[:1 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

        self.cnn_stage3 = ConvNextBasicLayer(
            in_chans=dims[2], out_chans=dims[3], layer=layers[2], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=True,
            drop_path_rate=dpr[sum(layers[:2]):sum(layers[:2 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

        self.cnn_stage4 = ConvNextBasicLayer(
            in_chans=dims[3], out_chans=dims[3], layer=layers[3], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=False,
            drop_path_rate=dpr[sum(layers[:3]):sum(layers[:3 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

    def get_stem(self):
        return self.cnn_stem

    def get_stage1(self):
        return self.cnn_stage1

    def get_stage2(self):
        return self.cnn_stage2

    def get_stage3(self):
        return self.cnn_stage3

    def get_stage4(self):
        return self.cnn_stage4

    def forward_embed(self, x):

        return self.cnn_stem(x)

    def forward_stage1(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''
        return self.cnn_stage1(x)

    def forward_stage2(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''
        return self.cnn_stage2(x)

    def forward_stage3(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''
        return self.cnn_stage3(x)

    def forward_stage4(self, x):
        '''
             input: x[B (H W) C] -> out: [B (H W) C]
        '''

        return self.cnn_stage4(x)

    def forward(self, x):
        print(f'cnn x shape {x.shape}')

        x = self.cnn_stem(x)
        print(f'cnn x stem shape {x.shape}')

        x = self.cnn_stage1(x)
        print(f'cnn stage1 {x.shape}')

        x = self.cnn_stage2(x)
        print(f'cnn stage2 {x.shape}')

        x = self.cnn_stage3(x)
        print(f'cnn stage3 {x.shape}')

        x = self.cnn_stage4(x)
        print(f'cnn stage4 {x.shape}')

        return x


'''
    def __init__(self, in_chan_1, in_chan_2, ratio, hidden_dim, drop=0.0, drop_path=0.0):
'''

class Bi_CnnFormer(nn.Module):
    def __init__(self, dims=None, layers=None, num_heads=None, img_size=224,
                 patch_size=4, in_chans=3, window_size=7, mlp_ratio=4, embed_dim=64,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, fused_window_process=False,
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, head_init_scale=1.):
        super(Bi_CnnFormer, self).__init__()

        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if dims is None:
            dims = [96, 192, 384, 768]
        if layers is None:
            layers = [2, 2, 6, 2]

        self.img_size = to_2tuple(img_size)

        # Transformer branch
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule

        self.transformer_branch = SwinTransformer_Branch(
            dims=dims, layers=layers, num_heads=num_heads, img_size=224,
            patch_size=patch_size, in_chans=in_chans, window_size=window_size, mlp_ratio=mlp_ratio, embed_dim=embed_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, ape=ape, patch_norm=patch_norm, fused_window_process=fused_window_process
        )

        # CNN branch
        self.cnn_branch = ConvNext_branch(
            in_chans=in_chans,
            layers=layers, dims=dims, drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale,
        )

        # feat Fusion module

        fusion_heads = [3, 6, 12, 24]

        self.AFB_stage1 = Attn_fusion_Block(in_chans=dims[0], num_heads=fusion_heads[0], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage2 = Attn_fusion_Block(in_chans=dims[1], num_heads=fusion_heads[1], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage3 = Attn_fusion_Block(in_chans=dims[2], num_heads=fusion_heads[2], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage4 = Attn_fusion_Block(in_chans=dims[3], num_heads=fusion_heads[3], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)

        self.norm = nn.Identity()

        self.decoder = nn.Identity()

        self.fuse_feat_proc = nn.Identity()

        self.head = nn.Identity()

    def _feature_forward(self, x):
        # embedding: [B C H W] -> [B C' H/4, W/4]
        x_c = self.cnn_branch.forward_embed(x)
        x_t = self.transformer_branch.forward_embed(x)

        # Stage1: [B C' H/4 W/4] -> [B C'*2 H/8, W/8]
        x_c = self.cnn_branch.forward_stage1(x_c)
        x_t = self.transformer_branch.forward_stage2(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)

        # Stage1 Fusion [B C'*2 H/8, W/8] -> [B C'*2 H/8, W/8]
        stage1_fuse = self.AFB_stage1.forward(x_c, x_t)


        # Stage2: [B C'*2 H/8 W/8] -> [B C'*4 H/16, W/16]
        x_c = self.cnn_branch.forward_stage2(x_c)
        x_t = self.transformer_branch.forward_stage2(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)

        # Stage2 Fusion [B C'*4 H/16, W/16] -> [B C'*2 H/16, W/16]
        stage2_fuse = self.AFB_stage2.forward(x_c, x_t)


        # Stage3: [B C'*4 H/16 W/16] -> [B C'*8 H/32, W/32]
        x_c = self.cnn_branch.forward_stage3(x_c)
        x_t = self.transformer_branch.forward_stage3(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)

        # Stage3 Fusion [B C'*8 H/32, W/32] -> [B C'*8 H/32, W/32]
        stage3_fuse = self.AFB_stage3.forward(x_c, x_t)


        # Stage4: [B C'*8 H/32, W/32] -> [B C'*16 H/32, W/32]
        x_c = self.cnn_branch.forward_stage4(x_c)
        x_t = self.transformer_branch.forward_stage4(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)

        # Stage4 Fusion [B C'*16 H/32, W/32] -> [B C'*16 H/32, W/32]
        stage4_fuse = self.AFB_stage4.forward(x_c, x_t)

    def forward(self, x):
        transformer_feat = self.transformer.forward(x)
        cnn_feat = self.cnn.forward(x)

        fuse_feats = self._feature_forward(x)

        for i in range(4):
            feats = self.fuse_feat_proc(fuse_feats[i])

        out = self.head(torch.cat([transformer_feat, cnn_feat, feats], dim=-1))

        return out



class Wnet(nn.Module):
    def __init__(self, dims=None, layers=None, num_heads=None, img_size=224,
                 patch_size=4, in_chans=3, window_size=7, mlp_ratio=4, embed_dim=64,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, fused_window_process=False,
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, head_init_scale=1.):
        super(Wnet, self).__init__()

        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if dims is None:
            dims = [96, 192, 384, 768]
        if layers is None:
            layers = [2, 2, 6, 2]

        self.img_size = to_2tuple(img_size)

        # Transformer branch
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule

        self.trans_stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=dims[0], norm_layer=norm_layer
        )

        self.trans_stage1 = BasicLayer(
            dim=dims[0],
            input_resolution=(self.img_size[0] // 4, self.img_size[1] // 4),
            depth=layers[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:0]):sum(layers[:0 + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging,
            fused_window_process=fused_window_process,
        )

        self.trans_stage2 = BasicLayer(
            dim=dims[0],
            input_resolution=(self.img_size[0] // 8, self.img_size[1] // 8),
            depth=layers[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:1]):sum(layers[:1 + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging,
            fused_window_process=fused_window_process,
        )

        self.trans_stage3 = BasicLayer(
            dim=dims[2],
            input_resolution=(self.img_size[0] // 16, self.img_size[1] // 16),
            depth=layers[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:2]):sum(layers[:2 + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging,
            fused_window_process=fused_window_process,
        )

        self.trans_stage4 = BasicLayer(
            dim=dims[3],
            input_resolution=(self.img_size[0] // 32, self.img_size[1] // 32),
            depth=layers[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(layers[:3]):sum(layers[:3 + 1])],
            norm_layer=norm_layer,
            downsample=None,
            fused_window_process=fused_window_process,
        )


        # CNN branch

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule


        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.cnn_stage1 = ConvNextBasicLayer(
            in_chans=dims[0], out_chans=dims[1], layer=layers[0], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=True,
            drop_path_rate=dpr[sum(layers[:0]):sum(layers[:0 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

        self.cnn_stage2 = ConvNextBasicLayer(
            in_chans=dims[1], out_chans=dims[2], layer=layers[1], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=True,
            drop_path_rate=dpr[sum(layers[:1]):sum(layers[:1 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

        self.cnn_stage3 = ConvNextBasicLayer(
            in_chans=dims[2], out_chans=dims[3], layer=layers[2], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=True,
            drop_path_rate=dpr[sum(layers[:2]):sum(layers[:2 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

        self.cnn_stage4 = ConvNextBasicLayer(
            in_chans=dims[3], out_chans=dims[3], layer=layers[3], eps=1e-6,
            data_format="channel_first", kernel_size=2, stride=2, downsample=False,
            drop_path_rate=dpr[sum(layers[:3]):sum(layers[:3 + 1])],
            layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale
        )

        fusion_heads = [3, 6, 12, 24]

        # feat Fusion module
        self.AFB_stage1 = Attn_fusion_Block(in_chans=dims[1], num_heads=fusion_heads[0], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage2 = Attn_fusion_Block(in_chans=dims[2], num_heads=fusion_heads[1], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage3 = Attn_fusion_Block(in_chans=dims[3], num_heads=fusion_heads[2], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage4 = Attn_fusion_Block(in_chans=dims[3], num_heads=fusion_heads[3], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)

        self.norm = nn.Identity()

        self.decoder = nn.Identity()

        self.fuse_feat_proc = nn.Identity()

        self.head = nn.Identity()

    def _feature_forward(self, x):
        # embedding: [B C H W] -> [B C' H/4, W/4]
        x_c = self.cnn_stem(x)
        x_t = self.trans_stem(x)

        print(f"input x: {x.shape}, embedding x_c: {x_c.shape}, embedding x_t: {x_t.shape}")

        # Stage1: [B C' H/4 W/4] -> [B C'*2 H/8, W/8]
        x_c = self.cnn_stage1(x_c)
        x_t = self.trans_stage1(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)
        print(f"stage 1: x_t: {x_t.shape}, x_c: {x_c.shape}")
        # Stage1 Fusion [B C'*2 H/8, W/8] -> [B C'*2 H/8, W/8]
        stage1_fuse = self.AFB_stage1.forward(x_c, x_t)
        x_t = einops.rearrange(x_t, "b c h w  -> b (h w) c", h=h, w=w)
        print(f"stage1 fuse: {stage1_fuse.shape}")


        # Stage2: [B C'*2 H/8 W/8] -> [B C'*4 H/16, W/16]
        x_c = self.cnn_stage2(x_c)
        x_t = self.trans_stage2(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)
        print(f"stage 2: x_t: {x_t.shape}, x_c: {x_c.shape}")
        # Stage2 Fusion [B C'*4 H/16, W/16] -> [B C'*2 H/16, W/16]
        stage2_fuse = self.AFB_stage2.forward(x_c, x_t)
        x_t = einops.rearrange(x_t, "b c h w -> b (h w) c", h=h, w=w)
        print(f"stage2 fuse: {stage2_fuse.shape}")

        # Stage3: [B C'*4 H/16 W/16] -> [B C'*8 H/32, W/32]
        x_c = self.cnn_stage3(x_c)
        x_t = self.trans_stage3(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)
        print(f"stage 3: x_t: {x_t.shape}, x_c: {x_c.shape}")
        # Stage3 Fusion [B C'*8 H/32, W/32] -> [B C'*8 H/32, W/32]
        stage3_fuse = self.AFB_stage3.forward(x_c, x_t)
        x_t = einops.rearrange(x_t, "b c h w -> b (h w) c", h=h, w=w)
        print(f"stage3 fuse: {stage3_fuse.shape}")

        # Stage4: [B C'*8 H/32, W/32] -> [B C'*16 H/32, W/32]
        x_c = self.cnn_stage4(x_c)
        x_t = self.trans_stage4(x_t)

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)
        print(f"stage 4: x_t: {x_t.shape}, x_c: {x_c.shape}")
        # Stage4 Fusion [B C'*16 H/32, W/32] -> [B C'*16 H/32, W/32]
        stage4_fuse = self.AFB_stage4.forward(x_c, x_t)
        print(f"stage4 fuse: {stage4_fuse.shape}")

        return torch.cat([stage1_fuse, stage2_fuse, stage3_fuse, stage4_fuse])

    def forward(self, x):

        forward_feat = self._feature_forward(x)


        return forward_feat




def Bi_CnnFormer(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios:
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = Bi_CnnFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = basic_config['poolformer_s']
    if pretrained:
        url = basic_config['poolformer_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    #cnb = ConvNext_branch()
    #y_cnb = cnb(x)
    #print(f"cnb ")
    model = Wnet()
    y = model(x)


