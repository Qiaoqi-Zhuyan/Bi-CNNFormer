import einops

from models.fusion import Attn_fusion_Block
from models.transformer.swin_transformer_fork import *
from models.cnn.convnet_fork import ConvNextBasicLayer, LayerNorm

from config.config import get_config

from torch import nn
import torch

basic_config = get_config()


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):

        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = activation if activation is not None else nn.Identity()
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = activation if activation is not None else nn.Identity()
        super().__init__(pool, flatten, dropout, linear, activation)


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


class DecoderBlock(nn.Module):
    def __init__(self, in_chans, skip_chans, out_chans,
                 act_layer=nn.ReLU, upsample=nn.UpsamplingBilinear2d(scale_factor=2)):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        self.act_layer = act_layer()
        self.in_chans = in_chans
        self.skip_chans = skip_chans
        self.out_chans = out_chans
        in_chans = in_chans + skip_chans
        self.conv_up = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=1)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            #print(f"x shape : {x.shape}, skip shape : {skip.shape}")
            x = torch.cat([x, skip], dim=1)
        x = self.conv_up(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = self.act_layer(x)

        return x


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
    def __init__(self, dims=None, layers=None, num_heads=None, num_classes=9, img_size=224,
                 patch_size=4, in_chans=3, window_size=7, mlp_ratio=4, embed_dim=64,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 cls_act_layer = None, seg_act_layer = None,
                 norm_layer=nn.LayerNorm, cls_head=False, patch_norm=True, fused_window_process=False,
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, head_init_scale=1.):
        super(Wnet, self).__init__()

        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if dims is None:
            dims = [96, 192, 384, 768]
        if layers is None:
            layers = [2, 2, 6, 2]

        self.img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.cls_head = cls_head

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

        '''
                num_heads = [3, 6, 12, 24]
        
                dims = [96, 192, 384, 768]
        '''
        self.AFB_stem   = Attn_fusion_Block(in_chans=dims[0], num_heads=fusion_heads[0], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage1 = Attn_fusion_Block(in_chans=dims[1], num_heads=fusion_heads[1], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage2 = Attn_fusion_Block(in_chans=dims[2], num_heads=fusion_heads[2], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage3 = Attn_fusion_Block(in_chans=dims[3], num_heads=fusion_heads[3], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)
        self.AFB_stage4 = Attn_fusion_Block(in_chans=dims[3], num_heads=fusion_heads[3], fuse_type="add",
                                            attn_drop=0.0,drop=0.0, drop_path=0.0, bias=True)

        # decoder

        self.decoder_out = DecoderBlock(in_chans=64, skip_chans=0, out_chans=16,
                                        act_layer=nn.ReLU, upsample=nn.UpsamplingBilinear2d(scale_factor=4))

        self.decoder_stage0 = DecoderBlock(in_chans=dims[0], skip_chans=dims[0], out_chans=64,
                                         act_layer=nn.ReLU, upsample=nn.UpsamplingBilinear2d(scale_factor=2))
        self.decoder_stage1 = DecoderBlock(in_chans=dims[1], skip_chans=dims[1], out_chans=dims[0],
                                         act_layer=nn.ReLU, upsample=nn.UpsamplingBilinear2d(scale_factor=2))
        self.decoder_stage2 = DecoderBlock(in_chans=dims[2], skip_chans=dims[2], out_chans=dims[1],
                                         act_layer=nn.ReLU, upsample=nn.UpsamplingBilinear2d(scale_factor=2))
        self.decoder_stage3 = DecoderBlock(in_chans=dims[3], skip_chans=dims[3], out_chans=dims[2],
                                         act_layer=nn.ReLU, upsample=nn.UpsamplingBilinear2d(scale_factor=1))
        self.decoder_stage4 = DecoderBlock(in_chans=dims[3], skip_chans=0, out_chans=dims[3],
                                         act_layer=nn.ReLU, upsample=nn.UpsamplingBilinear2d(scale_factor=1))
        self.norm = nn.Identity()

        self.segmentation_head = SegmentationHead(in_channels=16, out_channels=num_classes, kernel_size=3, activation=seg_act_layer, upsampling=1)

        if cls_head:
            self.classification_head = ClassificationHead(in_channels=dims[3], classes=self.num_classes, pooling="avg", dropout=0.2, activation=cls_act_layer)
        else:
            self.classification_head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _encoder_forward(self, x_c, x_t):

        # embedding: [B C H W] -> [B C' H/4, W/4]
        x_c = self.cnn_stem(x_c)
        x_t = self.trans_stem(x_t)

        print(f"input x_c: {x_c.shape}, x_t: {x_t.shape},\n"
              f"embedding x_c: {x_c.shape}, embedding x_t: {x_t.shape}")

        _, _, h, w = x_c.shape
        x_t = einops.rearrange(x_t, "b (h w) c -> b c h w", h=h, w=w)
        print(f"stage 1: x_t: {x_t.shape}, x_c: {x_c.shape}")
        # stem Fusion [B C' H/4, W/4] -> [B C' H/4, W/4]
        stem_fuse = self.AFB_stem.forward(x_c, x_t)
        x_t = einops.rearrange(x_t, "b c h w  -> b (h w) c", h=h, w=w)
        print(f"stage1 fuse: {stem_fuse.shape}")

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

        return stem_fuse,stage1_fuse, stage2_fuse, stage3_fuse, stage4_fuse

    def _decoder_forward(self, stem_fuse, stage1_fuse, stage2_fuse, stage3_fuse, stage4_fuse):
        #stem_fuse, stage1_fuse, stage2_fuse, stage3_fuse, stage4_fuse = self._encoder_forward(x_c, x_t)
        print(f"stem_fuse: {stem_fuse.shape} ,stage1_fuse: {stage1_fuse.shape}, stage2_fuse: {stage2_fuse.shape} \n, "
              f"stage3_fuse: {stage3_fuse.shape}, stage4_fuse: {stage4_fuse.shape} ")
        decode_feat = self.decoder_stage4(stage4_fuse, None)
        print(f'decoder_stage4 : {decode_feat.shape}')

        decode_feat = self.decoder_stage3(decode_feat, stage3_fuse)
        print(f'decoder_stage3 : {decode_feat.shape}')

        decode_feat = self.decoder_stage2(decode_feat, stage2_fuse)
        print(f'decoder_stage2 : {decode_feat.shape}')

        decode_feat = self.decoder_stage1(decode_feat, stage1_fuse)
        print(f'decoder_stage1 : {decode_feat.shape}')

        decode_feat = self.decoder_stage0(decode_feat, stem_fuse)
        print(f'decoder_stem : {decode_feat.shape}')

        decode_feat = self.decoder_out(decode_feat, None)
        print(f'decode_feat : {decode_feat.shape}')

        return decode_feat

    def forward(self, x_c, x_t):

        encoder_feat = self._encoder_forward(x_c, x_t)

        decoder_feat = self._decoder_forward(*encoder_feat)

        mask = self.segmentation_head(decoder_feat)

        if self.cls_head:
            cls_feat = encoder_feat[-1]
            #cls_feat = einops.rearrange(cls_feat, "b c h w -> b h w c")
            #print(f"cls_feat: {cls_feat.shape}")
            labels = self.classification_head(cls_feat)
            print(f"mask shape : {mask.shape}, labels shape : {labels.shape}")
            return mask, labels

        print(f"mask shape : {mask.shape}")

        return mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    x_c = torch.randn(1, 3, 224, 224)
    x_t = torch.randn(1, 3, 224, 224)

    model = Wnet(
        dims=None, layers=None, num_heads=None, num_classes=9, img_size=224,
        patch_size=4, in_chans=3, window_size=7, mlp_ratio=4, embed_dim=64,
        qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
        cls_act_layer=nn.ReLU(inplace=True), seg_act_layer=nn.ReLU(inplace=True),
        norm_layer=nn.LayerNorm, cls_head=True, patch_norm=True, fused_window_process=False,
        drop_path_rate=0.1, layer_scale_init_value=1e-6, head_init_scale=1.
    )


    params= count_parameters(model)
    print(f"model params : {params}")
    y = model(x_c, x_t)




