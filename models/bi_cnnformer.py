from models.fusion import BiFusion_block
from transformer.metaformer_fork import MetaFormer
from cnn.convnet_fork import ConvNeXt

from torch import nn
import torch

class Bi_CnnFormer(nn.Module):
    def __init__(self, ):
        super(Bi_CnnFormer, self).__init__()
        self.cnn = ConvNeXt(

        )

        self.transformer = MetaFormer(


        )

        self.fusion = BiFusion_block(

        )


        self.norm = nn.Identity()

        self.decoder = nn.Identity()

        self.fuse_feat_proc = nn.Identity()

        self.head = nn.Identity()


    def _feature_forward(self, x):
        transformer_feat_stage = self.transformer._fork_forward(x)
        cnn_feat_stage = self.cnn._fork_forward(x)

        fuse_feats = []
        for idx in range(4):
            fuse_feats[idx] = self.fusion.forward(cnn_feat_stage[idx], transformer_feat_stage[idx])

        return fuse_feats

    def forward(self, x):
        transformer_feat = self.transformer.forward(x)
        cnn_feat = self.cnn.forward(x)

        fuse_feats = self._feature_forward(x)

        for i in range(4):
            feats = self.fuse_feat_proc(fuse_feats[i])

        out = self.head(torch.cat([transformer_feat, cnn_feat, feats], dim=-1))

        return out

