import numpy as np
import copy
import torch
import spconv.pytorch as spconv
from torch import nn
from torch.nn import functional as F
from dqformer.models.thing_neck import ChannelAttention, SpatialAttention
from dqformer.models.transformer_arch.positional_encoding import SinePositionalEncoding
from dqformer.models.transformer_arch.mask_head import MaskHead
from dqformer.models.transformer_arch.dice_loss import DiceLoss
from dqformer.utils.misc import Sequential
from dqformer.utils.misc import get_world_size, is_dist_avail_and_initialized
from dqformer.utils.norm import build_norm_layer

class RPN_with_Decoder(nn.Module):
    def __init__(
        self,
        bb_cfg,
        decoder_cfg,
        layer_nums=[3, 3, 1],
        ds_num_filters=[256, 256, 128],
        num_input_features=256,
        use_gt_training=True,
    ):
        super(RPN_with_Decoder, self).__init__()
        self._layer_strides = [1, 2, -4]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features
        self.use_gt_training = use_gt_training
        self._norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.input_conv = spconv.SparseSequential(
            spconv.SparseConv3d(256, 128, kernel_size=(1, 1, 4), stride=(1, 1, 3), bias=False),  # D: 16 -> 5
            build_norm_layer(norm_cfg, 128)[1],
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseConv3d(128, 128, kernel_size=(1, 1, 3), stride=(1, 1, 2), bias=False),  # D: 5 -> 2
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [
            self._num_input_features,
            self._num_filters[0],
            self._num_filters[1],
        ]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.up = Sequential(
            nn.ConvTranspose2d(
                self._num_filters[0], self._num_filters[2], 2, stride=2, bias=False
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[2])[1],
            nn.ReLU(),
        )

        # query
        self.num_class = 20
        self.stuff_class = [9,10,11,12,13,14,15,16,17,18,19] 
        self.num_class_stuff = 11
        self.use_dynamic = True
        self.embed_dims = self._num_filters[2] * 2  # 256
        if self.use_dynamic:
            self.query_feat = nn.Embedding(self.num_class_stuff, self.embed_dims)
        else:
            self.stuff_query_proj = nn.Linear(bb_cfg.CHANNELS[-1], self.embed_dims)
        self.query_embed = nn.Embedding(self.num_class_stuff, self.embed_dims)

        # Transformer
        num_levels = 3  # feat memory level
        self.num_layers = 3  # decoder layer
        self.stuff_mask_head = MaskHead(d_model=self.embed_dims, 
                                        nhead=8,
                                        num_decoder_layers=self.num_layers,
                                        self_attn=True,)
        
        self.positional_encoding = SinePositionalEncoding(num_feats=self.embed_dims//2,
                                                          normalize=True,
                                                          offset=-0.5)
        self.level_embeds = nn.Parameter(torch.Tensor(num_levels, self.embed_dims))
        nn.init.normal_(self.level_embeds)

        # mask_loss
        # self.mask_loss = FastFocalLoss_Stuff()
        self.loss_mask = DiceLoss(loss_weight=2.0)
    
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        if stride > 0:
            block = Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )
        else:
            block = Sequential(
                nn.ConvTranspose2d(
                    inplanes, planes, -stride, stride=-stride, bias=False
                ),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
            )
            block.add(nn.ReLU())

        block.add(ChannelAttention(planes))
        block.add(SpatialAttention())

        return block, planes

    def forward(self, vox_feat, stuff_queries, example):
        # x: sp_tensor  B,C=256,H/8,W/8,D=16
        vox_feat = self.input_conv(vox_feat)    # B,C=128,H/8,W/8,D=2
        vox_feat = vox_feat.dense().permute(0,1,4,2,3)
        N, C, D, H, W = vox_feat.shape
        x = vox_feat.contiguous().view(N, C * D, H, W)

        # FPN
        x_mid = self.blocks[0](x)   # H,W=256
        x_down = self.blocks[1](x_mid)  # H,W=128
        x_up = torch.cat([self.blocks[2](x_down), self.up(x_mid)], dim=1)   # H,W=512

        # attn mask
        if self.use_gt_training and self.stuff_mask_head.training:
            mask_up = (example>0).float().to(x_up)
            mask_mid = F.interpolate(mask_up.unsqueeze(1), size=(mask_up.shape[1]//2, mask_up.shape[2]//2), mode='nearest').squeeze(1)
            mask_down = F.interpolate(mask_mid.unsqueeze(1), size=(mask_mid.shape[1]//2, mask_mid.shape[2]//2), mode='nearest').squeeze(1)
        else:
            mask_up = torch.cat([(feat.max(0)[0]==0).float()[None] for feat in x_up], dim=0)
            mask_mid = torch.cat([(feat.max(0)[0]==0).float()[None] for feat in x_mid], dim=0)
            mask_down = torch.cat([(feat.max(0)[0]==0).float()[None] for feat in x_down], dim=0)

        # memory
        mlvl_feats = [x_up, x_mid, x_down]
        mlvl_masks = [mask_up, mask_mid, mask_down]
        mlvl_pos_embeds = [self.positional_encoding(mask) 
                                     for mask in mlvl_masks]
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        hw_lvl = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            feat = feat.flatten(2).transpose(1, 2)  # B, HW, C
            mask = mask.flatten(1)  # B, HW
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # B, HW, C
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)   # B, HW, C
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            hw_lvl.append([h, w])
        feat_flatten = torch.cat(feat_flatten, 1)   # B, layer*HW, C
        mask_flatten = torch.cat(mask_flatten, 1)   # B, layer*HW
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # B, layer*HW, C

        # query
        if self.use_dynamic:
            stuff_query = self.query_feat.weight.unsqueeze(0).repeat(N, 1, 1)   # B,Nq,C
        else:
            stuff_query = self.stuff_query_proj(stuff_queries)
        stuff_query_pos = self.query_embed.weight.unsqueeze(0).repeat(N, 1, 1)

        # Transformer
        mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
            memory=feat_flatten,
            query_embed=stuff_query,
            pos_memory=lvl_pos_embed_flatten,
            pos_query=stuff_query_pos,
            mask_memory=mask_flatten,
            mask_query=None,
            hw_lvl=hw_lvl)
        
        mask_stuff = mask_stuff.squeeze(-1) # B, Nq, HW
        mask_inter_stuff = torch.stack(mask_inter_stuff, 0).squeeze(-1) # layer,B,Nq,HW

        mask_preds_stuff = []
        mask_preds_inter_stuff = [[] for _ in range(self.num_layers)]
        for i in range(N):
            tmp_i = mask_stuff[i].reshape(-1, *hw_lvl[0])   # Nq,H,W
            mask_preds_stuff.append(tmp_i)
            for j in range(self.num_layers):
                tmp_i_j = mask_inter_stuff[j][i].reshape(-1, *hw_lvl[0])
                mask_preds_inter_stuff[j].append(tmp_i_j)

        mask_preds_stuff = torch.cat(mask_preds_stuff, 0)
        mask_preds_inter_stuff = [
            torch.cat(each, 0) for each in mask_preds_inter_stuff
        ]

        # loss
        if example is not None:
            # example: B,H,W   bev_sem 0~19
            target_mask = F.one_hot(example.long(), num_classes=self.num_class)    # B,H,W -> B,H,W,C  one-hot
            target_mask = target_mask[..., self.stuff_class]
            target_mask = target_mask.permute(0, 3, 1, 2).float()  # [bsz, cls, h, w]
            target_mask = target_mask.flatten(0, 1)  # [bsz*cls, h, w]
            target_mask_weight = target_mask.max(2)[0].max(1)[0]  # [bsz*cls]
            # pos_num
            num_total_pos_stuff = target_mask_weight.sum()
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_total_pos_stuff)
            num_total_pos_stuff = torch.clamp(num_total_pos_stuff / get_world_size(), min=1).item()

            loss_dict = {}
            loss_mask_stuff = self.loss_mask(mask_preds_stuff,
                                             target_mask,
                                             target_mask_weight,
                                             avg_factor=num_total_pos_stuff)
            loss_dict['loss_bev_sem_mask'] = 5.0 * loss_mask_stuff
            
            for j in range(len(mask_preds_inter_stuff)):
                mask_preds_this_level = mask_preds_inter_stuff[j]
                loss_mask_j = self.loss_mask(mask_preds_this_level,
                                             target_mask,
                                             target_mask_weight,
                                             avg_factor=num_total_pos_stuff)
                loss_dict[f'd{j}.loss_bev_sem_mask'] = 5.0 * loss_mask_j

            return query_inter_stuff[-1], loss_dict

        else:
            loss_dict = {}
            return query_inter_stuff[-1], loss_dict
