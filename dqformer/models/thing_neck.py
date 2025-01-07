import time
import numpy as np

import torch
import spconv.pytorch as spconv
from torch import nn
from torch.nn import functional as F
from dqformer.utils.misc import Sequential
from dqformer.utils.norm import build_norm_layer
from dqformer.utils.visulize import vis_heatmap_pooled


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x


class SpatialAttention_mtf(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_mtf, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, curr, prev):
        avg_out = torch.mean(curr, dim=1, keepdim=True)
        max_out, _ = torch.max(curr, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * prev


class RPN_base_multitask(nn.Module):
    def __init__(
        self,
        dataset,
        layer_nums=[5, 5, 1],
        ds_num_filters=[256, 256, 128],
        num_input_features=256,
        hm_head_layer=2,
        assign_label_window_size=1,
        use_gt_training=True,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
    ):
        super(RPN_base_multitask, self).__init__()
        self._layer_strides = [1, 2, -4]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features
        self.score_threshold = score_threshold
        self.obj_num = obj_num
        self.use_gt_training = use_gt_training
        self.window_size = assign_label_window_size**2
        self.batch_id = None
        self._norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        # dataset config
        self.tasks = dataset.TASK
        self.obj_num = dataset.ASSIGNER.max_objs
        self.out_size_factor = dataset.ASSIGNER.out_size_factor
        self.voxel_size = dataset.ASSIGNER.voxel_size
        self.pc_range = dataset.ASSIGNER.pc_range


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
        # heatmap prediction
        self.hm_heads = nn.ModuleList()
        if 'kernel_size' in self.tasks[0].keys():
            self.hm_pool = True
            self.max_pool_s = nn.ModuleList()
        else:
            self.hm_pool = False
        for task in self.tasks:
            hm_head = Sequential()
            for i in range(hm_head_layer - 1):
                hm_head.add(
                    nn.Conv2d(
                        self._num_filters[-1] * 2,
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                hm_head.add(build_norm_layer(self._norm_cfg, 64)[1])
                hm_head.add(nn.ReLU())

            hm_head.add(
                nn.Conv2d(64, task['num_class'], kernel_size=3, stride=1, padding=1, bias=True)
            )
            hm_head[-1].bias.data.fill_(init_bias)
            self.hm_heads.append(hm_head)
            if self.hm_pool:
                self.max_pool_s.append(nn.MaxPool2d(kernel_size=task['kernel_size'], stride=task['stride']))

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

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m, gain=1)
    
    def get_center_pos(self, order, hm, task_id):
        # order: B, N_obj
        batch, _, H, W = hm.size()

        xs, ys = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])   # H,W
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(hm)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(hm)

        obj_num = order.shape[1]
        batch_id = np.indices((batch, obj_num))[0]
        batch_id = torch.from_numpy(batch_id).to(hm).type_as(order)

        xs = (
            xs.view(batch, -1, 1)[batch_id, order]    # bs,H*W,1 -> bs,N_obj  hm_idx_W
        )
        ys = (
            ys.view(batch, -1, 1)[batch_id, order]    # bs,H*W,1 -> bs,N_obj  hm_idx_H
        )

        if self.hm_pool:
            out_size_factor = self.out_size_factor * self.tasks[task_id]['stride']
        else:
            out_size_factor = self.out_size_factor
        xs = (
            xs * out_size_factor * self.voxel_size[0]
            + self.pc_range[0]
        )
        ys = (
            ys * out_size_factor * self.voxel_size[1]
            + self.pc_range[1]
            )
        
        xy = torch.cat([xs, ys], dim=2) # B, N_obj, 2
        return xy

    def forward(self, x, example=None):
        pass


class RPN_multitask(RPN_base_multitask):
    def __init__(
        self,
        dataset,
        layer_nums=[5, 5, 1],
        ds_num_filters=[256, 256, 128],
        num_input_features=256,
        hm_head_layer=2,
        assign_label_window_size=1,
        use_gt_training=True,
        init_bias=-2.19,
        score_threshold=0.1,
    ):
        super(RPN_multitask, self).__init__(
            dataset,
            layer_nums,
            ds_num_filters,
            num_input_features,
            hm_head_layer,
            assign_label_window_size,
            use_gt_training,
            init_bias,
            score_threshold,
        )

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


    def forward(self, vox_feat, example=None):
        # x: sp_tensor  B,C=256,H/8,W/8,D=16
        vox_feat = self.input_conv(vox_feat)    # B,C=128,H/8,W/8,D=2
        vox_feat = vox_feat.dense().permute(0,1,4,2,3)
        N, C, D, H, W = vox_feat.shape
        x = vox_feat.contiguous().view(N, C * D, H, W)

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        order_list = []
        out_dict_list = []

        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up)

            # find top K center location
            hm = torch.sigmoid(hm)  # B,C,512,512
            hm_pred = hm
            if self.hm_pool:
                hm = self.max_pool_s[idx](hm)
                x_up_task = self.max_pool_s[idx](x_up)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                if self.hm_pool:
                    gt_inds = example["ind_pooled"][idx][:, (self.window_size // 2) :: self.window_size].to(labels)
                else:
                    gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size].to(labels)
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ].to(labels)
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order) # B,N_obj
            labels = torch.gather(labels, 1, order) # B,N_obj
            mask = scores > self.score_threshold
            xy = self.get_center_pos(order, hm, idx)
            order_list.append(order)

            batch_id = torch.from_numpy(np.indices((batch, self.obj_num ))[0]).to(
                labels
            )
            ct_feat = (
                x_up_task.reshape(batch, -1, H * W)
                .transpose(2, 1)
                .contiguous()[batch_id, order]
            )  # B, tasks*obj_num, C

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm_pred,   # B,n_cls,H,W
                    "xy": xy,           # B,N_obj,2
                    "scores": scores,   # B,N_obj
                    "labels": labels,   # B,N_obj
                    "order": order,     # B,N_obj   index
                    "mask": mask,       # B,N_obj   scores > threshold
                    "ct_feat": ct_feat,
                }
            )
            out_dict_list.append(out_dict)

        return out_dict_list
