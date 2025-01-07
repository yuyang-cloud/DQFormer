import functools
import warnings
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.core import ConvAlgo
from collections import OrderedDict
from torch_scatter import scatter_mean
import torch.nn.functional as F
from dqformer.models.backbone.spherical_transformer import SphereFormer
from dqformer.utils.interpolate import knn_up

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = (pair_in != -1)
    valid_pair_in, valid_pair_out = pair_in[valid_mask].long(), pair_out[valid_mask].long()
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next


class UBlock(nn.Module):
    def __init__(self, nPlanes, 
        norm_fn, 
        block_reps, 
        block, 
        window_size, 
        window_size_sphere, 
        quant_size, 
        quant_size_sphere, 
        head_dim=16, 
        window_size_scale=[2.0, 2.0], 
        rel_query=True, 
        rel_key=True, 
        rel_value=True, 
        drop_path=0.0,
        indice_key_id=1, 
        grad_checkpoint_layers=[], 
        sphere_layers=[1,2,3,4,5],
        a=0.05*0.25,
    ):

        super().__init__()

        self.nPlanes = nPlanes
        self.indice_key_id = indice_key_id
        self.grad_checkpoint_layers = grad_checkpoint_layers
        self.sphere_layers = sphere_layers

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if indice_key_id in sphere_layers:
            self.window_size = window_size
            self.window_size_sphere = window_size_sphere
            num_heads = nPlanes[0] // head_dim
            self.transformer_block = SphereFormer(
                nPlanes[0], 
                num_heads, 
                window_size, 
                window_size_sphere, 
                quant_size, 
                quant_size_sphere,
                indice_key='sphereformer{}'.format(indice_key_id),
                rel_query=rel_query, 
                rel_key=rel_key, 
                rel_value=rel_value, 
                drop_path=drop_path[0],
                a=a,
            )

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )

            window_size_scale_cubic, window_size_scale_sphere = window_size_scale
            window_size_next = np.array([
                window_size[0]*window_size_scale_cubic, 
                window_size[1]*window_size_scale_cubic, 
                window_size[2]*window_size_scale_cubic
            ])
            quant_size_next = np.array([
                quant_size[0]*window_size_scale_cubic, 
                quant_size[1]*window_size_scale_cubic, 
                quant_size[2]*window_size_scale_cubic
            ])
            window_size_sphere_next = np.array([
                window_size_sphere[0]*window_size_scale_sphere, 
                window_size_sphere[1]*window_size_scale_sphere, 
                window_size_sphere[2]
            ])
            quant_size_sphere_next = np.array([
                quant_size_sphere[0]*window_size_scale_sphere, 
                quant_size_sphere[1]*window_size_scale_sphere, 
                quant_size_sphere[2]
            ])
            self.u = UBlock(nPlanes[1:], 
                norm_fn, 
                block_reps, 
                block, 
                window_size_next, 
                window_size_sphere_next, 
                quant_size_next, 
                quant_size_sphere_next, 
                window_size_scale=window_size_scale, 
                rel_query=rel_query, 
                rel_key=rel_key, 
                rel_value=rel_value, 
                drop_path=drop_path[1:],
                indice_key_id=indice_key_id+1, 
                grad_checkpoint_layers=grad_checkpoint_layers, 
                sphere_layers=sphere_layers,
                a=a
            )

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, inp, xyz, batch, output_list, xyz_list, batch_list):

        assert (inp.indices[:, 0] == batch).all()
        
        output = self.blocks(inp)

        # transformer
        if self.indice_key_id in self.sphere_layers:
            if self.indice_key_id in self.grad_checkpoint_layers:
                def run(feats_, xyz_, batch_):
                    return self.transformer_block(feats_, xyz_, batch_)
                transformer_features = torch.utils.checkpoint.checkpoint(run, output.features, xyz, batch)
            else:
                transformer_features = self.transformer_block(output.features, xyz, batch)
            output = output.replace_feature(transformer_features)

        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)

            # downsample
            indice_pairs = output_decoder.indice_dict['spconv{}'.format(self.indice_key_id)].indice_pairs
            xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)
            xyz_list.append(xyz_next)
            batch_list.append(batch_next)

            output_decoder, output_list, xyz_list, batch_list = self.u(output_decoder, xyz_next, batch_next.long(), output_list, xyz_list, batch_list)
            output_decoder = self.deconv(output_decoder)
            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        output_list.append(output)
        return output, output_list, xyz_list, batch_list


class Semantic(nn.Module):
    def __init__(self, 
        input_c=4, 
        m=32, 
        classes=19, 
        block_reps=2, 
        block_residual=True, 
        layers=[32,64,128,256,256], 
        window_size=np.array([0.3, 0.3, 0.3], dtype=np.float32), 
        window_size_sphere=np.array([2, 2, 80]), 
        quant_size=np.array([0.3, 0.3, 0.3], dtype=np.float32)/24, 
        quant_size_sphere=np.array([2, 2, 80])/24, 
        rel_query=True, 
        rel_key=True, 
        rel_value=True, 
        drop_path_rate=0.3, 
        window_size_scale=[2.0, 1.5], 
        grad_checkpoint_layers=[], 
        sphere_layers=[1,2,3,4,5],
        a=0.0125,
    ):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock(layers, 
            norm_fn, 
            block_reps, 
            block, 
            window_size, 
            window_size_sphere, 
            quant_size, 
            quant_size_sphere, 
            window_size_scale=window_size_scale, 
            rel_query=rel_query, 
            rel_key=rel_key, 
            rel_value=rel_value, 
            drop_path=dpr,
            indice_key_id=1,
            grad_checkpoint_layers=grad_checkpoint_layers, 
            sphere_layers=sphere_layers,
            a=a,
        )

        self.knn_up = knn_up(3)
        self.out_bnorm = nn.ModuleList([nn.BatchNorm1d(l) for l in [256, 128, 64, 32]])

        #### semantic segmentation
        self.stuff_class = [9,10,11,12,13,14,15,16,17,18,19] 
        self.sem_head = nn.Linear(m, 20)

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def pad_batch(self, coors, feats):
        """
        From a list of multi-level features create a list of batched tensors with
        features padded to the max number of points in the batch.

        returns:
            feats: List of batched feature Tensors per feature level
            coors: List of batched coordinate Tensors per feature level
            pad_masks: List of batched bool Tensors indicating padding
        """
        # get max number of points in the batch for each feature level
        maxs = [max([level.shape[0] for level in batch]) for batch in feats]
        # pad and batch each feature level in a single Tensor
        coors = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(coors)
        ]
        pad_masks = [
            torch.stack(
                [
                    F.pad(
                        torch.zeros_like(f[:, 0]), (0, maxs[i] - f.shape[0]), value=1
                    ).bool()
                    for f in batch
                ]
            )
            for i, batch in enumerate(feats)
        ]
        feats = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(feats)
        ]
        return feats, coors, pad_masks

    def prepare_data(self, input):
        offset_ = input["offset"].clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().to(input["vx_coord"])

        coord = input["vx_coord"]   # vox_coord
        coord = torch.cat([torch.clamp(coord[:,0],0,2048)[:,None], torch.clamp(coord[:,1],0,2048)[:,None], torch.clamp(coord[:,2],0,128)[:,None]], dim=-1)
        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        # coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)
        # spatial_shape = np.clip((coord.max(0)[0][1:] + 1).detach().cpu().numpy(), (2048, 2048, 128), None)  # (51.2-(-51.2)) / voxel_size_0.05 = 2048
        spatial_shape = np.array([2048, 2048, 128])
    
        xyz, feat = input["pt_coord"], input["feats"]
        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, batch_size=len(offset_))

        if 'inds_recons' in input.keys():
            return sinput, xyz, batch, input["inds_recons"]
        else:
            return sinput, xyz, batch, None

    def forward(self, input):
        '''
        :param input_map: (N), int, cuda
        '''
        sinput, xyz, batch, inds_recons = self.prepare_data(input)

        output = self.input_conv(sinput)
        output_list, xyz_list, batch_list = [], [xyz], [batch]
        output, output_list, xyz_list, batch_list = self.unet(output, xyz, batch, output_list, xyz_list, batch_list)

        output_list = output_list[1:]       # [N/8,256  N/4,128  N/2,64  N,32]
        xyz_list = xyz_list[::-1][1:]       # [N/8,3    N/4,3    N/2,3   N,3]
        batch_list = batch_list[::-1][1:]   # [N/8      N/4      N/2     N]

        vox_feats = []  # [N1/8,C N2/8,C    N1/4,C  N2/4,C   N1/2,C  N2/2,C   N1,C N2,C]
        vox_coors = []  # [N1/8,3 N2/8,3    N1/4,3  N2/4,3   N1/2,3  N2/2,3   N1,3 N2,3]
        for output, xyz, batch in zip(output_list, xyz_list, batch_list):
            vox_f  = [output.features[output.indices[:, 0]==b] for b in range(output.batch_size)]
            vox_c = [xyz[batch==b] for b in range(output.batch_size)]
            vox_feats.append(vox_f)
            vox_coors.append(vox_c)
        
        # knn -> point-wise
        pt_coors = [[xyz[batch==b] for b in range(output.batch_size)] for _ in range(4)]  # 4 layers
        pt_feats = [
            [
                bn(self.knn_up(vox_c, vox_f, pt_c))
                for vox_c, vox_f, pt_c in zip(vc, vf, pc)   # for batch
            ]
            for vc, vf, pc, bn in zip(vox_coors, vox_feats, pt_coors, self.out_bnorm) # for layers
        ]

        # inds -> org (val)
        if inds_recons:
            inds_recons = [inds_recons for _ in range(4)]
            for i, (pf_s, pc_s, idx_s) in enumerate(zip(pt_feats, pt_coors, inds_recons)):
                pt_feats[i] = [pf[idx, :] for pf, idx in zip(pf_s, idx_s)]
                pt_coors[i] = [pc[idx, :] for pc, idx in zip(pc_s, idx_s)]

        # padding
        pt_feats, pt_coors, pad_masks = self.pad_batch(pt_coors, pt_feats)

        #### semantic segmentation
        logits = self.sem_head(pt_feats[-1])
        batch_size = logits.shape[0]
        sem_queries = self.sem_head.weight.clone().unsqueeze(0).repeat(batch_size,1,1) # bs, 20, C
        stuff_queries = sem_queries[:, self.stuff_class, :]
        stuff_queries = sem_queries

        # last_vox_feat
        last_vox_feat = output_list[0]
        
        return last_vox_feat, pt_feats, pt_coors, stuff_queries, pad_masks, logits

