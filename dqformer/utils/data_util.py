import random
import torch
import numpy as np
import spconv.pytorch as spconv
import torch_scatter
import cv2
from torch_scatter import scatter_mean
from dqformer.utils.voxelize import voxelize

def collate_fn_limit(batch, max_batch_points, logger):
    coord, xyz, feat, label = list(zip(*batch))
    offset, count = [], 0
    
    new_coord, new_xyz, new_feat, new_label = [], [], [], []
    k = 0
    for i, item in enumerate(xyz):

        count += item.shape[0]
        if count > max_batch_points:
            break

        k += 1
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(label[i])

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in xyz])
        s_now = sum([x.shape[0] for x in new_xyz[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k]), torch.cat(new_label[:k]), torch.IntTensor(offset[:k])
    

def collation_fn_voxelmean(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, xyz, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    offset = []
    for i in range(len(coords)):
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        offset.append(accmulate_points_num)

    coords = torch.cat(coords)
    xyz = torch.cat(xyz)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    offset = torch.IntTensor(offset)
    inds_recons = torch.cat(inds_recons)

    return coords, xyz, feats, labels, offset, inds_recons

def collation_fn_voxelmean_tta(batch_list):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    samples = []
    batch_list = list(zip(*batch_list))

    for batch in batch_list:
        coords, xyz, feats, labels, inds_recons = list(zip(*batch))
        inds_recons = list(inds_recons)

        accmulate_points_num = 0
        offset = []
        for i in range(len(coords)):
            inds_recons[i] = accmulate_points_num + inds_recons[i]
            accmulate_points_num += coords[i].shape[0]
            offset.append(accmulate_points_num)

        coords = torch.cat(coords)
        xyz = torch.cat(xyz)
        feats = torch.cat(feats)
        labels = torch.cat(labels)
        offset = torch.IntTensor(offset)
        inds_recons = torch.cat(inds_recons)

        sample = (coords, xyz, feats, labels, offset, inds_recons)
        samples.append(sample)

    return samples

def data_prepare(coord, feat, sem_labels, ins_labels, masks, split='train', voxel_size=np.array([0.05, 0.05, 0.05]), voxel_max=120000, xyz_norm=False):
    # coord_min = np.min(coord, 0)
    coord_min = np.array([-51.2, -51.2, -4.0])
    # coord -= coord_min
    coord_norm = coord - coord_min
    if split == 'train':
        uniq_idx = voxelize(coord_norm, voxel_size)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))
        coord, feat, sem_labels, ins_labels, masks = coord[uniq_idx], feat[uniq_idx], sem_labels[uniq_idx], ins_labels[uniq_idx], masks[:, uniq_idx]
        if voxel_max and sem_labels.shape[0] > voxel_max:
            init_idx = np.random.randint(sem_labels.shape[0])
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            coord, feat, sem_labels, ins_labels, masks = coord[crop_idx], feat[crop_idx], sem_labels[crop_idx], ins_labels[crop_idx], masks[:, crop_idx]
            coord_voxel = coord_voxel[crop_idx]
    else:
        idx_recon = voxelize(coord_norm, voxel_size, mode=1)

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    # label = torch.LongTensor(label)
    if split == 'train':
        coord_voxel = torch.LongTensor(coord_voxel)
        return coord_voxel, coord, feat, sem_labels, ins_labels, masks
    else:
        coord_norm = torch.FloatTensor(coord_norm)
        idx_recon = torch.LongTensor(idx_recon)
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coords_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
        pt_coord_org = coord
        pt_coord_sub = scatter_mean(coord, idx_recon, dim=0)
        feat = scatter_mean(feat, idx_recon, dim=0)
        voxel_sem_labels = get_voxel_sem(idx_recon, sem_labels)
        return coords_voxel, pt_coord_org, pt_coord_sub, feat, voxel_sem_labels, sem_labels, ins_labels, masks, idx_recon

def get_voxel_sem(idx_recon, pt_sem_labels):
    pt_sem_labels = torch.from_numpy(pt_sem_labels)
    voxel_sem = torch.cat((idx_recon.unsqueeze(-1), pt_sem_labels), dim=1)
    unq_voxel_sem, unq_sem_count = torch.unique(voxel_sem, return_counts=True, dim=0)
    unq_voxel, unq_ind = torch.unique(unq_voxel_sem[:, 0], return_inverse=True, dim=0)
    label_max_ind = torch_scatter.scatter_max(unq_sem_count, unq_ind)[1]
    unq_sem = unq_voxel_sem[:, -1][label_max_ind]
    return unq_sem.unsqueeze(-1).numpy()  # N_voxel, 1

def get_bev_sem(coord_voxel, coord_sem, ds_factor=4):
    # coord_voxel: N, 3
    # coord_sem: N,1  np.array
    coord_voxel = torch.cat([torch.clamp(coord_voxel[:,0],0,2047)[:,None], torch.clamp(coord_voxel[:,1],0,2047)[:,None], torch.clamp(coord_voxel[:,2],0,128)[:,None]], dim=-1)
    coord_sem = torch.from_numpy(coord_sem)   # N,1
    bev_sem = torch.cat([coord_voxel[:, :2], coord_sem], dim=-1) # H,W,C
    unq_bev_sem, unq_sem_count = torch.unique(bev_sem, return_counts=True, dim=0)
    unq_bev, unq_ind = torch.unique(unq_bev_sem[:, :2], return_inverse=True, dim=0)
    label_max_ind = torch_scatter.scatter_max(unq_sem_count, unq_ind)[1]
    unq_sem = unq_bev_sem[:, -1][label_max_ind]

    batch = torch.zeros(unq_bev.shape[0]).unsqueeze(-1)
    bev_coord = torch.cat([batch, unq_bev], dim=1)
    bev_sem_labels = spconv.SparseConvTensor(unq_sem.unsqueeze(-1), bev_coord.int(), spatial_shape=[2048, 2048], batch_size=1)
    bev_sem_labels = bev_sem_labels.dense().squeeze().numpy() # H,W
    # resize
    bev_sem_labels = cv2.resize(bev_sem_labels, (bev_sem_labels.shape[0]//ds_factor, bev_sem_labels.shape[1]//ds_factor), interpolation=cv2.INTER_NEAREST)

    return bev_sem_labels