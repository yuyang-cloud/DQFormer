import os
import time
import numpy as np
import torch
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from dqformer.utils.data_util import data_prepare, get_bev_sem
from dqformer.datasets.preprocess import AssignLabel

class SemanticDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.things_ids = []
        self.color_map = []
        self.label_names = []
        self.dataset = cfg.MODEL.DATASET

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_set = SemanticDataset(
            self.cfg[self.cfg.MODEL.DATASET].PATH + "/sequences/",
            self.cfg[self.cfg.MODEL.DATASET].CONFIG,
            split="train",
            dataset=self.dataset,
        )
        self.train_mask_set = MaskSemanticDataset(
            dataset=train_set,
            split="train",
            min_pts=self.cfg[self.cfg.MODEL.DATASET].MIN_POINTS,
            space=self.cfg[self.cfg.MODEL.DATASET].SPACE,
            assigner_cfg=self.cfg[self.cfg.MODEL.DATASET].ASSIGNER,
            sub_pts=self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS,
            subsample=self.cfg.TRAIN.SUBSAMPLE,
            aug=self.cfg.TRAIN.AUG,
        )

        val_set = SemanticDataset(
            self.cfg[self.cfg.MODEL.DATASET].PATH + "/sequences/",
            self.cfg[self.cfg.MODEL.DATASET].CONFIG,
            split="valid",
            dataset=self.dataset,
        )
        self.val_mask_set = MaskSemanticDataset(
            dataset=val_set,
            split="valid",
            min_pts=self.cfg[self.cfg.MODEL.DATASET].MIN_POINTS,
            space=self.cfg[self.cfg.MODEL.DATASET].SPACE,
            assigner_cfg=self.cfg[self.cfg.MODEL.DATASET].ASSIGNER,
        )

        test_set = SemanticDataset(
            self.cfg[self.cfg.MODEL.DATASET].PATH + "/sequences/",
            self.cfg[self.cfg.MODEL.DATASET].CONFIG,
            split="test",
            dataset=self.dataset,
        )
        self.test_mask_set = MaskSemanticDataset(
            dataset=test_set,
            split="test",
            min_pts=self.cfg[self.cfg.MODEL.DATASET].MIN_POINTS,
            space=self.cfg[self.cfg.MODEL.DATASET].SPACE,
            use_tta=self.cfg[self.cfg.MODEL.DATASET].USE_TTA,
        )

        self.things_ids = train_set.things_ids
        self.color_map = train_set.color_map
        self.label_names = train_set.label_names

    def train_dataloader(self):
        dataset = self.train_mask_set
        collate_fn = BatchCollation_Train()
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.train_loader

    def val_dataloader(self):
        dataset = self.val_mask_set
        collate_fn = BatchCollation_Val()
        self.valid_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.valid_loader

    def test_dataloader(self):
        dataset = self.test_mask_set
        collate_fn = BatchCollation_Test_TTA() if dataset.use_tta else BatchCollation_Test()
        self.test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.test_loader


class SemanticDataset(Dataset):
    def __init__(self, data_path, cfg_path, split="train", dataset="KITTI"):
        yaml_path = cfg_path
        with open(yaml_path, "r") as stream:
            semyaml = yaml.safe_load(stream)

        self.things = get_things(dataset)
        self.stuff = get_stuff(dataset)

        self.label_names = {**self.things, **self.stuff}
        self.things_ids = get_things_ids(dataset)

        self.color_map = semyaml["color_map_learning"]
        self.labels = semyaml["labels"]
        self.learning_map = semyaml["learning_map"]
        self.inv_learning_map = semyaml["learning_map_inv"]
        self.split = split
        split = semyaml["split"][self.split]

        self.im_idx = []
        pose_files = []
        calib_files = []
        token_files = []
        fill = 2 if dataset == "KITTI" else 4
        for i_folder in split:
            self.im_idx += absoluteFilePaths(
                "/".join([data_path, str(i_folder).zfill(fill), "velodyne"])
            )
            pose_files.append(
                absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "poses.txt"])
                )
            )
            calib_files.append(
                absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "calib.txt"])
                )
            )
            if dataset == "NUSCENES":
                token_files.append(
                    absoluteDirPath(
                        "/".join(
                            [data_path, str(i_folder).zfill(fill), "lidar_tokens.txt"]
                        )
                    )
                )

        self.im_idx.sort()
        self.poses = load_poses(pose_files, calib_files)
        self.tokens = load_tokens(token_files)
        if self.split == 'test':
            select_im_idx = []
            select_poses = []
            for i, f_name in enumerate(self.im_idx):
                seq = f_name.split('/')[-3]
                frame = f_name.split('/')[-1][0:6]
                output_path = 'output/test/sequences/'+seq+'/predictions/'+frame+'.label'
                if os.path.exists(output_path) == False:
                    select_im_idx.append(self.im_idx[i])
                    select_poses.append(self.poses[i])
            
            self.im_idx = select_im_idx
            self.poses = select_poses

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        fname = self.im_idx[index]
        pose = self.poses[index]
        points = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        xyz = points[:, :3]
        intensity = points[:, 3]
        if len(intensity.shape) == 2:
            intensity = np.squeeze(intensity)
        token = "0"
        if len(self.tokens) > 0:
            token = self.tokens[index]
        if self.split == "test":
            annotated_data = np.expand_dims(
                np.zeros_like(points[:, 0], dtype=int), axis=1
            )
            sem_labels = annotated_data
            ins_labels = annotated_data
        else:
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
                dtype=np.int32,
            ).reshape((-1, 1))
            sem_labels = annotated_data & 0xFFFF
            ins_labels = annotated_data >> 16
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)

        return (xyz, sem_labels, ins_labels, intensity, fname, pose, token)


class MaskSemanticDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        min_pts,
        space,
        assigner_cfg=None,
        sub_pts=0,
        subsample=False,
        aug=False,
        use_tta=False,
        vote_num=4,
    ):
        self.dataset = dataset
        self.sub_pts = sub_pts
        self.split = split
        self.min_points = min_pts
        self.aug = aug
        self.subsample = subsample
        self.th_ids = dataset.things_ids
        self.xlim = space[0]
        self.ylim = space[1]
        self.zlim = space[2]
        self.use_tta = use_tta
        self.vote_num = vote_num
        if assigner_cfg is not None:
            self.Assigner = AssignLabel(assigner_cfg)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.use_tta:
            samples = []
            for i in range(self.vote_num):
                sample = tuple(self.get_single_sample(index, vote_idx=i))
                samples.append(sample)
            return tuple(samples)
        return self.get_single_sample(index)

    def get_single_sample(self, index, vote_idx=0):
        empty = True
        while empty == True:
            data = self.dataset[index]
            xyz, sem_labels, ins_labels, intensity, fname, pose, token = data

            if self.split == 'test':
                xyz[:, 0] = np.clip(xyz[:, 0], self.xlim[0], self.xlim[1])
                xyz[:, 1] = np.clip(xyz[:, 1], self.ylim[0], self.ylim[1])
                xyz[:, 2] = np.clip(xyz[:, 2], self.zlim[0], self.zlim[1])
            else:
                keep = np.argwhere(
                    (self.xlim[0] < xyz[:, 0])
                    & (xyz[:, 0] < self.xlim[1])
                    & (self.ylim[0] < xyz[:, 1])
                    & (xyz[:, 1] < self.ylim[1])
                    & (self.zlim[0] < xyz[:, 2])
                    & (xyz[:, 2] < self.zlim[1])
                )[:, 0]
                xyz = xyz[keep]
                sem_labels = sem_labels[keep]
                ins_labels = ins_labels[keep]
                intensity = intensity[keep]

            # skip scans without instances in train set
            if self.split != "train":
                empty = False
                break

            if len(np.unique(ins_labels)) == 1:
                empty = True
                index = np.random.randint(0, len(self.dataset))
            else:
                empty = False

        feats = np.concatenate((xyz, np.expand_dims(intensity, axis=1)), axis=1)

        if self.split == "test":
            xyz_org = torch.FloatTensor(xyz)    # for tta get ins_center
            if self.use_tta:
                xyz = self.pcd_augmentations(xyz, vote_idx)
                feats = np.concatenate((xyz, np.expand_dims(intensity, axis=1)), axis=1)
            # voxlization
            prepared_data = data_prepare(xyz, feats, sem_labels, ins_labels, masks=torch.tensor([]), split=self.split)
            coords, xyz, xyz_sub, feats, sem_labels, ins_labels, masks, inds_reconstruct = prepared_data
            return (
                    xyz_org,
                    coords,
                    xyz_sub,
                    feats, 
                    inds_reconstruct,
                    fname,
                    pose,
                    token,
                )

        # Subsample
        if self.split == "train" and self.subsample and len(xyz) > self.sub_pts:
            idx = np.random.choice(np.arange(len(xyz)), self.sub_pts, replace=False)
            xyz = xyz[idx]
            sem_labels = sem_labels[idx]
            ins_labels = ins_labels[idx]
            feats = feats[idx]
            intensity = intensity[idx]

        stuff_masks = np.array([]).reshape(0, xyz.shape[0])
        stuff_masks_ids = []
        things_masks = np.array([]).reshape(0, xyz.shape[0])
        things_cls = np.array([], dtype=int)
        things_masks_ids = []

        stuff_labels = np.asarray(
            [0 if s in self.th_ids else s for s in sem_labels[:, 0]]
        )
        stuff_cls, st_cnt = np.unique(stuff_labels, return_counts=True)
        # filter small masks
        keep_st = np.argwhere(st_cnt > self.min_points)[:, 0]
        stuff_cls = stuff_cls[keep_st][1:]
        if len(stuff_cls):
            stuff_masks = np.array(
                [np.where(stuff_labels == i, 1.0, 0.0) for i in stuff_cls]
            )
            stuff_masks_ids = [
                torch.from_numpy(np.where(m == 1)[0]) for m in stuff_masks
            ]
        # things masks
        ins_sems = np.where(ins_labels == 0, 0, sem_labels)
        _ins_labels = ins_sems + ((ins_labels << 16) & 0xFFFF0000).reshape(-1, 1)
        things_ids, th_idx, th_cnt = np.unique(
            _ins_labels[:, 0], return_index=True, return_counts=True
        )
        # filter small instances
        keep_th = np.argwhere(th_cnt > self.min_points)[:, 0]
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        # remove instances with wrong sem class
        keep_th = np.array(
            [i for i, idx in enumerate(th_idx) if sem_labels[idx] in self.th_ids],
            dtype=int,
        )
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        if len(th_idx):
            things_masks = np.array(
                [np.where(_ins_labels[:, 0] == i, 1.0, 0.0) for i in things_ids]
            )
            things_masks_ids = [
                torch.from_numpy(np.where(m == 1)[0]) for m in things_masks
            ]
            things_cls = np.array([sem_labels[i] for i in th_idx]).squeeze(1)

        masks = torch.from_numpy(np.concatenate((stuff_masks, things_masks)))
        masks_cls = torch.from_numpy(np.concatenate((stuff_cls, things_cls)))
        stuff_masks_ids.extend(things_masks_ids)
        masks_ids = stuff_masks_ids

        assert (
            masks.shape[0] == masks_cls.shape[0]
        ), f"not same number masks and classes: masks {masks.shape[0]}, classes {masks_cls.shape[0]} "

        if self.split == "train" and self.aug:
            xyz = self.pcd_augmentations(xyz)
            feats = np.concatenate((xyz, np.expand_dims(intensity, axis=1)), axis=1)
        
        # voxlization
        prepared_data = data_prepare(xyz, feats, sem_labels, ins_labels, masks, self.split)
        if self.split == 'train':
            coords, xyz, feats, sem_labels, ins_labels, masks = prepared_data
            bev_sem = get_bev_sem(coords, sem_labels)
        else:
            coords, xyz, xyz_sub, feats, vox_sem_labels, sem_labels, ins_labels, masks, inds_reconstruct = prepared_data
            bev_sem = get_bev_sem(coords, vox_sem_labels)
        # masks_ids
        masks_ids = [torch.from_numpy(np.where(m == 1)[0]) for m in masks]
        ins_masks_ids = masks_ids[len(stuff_cls):]

        # Assigner -> hm -> Th_dict
        Th_dict = self.Assigner(xyz, masks[len(stuff_cls):], masks_cls[len(stuff_cls):])

        # SphereFormer Data
        if self.split == 'train':
            return (
                coords,
                xyz,
                feats,
                sem_labels,
                ins_labels,
                masks[:len(stuff_cls)],     # stuff
                masks_cls[:len(stuff_cls)],
                masks_ids[:len(stuff_cls)],
                masks[len(stuff_cls):],     # thing
                masks_cls[len(stuff_cls):],
                masks_ids[len(stuff_cls):],
                ins_masks_ids,
                fname,
                pose,
                token,
                bev_sem,
                Th_dict,
            )
        
        elif self.split == 'valid':
            return (
                coords, # sub
                xyz_sub,   # sub
                feats,  # sub
                inds_reconstruct,   # sub -> org
                sem_labels,
                ins_labels,
                masks[:len(stuff_cls)],     # stuff
                masks_cls[:len(stuff_cls)],
                masks_ids[:len(stuff_cls)],
                masks[len(stuff_cls):],     # thing
                masks_cls[len(stuff_cls):],
                masks_ids[len(stuff_cls):],
                ins_masks_ids,
                fname,
                pose,
                token,
                bev_sem,
                Th_dict,
            )

    def pcd_augmentations(self, xyz, vote_idx=None):
        # rotation
        rotate_rad = np.deg2rad(np.random.random() * 360)
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        xyz[:, :2] = np.dot(xyz[:, :2], j)

        # flip
        if vote_idx is not None:
            flip_type = vote_idx % 4
        else:
            flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            xyz[:, 0] = -xyz[:, 0]
        elif flip_type == 2:
            xyz[:, 1] = -xyz[:, 1]
        elif flip_type == 3:
            xyz[:, 0] = -xyz[:, 0]
            xyz[:, 1] = -xyz[:, 1]

        # scale
        noise_scale = np.random.uniform(0.95, 1.05)
        xyz[:, 0] = noise_scale * xyz[:, 0]
        xyz[:, 1] = noise_scale * xyz[:, 1]

        # transform
        trans_std = [0.1, 0.1, 0.1]
        noise_translate = np.array(
            [
                np.random.normal(0, trans_std[0], 1),
                np.random.normal(0, trans_std[1], 1),
                np.random.normal(0, trans_std[2], 1),
            ]
        ).T
        xyz[:, 0:3] += noise_translate

        return xyz


class BatchCollation:
    def __init__(self):
        self.keys = [
            "pt_coord",
            "feats",
            "sem_label",
            "ins_label",
            "masks",
            "masks_cls",
            "masks_ids",
            "fname",
            "pose",
            "token",
        ]

    def __call__(self, data):
        return {self.keys[i]: list(x) for i, x in enumerate(zip(*data))}

class BatchCollation_Train:
    def __init__(self):
        self.keys = [
            "sem_label",
            "ins_label",
            "masks_st",
            "masks_cls_st",
            "masks_ids_st",
            "masks_th",
            "masks_cls_th",
            "masks_ids_th",
            "ins_masks_ids",
            "fname",
            "pose",
            "token",
        ]

    def __call__(self, data):
        vx_coord, xyz, feats = list(zip(*data))[:3]
        offset, count = [], 0
        
        new_vx_coord, new_xyz, new_feat = [], [], []
        k = 0
        for i, item in enumerate(xyz):
            count += item.shape[0]
            k += 1
            offset.append(count)
            new_vx_coord.append(vx_coord[i])
            new_xyz.append(xyz[i])
            new_feat.append(feats[i])
        
        out_dict = {self.keys[i]: list(x) for i, x in enumerate(list(zip(*data))[3:15])}
        out_dict["vx_coord"] = torch.cat(new_vx_coord[:k])
        out_dict["pt_coord"] = torch.cat(new_xyz[:k])
        out_dict["feats"] = torch.cat(new_feat[:k])
        out_dict["offset"] = torch.IntTensor(offset[:k])

        # B,tasks,[H,W] -> tasks,[B,H,W]
        Th_dict = {}
        hms, inds, masks, cats, Th_masks = [],[],[],[],[]
        hms_pooled, inds_pooled = [], []
        for task_id in range(len(data[0][-1]['ind'])):
            hm = [data_batch[-1]['hm'][task_id][np.newaxis, :] for data_batch in data]
            ind = [data_batch[-1]['ind'][task_id][np.newaxis, :] for data_batch in data]
            mask = [data_batch[-1]['mask'][task_id][np.newaxis, :] for data_batch in data]
            cat = [data_batch[-1]['cat'][task_id][np.newaxis, :] for data_batch in data]
            Th_mask = [torch.from_numpy(data_batch[-1]['Th_mask'][task_id]) for data_batch in data]
            hms.append(torch.from_numpy(np.concatenate(hm, axis=0)))
            inds.append(torch.from_numpy(np.concatenate(ind, axis=0)))
            masks.append(torch.from_numpy(np.concatenate(mask, axis=0)))
            cats.append(torch.from_numpy(np.concatenate(cat, axis=0)))
            Th_masks.append(Th_mask)
            if 'hm_pooled' in data[0][-1].keys():
                hm_pooled = [data_batch[-1]['hm_pooled'][task_id][np.newaxis, :] for data_batch in data]
                ind_pooled = [data_batch[-1]['ind_pooled'][task_id][np.newaxis, :] for data_batch in data]
                hms_pooled.append(torch.from_numpy(np.concatenate(hm_pooled, axis=0)))
                inds_pooled.append(torch.from_numpy(np.concatenate(ind_pooled, axis=0)))

        Th_dict.update({'hm': hms, 'ind': inds, 'mask': masks, 'cat': cats, 'Th_mask':Th_masks})    # hms: tasks,[B,C,H,W]   ind: tasks,[B,500]   Th_mask:taks,[B,[500,Np]]
        if len(hms_pooled) != 0:
            Th_dict.update({'hm_pooled': hms_pooled, 'ind_pooled': inds_pooled})
        
        bev_sem = torch.from_numpy(np.concatenate([data_batch[-2][None] for data_batch in data], axis=0))
        out_dict.update({'bev_sem': bev_sem})
        out_dict["Th_dict"] = Th_dict

        return out_dict

class BatchCollation_Val:
    def __init__(self):
        self.keys = [
            "sem_label",
            "ins_label",
            "masks_st",
            "masks_cls_st",
            "masks_ids_st",
            "masks_th",
            "masks_cls_th",
            "masks_ids_th",
            "ins_masks_ids",
            "fname",
            "pose",
            "token",
        ]

    def __call__(self, data):
        vx_coord, xyz, feats, inds_recons = list(zip(*data))[:4]
        inds = list(inds_recons)

        accmulate_points_num = 0
        offset = []
        for i in range(len(vx_coord)):
            inds[i] = accmulate_points_num + inds[i]
            accmulate_points_num += vx_coord[i].shape[0]
            offset.append(accmulate_points_num)

        out_dict = {self.keys[i]: list(x) for i, x in enumerate(list(zip(*data))[4:16])}
        out_dict["vx_coord"] = torch.cat(vx_coord)
        out_dict["pt_coord"] = torch.cat(xyz)
        out_dict["feats"] = torch.cat(feats)
        out_dict["offset"] = torch.IntTensor(offset)
        out_dict["inds_recons"] = list(inds_recons)

        # B,tasks,[H,W] -> tasks,[B,H,W]
        Th_dict = {}
        hms, inds, masks, cats, Th_masks = [],[],[],[],[]
        hms_pooled, inds_pooled = [], []
        for task_id in range(len(data[0][-1]['ind'])):
            hm = [data_batch[-1]['hm'][task_id][np.newaxis, :] for data_batch in data]
            ind = [data_batch[-1]['ind'][task_id][np.newaxis, :] for data_batch in data]
            mask = [data_batch[-1]['mask'][task_id][np.newaxis, :] for data_batch in data]
            cat = [data_batch[-1]['cat'][task_id][np.newaxis, :] for data_batch in data]
            Th_mask = [torch.from_numpy(data_batch[-1]['Th_mask'][task_id]) for data_batch in data]
            hms.append(torch.from_numpy(np.concatenate(hm, axis=0)))
            inds.append(torch.from_numpy(np.concatenate(ind, axis=0)))
            masks.append(torch.from_numpy(np.concatenate(mask, axis=0)))
            cats.append(torch.from_numpy(np.concatenate(cat, axis=0)))
            Th_masks.append(Th_mask)
            if 'hm_pooled' in data[0][-1].keys():
                hm_pooled = [data_batch[-1]['hm_pooled'][task_id][np.newaxis, :] for data_batch in data]
                ind_pooled = [data_batch[-1]['ind_pooled'][task_id][np.newaxis, :] for data_batch in data]
                hms_pooled.append(torch.from_numpy(np.concatenate(hm_pooled, axis=0)))
                inds_pooled.append(torch.from_numpy(np.concatenate(ind_pooled, axis=0)))

        Th_dict.update({'hm': hms, 'ind': inds, 'mask': masks, 'cat': cats, 'Th_mask':Th_masks})    # hms: tasks,[B,C,H,W]   ind: tasks,[B,500]   Th_mask:taks,[B,[500,Np]]
        if len(hms_pooled) != 0:
            Th_dict.update({'hm_pooled': hms_pooled, 'ind_pooled': inds_pooled})
        
        bev_sem = torch.from_numpy(np.concatenate([data_batch[-2][None] for data_batch in data], axis=0))
        out_dict.update({'bev_sem': bev_sem})
        out_dict["Th_dict"] = Th_dict

        return out_dict

class BatchCollation_Test:
    def __init__(self):
        self.keys = [
            "fname",
            "pose",
            "token",
        ]

    def __call__(self, data):
        vx_coord, xyz, feats, inds_recons = list(zip(*data))[:4]
        inds = list(inds_recons)

        accmulate_points_num = 0
        offset = []
        for i in range(len(vx_coord)):
            inds[i] = accmulate_points_num + inds[i]
            accmulate_points_num += vx_coord[i].shape[0]
            offset.append(accmulate_points_num)

        out_dict = {self.keys[i]: list(x) for i, x in enumerate(list(zip(*data))[4:])}
        out_dict["vx_coord"] = torch.cat(vx_coord)
        out_dict["pt_coord"] = torch.cat(xyz)
        out_dict["feats"] = torch.cat(feats)
        out_dict["offset"] = torch.IntTensor(offset)
        out_dict["inds_recons"] = list(inds_recons)
        return out_dict

class BatchCollation_Test_TTA:
    def __init__(self):
        self.keys = [
            "fname",
            "pose",
            "token",
        ]

    def __call__(self, data_list):
        out_list = []
        data_list = list(zip(*data_list))

        for data in data_list:
            vx_coord, xyz, feats, inds_recons = list(zip(*data))[:4]
            inds = list(inds_recons)

            accmulate_points_num = 0
            offset = []
            for i in range(len(vx_coord)):
                inds[i] = accmulate_points_num + inds[i]
                accmulate_points_num += vx_coord[i].shape[0]
                offset.append(accmulate_points_num)

            out_dict = {self.keys[i]: list(x) for i, x in enumerate(list(zip(*data))[4:])}
            out_dict["vx_coord"] = torch.cat(vx_coord)
            out_dict["pt_coord"] = torch.cat(xyz)
            out_dict["feats"] = torch.cat(feats)
            out_dict["offset"] = torch.IntTensor(offset)
            out_dict["inds_recons"] = list(inds_recons)
            out_list.append(out_dict)

        return out_list

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def absoluteDirPath(directory):
    return os.path.abspath(directory)


def parse_calibration(filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()
    return calib


def parse_poses(filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses


def load_poses(pose_files, calib_files):
    poses = []
    for i in range(len(pose_files)):
        calib = parse_calibration(calib_files[i])
        seq_poses_f64 = parse_poses(pose_files[i], calib)
        seq_poses = [pose.astype(np.float32) for pose in seq_poses_f64]
        poses += seq_poses
    return poses


def load_tokens(token_files):
    if len(token_files) == 0:
        return []
    token_files.sort()
    tokens = []
    for f in token_files:
        token_file = open(f)
        for line in token_file:
            token = line.strip()
            tokens.append(token)
        token_file.close()
    return tokens


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


def get_things(dataset):
    if dataset == "KITTI":
        things = {
            1: "car",
            2: "bicycle",
            3: "motorcycle",
            4: "truck",
            5: "other-vehicle",
            6: "person",
            7: "bicyclist",
            8: "motorcyclist",
        }
    elif dataset == "NUSCENES":
        things = {
            2: "bycicle",
            3: "bus",
            4: "car",
            5: "construction_vehicle",
            6: "motorcycle",
            7: "pedestrian",
            9: "trailer",
            10: "truck",
        }
    return things


def get_stuff(dataset):
    if dataset == "KITTI":
        stuff = {
            9: "road",
            10: "parking",
            11: "sidewalk",
            12: "other-ground",
            13: "building",
            14: "fence",
            15: "vegetation",
            16: "trunk",
            17: "terrain",
            18: "pole",
            19: "traffic-sign",
        }
    elif dataset == "NUSCENES":
        stuff = {
            1: "barrier",
            8: "traffic_cone",
            11: "driveable_surface",
            12: "other_flat",
            13: "sidewalk",
            14: "terrain",
            15: "manmade",
            16: "vegetation",
        }
    return stuff


def get_things_ids(dataset):
    if dataset == "KITTI":
        return [1, 2, 3, 4, 5, 6, 7, 8]
    elif dataset == "NUSCENES":
        return [2, 3, 4, 5, 6, 7, 9, 10]
