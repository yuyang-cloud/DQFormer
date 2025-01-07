import numpy as np
from dqformer.utils.visulize import vis_heatmap_pooled, vis_points_with_axis

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    top, bottom = min(x, radius), min(height - x, radius + 1)
    left, right = min(y, radius), min(width - y, radius + 1)

    masked_heatmap  = heatmap[x - top:x + bottom, y - left:y + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

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


class AssignLabel(object):
    def __init__(self, assigner_cfg):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.gt_kernel_size = assigner_cfg.gt_kernel_size
        self.dataset_type = assigner_cfg.dataset_type
        print('use gt label assigning kernel size ', self.gt_kernel_size)
        self.cfg = assigner_cfg

        self.things = get_things(self.dataset_type)
        self.stuff = get_stuff(self.dataset_type)

    def __call__(self, xyz, masks_binary, masks_cls):
        """
            xyz: Np,3     masks: N_mask,Np    masks_cls: N_mask
        """
        max_objs = self._max_objs
        gt_kernel_size = self.gt_kernel_size
        window_size = gt_kernel_size**2
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]
        # max_pooling factor
        if 'kernel_size' in self.tasks[0].keys():
            maxpool = True
            maxpool_kernel = [t.kernel_size for t in self.tasks]
            maxpool_stride = [t.stride for t in self.tasks]
        else:
            maxpool = False

        example = {}

        pc_range = np.array(self.cfg['pc_range'], dtype=np.float32)
        voxel_size = np.array(self.cfg['voxel_size'], dtype=np.float32)
        grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        # BEV map down sample scale
        ds_factor=self.out_size_factor  # 4
        # get width and height
        H,W=(pc_range[3] - pc_range[0]) / voxel_size[0]/ ds_factor, (pc_range[4] - pc_range[1]) / voxel_size[1]/ ds_factor
        H,W=np.round(H).astype(int),np.round(W).astype(int)
        feature_map_size = grid_size[:2] // self.out_size_factor

        # reorganize the gt_dict by tasks
        task_idxs = []
        flag = 0
        for class_name in class_names_by_task:
            task_idxs.append(
                [
                    np.where(
                        masks_cls == class_name.index(i) + 1 + flag
                    )
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_masks = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_idxs):
            task_mask = []
            task_class = []
            for m in mask:
                task_mask.append(masks_binary[m])
                task_class.append(masks_cls[m] - flag2)
            task_masks.append(np.concatenate(task_mask, axis=0))
            task_classes.append(np.concatenate(task_class))
            flag2 += len(mask)

        gt_dict = {}
        gt_dict["gt_classes"] = task_classes
        gt_dict["gt_masks"] = task_masks

        draw_gaussian = draw_umich_gaussian
        hms, inds, masks, cats, Th_masks = [], [], [], [], []
        hms_pooled, inds_pooled = [], []

        for idx, task in enumerate(self.tasks):
            if maxpool:
                hm_size_H = int((feature_map_size[0]-maxpool_kernel[idx])/maxpool_stride[idx]+1)
                hm_size_W = int((feature_map_size[1]-maxpool_kernel[idx])/maxpool_stride[idx]+1)
                hm_pooled = np.zeros((len(class_names_by_task[idx]), hm_size_H, hm_size_W),
                            dtype=np.float32)
                ind_pooled = np.zeros((max_objs*window_size), dtype=np.int64)
            
            hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[0], feature_map_size[1]),
                            dtype=np.float32)
            ind = np.zeros((max_objs*window_size), dtype=np.int64)
            mask = np.zeros((max_objs*window_size), dtype=np.uint8)
            cat = np.zeros((max_objs*window_size), dtype=np.int64)
            Th_mask = np.zeros((max_objs*window_size, masks_binary.shape[-1]), dtype=np.float32)

            num_objs = min(gt_dict['gt_classes'][idx].shape[0], max_objs)  

            for k in range(num_objs):
                cls_id = gt_dict['gt_classes'][idx][k] - 1

                cur_ins_mask = gt_dict['gt_masks'][idx][k].astype(np.bool_)  # Np -> Np_inst
                cur_ins_points = xyz[cur_ins_mask]
                l = cur_ins_points[:, 0].max() - cur_ins_points[:, 0].min() # l = delta_x
                w = cur_ins_points[:, 1].max() - cur_ins_points[:, 1].min() # w = delta_y
                l, w = l / voxel_size[0] / self.out_size_factor, w / voxel_size[1] / self.out_size_factor
                if maxpool:
                    l_pooled, w_pooled = l / maxpool_stride[idx], w / maxpool_stride[idx]
                if w > 0 and l > 0:
                    radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                    radius = max(self._min_radius, int(radius))

                    # be really careful for the coordinate system of your box annotation. 
                    x = cur_ins_points[:, 0].mean() # center_x
                    y = cur_ins_points[:, 1].mean() # center_y

                    coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                        (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                    ct = np.array(
                        [coor_x, coor_y], dtype=np.float32)  
                    ct_int = ct.astype(np.int32)

                    # throw out not in range objects to avoid out of array area when creating the heatmap
                    if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                        continue 

                    draw_gaussian(hm[cls_id], ct, radius)

                    new_idx = k
                    x, y = np.arange(ct_int[0]-gt_kernel_size//2,ct_int[0]+1+gt_kernel_size//2), np.arange(ct_int[1]-gt_kernel_size//2,ct_int[1]+1+gt_kernel_size//2)
                    x, y = np.meshgrid(x, y)
                    x = x.reshape(-1)
                    y = y.reshape(-1)

                    if maxpool:
                        radius_pooled = gaussian_radius((l_pooled, w_pooled), min_overlap=self.gaussian_overlap)
                        radius_pooled = max(self._min_radius, int(radius_pooled))
                        # be really careful for the coordinate system of your box annotation. 
                        ct_pooled = ct / maxpool_stride[idx]
                        ct_pooled_int = ct_pooled.astype(np.int32)
                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_pooled_int[0] < hm_size_H and 0 <= ct_pooled_int[1] < hm_size_W):
                            continue 
                        draw_gaussian(hm_pooled[cls_id], ct_pooled, radius_pooled)

                        x_pooled, y_pooled = np.arange(ct_pooled_int[0]-gt_kernel_size//2,ct_pooled_int[0]+1+gt_kernel_size//2), np.arange(ct_pooled_int[1]-gt_kernel_size//2,ct_pooled_int[1]+1+gt_kernel_size//2)
                        x_pooled, y_pooled = np.meshgrid(x_pooled, y_pooled)
                        x_pooled = x_pooled.reshape(-1)
                        y_pooled = y_pooled.reshape(-1)

                    for j in range(window_size):
                        cat[new_idx*window_size+j] = cls_id
                        ind[new_idx*window_size+j] = x[j] * feature_map_size[1] + y[j]
                        mask[new_idx*window_size+j] = 1
                        Th_mask[new_idx*window_size+j] = cur_ins_mask
                        if maxpool:
                            ind_pooled[new_idx*window_size+j] = x_pooled[j] * hm_size_W + y_pooled[j]

            hms.append(hm)
            masks.append(mask)
            inds.append(ind)
            cats.append(cat)
            Th_masks.append(Th_mask)
            if maxpool:
                hms_pooled.append(hm_pooled)
                inds_pooled.append(ind_pooled)

        example.update({'hm': hms, 'ind': inds, 'mask': masks, 'cat': cats, 'Th_mask':Th_masks})
        if maxpool:
            example.update({'hm_pooled': hms_pooled, 'ind_pooled': inds_pooled})
        return example

