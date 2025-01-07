import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
import os
import time

def load_view_point(filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd')
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    render_option = vis.get_render_option()

    render_option.point_size = 1.0
    render_option.background_color = np.ones(3)
    ctr.convert_from_pinhole_camera_parameters(param)
    return vis

def vis_panoptic_seg(x, semantic_labels, instance_labels):
    # points: B,[N,3]   instance_labels: B,[N]
    if 'inds_recons' in x.keys():
        points = x['pt_coord'][:x['inds_recons'][0].max()+1][x['inds_recons'][0], :]  # down-sample -> org-nums

    # Batch -> 1
    points = points.detach().cpu().numpy()
    semantic_labels = semantic_labels[0].squeeze()
    instance_labels = instance_labels[0].squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # semantic colors
    colors = np.array([
        [0.5       , 0.5       , 0.5       ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [1.        , 0.        , 1.        ],
        [1.        , 0.58823529, 1.        ],
        [0.78431372, 0.        , 0.78431372],
        [0.68627451, 0.        , 0.29411765],
        [1.        , 0.78431373, 0.        ],
        [1.        , 0.47058824, 0.19607843],
        [0.        , 0.68627451, 0.        ],
        [0.52941176, 0.23529412, 0.        ],
        [0.58823529, 0.94117647, 0.31372549],
        [1.        , 0.94117647, 0.58823529],
        [1.        , 0.        , 0.        ],
        [0.5       , 0.5       , 0.5       ],
    ])
    semantic_colors = np.array([colors[sem] for sem in semantic_labels])

    # instance colors
    ins_ids = np.unique(instance_labels)
    colors = np.random.rand(max(ins_ids)+1, 3)
    colors[0] = np.array([0., 0., 0.])   # background gray
    instance_colors = np.array([colors[instance] for instance in instance_labels])

    # combine colors
    combined_colors = np.array([semantic_color + instance_color for semantic_color, instance_color in zip(semantic_colors, instance_colors)])
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # view_control
    view_control = vis.get_view_control()
    view_control.rotate(0.0, 0.0)
    view_control.translate(0.0, 0.0)
    view_control.scale(1.0)

    # run
    vis.run()
    vis.destroy_window()


    ## 保存viewpoint.json
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # open3d.io.write_pinhole_camera_parameters('viewpoint.json', param)
    # vis.destroy_window()

    ## 读取viewpoint.json进行渲染 并保存图片 （使用时屏蔽上边Visualizer, view_control, run）
    # vis = load_view_point('viewpoint.json')
    # vis.add_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()
    # vis_image = vis.capture_screen_float_buffer(do_render=True)
    # vis_image = (np.array(vis_image) * 255).astype(np.uint8)
    # vis_image = Image.fromarray(vis_image)
    # vis_image.save(os.path.join("./output", str(idx).zfill(5)+'.png'))
    # time.sleep(0.1)
    # vis.destroy_window()

def vis_instance_seg(x, instance_labels):
    # points: B,[N,3]   instance_labels: B,[N]
    if 'inds_recons' in x.keys():
        points = x['pt_coord'][:x['inds_recons'][0].max()+1][x['inds_recons'][0], :]  # down-sample -> org-nums

    # Batch -> 1
    points = points.detach().cpu().numpy()
    instance_labels = instance_labels[0].squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # colors
    ins_ids = np.unique(instance_labels)
    colors = np.random.rand(max(ins_ids)+1, 3)
    colors[0] = np.array([0.5, 0.5, 0.5])   # background gray
    colored_points = np.array([colors[instance] for instance in instance_labels])
    pcd.colors = o3d.utility.Vector3dVector(colored_points)

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # view_control
    view_control = vis.get_view_control()
    view_control.rotate(0.0, 0.0)
    view_control.translate(0.0, 0.0)
    view_control.scale(1.0)

    # run
    vis.run()
    vis.destroy_window()


def vis_semantic_seg(x, semantic_labels):
    # points: B,[N,3]   instance_labels: B,[N]
    if 'inds_recons' in x.keys():
        points = x['pt_coord'][:x['inds_recons'][0].max()+1][x['inds_recons'][0], :]  # down-sample -> org-nums

    # Batch -> 1
    points = points.detach().cpu().numpy()
    semantic_labels = semantic_labels[0].squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # colors
    colors = np.array([
        [0.5       , 0.5       , 0.5       ],
        [0.39215686, 0.58823529, 0.96078431],
        [0.39215686, 0.90196078, 0.96078431],
        [0.11764706, 0.23529412, 0.58823529],
        [0.31372549, 0.11764706, 0.70588235],
        [0.39215686, 0.31372549, 0.98039216],
        [1.        , 0.11764706, 0.11764706],
        [1.        , 0.15686275, 0.78431373],
        [0.58823529, 0.11764706, 0.35294118],
        [1.        , 0.        , 1.        ],
        [1.        , 0.58823529, 1.        ],
        [0.29411765, 0.        , 0.29411765],
        [0.68627451, 0.        , 0.29411765],
        [1.        , 0.78431373, 0.        ],
        [1.        , 0.47058824, 0.19607843],
        [0.        , 0.68627451, 0.        ],
        [0.52941176, 0.23529412, 0.        ],
        [0.58823529, 0.94117647, 0.31372549],
        [1.        , 0.94117647, 0.58823529],
        [1.        , 0.        , 0.        ],
        [0.5       , 0.5       , 0.5       ],
    ])
    colored_points = np.array([colors[sem] for sem in semantic_labels])
    pcd.colors = o3d.utility.Vector3dVector(colored_points)

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # view_control
    view_control = vis.get_view_control()
    view_control.rotate(0.0, 0.0)
    view_control.translate(0.0, 0.0)
    view_control.scale(1.0)

    # run
    vis.run()
    vis.destroy_window()

def vis_seg_attention(x, th_masks, th_sem_labels, stuff=False):
    # points: B,[N,3]   instance_labels: B,[N]
    if 'inds_recons' in x.keys():
        points = x['pt_coord'][:x['inds_recons'][0].max()+1][x['inds_recons'][0], :]  # down-sample -> org-nums

    # Batch -> 1
    # for cur_label in th_masks:
    cur_label = th_masks[0]
    print(th_sem_labels)
    print('choose vis which instance by: cur_label = th_masks[i]')
    breakpoint()

    # stuff
    if stuff:
        cur_mask = cur_label > 0.5
        noise = torch.from_numpy(np.random.rand(cur_mask.sum())).to(cur_label)
        cur_label[cur_mask] += noise * 0.5

        cur_mask = cur_label < 0.5
        noise = torch.from_numpy(np.random.rand(cur_mask.sum())).to(cur_label)
        cur_label[cur_mask] += noise * 0.5

        cur_label = cur_label.detach().cpu().numpy()
    
    # instance
    else:
        # obj_cente
        cur_mask = cur_label > 0.5
        obj_center = points[cur_mask].mean(0)
        distances = torch.norm(points-obj_center, dim=1)
        ## 过滤距离obj_center过远的点
        # indices = torch.where(distances <= 20)[0]
        # distances = distances[indices]
        # points = points[indices]
        # cur_label = cur_label[indices]
        # 根据距离obj_center的距离 添加高斯噪声
        std = 0.1 * (1 - distances / distances.max())
        noise = np.random.normal(0, std.detach().cpu().numpy(), size=cur_label.shape[0])
        cur_label = (cur_label.detach().cpu().numpy() + noise)

    select_points = points.detach().cpu().numpy()
    cur_label = (cur_label - cur_label.min()) / (cur_label.max() - cur_label.min())

    # jet colormap
    color_map = cm.jet(cur_label)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(select_points)
    pcd.colors = o3d.utility.Vector3dVector(color_map)

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # view_control
    view_control = vis.get_view_control()
    view_control.rotate(0.0, 0.0)
    view_control.translate(0.0, 0.0)
    view_control.scale(1.0)

    # run
    vis.run()
    vis.destroy_window()


def vis_centers(point_cloud, centers, semantics):
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.detach().cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.detach().cpu().numpy()

    cloud_gray = o3d.geometry.PointCloud()
    cloud_gray.points = o3d.utility.Vector3dVector(point_cloud)
    gray_color = [0.5, 0.5, 0.5]
    cloud_gray.paint_uniform_color(gray_color)

    colors = np.array([
        [0.        , 0.        , 1.        ],
        [0.39215686, 0.90196078, 0.96078431],
        [0.11764706, 0.23529412, 0.58823529],
        # [0.31372549, 0.11764706, 0.70588235],
        [1.        , 1.        , 0.        ],
        # [0.39215686, 0.31372549, 0.98039216],
        [0         , 0.98      , 0.1       ],
        [1.        , 0.        , 0.        ],
        [1.        , 0.15686275, 0.78431373],
        [0.58823529, 0.11764706, 0.35294118],
    ])
    spheres = []
    for center, sem in zip(centers, semantics):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3, resolution=20)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(colors[sem])
        sphere.translate(center)
        spheres.append(sphere)

    pointclouds = [cloud_gray] + spheres

    o3d.visualization.draw_geometries(pointclouds)


def vis_heatmap(gt, pred, task=0, channel=0):
    gt_heatmap = gt['Th_dict']['hm'][task][0][channel].detach().cpu().numpy()
    pred_heatmap = pred[task]['hm'][0][channel].detach().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(gt_heatmap, cmap='hot')
    ax1.set_title('GT Heatmap')

    ax2.imshow(pred_heatmap, cmap='hot')
    ax2.set_title('Pred Heatmap')

    plt.tight_layout()

    plt.show()

def vis_heatmap_pooled(gt, pred, task=0, channel=0):
    if torch.is_tensor(gt[0]) and gt[0].is_cuda:
        gt_heatmap = gt[0][channel].detach().cpu().numpy()
        pred_heatmap = pred[0][channel].detach().cpu().numpy()
    else:
        gt_heatmap = gt[channel]
        pred_heatmap = pred[channel]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(gt_heatmap, cmap='hot')
    ax1.set_title('GT Heatmap')

    ax2.imshow(pred_heatmap, cmap='hot')
    ax2.set_title('Pred Heatmap')

    plt.tight_layout()

    plt.show()

def vis_points_with_axis(points, mask):
    if torch.is_tensor(points):
        points = points.numpy()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # colors
    mask = mask.astype(float)
    colors = np.zeros((len(mask), 3))
    colors[mask == 1] = [1, 0, 0]
    colors[mask == 0] = [0.5, 0.5, 0.5]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.mean(points, axis=0))
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(point_cloud)
    vis.add_geometry(axis)

    render_options = vis.get_render_option()
    render_options.point_size = 2
    render_options.show_coordinate_frame = True

    vis.update_renderer()
    vis.run()

