"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import keyboard

box_colormap = [
    [1, 1, 1],
    [0, 1, 0], # Car
    [0, 1, 1], # Pedestrian
    [1, 1, 0], # Cyclist
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, filename=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # 需要debug查看ref_labels和ref_scores的内容
    if isinstance(ref_labels, torch.Tensor):
        ref_labels = ref_labels.cpu().numpy()
    if isinstance(ref_scores, torch.Tensor):
        ref_scores = ref_scores.cpu().numpy()

    def filter_boxes(ref_labels, ref_scores, ref_boxes):
        invalid_labels = []
        keep = []

        for label, score in zip(ref_labels, ref_scores):
            if label == 1 and score > 0.7:
                keep.append(True)
            elif (label == 2 or label == 3) and score > 0.5:
                keep.append(True)
            else:
                keep.append(False)

            if label not in [1, 2, 3]:
                invalid_labels.append(label)

        keep = np.array(keep)
        if invalid_labels:
            print("Invalid ref_labels detected:", invalid_labels)

        return ref_boxes[keep], ref_labels[keep], ref_scores[keep]

    # 过滤 ref_boxes, ref_labels 和 ref_scores
    if ref_boxes is not None and ref_labels is not None and ref_scores is not None:
        ref_boxes, ref_labels, ref_scores = filter_boxes(ref_labels, ref_scores, ref_boxes)

    print("ref_boxes: " + str(ref_boxes) + "\n")
    print("ref_labels: " + str(ref_labels) + "\n")
    print("ref_scores: " + str(ref_scores) + "\n")

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))    # 真值框为红色

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)   # 预测框为绿色

    vis.run()

    # 额外添加保存结果功能，在关闭可视化窗口时保存
    if filename is not None:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filename)

    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
