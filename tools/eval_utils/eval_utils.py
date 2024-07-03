import pickle
import time

import numpy as np
import torch
import tqdm
import json
import cv2
import os
import pcdet.utils.box_utils as box_utils

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

# def draw_2d_bboxes_on_image(image, bboxes, output_path):
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox.astype(int)
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.imwrite(output_path, image)

def draw_3d_bboxes_on_image(image, corners_in_image, gt_corners_in_image, output_path, thickness = 2):
    """
    在图像上绘制3D框的2D投影，并将图像保存到指定路径。
    
    :param image: 输入的图像 (H, W, 3) 格式的 numpy 数组
    :param corners_in_image: 形状为 (N, 8, 2) 的 3D 框的 2D 投影点
    :param output_path: 图像保存的路径
    :param colors: 颜色列表，包含每个框的颜色 (B, G, R) 格式
    """
    
    # 绘制检测结果对应的绿色3D候选框
    for corners in corners_in_image:
        corners = corners.astype(np.int32)
        # 绘制3D框的12条边
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), thickness)  # 底面
            cv2.line(image, tuple(corners[i + 4]), tuple(corners[(i + 1) % 4 + 4]), (0, 255, 0), thickness)  # 顶面
            cv2.line(image, tuple(corners[i]), tuple(corners[i + 4]), (0, 255, 0), thickness)  # 侧面

    # 绘制真值对应的红色3D候选框
    for corners in gt_corners_in_image:
        corners = corners.cpu().numpy().astype(np.int32)  # 先转成numpy格式，再转成int32
        # corners = corners.to(torch.int32) # torch变量
        # 绘制3D框的12条边
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), thickness)  # 底面
            cv2.line(image, tuple(corners[i + 4]), tuple(corners[(i + 1) % 4 + 4]), (0, 0, 255), thickness)  # 顶面
            cv2.line(image, tuple(corners[i]), tuple(corners[i + 4]), (0, 0, 255), thickness)  # 侧面

    # 保存图像
    cv2.imwrite(output_path, image)

def draw_scenes(batch_dict, annos):
    for i in range(batch_dict['batch_size']):
        corners_in_image = annos[i]['corners_in_image']
        image = batch_dict['images'][i]
        gt_corners_in_image = batch_dict['gt_corners_in_image'][i]

        # convert to opencv format
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0)) # 从PyTorch的(C, H, W)格式转换为OpenCV的(H, W, C)格式
        image_np = (image_np * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # construct output path
        prefix = '/media/gx/tmp/OpenPCDet/data/kitti/training/2d_visualize/'
        os.makedirs(prefix, exist_ok=True)  # 确保目录存在
        output_path = os.path.join(prefix, '{}.png'.format(batch_dict['frame_id'][i]))
        
        # 在图像上绘制检测结果对应的绿色3D候选框和真值对应的红色3D候选框
        draw_3d_bboxes_on_image(image_bgr, corners_in_image, gt_corners_in_image, output_path)

def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )

        # 在图像上绘制3D候选框
        draw_scenes(batch_dict, annos)

        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
