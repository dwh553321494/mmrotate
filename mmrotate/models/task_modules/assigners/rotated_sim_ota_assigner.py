# Copyright (c) OpenMMLab. All rights reserved.
from calendar import c
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.assigners import SimOTAAssigner
from mmdet.utils import ConfigType

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class RotatedSimOTAAssigner(SimOTAAssigner):
    def __init__(self,
                 center_radius: float = 2.5,
                 candidate_topk: int = 10,
                 iou_weight: float = 3.0,
                 cls_weight: float = 1.0,
                 iou_calculator: ConfigType = dict(type='RBbox2HBboxOverlaps2D')):
        super().__init__(center_radius, candidate_topk, iou_weight, cls_weight, iou_calculator)
        
    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        # Get the center distance between priors and gt_bboxes
        num_gt = gt_bboxes.shape[0]
        
        gt_angles = gt_bboxes[:, 4]
        gt_xy = gt_bboxes[:, :2]
        gt_wh = gt_bboxes[:, 2:4]
        
        grid_xy = priors[:, :2]
        grid_wh = priors[:, 2:4]
        
        # in box
        Cos, Sin = torch.cos(gt_angles), torch.sin(gt_angles)
        Matric = torch.stack(
            [Cos, -Sin, Sin, Cos], dim=1).reshape(-1, 2, 2)
        grid_xy = grid_xy.unsqueeze(1).expand(-1, num_gt, -1)
        offset = grid_xy - gt_xy.unsqueeze(0).expand_as(grid_xy)
        offset = offset.reshape(-1, 2)
        offset = torch.matmul(offset, Matric)
        offset = offset.reshape(-1, num_gt, 2)
        
        b_lt = gt_wh / 2 + offset
        b_rb = gt_wh / 2 - offset
        
        bbox_delta = torch.cat([b_lt, b_rb], dim=1)
        
        is_in_gts = bbox_delta.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0
        
        
        # is prior centers in gt centers
        c_dist = self.center_radius * gt_wh
        c_lt = grid_xy - (gt_xy - c_dist).unsqueeze(0).expand_as(grid_xy)
        c_rb = grid_xy - (gt_xy + c_dist).unsqueeze(0).expand_as(grid_xy)
        c_bbox_delta = torch.cat([c_lt, c_rb], dim=1)
        
        is_in_cts = c_bbox_delta.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers
        
    
    

        
        
    
    