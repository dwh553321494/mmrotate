# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import shapely
import numpy as np

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor, pairwise_distance

from mmrotate.registry import TASK_UTILS
from mmdet.utils import ConfigType
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner

INF = 100000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class QuadriSimOTAAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (float): Ground truth center size
            to judge whether a prior is in center. Defaults to 2.5.
        candidate_topk (int): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Defaults to 10.
        iou_weight (float): The scale factor for regression
            iou cost. Defaults to 3.0.
        cls_weight (float): The scale factor for classification
            cost. Defaults to 1.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(self,
                 center_radius: float = 2.5,
                 candidate_topk: int = 10,
                 iou_weight: float = 3.0,
                 cls_weight: float = 1.0,
                 iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D')):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors using SimOTA.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # 这里需要改
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            # No valid bboxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # poly的IoU计算
        # pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        # pairwise_ious = self.polygon_box_iou(
        #     gt_bboxes, valid_decoded_bbox, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device='cpu', ordered=False)
        pairwise_ious = torch.cat([self.polygon_box_iou(
            gt_bboxes[i].unsqueeze(0), valid_decoded_bbox, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device='cpu', ordered=False).unsqueeze(0) for i in range(num_gt)], dim=0)
    
        iou_cost = -torch.log(pairwise_ious + EPS)
        iou_cost = iou_cost.permute(1, 2, 0).squeeze().cuda()
        pairwise_ious = pairwise_ious.permute(1, 2, 0).squeeze().cuda()
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        # disable AMP autocast and calculate BCE with FP32 to avoid overflow
        with torch.cuda.amp.autocast(enabled=False):
            cls_cost = (
                F.binary_cross_entropy(
                    valid_pred_scores.to(dtype=torch.float32),
                    gt_onehot_label,
                    reduction='none',
                ).sum(-1).to(dtype=valid_pred_scores.dtype))

        cost_matrix = (
            cls_cost * self.cls_weight + iou_cost * self.iou_weight +
            (~is_in_boxes_and_center) * INF)

        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious.float()
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(
            self, priors: Tensor, gt_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the information of which prior is in gt bboxes and gt center
        priors."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        # l_ = repeated_x - gt_bboxes[:, 0]
        # t_ = repeated_y - gt_bboxes[:, 1]
        # r_ = gt_bboxes[:, 2] - repeated_x
        # b_ = gt_bboxes[:, 3] - repeated_y
        gt_bboxes_l = torch.min(gt_bboxes[:, 0], gt_bboxes[:, 2])
        gt_bboxes_r = torch.max(gt_bboxes[:, 4], gt_bboxes[:, 6])
        gt_bboxes_t = torch.min(gt_bboxes[:, 3], gt_bboxes[:, 5])
        gt_bboxes_b = torch.max(gt_bboxes[:, 1], gt_bboxes[:, 7])
        # gt_bboxes_r = max(gt_bboxes[:, 4], gt_bboxes[:, 6])
        # gt_bboxes_t = min(gt_bboxes[:, 3], gt_bboxes[:, 5])
        # gt_bboxes_b = max(gt_bboxes[:, 7], gt_bboxes[:, 1])
        l_ = repeated_x - gt_bboxes_l
        t_ = repeated_y - gt_bboxes_t
        r_ = gt_bboxes_r - repeated_x
        b_ = gt_bboxes_b - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        # gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        # gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2] + gt_bboxes[:, 4] +
                  gt_bboxes[:, 6]) / 4.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3] + gt_bboxes[:, 5] +
                  gt_bboxes[:, 7]) / 4.0
            
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,num_gt: int,valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
    # def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds



    def order_corners(self, boxes):
        """
            Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
            Sorted corners have the following restrictions:
                                    y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
        """
        boxes = boxes.view(-1, 4, 2)
        x = boxes[..., 0]
        y = boxes[..., 1]
        y_sorted, y_indices = torch.sort(y)  # sort y
        x_sorted = torch.zeros_like(x, dtype=x.dtype)
        for i in range(x.shape[0]):
            x_sorted[i] = x[i, y_indices[i]]
        x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
        x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
        for i in range(y.shape[0]):
            y_sorted[i, :2] = y_sorted[i, :2][x_bottom_indices[i]]
            y_sorted[i, 2:4] = y_sorted[i, 2:4][x_top_indices[i]]
        return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()

    def polygon_inter_union_cpu(self, boxes1, boxes2):
        """
            Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
            iou computation (polygon) with cpu;
            Boxes have shape nx8 and Anchors have mx8;
            Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
        """

        n, m = boxes1.shape[0], boxes2.shape[0]
        inter = torch.zeros(n, m)
        union = torch.zeros(n, m)
        for i in range(n):
            polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4, 2)).convex_hull
            for j in range(m):
                polygon2 = shapely.geometry.Polygon(boxes2[j, :].view(4, 2)).convex_hull
                if polygon1.intersects(polygon2):
                    try:
                        inter[i, j] = polygon1.intersection(polygon2).area
                        union[i, j] = polygon1.union(polygon2).area
                    except shapely.geos.TopologicalError:
                        print('shapely.geos.TopologicalError occured')
        return inter, union

    def polygon_box_iou(self, gt_box, pred_boxes, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu", ordered=False):
        """
            Compute iou of polygon boxes via cpu or cuda;
            For cuda code, please refer to files in ./iou_cuda
            Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
        """
        if gt_box[0] == 'inf':
            print(66)
        ious = []
        boxes1 = gt_box
        pred_boxes = pred_boxes.cpu().numpy().tolist()
        for i in range(len(pred_boxes)):
            boxes2 = pred_boxes[i]
            boxes2 = np.array(boxes2)
            boxes2 = torch.from_numpy(boxes2).unsqueeze(0)
            # For testing this function, please use ordered=False
            # if not ordered:
            #     boxes1, boxes2 = self.order_corners(boxes1.clone().to(device)), self.order_corners(boxes2.clone().to(device))
            # else:
            boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)

            if torch.cuda.is_available() and boxes1.is_cuda:
                # using cuda extension to compute
                # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
                boxes1_ = boxes1.float().contiguous().view(-1)
                boxes2_ = boxes2.float().contiguous().view(-1)
                inter, union = self.polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.

                inter_nan, union_nan = inter.isnan(), union.isnan()
                if inter_nan.any() or union_nan.any():
                    inter2, union2 = self.polygon_inter_union_cuda(boxes1_,
                                                              boxes2_)  # Careful that order should be: boxes1_, boxes2_.
                    inter2, union2 = inter2.T, union2.T
                    inter = torch.where(inter_nan, inter2, inter)
                    union = torch.where(union_nan, union2, union)
            else:
                # using shapely (cpu) to compute
                inter, union = self.polygon_inter_union_cpu(boxes1, boxes2)
            union += eps
            iou = inter / union
            iou[torch.isnan(inter)] = 0.0
            iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
            iou[torch.isnan(iou)] = 0.0
            iou = iou.cpu().numpy().tolist()
            ious.append(iou)  # IoU
        ious = np.array(ious)
        ious = torch.from_numpy(ious)
        ious = ious.squeeze(1)
        ious = ious.permute(1, 0)
        return ious

# Copyright (c) OpenMMLab. All rights reserved.
# from typing import Optional, Tuple

# import torch
# import torch.nn.functional as F
# from mmengine.structures import InstanceData
# from torch import Tensor

# # from mmdet.registry import TASK_UTILS
# from mmdet.utils import ConfigType
# # from .assign_result import AssignResult
# # from .base_assigner import BaseAssigner

# INF = 100000.0
# EPS = 1.0e-7


# def qbox2hbox(boxes: Tensor) -> Tensor:
#     """Convert quadrilateral boxes to horizontal boxes.

#     Args:
#         boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

#     Returns:
#         Tensor: Horizontal box tensor with shape of (..., 4).
#     """
#     boxes = boxes.view(*boxes.shape[:-1], 4, 2)
#     x1y1, _ = boxes.min(dim=-2)
#     x2y2, _ = boxes.max(dim=-2)
#     return torch.cat([x1y1, x2y2], dim=-1)


# @TASK_UTILS.register_module()
# class QuadriSimOTAAssigner(BaseAssigner):
#     """Computes matching between predictions and ground truth.

#     Args:
#         center_radius (float): Ground truth center size
#             to judge whether a prior is in center. Defaults to 2.5.
#         candidate_topk (int): The candidate top-k which used to
#             get top-k ious to calculate dynamic-k. Defaults to 10.
#         iou_weight (float): The scale factor for regression
#             iou cost. Defaults to 3.0.
#         cls_weight (float): The scale factor for classification
#             cost. Defaults to 1.0.
#         iou_calculator (ConfigType): Config of overlaps Calculator.
#             Defaults to dict(type='BboxOverlaps2D').
#     """

#     def __init__(self,
#                  center_radius: float = 2.5,
#                  candidate_topk: int = 10,
#                  iou_weight: float = 3.0,
#                  cls_weight: float = 1.0,
#                  iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D')):
#         self.center_radius = center_radius
#         self.candidate_topk = candidate_topk
#         self.iou_weight = iou_weight
#         self.cls_weight = cls_weight
#         self.iou_calculator = TASK_UTILS.build(iou_calculator)

#     def assign(self,
#                pred_instances: InstanceData,
#                gt_instances: InstanceData,
#                gt_instances_ignore: Optional[InstanceData] = None,
#                **kwargs) -> AssignResult:
#         """Assign gt to priors using SimOTA.

#         Args:
#             pred_instances (:obj:`InstanceData`): Instances of model
#                 predictions. It includes ``priors``, and the priors can
#                 be anchors or points, or the bboxes predicted by the
#                 previous stage, has shape (n, 4). The bboxes predicted by
#                 the current model or stage will be named ``bboxes``,
#                 ``labels``, and ``scores``, the same as the ``InstanceData``
#                 in other places.
#             gt_instances (:obj:`InstanceData`): Ground truth of instance
#                 annotations. It usually includes ``bboxes``, with shape (k, 4),
#                 and ``labels``, with shape (k, ).
#             gt_instances_ignore (:obj:`InstanceData`, optional): Instances
#                 to be ignored during training. It includes ``bboxes``
#                 attribute data that is ignored during training and testing.
#                 Defaults to None.
#         Returns:
#             obj:`AssignResult`: The assigned result.
#         """
#         gt_bboxes = gt_instances.bboxes
#         gt_bboxes = qbox2hbox(gt_bboxes)
        
#         gt_labels = gt_instances.labels
#         num_gt = gt_bboxes.size(0)

#         decoded_bboxes = pred_instances.bboxes
#         decoded_bboxes = qbox2hbox(decoded_bboxes)
        
#         pred_scores = pred_instances.scores
#         priors = pred_instances.priors
#         num_bboxes = decoded_bboxes.size(0)

#         # assign 0 by default
#         assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
#                                                    0,
#                                                    dtype=torch.long)
#         if num_gt == 0 or num_bboxes == 0:
#             # No ground truth or boxes, return empty assignment
#             max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
#             assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
#                                                       -1,
#                                                       dtype=torch.long)
#             return AssignResult(
#                 num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

#         # 这里需要改
#         valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
#             priors, gt_bboxes)
#         valid_decoded_bbox = decoded_bboxes[valid_mask]
#         valid_pred_scores = pred_scores[valid_mask]
#         num_valid = valid_decoded_bbox.size(0)
#         if num_valid == 0:
#             # No valid bboxes, return empty assignment
#             max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
#             assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
#                                                       -1,
#                                                       dtype=torch.long)
#             return AssignResult(
#                 num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

#         # poly的IoU计算
#         pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
#         print(pairwise_ious.size())
#         iou_cost = -torch.log(pairwise_ious + EPS)

#         gt_onehot_label = (
#             F.one_hot(gt_labels.to(torch.int64),
#                       pred_scores.shape[-1]).float().unsqueeze(0).repeat(
#                           num_valid, 1, 1))

#         valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
#         # disable AMP autocast and calculate BCE with FP32 to avoid overflow
#         with torch.cuda.amp.autocast(enabled=False):
#             cls_cost = (
#                 F.binary_cross_entropy(
#                     valid_pred_scores.to(dtype=torch.float32),
#                     gt_onehot_label,
#                     reduction='none',
#                 ).sum(-1).to(dtype=valid_pred_scores.dtype))

#         cost_matrix = (
#             cls_cost * self.cls_weight + iou_cost * self.iou_weight +
#             (~is_in_boxes_and_center) * INF)

#         matched_pred_ious, matched_gt_inds = \
#             self.dynamic_k_matching(
#                 cost_matrix, pairwise_ious, num_gt, valid_mask)

#         # convert to AssignResult format
#         assigned_gt_inds[valid_mask] = matched_gt_inds + 1
#         assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
#         assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
#         max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
#                                                  -INF,
#                                                  dtype=torch.float32)
#         max_overlaps[valid_mask] = matched_pred_ious
#         return AssignResult(
#             num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

#     def get_in_gt_and_in_center_info(
#             self, priors: Tensor, gt_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
#         """Get the information of which prior is in gt bboxes and gt center
#         priors."""
#         num_gt = gt_bboxes.size(0)

#         repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
#         repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
#         repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
#         repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

#         # is prior centers in gt bboxes, shape: [n_prior, n_gt]
#         l_ = repeated_x - gt_bboxes[:, 0]
#         t_ = repeated_y - gt_bboxes[:, 1]
#         r_ = gt_bboxes[:, 2] - repeated_x
#         b_ = gt_bboxes[:, 3] - repeated_y

#         deltas = torch.stack([l_, t_, r_, b_], dim=1)
#         is_in_gts = deltas.min(dim=1).values > 0
#         is_in_gts_all = is_in_gts.sum(dim=1) > 0

#         # is prior centers in gt centers
#         gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
#         gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
#         ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
#         ct_box_t = gt_cys - self.center_radius * repeated_stride_y
#         ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
#         ct_box_b = gt_cys + self.center_radius * repeated_stride_y

#         cl_ = repeated_x - ct_box_l
#         ct_ = repeated_y - ct_box_t
#         cr_ = ct_box_r - repeated_x
#         cb_ = ct_box_b - repeated_y

#         ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
#         is_in_cts = ct_deltas.min(dim=1).values > 0
#         is_in_cts_all = is_in_cts.sum(dim=1) > 0

#         # in boxes or in centers, shape: [num_priors]
#         is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

#         # both in boxes and centers, shape: [num_fg, num_gt]
#         is_in_boxes_and_centers = (
#             is_in_gts[is_in_gts_or_centers, :]
#             & is_in_cts[is_in_gts_or_centers, :])
#         return is_in_gts_or_centers, is_in_boxes_and_centers

#     def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
#                            num_gt: int,
#                            valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
#         """Use IoU and matching cost to calculate the dynamic top-k positive
#         targets."""
#         matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
#         # select candidate topk ious for dynamic-k calculation
#         candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
#         topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
#         # calculate dynamic k for each gt
#         dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
#         for gt_idx in range(num_gt):
#             _, pos_idx = torch.topk(
#                 cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
#             matching_matrix[:, gt_idx][pos_idx] = 1

#         del topk_ious, dynamic_ks, pos_idx

#         prior_match_gt_mask = matching_matrix.sum(1) > 1
#         if prior_match_gt_mask.sum() > 0:
#             cost_min, cost_argmin = torch.min(
#                 cost[prior_match_gt_mask, :], dim=1)
#             matching_matrix[prior_match_gt_mask, :] *= 0
#             matching_matrix[prior_match_gt_mask, cost_argmin] = 1
#         # get foreground mask inside box and center prior
#         fg_mask_inboxes = matching_matrix.sum(1) > 0
#         valid_mask[valid_mask.clone()] = fg_mask_inboxes

#         matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
#         matched_pred_ious = (matching_matrix *
#                              pairwise_ious).sum(1)[fg_mask_inboxes]
#         return matched_pred_ious, matched_gt_inds
