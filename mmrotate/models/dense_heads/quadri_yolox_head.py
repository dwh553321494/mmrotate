import math
from turtle import forward
from typing import List, Optional, Sequence, Tuple, Union


from sympy import flatten
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.ops.nms import batched_nms, nms_quadri

from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmdet.utils import (ConfigType, InstanceList, OptConfigType, OptMultiConfig, OptInstanceList, reduce_mean)

from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures.bbox import QuadriBoxes, hbox2qbox, qbox2hbox

torch.pi = torch.acos(torch.zeros(1)).item() * 2 


@MODELS.register_module()
class QuadriYOLOXHead(BaseDenseHead):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32),
        use_depthwise: bool = False,
        dcn_on_last_conv: bool = False,
        conv_bias: Union[bool, str] = 'auto',
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        loss_cls: ConfigType = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox: ConfigType = dict(
            type='mmdet.IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj: ConfigType = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1: ConfigType = dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1.0),
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = dict(
            type='mmdet.Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_obj = MODELS.build(loss_obj)
        
        self.use_l1 = False
        self.loss_l1 = MODELS.build(loss_l1)
        
        self.prior_generator = MlvlPointGenerator(strides=strides, offset=0)
    
        self.test_cfg = ConfigDict() if test_cfg is None else test_cfg
        self.train_cfg = ConfigDict() if train_cfg is None else train_cfg
        
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = PseudoSampler()
        self._init_layers()
    
    def _init_layers(self):
        self.multi_level_cls_convs = torch.nn.ModuleList()
        self.multi_level_reg_convs = torch.nn.ModuleList()
        
        self.multi_level_conv_cls = torch.nn.ModuleList()
        self.multi_level_conv_reg = torch.nn.ModuleList()
        self.multi_level_conv_obj = torch.nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)
            
            
    def _build_stacked_convs(self) -> torch.nn.Sequential:
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return torch.nn.Sequential(*stacked_convs)

    def _build_predictor(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        """Initialize predictor layers of a single level head."""
        conv_cls = torch.nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = torch.nn.Conv2d(self.feat_channels, 8, 1)
        conv_obj = torch.nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj
    
    def init_weights(self):
        super(QuadriYOLOXHead, self).init_weights()
        
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls, self.multi_level_conv_obj):
            if isinstance(conv_cls, torch.nn.Conv2d):
                conv_cls.bias.data.fill_(bias_init)
            if isinstance(conv_obj, torch.nn.Conv2d):
                conv_obj.bias.data.fill_(bias_init)
    
    
    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj):
        """Forward feature of a single level."""
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)
        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        obj_pred = conv_obj(reg_feat)
        return cls_score, bbox_pred, obj_pred
    
    def forward(self, x):
        return multi_apply(self.forward_single, x, self.multi_level_cls_convs, self.multi_level_reg_convs, self.multi_level_conv_cls, self.multi_level_conv_reg, self.multi_level_conv_obj)
    
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        
        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.size()[-2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        
        # flatten cls_scores, bbox_preds and objectnesses
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 8)
            for bbox_pred in bbox_preds
        ]
        flatten_objectnesses = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, 1).sigmoid()
        flatten_objectnesses = torch.cat(flatten_objectnesses, 1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, 1)
        flatten_priors = torch.cat(mlvl_priors)
        
        # decode bbox
        flatten_bbox_preds = self._bbox_decode(flatten_priors, flatten_bbox_preds) 
        
        result_list = []

        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], dim=1)
            valid_mask = flatten_objectnesses[img_id] * max_scores > cfg.score_thr
            results = InstanceData(
                bboxes=flatten_bbox_preds[img_id][valid_mask],
                scores=max_scores[valid_mask] * flatten_objectnesses[img_id][valid_mask],
                labels=labels[valid_mask])
            result_list.append(
                self._bbox_post_process(results, cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta))
        return result_list
    
    def _bbox_decode(self, priors, bbox_preds, max_shape = None):
        """Decode qbox predictions."""
        # x1y1 = bbox_preds[..., :2] * priors[:, 2:] + priors[:, :2]
        # x2y2 = bbox_preds[..., 2:4] * priors[:, 2:] + priors[:, :2]
        # x3y3 = bbox_preds[..., 4:6] * priors[:, 2:] + priors[:, :2]
        # x4y4 = bbox_preds[..., 6:8] * priors[:, 2:] + priors[:, :2]
        # qbox = torch.stack([x1y1, x2y2, x3y3, x4y4], dim=-1)
        
        # x1 = priors[:, 0] - torch.exp(bbox_preds[..., 0]) * priors[:, 2]
        # y1 = priors[:, 1] + torch.exp(bbox_preds[..., 1]) * priors[:, 3]
        # x2 = priors[:, 0] - torch.exp(bbox_preds[..., 2]) * priors[:, 2]
        # y2 = priors[:, 1] - torch.exp(bbox_preds[..., 3]) * priors[:, 3]
        # x3 = priors[:, 0] + torch.exp(bbox_preds[..., 4]) * priors[:, 2]
        # y3 = priors[:, 1] - torch.exp(bbox_preds[..., 5]) * priors[:, 3]
        # x4 = priors[:, 0] + torch.exp(bbox_preds[..., 6]) * priors[:, 2]
        # y4 = priors[:, 1] + torch.exp(bbox_preds[..., 7]) * priors[:, 3]
        
        x1 = priors[:, 0] + (bbox_preds[..., 0]) ** 3 * priors[:, 2]
        y1 = priors[:, 1] + (bbox_preds[..., 1]) ** 3 * priors[:, 3]
        x2 = priors[:, 0] + (bbox_preds[..., 2]) ** 3 * priors[:, 2]
        y2 = priors[:, 1] + (bbox_preds[..., 3]) ** 3 * priors[:, 3]
        x3 = priors[:, 0] + (bbox_preds[..., 4]) ** 3 * priors[:, 2]
        y3 = priors[:, 1] + (bbox_preds[..., 5]) ** 3 * priors[:, 3]
        x4 = priors[:, 0] + (bbox_preds[..., 6]) ** 3 * priors[:, 2]
        y4 = priors[:, 1] + (bbox_preds[..., 7]) ** 3 * priors[:, 3]
        
        
        
        # x1 = priors[:, 0] + F.leaky_relu(bbox_preds[..., 0], 0.1) * priors[:, 2] * torch.exp(bbox_preds[..., 0])
        # y1 = priors[:, 1] + F.leaky_relu(bbox_preds[..., 1], 0.1) * priors[:, 3] * torch.exp(bbox_preds[..., 1])
        # x2 = priors[:, 0] + F.leaky_relu(bbox_preds[..., 2], 0.1) * priors[:, 2] * torch.exp(bbox_preds[..., 2])
        # y2 = priors[:, 1] + F.leaky_relu(bbox_preds[..., 3], 0.1) * priors[:, 3] * torch.exp(bbox_preds[..., 3]) 
        # x3 = priors[:, 0] + F.leaky_relu(bbox_preds[..., 4], 0.1) * priors[:, 2] * torch.exp(bbox_preds[..., 4])
        # y3 = priors[:, 1] + F.leaky_relu(bbox_preds[..., 5], 0.1) * priors[:, 3] * torch.exp(bbox_preds[..., 5])
        # x4 = priors[:, 0] + F.leaky_relu(bbox_preds[..., 6], 0.1) * priors[:, 2] * torch.exp(bbox_preds[..., 6])
        # y4 = priors[:, 1] + F.leaky_relu(bbox_preds[..., 7], 0.1) * priors[:, 3] * torch.exp(bbox_preds[..., 7])
        # x1 = priors[:, 0] + bbox_preds[..., 0] * priors[:, 2] * torch.exp(bbox_preds[..., 0])
        # y1 = priors[:, 1] + bbox_preds[..., 1] * priors[:, 3] * torch.exp(bbox_preds[..., 1])
        # x2 = priors[:, 0] + bbox_preds[..., 2] * priors[:, 2] * torch.exp(bbox_preds[..., 2])
        # y2 = priors[:, 1] + bbox_preds[..., 3] * priors[:, 3] * torch.exp(bbox_preds[..., 3]) 
        # x3 = priors[:, 0] + bbox_preds[..., 4] * priors[:, 2] * torch.exp(bbox_preds[..., 4])
        # y3 = priors[:, 1] + bbox_preds[..., 5] * priors[:, 3] * torch.exp(bbox_preds[..., 5])
        # x4 = priors[:, 0] + bbox_preds[..., 6] * priors[:, 2] * torch.exp(bbox_preds[..., 6])
        # y4 = priors[:, 1] + bbox_preds[..., 7] * priors[:, 3] * torch.exp(bbox_preds[..., 7])
        
        qbox = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)

        # (B, N, 8, 1) -> (B, N, 8)
        qbox = qbox.view(qbox.size(0), qbox.size(1), -1)
        # qbox = qbox.reshape(-1, 8).view()
        return qbox
    
    def _bbox_post_process(self, results, cfg, rescale = False, with_nms = True, img_meta = None):
        """bbox post-processing method.
        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.
        
        mmcv has supported the nms operation for quadri boxes! So just in test_cfg use nms_quadri type~
        
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            results.bboxes /= results.bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 4))

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results
    
    def loss_by_feat(self, cls_scores, bbox_preds, objectnesses, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore=None, **kwargs):
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
            
        featmap_sizes = [cls_score.size()[-2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 8)
            for bbox_pred in bbox_preds
        ]
        
        flatten_objectnesses = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        
        flatten_cls_scores = torch.cat(flatten_cls_scores, 1)
        flatten_objectnesses = torch.cat(flatten_objectnesses, 1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, 1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        
        # flatten_hboxes = qbox2hbox(flatten_bboxes)
        
        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, num_pos_per_img) = multi_apply(
            self._get_targets_single,
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_cls_scores.detach(),
            flatten_bboxes.detach(),
            flatten_objectnesses.detach(),
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)
        
        num_pos = torch.tensor(sum(num_pos_per_img), dtype=torch.float, device=cls_targets[0].device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)
        
        pos_masks = torch.cat(pos_masks, dim=0)
        cls_targets = torch.cat(cls_targets, dim=0)
        obj_targets = torch.cat(obj_targets, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        # hbox_targets = qbox2hbox(bbox_targets)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, dim=0)
            
        loss_obj = self.loss_obj(flatten_objectnesses.view(-1, 1), obj_targets) / num_total_samples
        if num_pos > 0:
            loss_cls = self.loss_cls(
                flatten_cls_scores.view(-1, self.num_classes)[pos_masks],
                cls_targets) / num_total_samples
            # loss_bbox = self.loss_bbox(
            #     flatten_hboxes.view(-1, 4)[pos_masks],
            #     hbox_targets) / num_total_samples
            loss_bbox = self.loss_bbox(
                flatten_bboxes.view(-1, 8)[pos_masks],
                bbox_targets) / num_total_samples
            if self.use_l1:
                loss_l1 = self.loss_l1(
                    flatten_bbox_preds.view(-1, 8)[pos_masks],
                    l1_targets) / num_total_samples
        else:
            loss_cls = flatten_cls_scores.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0
            loss_l1 = flatten_bbox_preds.sum() * 0
        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,)
        if self.use_l1:
            losses.update(loss_l1=loss_l1)
        
        return losses

    
    @torch.no_grad()
    def _get_targets_single(self, priors, cls_preds: Tensor, decoded_bboxex, objectness: Tensor, gt_instances, img_meta, gt_instances_ignore=None):
        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        if num_gts == 0:
            cls_target = cls_preds.new_zeros(num_priors, self.num_classes)
            bbox_target = cls_preds.new_zeros(num_priors, 8)
            objectness_target = cls_preds.new_zeros(num_priors)
            l1_target = cls_preds.new_zeros(num_priors, 8)
            foreground_mask = cls_preds.new_zeros(num_priors, dtype=torch.uint8).bool()
            return (foreground_mask, cls_target, bbox_target, objectness_target, l1_target, 0) # pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, num_fg_imgs

        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim= -1)
        
        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(
            bboxes = decoded_bboxex, scores = scores.sqrt_(), priors = offset_priors
        )
        assign_result = self.assigner.assign(
            pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)
        
        pos_ious = assign_result.max_overlaps[pos_inds]
        cls_target = F.one_hot(
            sampling_result.pos_gt_labels, self.num_classes) * pos_ious.view(-1, 1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 8))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness, dtype=torch.uint8).bool()
        foreground_mask[pos_inds] = 1
        return (
            foreground_mask, cls_target, obj_target, bbox_target, l1_target, num_pos_per_img
        )
            
            
    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps = 1e-8):
        l1_target[:, :2] = (gt_bboxes[..., :2] - priors[:, :2]) / (priors[:, 2:] + eps)
        l1_target[:, 2:4] = (gt_bboxes[..., 2:4] - priors[:, :2]) / (priors[:, 2:] + eps)
        l1_target[:, 4:6] = (gt_bboxes[..., 4:6] - priors[:, :2]) / (priors[:, 2:] + eps)
        l1_target[:, 6:8] = (gt_bboxes[..., 6:8] - priors[:, :2]) / (priors[:, 2:] + eps)
        

        l1_target = l1_target.reshape(-1, 8)
        return l1_target
        
        # pts = gt_bboxes.view(-1, 4, 2)
        # dx = pts[..., 0] - priors[:, 0]
        # dy = pts[..., 1] - priors[:, 1]
        
        # distance = torch.sqrt(dx * dx + dy * dy)
        # t1 = distance / (priors[:, 2] + eps)
        # t2 = torch.atan2(dy, dx) + torch.pi
        # t2 = t2 / (2 * torch.pi)
        # l1_target[:, 0::2] = t1
        # l1_target[:, 1::2] = t2
        # l1_target = l1_target.view(-1, 8)
        # return l1_target
        
        
        

        
    
    
    
    
    
    