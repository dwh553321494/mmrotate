import math
from turtle import forward
from typing import List, Optional, Sequence, Tuple, Union


import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.ops.nms import batched_nms, nms_quadri

from mmdet.structures.bbox import scale_boxes
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
from mmrotate.structures.bbox.rotated_boxes import RotatedBoxes

# from mmrotate.models.utils import gt_instances_preprocess


# 4.22 rotated YOLOX head 
@MODELS.register_module()
class RotatedYOLOXHead(BaseDenseHead):
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
        
        # angle_version: str = 'le90',
        # bbox_coder: ConfigType = dict(
        #     type='DeltaXYWHTHBBoxCoder',
        #     target_means=(0., 0., 0., 0., 0.),
        #     target_stds=(1., 1., 1., 1., 1.),
        #     angle_version='oc',
        #     # proj_xy=True,
        # ),
        
        loss_cls: ConfigType = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox: ConfigType = dict(
            type='RotatedIoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj: ConfigType = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        # loss_ang = None,
        # use_hbbox_loss: bool = False,
        # loss_w: ConfigType = dict(
        #     type='mmdet.SmoothL1Loss',
        #     reduction='sum',
        #     loss_weight=1),
        # loss_h: ConfigType = dict(
        #     type='mmdet.SmoothL1Loss',
        #     reduction='sum',
        #     loss_weight=1),
        # loss_angle: ConfigType = dict(
        #     type='mmdet.SmoothL1Loss',
        #     beta=1.0,
        #     reduction='sum',
        #     loss_weight=2.5),
        
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
        
        # bbox coder
        # self.bbox_coder = TASK_UTILS.build(bbox_coder)
        # self.angle_version = angle_version
        # self.angle_coder = TASK_UTILS.build(angle_coder)
        # self.use_hbbox_loss = use_hbbox_loss
        
        
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_obj = MODELS.build(loss_obj)
        # if loss_ang is not None:
        #     self.loss_ang = MODELS.build(loss_ang)
        # else:
        #     self.loss_ang = None
        
        # self.loss_w = MODELS.build(loss_w)
        # self.loss_h = MODELS.build(loss_h)
        # self.loss_angle = MODELS.build(loss_angle)
        
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
        # self.multi_level_conv_angle = torch.nn.ModuleList()
        self.multi_level_conv_obj = torch.nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            # self.multi_level_conv_angle.append(conv_angle)
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

    def _build_predictor(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        """Initialize predictor layers of a single level head."""
        conv_cls = torch.nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = torch.nn.Conv2d(self.feat_channels, 5, 1)
        # conv_angle = torch.nn.Conv2d(self.feat_channels, 1, 1)
        conv_obj = torch.nn.Conv2d(self.feat_channels, 1, 1)
        # return conv_cls, conv_reg, conv_angle, conv_obj
        return conv_cls, conv_reg, conv_obj
        
    
    def init_weights(self):
        super(RotatedYOLOXHead, self).init_weights()
        
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls, self.multi_level_conv_obj):
            if isinstance(conv_cls, torch.nn.Conv2d):
                conv_cls.bias.data.fill_(bias_init)
            if isinstance(conv_obj, torch.nn.Conv2d):
                conv_obj.bias.data.fill_(bias_init)
            
        
                
    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj):
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)
        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        # angle_pred = conv_angle(reg_feat)
        obj_pred = conv_obj(reg_feat)
        
        return cls_score, bbox_pred, obj_pred
    
    def forward(self, feats: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return multi_apply(
            self.forward_single,
            feats,
            self.multi_level_cls_convs,
            self.multi_level_reg_convs,
            self.multi_level_conv_cls,
            self.multi_level_conv_reg,
            # self.multi_level_conv_angle,
            self.multi_level_conv_obj)
        

    def _bbox_post_process(self, results, cfg, rescale = False, with_nms = True, img_meta = None):
        """bbox post-processing method.
        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.
        
        mmcv has supported the nms operation for quadri boxes! So just in test_cfg use nms_quadri type~
        
        """
        # if rescale:
        #     assert img_meta.get('scale_factor') is not None
        #     results.bboxes /= results.bboxes.new_tensor(
        #         img_meta['scale_factor']).repeat((1, 4))
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes.tensor, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results
    
    def predict_by_feat(self, 
                        cls_scores, 
                        bbox_preds, 
                        # angle_preds, 
                        objectnesses, 
                        batch_img_metas: Optional[List[dict]] = None, 
                        cfg: Optional[ConfigDict] = None, 
                        rescale: bool = False,
                        with_nms: bool = True):
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        
        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.size()[-2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        

        
        # flatten cls_scores, bbox_preds, angle_preds, objectnesses
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 5)
            for bbox_pred in bbox_preds
        ]
        # flatten_angle_preds = [
        #     angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
        #     for angle_pred in angle_preds
        # ]
        flatten_objectnesses = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_objectnesses = torch.cat(flatten_objectnesses, dim=1).sigmoid()
        
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_bbox_preds = flatten_bbox_preds[..., 4:5]
        flatten_bbox_preds[..., 4:5] = self.angle_coder.decode(flatten_bbox_preds,  keepdim=True)
        # flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
    
        # flatten_bbox_preds = torch.cat([flatten_bbox_preds, flatten_angle_preds], dim=2)
        # flatten_bbox_preds = flatten_bbox_preds.view(num_imgs, -1, 5)  # (dx, dy, dw, dh, dt)
        
        flatten_priors = torch.cat(mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        
        # priors_xyxy = flatten_priors.clone()
        # priors_xyxy[:, 2] = priors_xyxy[:, 0] + priors_xyxy[:, 2]
        # priors_xyxy[:, 3] = priors_xyxy[:, 1] + priors_xyxy[:, 3]
        
        
        # decode bbox
        # flatten_bbox_preds = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        flatten_bbox = self.bbox_coder.decode(flatten_priors, flatten_bbox_preds, max_shape=None)
        
        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], dim=1)
            valid_mask = flatten_objectnesses[img_id] * max_scores > cfg.score_thr
            results = InstanceData(
                bboxes=RotatedBoxes(flatten_bbox_preds[img_id][valid_mask]),
                # bboxes=flatten_bbox_preds[img_id][valid_mask],
                scores=max_scores[valid_mask] * flatten_objectnesses[img_id][valid_mask],
                labels=labels[valid_mask])
            result_list.append(
                self._bbox_post_process(results, cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta))
        return result_list
    
    def _bbox_decode(self, priors: Tensor, bbox_preds: Tensor):
        """Decode rbox predictions."""
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:4].exp() * priors[:, 2:]
        # angle = bbox_preds[..., 4:5].relu() * math.pi / 2
        angle = (bbox_preds[..., 4:5].sigmoid() - 1) * math.pi / 2 
        return torch.cat([xys, whs, angle], dim=-1)
    
    def loss_by_feat(self, cls_scores, bbox_preds,objectnesses, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore=None, **kwargs):
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
            
        featmap_sizes = [cls_score.size()[-2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        
        # flatten cls_scores, bbox_preds, angle_preds, objectnesses
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 5)
            for bbox_pred in bbox_preds
        ]
        # flatten_angle_preds = [
        #     angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
        #     for angle_pred in angle_preds
        # ]
        flatten_objectnesses = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_objectnesses = torch.cat(flatten_objectnesses, dim=1)
        
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        # flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
    
        # flatten_bbox_preds = torch.cat([flatten_bbox_preds, flatten_angle_preds], dim=2)
        # flatten_bbox_preds = flatten_bbox_preds.view(num_imgs, -1, 5)  # (dx, dy, dw, dh, dt)
        
        flatten_priors = torch.cat(mlvl_priors)
        # flatten_priors_xyxy = flatten_priors.clone()
        # flatten_priors_xyxy[:, 2] = flatten_priors_xyxy[:, 0] + flatten_priors_xyxy[:, 2]
        # flatten_priors_xyxy[:, 3] = flatten_priors_xyxy[:, 1] + flatten_priors_xyxy[:, 3]
        
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        # flatten_bboxes = self.bbox_coder.decode(flatten_priors_xyxy,flatten_bbox_preds, max_shape=None)
        # flatten_bboxes = flatten_bboxes.tensor
        
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
                flatten_bboxes.view(-1, 5)[pos_masks],
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
    def _get_targets_single(
        self, priors, cls_preds: Tensor, decoded_bboxex, objectness: Tensor, gt_instances, img_meta, gt_instances_ignore=None
    ):
        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        if num_gts == 0:
            cls_target = cls_preds.new_zeros(num_priors, self.num_classes)
            bbox_target = cls_preds.new_zeros(num_priors, 5)
            objectness_target = cls_preds.new_zeros(num_priors)
            l1_target = cls_preds.new_zeros(num_priors, 5)
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
        l1_target = cls_preds.new_zeros((num_pos_per_img, 5))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness, dtype=torch.uint8).bool()
        foreground_mask[pos_inds] = 1
        return (
            foreground_mask, cls_target, obj_target, bbox_target, l1_target, num_pos_per_img
        )
        
    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps = 1e-8):
        priors_xyxy = priors.clone()
        priors_xyxy[:, 2] = priors_xyxy[:, 0] + priors_xyxy[:, 2]
        priors_xyxy[:, 3] = priors_xyxy[:, 1] + priors_xyxy[:, 3]
        l1_target = l1_target.view(-1, 4)
        
        l1_target = self.bbox_coder.encode(
            priors_xyxy, gt_bboxes)
        return l1_target