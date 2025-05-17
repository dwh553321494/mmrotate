# Copyright (c) OpenMMLab. All rights reserved.
from .convex_assigner import ConvexAssigner
from .max_convex_iou_assigner import MaxConvexIoUAssigner
from .rotate_iou2d_calculator import (FakeRBboxOverlaps2D,
                                      QBbox2HBboxOverlaps2D,
                                      RBbox2HBboxOverlaps2D, RBboxOverlaps2D)
from .rotated_atss_assigner import RotatedATSSAssigner
from .quadri_sim_ota_assigner import QuadriSimOTAAssigner
from .rotated_sim_ota_assigner import RotatedSimOTAAssigner
from .batch_dsl_assigner import BatchDynamicSoftLabelAssigner
from .sas_assigner import SASAssigner


__all__ = [
    'ConvexAssigner', 'MaxConvexIoUAssigner', 'SASAssigner',
    'RotatedATSSAssigner', 'RBboxOverlaps2D', 'FakeRBboxOverlaps2D',
    'RBbox2HBboxOverlaps2D', 'QBbox2HBboxOverlaps2D', 'QuadriSimOTAAssigner', 'RotatedSimOTAAssigner',
    'BatchDynamicSoftLabelAssigner'
]
