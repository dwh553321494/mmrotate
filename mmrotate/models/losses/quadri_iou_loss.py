from numpy import poly
import torch
import torch.nn as nn
from mmrotate.registry import MODELS


def sort_polygon_clockwise(polygon: torch.Tensor) -> torch.Tensor:
    """
    Sort 4 points in clockwise order starting from top-left.
    Args:
        polygon: (n, 8) tensor
    Returns:
        sorted_polygon: (n, 8) tensor sorted in clockwise order
    """
    polygon = polygon.view(-1, 4, 2)  # (n, 4, 2)
    center = torch.mean(polygon, dim=1)  # (n, 2)
    diff = polygon - center.unsqueeze(1)  # (n, 4, 2)
    distances = torch.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)  # (n, 4)
    angles = torch.atan2(diff[:, :, 1], diff[:, :, 0])  # (n, 4)
    sort_idx = torch.argsort(angles, dim=1, descending=True)  # (n, 4)

    sorted_polygon = polygon.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))  # (n, 4, 2)

    sorted_polygon[:, 1], sorted_polygon[:, 3] = sorted_polygon[:, 3].clone(), sorted_polygon[:, 1].clone()

    sorted_polygon = sorted_polygon.view(-1, 8)  # (n, 8)
    return sorted_polygon
    

def get_bbox(x1y1, x2y2):
    x1 = x1y1[:, 0]
    y1 = x1y1[:, 1]
    x2 = x2y2[:, 0]
    y2 = x2y2[:, 1]
    
    x_min = torch.min(x1, x2)
    y_min = torch.min(y1, y2)
    x_max = torch.max(x1, x2)
    y_max = torch.max(y1, y2)
    bbox = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    return bbox
    
    
@MODELS.register_module()
class QuadriLoss(nn.Module):
    def __init__(self, 
                 loss_type=dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=5.0
                 ),
                 loss_weight=1.0):
        super(QuadriLoss, self).__init__()
        self.loss = MODELS.build(loss_type)
        self.loss_weight = loss_weight
        
    def forward(self, pred, target, img_shape=None):
        if img_shape is None:
            W, H = 640, 640
        else:
            W, H = img_shape[2], img_shape[3]
        
        # 因为有旋转，所以重新排序
        target = sort_polygon_clockwise(target)
        P1_x = pred[:, 0] 
        P1_y = pred[:, 1]
        P2_x = pred[:, 2] 
        P2_y = pred[:, 3]
        P3_x = pred[:, 4] 
        P3_y = pred[:, 5]
        P4_x = pred[:, 6] 
        P4_y = pred[:, 7]
        
        T1_x = target[:, 0] 
        T1_y = target[:, 1]
        T2_x = target[:, 2] 
        T2_y = target[:, 3]
        T3_x = target[:, 4] 
        T3_y = target[:, 5]
        T4_x = target[:, 6] 
        T4_y = target[:, 7]
        loss1 = self.loss(P1_x,T1_x)
        loss2 = self.loss(P1_y,T1_y)
        loss3 = self.loss(P2_x,T2_x)
        loss4 = self.loss(P2_y,T2_y)
        loss5 = self.loss(P3_x,T3_x)
        loss6 = self.loss(P3_y,T3_y)
        loss7 = self.loss(P4_x,T4_x)
        loss8 = self.loss(P4_y,T4_y)

        # zero = torch.zeros_like(P1_y)
        loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8) / 8
        # loss += (torch.max(zero, P1_y - P3_y) ** 2).mean() / 6 + \
        #         (torch.max(zero, P4_y - P3_y) ** 2).mean() / 6 + \
        #         (torch.max(zero, P1_y - P2_y) ** 2).mean() / 6 + \
        #         (torch.max(zero, P4_y - P2_y) ** 2).mean() / 6 + \
        #         (torch.max(zero, P4_x - P1_x) ** 2).mean() / 6 + \
        #         (torch.max(zero, P3_x - P2_x) ** 2).mean() / 6
        
        if self.loss_weight is not None:
            loss = loss * self.loss_weight
        return loss

@MODELS.register_module()
class QuadriIoULoss(nn.Module):
    """QuadriIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 iou_loss_type=dict(
                    type='mmdet.IoULoss',
                    mode='square',
                    eps=1e-16,
                    reduction='sum',
                    loss_weight=5.0,
                 ),
                 loss_weight=1.0,):
        super(QuadriIoULoss, self).__init__()
        self.loss_1 = MODELS.build(iou_loss_type)
        self.loss_2 = MODELS.build(iou_loss_type)
        self.loss_weight = loss_weight
    
    
    
    def get_double_bbox(self, qbox):
        x1y1 = qbox[:, 0:2]
        x2y2 = qbox[:, 2:4]
        x3y3 = qbox[:, 4:6]
        x4y4 = qbox[:, 6:8]
        bbox1 = get_bbox(x1y1, x4y4)
        bbox2 = get_bbox(x2y2, x3y3)
        
        return bbox1, bbox2
        
    
    def forward(self, pred, target):
        pred_bbox1, pred_bbox2 = self.get_double_bbox(pred)
        target_bbox1, target_bbox2 = self.get_double_bbox(target)
        
        loss1 = self.loss_1(
            pred_bbox1,
            target_bbox1)
        loss2 = self.loss_2(
            pred_bbox2,
            target_bbox2)
        loss = (loss1 + loss2)
        if self.loss_weight is not None:
            loss = loss * self.loss_weight

        return loss
        
        
if __name__ == '__main__':
    bbox = torch.tensor([
        [743,855, 915,891, 802,569, 974,606],
        [558,820,616,535,788,571,729,855]
    ]).float()
    print(sort_polygon_clockwise(bbox))


            
            