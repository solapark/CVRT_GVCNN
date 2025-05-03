import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES

def cal_iou(boxes1, boxes2):
    """
    boxes1: (N, 4), boxes2: (M, 4)
    return: (N, M) IoU matrix
    """
    # boxes: (cx, cy, w, h) → (x1, y1, x2, y2)
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

@LOSSES.register_module()
class IntraViewLoss(nn.Module):
    def __init__(self, alpha, loss_weight, pos_thresh, neg_thresh):
        super(IntraViewLoss, self).__init__()
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.triplet_loss = nn.TripletMarginLoss(margin=self.alpha, p=1)

    def forward(self, attn, query_is_val, query, key):
        # (900, 3, 300), (900, 3), (900, 3, 4), (300, 3, 4)
        num_sample, num_view, num_key = attn.shape

        query_is_val = query_is_val.transpose(1, 0)  # (3, 900)
        query = query.transpose(1, 0)  # (3, 900, 4)
        key = key.transpose(1, 0)      # (3, 300, 4)
        attn = attn.transpose(1, 0)    # (3, 900, 300)

        query2rpn = []
        for v in range(num_view):
            iou = cal_iou(query[v], key[v])  # (900, 300)
            query2rpn.append(iou)

        query2rpn = torch.stack(query2rpn)  # (3, 900, 300)

        query_is_val = query_is_val.reshape(num_view * num_sample).to(torch.bool)
        attn = attn.reshape(num_view * num_sample, num_key)[query_is_val]  # (N, 300)
        query2rpn = query2rpn.reshape(num_view * num_sample, num_key)[query_is_val]  # (N, 300)

        sorted_dist, sorted_idx = torch.sort(query2rpn, dim=-1, descending=True)  # IoU descending order 

        num_val_sample = attn.shape[0]

        # randomly choose  positive, negative index (pos: [0, pos_thresh), neg: [pos_thresh, neg_thresh))
        pos_idx = torch.randint(0, self.pos_thresh, (num_val_sample,), device=attn.device)
        neg_idx = torch.randint(self.pos_thresh, self.neg_thresh, (num_val_sample,), device=attn.device)

        sample_idx = torch.arange(num_val_sample, device=attn.device)

        pos = attn[sample_idx, sorted_idx[sample_idx, pos_idx]].sigmoid().unsqueeze(-1)
        neg = attn[sample_idx, sorted_idx[sample_idx, neg_idx]].sigmoid().unsqueeze(-1)
        anchor = torch.ones_like(pos)  # dummy anchor (1)

        return self.loss_weight * self.triplet_loss(anchor, pos, neg)
