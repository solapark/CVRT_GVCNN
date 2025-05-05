import torch
import torch.nn as nn

# 1x1 convolution used instead of FC layer for scoring
class OneConvFc(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Input shape: [N, C] or [N, C, 1, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # [N, C] → [N, C, 1, 1]
        return self.conv(x).squeeze()  # Output: [N]

# Scoring module for view discrimination
class GroupSchema(nn.Module):
    def __init__(self, in_channels=256):
        super(GroupSchema, self).__init__()
        self.score_layer = OneConvFc(in_channels=in_channels)
        self.sft = nn.Softmax(dim=1)

    def forward(self, raw_view):  # raw_view: [N, V, C]
        scores = []
        for batch_view in raw_view:  # batch_view: [V, C]
            y = self.score_layer(batch_view)  # [V]
            y = torch.sigmoid(torch.log(torch.abs(y) + 1e-6))  # Log-stabilized score
            scores.append(y)
        view_scores = torch.stack(scores, dim=0)  # [N, V]
        return self.sft(view_scores)  # Softmax-normalized scores

# Group-based view pooling
def view_pool(ungrp_views, view_scores, num_grps):
    def calc_scores(scores):
        n = len(scores)
        s = torch.ceil(scores[0] * n)
        for score in scores[1:]:
            s += torch.ceil(score * n)
        s /= n
        return s

    interval = 1 / (num_grps + 1)
    view_grps = [[] for _ in range(num_grps)]
    score_grps = [[] for _ in range(num_grps)]

    for view, score in zip(ungrp_views, view_scores):
        begin = 0
        for j in range(num_grps):
            right = 1.1 if j == num_grps - 1 else begin + interval
            if begin <= score < right:
                view_grps[j].append(view)
                score_grps[j].append(score)
                break
            begin += interval

    # Mean pooling within each group
    view_grps = [torch.stack(views).mean(0) for views in view_grps if len(views) > 0]
    score_grps = [calc_scores(scores) for scores in score_grps if len(scores) > 0]

    # Weighted sum of group descriptors
    shape_des = sum([v * s for v, s in zip(view_grps, score_grps)]) / sum(score_grps)
    return shape_des

# Wrapper for processing a batch of inputs
def group_pool(final_view, scores, num_grps):
    shape_descriptors = []
    for views, score in zip(final_view, scores):  # views: [V, C], score: [V]
        shape_descriptors.append(view_pool(views, score, num_grps))
    return torch.stack(shape_descriptors, dim=0)  # [N, C]

# Complete GVCNN pipeline
class GVCNN(nn.Module):
    def __init__(self, in_channels=256, num_grps=7):
        super().__init__()
        self.num_grps = num_grps
        self.score_net = GroupSchema(in_channels=in_channels)

    def forward(self, RPs):  # RPs: [N, V, C]
        view_scores = self.score_net(RPs)  # [N, V]
        shape_des = group_pool(RPs, view_scores, self.num_grps)  # [N, C]
        return shape_des

