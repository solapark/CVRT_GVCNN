import torch
import torch.nn as nn

# 1x1 convolution used instead of FC layer for scoring
class OneConvFc(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Input shape: [N, C]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [N, C] → [N, C, 1, 1]
        return self.conv(x).squeeze()  # Output: [N]

# Scoring module for view discrimination
class GroupSchema(nn.Module):
    def __init__(self, in_channels=256):
        super(GroupSchema, self).__init__()
        self.score_layer = OneConvFc(in_channels=in_channels)

    def forward(self, raw_view):  # raw_view: [N, V, C]
        N,V,C = raw_view.shape
        y = self.score_layer(raw_view.reshape(-1, C)) # [N*V]
        y = torch.sigmoid(torch.log(torch.abs(y) + 1e-6))  # Log-stabilized score
        return y.reshape(N,V) # Softmax-normalized scores

def group_pool(final_view, scores):
    # final_view: (N, V, C), scores: (N, V)
    N, V, C = final_view.shape

    groupID = (scores * 10).long().clamp(max=9) // 2  # (N, V), values 0~4

    group_view_sum = torch.zeros((N, C), device=final_view.device)
    group_score_sum = torch.zeros((N, 1), device=final_view.device)

    for i in range(5):  # groups: 0~4
        mask = (groupID == i).float()  # (N, V), float mask
        mask_unsq = mask.unsqueeze(-1)  # (N, V, 1)

        group_size = mask.sum(dim=1, keepdim=True)  # (N, 1), count per row

        masked_view = final_view * mask_unsq  # (N, V, C)
        group_view = masked_view.sum(dim=1) / (group_size + 1e-6)  # (N, C)

        masked_score = scores * mask  # (N, V)
        group_score = torch.ceil(masked_score * group_size).sum(1, keepdim=True) / (group_size + 1e-6) #(N,1)

        group_view_sum += group_view * group_score
        group_score_sum += group_score

    group_view_mean = group_view_sum / (group_score_sum + 1e-6)  # (N, C)
    return group_view_mean

# Complete GVCNN pipeline
class GVCNN(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.score_net = GroupSchema(in_channels=in_channels)

    def forward(self, RPs):  # RPs: [N, V, C]
        view_scores = self.score_net(RPs)  # [N, V]
        shape_des = group_pool(RPs, view_scores)  # [N, C]
        return shape_des

