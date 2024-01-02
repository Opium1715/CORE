import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self, margin, lambda_ortho, lambda_dist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.lambda_ortho = lambda_ortho
        self.lambda_dist = lambda_dist

    def forward(self, ortho_short_interest, proxy, ortho_item, norm_vector, y_true):
        reg_ortho = torch.abs(torch.sum(norm_vector * proxy, dim=-1)) / torch.norm(proxy, dim=-1)
        item_positive = ortho_item[y_true]
        item_negative = torch.concat([ortho_item[:y_true], ortho_item[y_true+1:]], dim=0)
        reg_dist = self.distance(proxy + ortho_short_interest, item_positive)  # sum
        loss = torch.relu(self.margin + self.distance(proxy + ortho_short_interest, item_positive) +
                          self.distance(proxy + ortho_short_interest, item_negative))

        final_loss = loss + self.lambda_dist * reg_dist + self.lambda_ortho * reg_ortho
        return final_loss

    def distance(self, x, y):
        return torch.pow(torch.norm(x - y, dim=-1), 2)
