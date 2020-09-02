import torch

from torch import nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, cfg):
        super().__init__()
        self.num_classes = num_classes

        self.cfg = cfg
        self.feat_dim = num_classes# self.cfg["hidden_size"] * (2 if self.cfg["bidirectional"] else 1)
        if self.cfg["finetune"] is not None:
            if "concat_hidden" in self.cfg["finetune_type"]:
                self.feat_dim += self.cfg["hidden_size"] * (2 if self.cfg["bidirectional"] else 1)
            elif "concat_final" in self.cfg["finetune_type"]:
                self.feat_dim += 64

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, dtype=torch.double, device=self.cfg["device"]))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t() - 2*x@self.centers.t()
#        distmat.addmm_(1, -2, x, self.centers.t())

        #classes = torch.arange(self.num_classes).long().to(self.cfg["device"])
        #labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        #mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * labels.double()
        loss = dist.clamp(min=1e-12, max=1e+12)

        return loss
