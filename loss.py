import torch
import numpy as np
from torch import nn

from center_loss import CenterLoss


class Loss:

    def __init__(self, cfg, num_classes, samples_per_cls, experiment):
        self.samples_per_cls = samples_per_cls
        self.num_classes = num_classes
        self.cfg = cfg
        self.experiment = experiment
        loss_name = self.cfg["loss"].replace("_center", "")
        self.loss = getattr(self, loss_name + "_loss")
        self.weighter = getattr(self, "calculate_" + self.cfg["weights"] + "_weights") if (self.cfg["weights"] is not None and self.cfg["weights"].lower() != "none") else None

        self.weights = None

        self.bce = nn.BCEWithLogitsLoss(reduction="none").to(self.cfg["device"]).double()

        self.losses = torch.ones(self.num_classes, dtype=torch.double, requires_grad=False, device=self.cfg["device"]) / self.num_classes

        if "center" in self.cfg["loss"]:
            self.center_loss_instance = CenterLoss(num_classes=self.num_classes, cfg=self.cfg)

    def calculate(self, logits, y_batch, hiddens=None, reduce=True):
        loss = self.loss(logits, y_batch)

        if "center" in self.cfg["loss"]:
            loss += self.cfg["lambda"] * self.center_loss_instance.forward(logits, y_batch) / self.num_classes

        if reduce:
            if self.cfg["weights"] == "dynamic_loss_size":
                self.calculate_dynamic_loss_size_weights(loss)
            elif self.cfg["weights"] == "equal_entropy":
                self.calculate_equal_entropy_weights(logits)

            if self.weights is not None:
                loss *= self.weights.reshape(1, -1).expand_as(loss)

            class_loss = loss.detach().cpu().numpy().sum(axis=0)

            loss = torch.mean(loss)

            return loss, class_loss

        return loss

    def calculate_dynamic_loss_size_weights(self, loss):
        self.losses = self.cfg["weight_smoothing"] * loss.mean(dim=0).detach() + (1 - self.cfg["weight_smoothing"])*self.losses
        weights = 1 / (0.00001 + self.losses)

        self.weights = weights / torch.sum(weights)

    def calculate_equal_entropy_weights(self, logits):
        r = torch.sigmoid(logits.detach())
        self.losses =  ((-r) * torch.log(r)).mean(dim=0)
        weights = 1 / (0.00001 + self.losses)

        self.weights = weights / torch.sum(weights)

    def focal_loss(self, logits, y_batch):
        logpt = self.bce_loss(logits, y_batch)
        pt = torch.exp(-logpt)

        loss = self.cfg["alpha"] * ( (1 - pt)**self.cfg["gamma"]) * logpt

        return loss

    def bce_loss(self, logits, y_batch):

        loss = self.bce(logits, y_batch) # -(y_batch*torch.log(torch.sigmoid(logits)) + (1 - y_batch)*torch.log(1 - torch.sigmoid(logits)))

        return loss

    def center_loss(self, logits, y_batch):
        return 0.0

    def calculate_epochwise_weights(self):
        if self.weighter is not None and self.cfg["weights"] not in ["equal_entropy", "dynamic_loss_size"]:
            self.weighter()

    def calculate_uniform_weights(self):

        if self.weights is None:
            self.weights = torch.ones(self.num_classes, dtype=torch.double, requires_grad=False, device=self.cfg["device"]) / self.num_classes

    def calculate_cost_sensitive_weights(self):
        """
        https://link.springer.com/content/pdf/10.1007/s11063-018-09977-1.pdf
        Learning from Imbalanced Data Sets with Weighted Cross-Entropy Function
        """
        if self.weights is None:
            weights = torch.tensor(self.samples_per_cls, dtype=torch.double, device=self.cfg["device"], requires_grad=False)
            self.weights = weights/torch.sum(weights)

    def calculate_loss_size_weights(self):
        self.experiment.model.eval()
        losses = torch.zeros(self.num_classes, dtype=torch.double, device=self.cfg["device"], requires_grad = False)
        for X_batch, y_batch, mask_batch in self.experiment.train_loader:
            X_batch, y_batch, mask_batch = self.experiment.utils.to_gpu(X_batch, y_batch, mask_batch)
            logits, hiddens = self.experiment.model.forward(X_batch, mask_batch, return_hidden=True)

            for cls_idx in range(self.num_classes):
                indices = y_batch[:, cls_idx] > 0
                if torch.sum(indices) > 0:
                    losses[cls_idx] += self.calculate(logits[indices], y_batch[indices], hiddens[indices]).detach()

        weights = torch.sum(losses) / losses
        self.weights = weights / torch.sum(weights)

    def calculate_inverse_weights(self):
        if self.weights is None:
            weights = torch.tensor(self.samples_per_cls, dtype=torch.double, device=self.cfg["device"], requires_grad=False).pow(-1)
            self.weights = weights / torch.sum(weights)

    def calculate_class_balanced_weights(self):
        # https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
        if self.weights is None:
            effective_num = 1.0 - torch.pow(self.cfg["beta"], torch.tensor(self.samples_per_cls, dtype=torch.double, device=self.cfg["device"]))
            weights = (1.0 - self.cfg["beta"]) / torch.tensor(effective_num, dtype=torch.double, device=self.cfg["device"])
            self.weights = weights / torch.sum(weights)

    def calculate_histogram_volume_weights(self):
        # Naive bayes-like assumption
        self.experiment.model.eval()
        bins = np.zeros((self.num_classes, self.cfg["hidden_size"], self.cfg["num_bins"]), dtype=np.dtype(bool))
        for X_batch, y_batch, mask_batch in self.experiment.train_loader:
            X_batch, y_batch, mask_batch = self.experiment.utils.to_gpu(X_batch, y_batch, mask_batch)
            hiddens = torch.sigmoid(self.experiment.model.forward_hidden(X_batch, mask_batch))

            for i in range(self.cfg["hidden_size"]):
                for cls_idx in range(self.num_classes):
                    indices = y_batch[:, cls_idx] > 0
                    hist, _ = np.histogram(hiddens[indices, i].flatten().detach().cpu().numpy(), bins=self.cfg["num_bins"], range=(0.0, 1.0))
                    bins[cls_idx, i, (hist > 0)] = True

        weights = np.sum(bins, axis=2)
        weights = np.sum(np.log(weights), axis=1)
        weights = np.sum(weights) / weights
        self.weights = torch.from_numpy(weights / np.sum(weights)).to(self.cfg["device"])


