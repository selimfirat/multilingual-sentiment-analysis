import copy
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TextClassificationDataset
from feature_extraction import FeatureExtraction
from loss import Loss
from models.xml_cnn import XMLCNN
from models.recurrent_network import RecurrentNetwork
import numpy as np

from predictor import Predictor
from utils.utils import Utils
from metric_evaluator import MetricEvaluator


class ExperimentMultiLabelClassification:

    def __init__(self, cfg):

        self.cfg = cfg
        # Data attributes
        if self.cfg['prepro'] == 'None':
            h5_path = f"data/{self.cfg['data']}-{self.cfg['pretrained'].replace('/', '_')}.h5"
        else:
            h5_path = f"data/{self.cfg['data']}-{self.cfg['pretrained'].replace('/', '_')}-prepro_{self.cfg['prepro']}.h5"
        if not os.path.exists(h5_path):
            fe = FeatureExtraction(self.cfg, h5_path)
            fe.extract()

        self.train_data = TextClassificationDataset(self.cfg, "train")
        self.val_data = TextClassificationDataset(self.cfg, "val")
        self.test_data = TextClassificationDataset(self.cfg, "test")

        self.threshold_data = self.val_data.split("threshold", 0.3)

        self.num_classes = self.train_data.num_classes
        self.samples_per_cls = self.train_data.samples_per_cls

        self.utils = Utils(self.cfg)
        self.utils.start_mlflow()

        self.train_loader = DataLoader(self.train_data, shuffle=True, batch_size=self.cfg["batch_size"], num_workers=1, collate_fn=self.utils.collate_fn)
        self.val_loader = DataLoader(self.val_data, shuffle=False, batch_size=self.cfg["batch_size"], num_workers=1, collate_fn=self.utils.collate_fn)
        self.threshold_loader = DataLoader(self.threshold_data, shuffle=False, batch_size=self.cfg["batch_size"], num_workers=1, collate_fn=self.utils.collate_fn)
        self.test_loader = DataLoader(self.test_data, shuffle=False, batch_size=self.cfg["batch_size"], num_workers=1, collate_fn=self.utils.collate_fn)

        self.model = self.init_model()
        self.loss = self.init_loss()

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        if "center" in self.cfg["loss"]:
            self.optimizer_centloss = torch.optim.SGD(self.loss.center_loss_instance.parameters(), lr=self.cfg["lr_center"])

        #for name, _ in list(self.model.named_parameters()):
        #    print(name)

        self.predictor = self.init_predictor()

    def init_loss(self):
        loss = Loss(self.cfg, self.num_classes, self.samples_per_cls, self)

        if self.cfg["finetune"] is not None and self.cfg["no_train"]:
            loss.weights = torch.load(os.path.join("pretrained", self.cfg["finetune"], "best_model_loss_weights.pt"), map_location=self.cfg["device"])

        return loss

    def init_predictor(self):
        if self.cfg["finetune"] is not None:
            predictor = torch.load(os.path.join("pretrained", self.cfg["finetune"], "best_model_predictor.pt"), map_location=self.cfg["device"])
            predictor.experiment = self
        else:
            predictor = Predictor(self.cfg, self)

        return predictor

    def init_model(self):

        if self.cfg["model"] in ["rnn", "gru", "lstm"]:
            model = RecurrentNetwork(cfg=self.cfg, output_size=self.num_classes, embedding_size=self.train_data.embedding_size)
        elif self.cfg["model"] == "xmlcnn":
            model = XMLCNN(output_size=self.num_classes, device=self.cfg["device"], sequence_length=self.train_data.max_len, embedding_size=self.train_data.embedding_size).double()
        else:
            raise Exception("Unknown model")

        if self.cfg["finetune"] is not None:
            ft_model = torch.load(os.path.join("pretrained", self.cfg["finetune"], "best_model.pt"), map_location="cpu").to(self.cfg["device"])

            if self.cfg["no_train"]:
                model = ft_model
            else:
                if self.cfg["finetune_type"] == "change_final":
                    ft_model.output_size = self.num_classes
                    ft_model.init_final()
                    model = ft_model
                elif self.cfg["finetune_type"] == "new_final":
                    ft_model.add_finetune_layer(self.num_classes)
                    model = ft_model
                elif self.cfg["finetune_type"] == "new_final_last":
                    ft_model.add_finetune_layer(self.num_classes)
                    for param in ft_model.parameters():
                        param.requires_grad = False
                    for param in ft_model.final_ft.parameters():
                        param.requires_grad = True
                    model = ft_model
                elif self.cfg["finetune_type"] == "concat_hidden":
                    model.ft_model = ft_model
                    model.init_final()
                elif self.cfg["finetune_type"] == "concat_hidden_freezed":
                    model.ft_model = ft_model
                    for param in ft_model.parameters():
                        param.requires_grad = False
                    model.init_final()
                elif self.cfg["finetune_type"] == "concat_final":
                    model.ft_model = ft_model
                    model.init_final()
                elif self.cfg["finetune_type"] == "concat_final_freezed":
                    model.ft_model = ft_model
                    for param in ft_model.parameters():
                        param.requires_grad = False
                    model.init_final()

        return model

    def run(self):

        if self.cfg["no_train"]:
            best_val_metrics, class_losses, val_scores = {}, [], []
            best_val_metrics, val_y_pred, val_y_pred_labels = self.eval(self.val_data, self.val_loader)
        else:
            best_val_metrics, class_losses, val_scores, val_y_pred, val_y_pred_labels = self.train()

        test_metrics, y_test_pred, y_test_pred_labels = self.eval(self.test_data, self.test_loader)

        best_val_metrics = {f"val_{k}": v for k,v in best_val_metrics.items()}
        test_metrics.update(best_val_metrics)

        self.utils.finish_run(self.test_data.y_true, y_test_pred, y_test_pred_labels, test_metrics, self.model, self.predictor, self.loss.weights.to("cpu"), class_losses, val_scores, val_y_pred, val_y_pred_labels, self.val_data.y_true)

        return test_metrics

    def train(self):
        val_y_pred = None
        val_y_pred_labels = None
        class_losses = []
        val_scores = []
        best_val_score = -float("inf")
        best_val_metrics = {}
        best_model = None
        patience_left = self.cfg["patience"]
        epochs = tqdm(range(1, self.cfg["num_epochs"] + 1))
        for epoch_idx in epochs:
            self.loss.calculate_epochwise_weights()

            self.model.train()

            for X_batch, y_batch, mask_batch in self.train_loader:
                X_batch, y_batch, mask_batch = self.utils.to_gpu(X_batch, y_batch, mask_batch)
                self.optimizer.zero_grad()
                if "center" in self.cfg["loss"]:
                    self.optimizer_centloss.zero_grad()

                logits, hiddens = self.model.forward(X_batch, mask_batch, return_hidden=True)

                loss, class_loss = self.loss.calculate(logits, y_batch, hiddens)

                class_losses.append((epoch_idx, X_batch.shape[0], class_loss))

                loss.backward()
                if "center" in self.cfg["loss"]:
                    for param in self.loss.center_loss_instance.parameters():
                        if self.cfg["lambda"] > 0:
                            param.grad.data *= (1 / (self.cfg["lambda"]))

                    self.optimizer_centloss.step()
                self.optimizer.step()

            self.predictor.calculate_epochwise_thresholds()

            val_metrics, val_y_pred, val_y_pred_labels = self.eval(self.val_data, self.val_loader)
            val_score = val_metrics["macro_f1"]
            val_scores.append(val_score)
            epochs.set_description(f"Validation Score: {np.round(val_score, 4)}")

            if val_score > best_val_score:
                best_model = copy.deepcopy(self.model.to("cpu"))
                self.model.to(self.cfg["device"])
                self.predictor.experiment = None
                best_predictor = copy.deepcopy(self.predictor)
                self.predictor.experiment = self
                best_loss_weights = self.loss.weights.to("cpu")
                best_val_score = val_score
                best_val_metrics = val_metrics
                patience_left = self.cfg["patience"]
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        self.model.to("cpu")
        self.model = best_model.to(self.cfg["device"])
        self.predictor = best_predictor
        self.predictor.experiment = self
        self.loss.weights = best_loss_weights.to(self.cfg["device"])

        return best_val_metrics, class_losses, val_scores, val_y_pred, val_y_pred_labels

    def eval(self, data, data_loader):
        length = data.__len__()
        y_pred = np.empty((length, self.num_classes))
        y_pred_labels = np.empty((length, self.num_classes), dtype=np.bool)

        self.model.eval()
        for i, (X_batch, y_batch, mask_batch) in enumerate(data_loader):
            X_batch, y_batch, mask_batch = self.utils.to_gpu(X_batch, y_batch, mask_batch)
            start_idx, end_idx = i*self.cfg["batch_size"], min(length, (i+1)*self.cfg["batch_size"])

            logits, hiddens = self.model.forward(X_batch, mask_batch, return_hidden=True)

            y_pred[start_idx:end_idx], y_pred_labels[start_idx:end_idx] = self.predictor.calculate(logits, hiddens)

        me = MetricEvaluator()
        test_metrics = me.eval_all_metrics(data.y_true, y_pred, y_pred_labels)

        return test_metrics, y_pred, y_pred_labels
