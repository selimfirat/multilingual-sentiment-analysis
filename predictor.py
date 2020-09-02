import torch

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_curve, f1_score, jaccard_score
from torch import nn

from metric_evaluator import MetricEvaluator


class Predictor:

    def __init__(self, cfg, experiment):
        self.cfg = cfg
        self.experiment = experiment
        self.global_threshold = None
        self.sigmoid = nn.Sigmoid().to(self.cfg["device"])

    def get_scores(self, data, data_loader):
        self.experiment.model.eval()
        with torch.no_grad():
            length = data.__len__()
            y_true = data.y_true
            y_pred = np.zeros((length,self.experiment.num_classes))
            for i, (X_batch, y_batch, mask_batch) in enumerate(data_loader):
                X_batch, y_batch, mask_batch = self.experiment.utils.to_gpu(X_batch, y_batch, mask_batch)
                start_idx, end_idx = i*self.cfg["batch_size"], min(length, (i+1)*self.cfg["batch_size"])

                logits, hiddens = self.experiment.model.forward(X_batch, mask_batch, return_hidden=True)
                y_pred[start_idx:end_idx, :] = self.sigmoid(logits).detach().cpu().numpy()

        return y_true, y_pred

    def get_losses(self, data, data_loader):
        self.experiment.model.eval()
        with torch.no_grad():
            length = data.__len__()
            y_true = data.y_true
            y_pred = np.zeros((length,self.experiment.num_classes))
            for i, (X_batch, y_batch, mask_batch) in enumerate(data_loader):
                X_batch, y_batch, mask_batch = self.experiment.utils.to_gpu(X_batch, y_batch, mask_batch)
                y_batch = torch.ones(y_batch.shape, dtype=torch.double, device=self.cfg["device"])

                start_idx, end_idx = i*self.cfg["batch_size"], min(length, (i+1)*self.cfg["batch_size"])

                logits, hiddens = self.experiment.model.forward(X_batch, mask_batch, return_hidden=True)
                y_pred[start_idx:end_idx, :] = -self.experiment.loss.calculate(logits, y_batch, hiddens, reduce=False).cpu().numpy() # self.sigmoid(logits).detach().cpu().numpy()

        return y_true, y_pred

    def get_optimal_threshold(self, y_true, y_pred):
        #y_pred = np.clip(y_pred, a_min=1e-12, a_max=1e+12)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        indices = np.logical_and.reduce([np.logical_or(precision > 0, recall > 0), np.isfinite(precision), np.isfinite(recall)])
        precision = precision[indices]
        recall = recall[indices]
        f1 = 2 * precision * recall / (precision + recall)
        argmax_f1 = np.argmax(f1)
        return thresholds[argmax_f1]

    def get_optimal_threshold_jaccard(self, y_true, y_pred):
        minval = y_pred.min()
        maxval = y_pred.max()
        best_threshold = minval
        best_score = -1
        for threshold in np.arange(minval, maxval, 0.01):
            score = MetricEvaluator.jaccard_index(y_true, y_pred >= threshold)
            if score >= best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def calculate_epochwise_thresholds(self):
        if self.cfg["thresholding"] == "global":
            y_true, y_pred = self.get_scores(self.experiment.threshold_data, self.experiment.threshold_loader)
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            self.global_threshold = self.get_optimal_threshold(y_true, y_pred)
        elif self.cfg["thresholding"] == "class_specific":
            y_true, y_pred = self.get_scores(self.experiment.threshold_data, self.experiment.threshold_loader)
            thresholds = np.empty((self.experiment.num_classes,), dtype=np.double)
            for cls_idx in range(self.experiment.num_classes):
                thresholds[cls_idx] = self.get_optimal_threshold(y_true[:, cls_idx], y_pred[:, cls_idx])

            self.class_specific_thresholds = thresholds
        elif self.cfg["thresholding"] == "anomaly":
            y_true, y_pred = self.get_scores(self.experiment.threshold_data, self.experiment.threshold_loader)
            self.anomaly_detectors = [KNN(n_neighbors=5, method="median") for _ in range(self.experiment.num_classes)]
            self.anomaly_detectors_other = [KNN(n_neighbors=5, method="median") for _ in range(self.experiment.num_classes)]

            y_pred_anomaly = np.empty(y_pred.shape, dtype=np.double)
            for cls_idx, detector in enumerate(self.anomaly_detectors):
                detector.fit(y_pred[y_true[:, cls_idx] > 0.5, :])
                y_pred_anomaly[:, cls_idx] = -detector.decision_function(y_pred)

            y_pred_anomaly_other = np.empty(y_pred.shape, dtype=np.double)
            for cls_idx, detector in enumerate(self.anomaly_detectors_other):
                detector.fit(y_pred[y_true[:, cls_idx] < 0.5, :])
                y_pred_anomaly_other[:, cls_idx] = -detector.decision_function(y_pred)

            y_pred_anomaly -= y_pred_anomaly_other

            thresholds = np.empty((self.experiment.num_classes,), dtype=np.double)
            for cls_idx in range(self.experiment.num_classes):
                thresholds[cls_idx] = self.get_optimal_threshold(y_true[:, cls_idx], y_pred_anomaly[:, cls_idx])

            self.anomaly_thresholds = thresholds

        elif self.cfg["thresholding"] == "class_specific_frequency":
            self.class_specific_frequency_thresholds = (self.experiment.samples_per_cls / self.experiment.train_data.__len__())
        elif self.cfg["thresholding"] == "minimized_expectation":
            y_true, y_pred = self.get_losses(self.experiment.threshold_data, self.experiment.threshold_loader)
            thresholds = np.empty((self.experiment.num_classes,), dtype=np.float)
            for cls_idx in range(self.experiment.num_classes):
                thresholds[cls_idx] = self.get_optimal_threshold(y_true[:, cls_idx], y_pred[:, cls_idx])

            self.minimized_expectation_thresholds = thresholds
        elif self.cfg["thresholding"] == "simplex":
            y_true, y_pred = self.get_losses(self.experiment.threshold_data, self.experiment.threshold_loader)

            def scalar_func(x):
                y_pred_labels = y_pred >= np.repeat(np.expand_dims(x, axis=0), y_pred.shape[0], axis=0)
                res = MetricEvaluator.jaccard_index(y_true, y_pred_labels)

                return -res

            self.simplex_thresholds = minimize(scalar_func, np.mean(y_pred, axis=0), method='Powell', options={"ftol": 1e-3, "xtol": 1e-2, 'disp': True}).x

    def calculate(self, logits, hiddens):

        return getattr(self, f"predict_{self.cfg['thresholding']}")(logits, hiddens)

    def predict_half(self, logits, hiddens):
        y_pred = self.sigmoid(logits).detach().cpu().numpy()
        y_pred_labels = y_pred >= 0.5

        return y_pred, y_pred_labels

    def predict_top1(self, logits, hiddens):
        y_pred = self.sigmoid(logits).detach().cpu().numpy()
        y_pred_labels = np.argmax(y_pred, dim=1)

        return y_pred, y_pred_labels

    def predict_global(self, logits, hiddens):
        y_pred = self.sigmoid(logits).detach().cpu().numpy()
        y_pred_labels = y_pred >= self.global_threshold

        return y_pred, y_pred_labels

    def predict_class_specific(self, logits, hiddens):
        y_pred = self.sigmoid(logits).detach().cpu().numpy()
        y_pred_labels = y_pred >= np.repeat(np.expand_dims(self.class_specific_thresholds, axis=0), y_pred.shape[0], axis=0)

        return y_pred, y_pred_labels

    def predict_class_specific_frequency(self, logits, hiddens):
        y_pred = self.sigmoid(logits).detach().cpu().numpy()
        y_pred_labels = y_pred >= np.repeat(np.expand_dims(self.class_specific_frequency_thresholds, axis=0), y_pred.shape[0], axis=0)

        return y_pred, y_pred_labels

    def predict_minimized_expectation(self, logits, hiddens):
        self.experiment.model.eval()
        with torch.no_grad():
            y_batch = torch.ones(logits.shape, dtype=torch.float, device=self.cfg["device"])
            y_pred = -self.experiment.loss.calculate(logits, y_batch, hiddens, reduce=False).cpu().numpy()

            y_pred_labels = y_pred >= np.repeat(np.expand_dims(self.minimized_expectation_thresholds, axis=0), y_pred.shape[0], axis=0)

        return y_pred, y_pred_labels

    def predict_simplex(self, logits, hiddens):
        self.experiment.model.eval()
        with torch.no_grad():
            y_batch = torch.ones(logits.shape, dtype=torch.float, device=self.cfg["device"])
            y_pred = -self.experiment.loss.calculate(logits, y_batch, hiddens, reduce=False).cpu().numpy()
            y_pred_labels = y_pred >= np.repeat(np.expand_dims(self.simplex_thresholds, axis=0),
                                               y_pred.shape[0], axis=0)

        return y_pred, y_pred_labels

    def predict_anomaly(self, logits, hiddens):
        y_pred = self.sigmoid(logits).detach().cpu().numpy()
        y_pred_anomaly = np.empty(y_pred.shape, dtype=np.double)
        for cls_idx, detector in enumerate(self.anomaly_detectors):
            y_pred_anomaly[:, cls_idx] = -detector.decision_function(y_pred)

        y_pred_anomaly_other = np.empty(y_pred.shape, dtype=np.double)
        for cls_idx, detector in enumerate(self.anomaly_detectors_other):
            y_pred_anomaly_other[:, cls_idx] = -detector.decision_function(y_pred)

        y_pred_anomaly -= y_pred_anomaly_other

        y_pred_labels = y_pred_anomaly >= np.repeat(np.expand_dims(self.anomaly_thresholds, axis=0), y_pred.shape[0], axis=0)

        return y_pred, y_pred_labels
