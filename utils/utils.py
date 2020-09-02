import os

import mlflow
import torch
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Utils:

    def __init__(self, cfg):
        self.cfg = cfg

    def collate_fn(self, samples):
        X_batch = []
        y_batch = []
        mask_batch = []
        for x, y, m in samples:
            X_batch.append(torch.from_numpy(x))
            y_batch.append(torch.from_numpy(y).unsqueeze(0))
            mask_batch.append(m)

        X_batch = pad_sequence(X_batch, batch_first=True)
        mask_batch = torch.tensor(mask_batch, dtype=torch.long)
        y_batch = torch.cat(y_batch)

        return X_batch, y_batch, mask_batch

    def to_gpu(self, X_batch, y_batch, mask_batch):
        X_batch, y_batch, mask_batch = X_batch.to(self.cfg["device"]), y_batch.to(self.cfg["device"]), mask_batch.to(self.cfg["device"])

        return X_batch, y_batch, mask_batch
    
    def start_mlflow(self):
        mlflow.set_experiment(self.cfg["experiment"])
        mlflow.start_run()
        run = mlflow.active_run()._info
        self.uuid = run.run_uuid

        import copy
        copy_cfg = copy.deepcopy(self.cfg)
        copy_cfg.pop('all_datasets', None)
        mlflow.log_params(copy_cfg)
        self.artifacts_path = os.path.join("pretrained", self.uuid)
        os.makedirs(self.artifacts_path)

    def finish_run(self, y_test, y_test_pred, y_test_pred_labels, test_metrics, model, predictor, loss_weights, class_losses, val_scores, val_y_pred, val_y_pred_labels, y_val):
        print(f"Experiment {self.uuid}")

        print(test_metrics)
        mlflow.log_metrics(test_metrics)

        torch.save(y_test_pred, os.path.join(self.artifacts_path, "y_test_pred.pt"))
        torch.save(y_test_pred_labels, os.path.join(self.artifacts_path, "y_test_pred_labels.pt"))
        torch.save(val_y_pred, os.path.join(self.artifacts_path, "y_val_pred.pt"))
        torch.save(y_test, os.path.join(self.artifacts_path, "y_test.pt"))
        torch.save(y_val, os.path.join(self.artifacts_path, "y_val.pt"))
        torch.save(val_y_pred_labels, os.path.join(self.artifacts_path, "y_val_pred_labels.pt"))

        torch.save(model, os.path.join(self.artifacts_path,"best_model.pt"))
        predictor.experiment = None
        torch.save(predictor, os.path.join(self.artifacts_path,"best_model_predictor.pt"))
        torch.save(loss_weights, os.path.join(self.artifacts_path,"best_model_loss_weights.pt"))
        torch.save(class_losses, os.path.join(self.artifacts_path,"class_losses.pt"))
        torch.save(val_scores, os.path.join(self.artifacts_path,"val_scores.pt"))

        mlflow.end_run()

        #mlflow.log_artifacts(self.artifacts_path)
