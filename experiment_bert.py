import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
from dataset import Dataset
from tqdm import tqdm

from metric_evaluator import MetricEvaluator
from utils.utils import Utils


class ExperimentBERT:

    def __init__(self, cfg):

        self.cfg = cfg

        self.dataset = Dataset(self.cfg)

        self.texts, self.labels, self.train_ind, self.val_ind, self.test_ind = self.dataset.get()

        self.train_ind = np.array(self.train_ind)
        self.val_ind = np.array(self.val_ind)
        self.test_ind = np.array(self.test_ind)

        self.num_classes = self.dataset.num_classes

        self.bce = nn.BCEWithLogitsLoss(reduction="mean").to(self.cfg["device"]).double()

        if "bert" in self.cfg["pretrained"]:
            self.model_cls = BertForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["pretrained"])
        self.model = self.model_cls.from_pretrained(self.cfg["pretrained"], num_labels=self.num_classes)

        self.len_train = len(self.train_ind)
        self.len_val = len(self.val_ind)
        self.len_test = len(self.test_ind)

        self.utils = Utils(self.cfg)
        self.utils.start_mlflow()

    def preprocess(self):
        encoded_texts = []

        maxlen = -1
        for text in self.texts:
            e = self.tokenizer.encode(text, add_special_tokens=True)
            maxlen = max(maxlen, len(e))
            encoded_texts.append(e)

        padded_X = torch.zeros(len(self.texts), maxlen, dtype=torch.long)

        for i in range(len(encoded_texts)):
            padded_X[i, :len(encoded_texts[i])] = torch.tensor(encoded_texts[i], dtype=torch.long)

        return padded_X, torch.tensor(self.labels)

    def run(self):

        X, y = self.preprocess()
        self.model.to(self.cfg["device"])

        patience_left = self.cfg["patience"]
        best_val_f1 = -float("inf")
        best_model = None

        if self.cfg["freeze_bert"]:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = False

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])

        epochs = tqdm(range(self.cfg["num_epochs"]))
        for epoch_idx in epochs:
            epoch_train_indices = list(range(self.len_train))
            np.random.shuffle(epoch_train_indices)
            # Train

            self.model.train()
            for start_idx in range(0, self.len_train, self.cfg["batch_size"]):
                self.optimizer.zero_grad()

                batch_indices = self.train_ind[epoch_train_indices[start_idx:min(self.len_train, start_idx+self.cfg["batch_size"])]]
                X_batch = X[batch_indices].to(self.cfg["device"])
                y_batch = y[batch_indices].to(self.cfg["device"]).double()

                logits, = self.model.forward(X_batch)

                loss = self.bce(logits, y_batch)

                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                y_val = y[self.val_ind].numpy()
                y_val_pred = np.zeros((self.len_val, self.num_classes))

                for start_idx in range(0, self.len_val, self.cfg["batch_size"]):
                    batch_indices = self.val_ind[start_idx:min(self.len_val, start_idx + self.cfg["batch_size"])]
                    X_batch = X[batch_indices].to(self.cfg["device"])

                    logits, = self.model.forward(X_batch)
                    y_val_pred[start_idx:min(self.len_val, start_idx + self.cfg["batch_size"])] = (logits >= 0.5).cpu().numpy()

                val_f1 = f1_score(y_val, y_val_pred, labels=list(range(self.num_classes)), average="macro")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1

                    best_model = self.model_cls(self.model.config)
                    best_model.load_state_dict(self.model.state_dict())

                    patience_left = self.cfg["patience"]
                else:
                    patience_left -= 1

                if patience_left == 0:
                    print("Early stopping...")
                    break

                epochs.set_description(f"Val F1: {val_f1}")

        self.model.eval()
        with torch.no_grad():
            self.model = best_model.to(self.cfg["device"])
            y_test = y[self.test_ind].numpy()
            y_test_pred = np.zeros((self.len_test, self.num_classes))

            self.model.train()
            for start_idx in range(0, self.len_test, self.cfg["batch_size"]):
                batch_indices = self.test_ind[start_idx:min(self.len_test, start_idx+self.cfg["batch_size"])]

                X_batch = X[batch_indices].to(self.cfg["device"])
                # y_batch = y[batch_indices].to(self.cfg["device"])

                logits, = self.model.forward(X_batch)

                y_test_pred[start_idx:min(self.len_test, start_idx+self.cfg["batch_size"])] = logits.cpu().numpy()

            y_test_pred_labels = (y_test_pred >= 0.5)

            me = MetricEvaluator()
            metric_dict = me.eval_all_metrics(y_test, y_test_pred, y_test_pred_labels)

            metric_dict["val_f1"] = best_val_f1

            print(f"Test F1: {metric_dict['f1']} (Best val F1: {metric_dict['val_f1']})")

            self.utils.finish_run(y_test, y_test_pred, y_test_pred_labels, metric_dict, self.model)
