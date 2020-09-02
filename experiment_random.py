import os
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from feature_extraction import FeatureExtraction


class ExperimentRandom:

    def __init__(self, cfg):

        self.cfg = cfg

        # Data attributes
        pt_path = f"data/{self.cfg['data']}_{self.cfg['pretrained'].replace('/', '_')}.pt"
        if not os.path.exists(pt_path):
            fe = FeatureExtraction(self.cfg)
            fe.extract()

        self.texts, self.features, self.labels, self.train_ind, self.val_ind, self.test_ind, self.num_classes = torch.load(f"data/{self.cfg['data']}_{self.cfg['pretrained'].replace('/', '_')}.pt", map_location="cpu")

        self.len_train = len(self.train_ind)
        self.len_val = len(self.val_ind)
        self.len_test = len(self.test_ind)

    def evaluate(self, y_test, y_pred):
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)

        print(f"F1 Score: {np.round(f1, 4)}")
        print(f"Accuracy Score: {np.round(acc, 4)}")

    def run(self):
        print(f"Random experiment for data {self.cfg['data']}")
        y_test = self.labels[self.test_ind]

        y_test = y_test.detach().cpu().numpy()

        print("---")

        # Random per class
        print("Random per class")
        y_pred = np.random.randint(0,self.num_classes, size=y_test.shape)
        y_pred = y_pred >= 0.5
        self.evaluate(y_test, y_pred)

        print("---")


        # Random per instance
        print("Random per instance")
        y_pred = np.random.randint(0,self.num_classes, size=y_test.shape[0])
        y_pred = np.eye(self.num_classes)[y_pred, :]
        y_pred = y_pred >= 0.5

        self.evaluate(y_test, y_pred)
        print("------------------")
