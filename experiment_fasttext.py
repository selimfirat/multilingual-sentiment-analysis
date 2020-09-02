import os
from dataset import Dataset
import torch
import fasttext
from metric_evaluator import MetricEvaluator
import numpy as np
from sklearn.metrics import precision_recall_curve

class ExperimentFasttext:

    def __init__(self, cfg):

        self.cfg = cfg
        
        """
        self.fasttext_path = f"data/{self.cfg['data']}/fasttext."
        """
        if 'Spanish' in self.cfg['data']:
            self.pretrained_vector_path = 'cc.es.300.vec'
        elif 'Arabic' in self.cfg['data']:
            self.pretrained_vector_path = 'cc.ar.300.vec'
        else:
            self.pretrained_vector_path = 'cc.en.300.vec'
    
        print(self.pretrained_vector_path)
        fasttext_folder_path = f"data/fasttext/{self.cfg['data']}/"
        if not os.path.exists(fasttext_folder_path):
            os.makedirs(fasttext_folder_path)
        
        self.fasttext_path = fasttext_folder_path + "fasttext."
        
        self.dataset = Dataset(self.cfg)
        self.num_classes = self.dataset.num_classes
        texts, labels, train_ind, val_ind, test_ind = self.dataset.get()
        
        labels = labels.tolist()
        
        # https://github.com/facebookresearch/fastText/issues/363
        # https://github.com/facebookresearch/fasttext/issues/72
        fasttext_instances = []
        for instance_idx, instance_label in enumerate(labels):
            instance_label_str = []
            for label_idx, label in enumerate(instance_label):
                if label:
                    instance_label_str.append('__label__' + str(label_idx) + ', ')
                    
            instance_str = ''.join(instance_label_str) + ' ' + texts[instance_idx]
            fasttext_instances.append(instance_str)
            
            
        def create_dicts(texts, labels, fasttext_instances, idxs):
            texts = [texts[i] for i in idxs]
            labels = [labels[i] for i in idxs]
            data = [fasttext_instances[i] for i in idxs]
            
            return {'texts': texts, 'labels': labels, 'data': data}
            
        train_dict = {'partition': 'train', 'idxs': train_ind}
        val_dict = {'partition': 'val', 'idxs': val_ind}
        test_dict = {'partition': 'test', 'idxs': test_ind}
        
        self.dicts = [train_dict, val_dict, test_dict]
        
        for dic in self.dicts:
            dic.update(create_dicts(texts, labels, fasttext_instances, dic['idxs']))
        
            with open(self.fasttext_path + dic['partition'], "w") as f:
                for fi in dic['data']:
                    f.write(fi + '\n')

        
    def run(self):
        #self.model = fasttext.train_supervised(input=self.fasttext_path + 'train', lr=self.cfg['lr'], epoch=self.cfg['num_epochs'], wordNgrams=2, bucket=200000, dim=50, loss='ova')
        self.model = fasttext.train_supervised(input=self.fasttext_path + 'train', lr=0.5, epoch=self.cfg['num_epochs'], wordNgrams=2, bucket=200000, dim=300, loss='ova', pretrainedVectors='cc.en.300.vec')
        
        def print_results(N, p, r):
            print("N\t" + str(N))
            print("P@{}\t{:.3f}".format(1, p))
            print("R@{}\t{:.3f}".format(1, r))
        
        for dic in self.dicts[1:3]:
            metrics = self.eval(dic['texts'], dic['labels'])
            print(dic['partition'] + ' metrics', metrics)
            
            print('fasttext ' + dic['partition'] + ' eval:')
            print_results(*self.model.test(self.fasttext_path + dic['partition']))

    def eval(self, texts, y_true):
        #predictions = self.model.predict(texts, k=-1, threshold=0.5)[0] #[1] has confidences
        
        y_true = np.array(y_true)
        
        y_pred = np.array(self.model.predict(texts, k=-1)[1])
        
        thresholds = np.empty((self.num_classes,), dtype=np.double)
        for cls_idx in range(self.num_classes):
            thresholds[cls_idx] = self.get_optimal_threshold(y_true[:, cls_idx], y_pred[:, cls_idx])
        
        """
        y_pred = np.zeros((len(predictions), self.num_classes), dtype=int)
        for idx, p_instance in enumerate(predictions):
            for p in p_instance:
                true_indices = p.split('__label__')[1:]
                true_indices = [int(x[:-1]) for x in true_indices]
                for i in true_indices:
                    y_pred[idx, i] = 1
        """
        
        y_pred_labels = y_pred >= thresholds
        
        me = MetricEvaluator()
        metric_dict = me.eval_all_metrics(y_true, y_pred, y_pred_labels)
        return metric_dict
    
    def get_optimal_threshold(self, y_true, y_pred):
        #y_pred = np.clip(y_pred, a_min=1e-12, a_max=1e+12)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        indices = np.logical_and.reduce([np.logical_or(precision > 0, recall > 0), np.isfinite(precision), np.isfinite(recall)])
        precision = precision[indices]
        recall = recall[indices]
        f1 = 2 * precision * recall / (precision + recall)
        argmax_f1 = np.argmax(f1)
        return thresholds[argmax_f1]
    
    def run_auto(self):
        runtime = 500 * 60
        self.model = fasttext.train_supervised(input=self.fasttext_path + 'train', autotuneValidationFile=self.fasttext_path + 'val', autotuneDuration=runtime, pretrainedVectors=self.pretrained_vector_path, dim=300)
        self.model.save_model(self.pretrained_vector_path[3:5] + '_automodel.bin')
        
        def print_results(N, p, r):
            print("N\t" + str(N))
            print("P@{}\t{:.3f}".format(1, p))
            print("R@{}\t{:.3f}".format(1, r))
        
        for dic in self.dicts[1:3]:
            metrics = self.eval(dic['texts'], dic['labels'])
            print(dic['partition'] + ' metrics', metrics)
            
            print('fasttext ' + dic['partition'] + ' eval:')
            print_results(*self.model.test(self.fasttext_path + dic['partition']))
