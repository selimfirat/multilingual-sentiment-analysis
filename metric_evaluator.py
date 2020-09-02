from sklearn.metrics import f1_score, accuracy_score, classification_report, dcg_score, ndcg_score, confusion_matrix
from utils.letor_metrics import ranking_precision_score#, dcg_score, ndcg_score
import numpy as np
import sys


class MetricEvaluator:
    
    def __init__(self, k=3):
        self.k = k

    @staticmethod
    def jaccard_index(y_true, y_pred_labels):

        y_true = y_true.astype(np.double)
        y_pred_labels = y_pred_labels.astype(np.double)
        intersection = (y_true * y_pred_labels).sum(axis=1)
        union = ((y_true + y_pred_labels)>0).astype(np.double).sum(axis=1)
        union[union==0] = 1
        intersection[union==0] = 1
        res = np.mean(intersection/union)

        return res

    def eval_all_metrics(self, y_true, y_pred, y_pred_labels):
        
        print(y_true.shape, y_pred.shape, y_pred_labels.shape)
        
        macro_f1 = f1_score(y_true, y_pred_labels, average="macro")
        micro_f1 = f1_score(y_true, y_pred_labels, average="micro")
        acc = accuracy_score(y_true, y_pred_labels)
        jaccard_index = MetricEvaluator.jaccard_index(y_true, y_pred_labels)
        
        matching_labels = y_true == y_pred_labels                       # True if pred = label, False else
        matching_labels = matching_labels * 1                           # T,F to 1,0
        correctly_predicted_labels = matching_labels.sum(axis=0)        # Num. of correct predictions per label (C x 1)
        per_class_accs = correctly_predicted_labels / y_true.shape[0]   
        
        if sys.version_info[0] > 3:
            cr = classification_report(y_true, y_pred_labels, output_dict=True)
            cr.pop("micro avg", None)
            cr.pop("macro avg", None)
            cr.pop("weighted avg", None)
            cr.pop("samples avg", None)
            #per_class_f1_dict = dict()
            #for c in cr:
                #per_class_f1_dict[c] = cr[str(c)]['f1-score']
                #metric_dict['perclassf1_' + c] = cr[str(c)]['f1-score']


        metric_dict = {}

        #metric_dict['acc'] = acc
        #metric_dict['f1'] = macro_f1
        metric_dict['macro_f1'] = macro_f1
        metric_dict['micro_f1'] = micro_f1
        metric_dict['jaccard_index'] = jaccard_index
        #metric_dict['per_class_f1_dict'] = per_class_f1_dict
        if sys.version_info[0] > 3:
            for idx, c in enumerate(cr):
                metric_dict['perclass_f1_' + c] = cr[str(c)]['f1-score']
                metric_dict['perclass_acc_' + c] = per_class_accs[idx]
        
        #metric_dict['P@' + str(self.k)] = self.PatK(y_true, y_pred, self.k)
        #metric_dict['DCG@' + str(self.k)] = self.DCGatK(y_true, y_pred, self.k)
        #metric_dict['NDCG@' + str(self.k)] = self.NDCGatK(y_true, y_pred, self.k)
        return metric_dict
    
    
    """
        y_test --> y
        y_pred --> yhat
        rk_yhat = is the set of rank indices of the truly relevant labels among the top-k portion of the system-predicted ranked list for a document
        y0 = counts the number of relevant labels in the ground truth label vector y
    """
    def PatK(self, y_true, y_pred, k):
        return ranking_precision_score(y_true, y_pred, k=k)
    
    def DCGatK(self, y_true, y_pred, k):
        #return dcg_score(y_true, y_pred, k=k, gains="exponential")
        return dcg_score(y_true, y_pred, k=k)
    
    def NDCGatK(self, y_true, y_pred, k):
        #return ndcg_score(y_true, y_pred, k=k, gains="exponential")
        return ndcg_score(y_true, y_pred, k=k)
    
if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Sentiment Deep Metric Learning')
    parser.add_argument("--data", type=str, default="SS-Youtube")
    cfg = vars(parser.parse_args())

    me = MetricEvaluator()

    for data_name in ['val', 'test']:
        dm_folder_path = "data/deepmoji/" + cfg['data'] + "/" + data_name + '-'
        y_true = np.load(dm_folder_path + 'labels.npy')
        y_pred = np.load(dm_folder_path + 'y_pred.npy')
        y_pred_labels = np.load(dm_folder_path + 'y_pred_labels.npy')
        metric_dict = me.eval_all_metrics(y_true, y_pred, y_pred_labels)
        print(data_name + ': ', metric_dict)


