from __future__ import print_function

import numpy as np
import json
import argparse
from sklearn.metrics import f1_score

from metric_evaluator import MetricEvaluator
from keras.preprocessing import sequence
#import gpu_DeepMoji.examples.example_helper
from dataset_deepmoji import Dataset

import os

class ExperimentDeepMoji:

    def __init__(self, cfg):

        self.cfg = cfg
        
        if self.cfg["device"] != "cpu":
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg["device"]
            self.deepmoji_name = "gpu"
            from gpu_DeepMoji.deepmoji.model_def import deepmoji_transfer
            from gpu_DeepMoji.deepmoji.global_variables import PRETRAINED_PATH
            from gpu_DeepMoji.deepmoji.finetuning import (load_benchmark, finetune)
            from gpu_DeepMoji.deepmoji.class_avg_finetuning import class_avg_finetune
            
            from gpu_DeepMoji.deepmoji.sentence_tokenizer import SentenceTokenizer
            from gpu_DeepMoji.deepmoji.model_def import deepmoji_architecture

            import tensorflow as tf

            tf.get_logger().setLevel('ERROR')

            #gpus = tf.config.experimental.list_physical_devices('GPU')
            #for gpu in gpus:
                #tf.config.experimental.set_memory_growth(gpu, True)
            #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

        else:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.deepmoji_name = "cpu"
            from cpu_DeepMoji.deepmoji.model_def import deepmoji_transfer
            from cpu_DeepMoji.deepmoji.global_variables import PRETRAINED_PATH
            from cpu_DeepMoji.deepmoji.finetuning import (load_benchmark, finetune)
            from cpu_DeepMoji.deepmoji.class_avg_finetuning import class_avg_finetune
            
            from cpu_DeepMoji.deepmoji.sentence_tokenizer import SentenceTokenizer
            from cpu_DeepMoji.deepmoji.model_def import deepmoji_architecture
        
        if self.cfg['dm_mode'] == 'ft':
            DATASET_PATH = 'data/' + self.cfg["data"] + '/dm_raw.pickle'
            DATASET_PATH = 'data/' + self.cfg["data"] + '/raw.pickle'
            
            with open(self.deepmoji_name + '_DeepMoji/model/vocabulary.json', 'r') as f:
                vocab = json.load(f)

            data = load_benchmark(DATASET_PATH, vocab, extend_with=10000)
            self.batch_size = data['batch_size']
            #print(data)
            
            texts = data['texts']
            labels = data['labels']
            
            lab_shape = np.array(labels[0]).shape
            try:
                nb_classes = lab_shape[1]
            except IndexError:
                nb_classes = 2
                
            import time
            metrics = []
            #for i in range(5):
            np.random.seed(int(time.time()))
            
            #self.model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH, extend_embedding=data['added'], 
                                        #embed_dropout_rate=self.cfg['embed_dropout_rate'], final_dropout_rate=self.cfg['final_dropout_rate'], embed_l2=self.cfg['embed_l2'])
            self.model = deepmoji_transfer(2, data['maxlen'], PRETRAINED_PATH, extend_embedding=data['added'],
                                           embed_dropout_rate=self.cfg['embed_dropout_rate'], final_dropout_rate=self.cfg['final_dropout_rate'], embed_l2=self.cfg['embed_l2'])
            
            #self.model = deepmoji_transfer(nb_classes, data['maxlen'], None, extend_embedding=data['added'])
            self.model.summary()
            #self.model, acc = finetune(self.model, texts, labels, nb_classes, self.batch_size, method='new')#, metric='acc')
            #self.model, acc = finetune(self.model, texts, labels, nb_classes, self.batch_size, method='chain-thaw', lr=self.cfg['initial_lr'], next_lr=self.cfg['next_lr'])#, metric='acc')

            self.model, f1, all_y_pred_val, all_y_pred_test, all_ts = class_avg_finetune(self.model, data['texts'], data['labels'], nb_classes, self.batch_size, method='chain-thaw', lr=self.cfg['initial_lr'], next_lr=self.cfg['next_lr'], verbose=self.cfg['verbose'])

            (X_train, y_train) = (texts[0], labels[0])
            (X_val, y_val) = (texts[1], labels[1])
            (X_test, y_test) = (texts[2], labels[2])

            #acc = self.evaluate_using_acc(self.model, X_test, y_test, batch_size=self.batch_size)
            #weighted_f1 = self.evaluate_using_weighted_f1(self.model, X_test, y_test, X_val, y_val, batch_size=self.batch_size)
            #print('Acc: {} Weighted F1: {} f1: {}'.format(acc, weighted_f1, f1))
            print('f1: {}'.format(f1))
            
            y_pred_labels = all_y_pred_val >= all_ts
            
            me = MetricEvaluator()
            val_metric_dict = me.eval_all_metrics(np.array(y_val), all_y_pred_val, y_pred_labels)
            self.val_metrics = val_metric_dict
            
            y_pred_labels = all_y_pred_test >= all_ts
            metric_dict = me.eval_all_metrics(np.array(y_test), all_y_pred_test, y_pred_labels)
            self.test_metrics = metric_dict
            
            val_f1 = val_metric_dict["macro_f1"]
            print("trial result: ", val_f1)
            
            #val_metrics = self.eval(texts[1], labels[1], 'val')
            #test_metrics = self.eval(texts[2], labels[2], 'test')
            #print('val metrics:', val_metrics)
            #print('test metrics:', test_metrics)
            #metrics.append([acc, weighted_f1, val_metrics['acc'], val_metrics['f1'], val_metrics['P@3'], val_metrics['DCG@3'], val_metrics['NDCG@3'], test_metrics['acc'], test_metrics['f1'], test_metrics['P@3'], test_metrics['DCG@3'], test_metrics['NDCG@3']])
            #metrics.append([acc, weighted_f1])
                
            #print(np.mean(np.array(metrics), axis=0))
            
            #self.val_metrics = val_metrics
            
            """
            import json
            with open('ex_dm_val_out.txt', 'w') as file:
                 json.dump(val_metric_dict)
            """
            
        elif self.cfg['dm_mode'] == 'scratch':

            tokenization_length = 30 # Try to find this by brute force?
            nb_tokens = 60000 # Find this programatically
            
            self.dataset = Dataset(self.cfg)
            self.num_classes = self.dataset.num_classes
            
            """
            import codecs
            from DeepMoji.deepmoji.create_vocab import VocabBuilder
            from DeepMoji.deepmoji.word_generator import TweetWordGenerator
            
            
            with codecs.open('../../twitterdata/tweets.2016-09-01', 'rU', 'utf-8') as stream:
                wg = TweetWordGenerator(stream)
                vb = VocabBuilder(wg)
                vb.count_all_words()
                vb.save_vocab()
            """
            
            with open(self.deepmoji_name + '_DeepMoji/model/vocabulary.json', 'r') as f:
                vocabulary = json.load(f)

            st = SentenceTokenizer(vocabulary, tokenization_length)

            texts, labels, train_ind, val_ind, test_ind = self.dataset.get()
            texts = [unicode(t) for t in texts]
            labels = labels.tolist()
            
            def create_dicts(texts, labels, idxs):
                texts = [texts[i] for i in idxs]
                #texts = [text.encode(encoding='UTF-8',errors='strict') for text in texts]
                tokens, _, _ = st.tokenize_sentences(texts)
                
                labels = [labels[i] for i in idxs]
                
                #deepmoji_instances = []
                dm_texts = []
                dm_tokens = []
                dm_labels = []
                for instance_idx, instance_label in enumerate(labels):
                    for label_idx, label in enumerate(instance_label):
                        if label:
                            dm_texts.append(texts[instance_idx])
                            dm_tokens.append(tokens[instance_idx])
                            dm_labels.append(int(label_idx))
                            #instance_str = [str(label_idx), tokens[instance_idx]]
                            #deepmoji_instances.append(instance_str)
                
                
                #return {'texts': texts, 'labels': labels, 'tokens': tokens, 'data': deepmoji_instances}
                return {'dm_texts': dm_texts, 'dm_tokens': dm_tokens, 'dm_labels': np.array(dm_labels)}
                
            train_dict = {'partition': 'train', 'idxs': train_ind}
            val_dict = {'partition': 'val', 'idxs': val_ind}
            test_dict = {'partition': 'test', 'idxs': test_ind}
            
            self.dicts = [train_dict, val_dict, test_dict]
            
            for dic in self.dicts:
                dic.update(create_dicts(texts, labels, dic['idxs']))

            train_dict['dm_tokens_pad'] = sequence.pad_sequences(train_dict['dm_tokens'], maxlen=tokenization_length)
            val_dict['dm_tokens_pad'] = sequence.pad_sequences(val_dict['dm_tokens'], maxlen=tokenization_length)
            test_dict['dm_tokens_pad'] = sequence.pad_sequences(test_dict['dm_tokens'], maxlen=tokenization_length)
            
            
            epochs = 5
            
            import time
            metrics = []
            for i in range(5):
                np.random.seed(int(time.time()))
            
                print('Build model...')
                model = deepmoji_architecture(nb_classes=2, nb_tokens=nb_tokens, maxlen=tokenization_length)
                model.summary()

                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                print('Train...')
                #print(len(train_dict['dm_tokens_pad']))
                #print(len(val_dict['dm_tokens_pad']))
                #print(len(test_dict['dm_tokens_pad']))
                model.fit(train_dict['dm_tokens_pad'], train_dict['dm_labels'], batch_size=self.cfg['batch_size'], epochs=epochs, validation_data=(val_dict['dm_tokens_pad'], val_dict['dm_labels']))
                score, acc = model.evaluate(test_dict['dm_tokens_pad'], test_dict['dm_labels'], batch_size=self.cfg['batch_size'])
                print('Test score:', score)
                print('Test accuracy:', acc)

                metrics.append([score, acc])
                
            print(np.mean(np.array(metrics), axis=0))
        else:
            raise Exception("Unknown DeepMoji Mode")

    def run(self):
        pass

    def eval(self, texts, dm_labels, data_name):
        dm_y_pred = np.array(self.model.predict(texts, batch_size=self.batch_size))
        print(dm_y_pred)
        
        y_pred = []
        for yp in dm_y_pred:
            yp = yp[0]
            new_guy = [1-yp, yp]
            """
            if yp >= 0.5:
                new_guy = [1-yp, yp]
            else:
                new_guy = [yp, 1-yp]
            """
            y_pred.append(new_guy)
        
        y_pred = np.array(y_pred)
        y_pred_labels = (y_pred >= 0.5)
        print(dm_labels)
        
        labels = []
        for lb in dm_labels:
            if lb == 0:
                new_guy = [True, False]
            else:
                new_guy = [False, True]
            labels.append(new_guy)
        labels = np.array(labels)
        
        dm_folder_path = "data/deepmoji/" + self.cfg['data'] + "/"
        if not os.path.exists(dm_folder_path):
            os.makedirs(dm_folder_path)
            
        dm_folder_path = dm_folder_path + data_name + '-'
            
        np.save(dm_folder_path + 'labels.npy', labels)
        np.save(dm_folder_path + 'y_pred.npy', y_pred)
        np.save(dm_folder_path + 'y_pred_labels.npy', y_pred_labels)
        
        #me = MetricEvaluator()
        #metric_dict = me.eval_all_metrics(labels, y_pred, y_pred_labels)
        #return metric_dict
        
    def find_f1_threshold(self, y_val, y_pred_val, y_test, y_pred_test, average='binary'):
        """ Choose a threshold for F1 based on the validation dataset
            (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/
            for details on why to find another threshold than simply 0.5)

        # Arguments:
            y_val: Outputs of the validation dataset.
            y_pred_val: Predicted outputs of the validation dataset.
            y_test: Outputs of the testing dataset.
            y_pred_test: Predicted outputs of the testing dataset.

        # Returns:
            F1 score for the given data and
            the corresponding F1 threshold
        """
        thresholds = np.arange(0.01, 0.5, step=0.01)
        f1_scores = []

        for t in thresholds:
            y_pred_val_ind = (y_pred_val > t)
            f1_val = f1_score(y_val, y_pred_val_ind, average=average)
            f1_scores.append(f1_val)

        best_t = thresholds[np.argmax(f1_scores)]
        y_pred_ind = (y_pred_test > best_t)
        f1_test = f1_score(y_test, y_pred_ind, average=average)
        return f1_test, best_t
        
    def evaluate_using_weighted_f1(self, model, X_test, y_test, X_val, y_val, batch_size):
        """ Evaluation function using macro weighted F1 score.

        # Arguments:
            model: Model to be evaluated.
            X_test: Inputs of the testing set.
            y_test: Outputs of the testing set.
            X_val: Inputs of the validation set.
            y_val: Outputs of the validation set.
            batch_size: Batch size.

        # Returns:
            Weighted F1 score of the given model.
        """
        y_pred_test = np.array(model.predict(X_test, batch_size=batch_size))
        y_pred_val = np.array(model.predict(X_val, batch_size=batch_size))

        f1_test, _ = self.find_f1_threshold(y_val, y_pred_val, y_test, y_pred_test,
                                    average='weighted')
        return f1_test


    def evaluate_using_acc(self, model, X_test, y_test, batch_size):
        """ Evaluation function using accuracy.

        # Arguments:
            model: Model to be evaluated.
            X_test: Inputs of the testing set.
            y_test: Outputs of the testing set.
            batch_size: Batch size.

        # Returns:
            Accuracy of the given model.
        """
        _, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

        return acc
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepMoji Experiment')

    parser.add_argument("--data", type=str, default="SE0714")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dm_mode", type=str, default="ft")
    parser.add_argument("--verbose", type=str, default="False")
    
    parser.add_argument("--embed_dropout_rate", type=float)
    parser.add_argument("--final_dropout_rate", type=float)
    parser.add_argument("--embed_l2", type=float)
    parser.add_argument("--initial_lr", type=float)
    parser.add_argument("--next_lr", type=float)

    cfg = vars(parser.parse_args())
    
    cfg['verbose'] = cfg['verbose'] == 'True'
    
    print(cfg)
    
    experiment = ExperimentDeepMoji(cfg)
    experiment.run()
