import h5py
import pickle
import numpy as np
import torch
from utils.emoji_tweets_preprocessing import num_emojis


class Dataset:

    def __init__(self, cfg):

        self.cfg = cfg

        self.datasets = {
            'SE0714': ('data/SE0714/raw.pickle', 3),
            'Olympic': ('data/Olympic/raw.pickle', 4),
            'PsychExp': ('data/PsychExp/raw.pickle', 7),
            'SS-Twitter': ('data/SS-Twitter/raw.pickle', 2),
            'SS-Youtube': ('data/SS-Youtube/raw.pickle', 2),
            'SCv1': ('data/SCv1/raw.pickle', 2),
            'SCv2-GEN': ('data/SCv2-GEN/raw.pickle', 2),
            'kaggle-insults': ('data/kaggle-insults/raw.pickle', 2),
            'emoji-tweets': ('data/emoji-tweets/raw.pickle', num_emojis),
            'SemEval_Arabic': ('data/SemEval_Arabic/raw.pickle', 11),
            'SemEval_English': ('data/SemEval_English/raw.pickle', 11),
            'SemEval_Spanish': ('data/SemEval_Spanish/raw.pickle', 11),
            'SemEval_Arabic_English': ('data/SemEval_Arabic_English/raw.pickle', 11),
            'SemEval_Arabic_Spanish': ('data/SemEval_Arabic_Spanish/raw.pickle', 11),
            'SemEval_English_Spanish': ('data/SemEval_English_Spanish/raw.pickle', 11),
            'SemEval_English_Turkish': ('data/SemEval_English_Turkish/raw.pickle', 11),
            'SemEval_Arabic_English_Spanish': ('data/SemEval_Arabic_English_Spanish/raw.pickle', 11),
            'SemEval_Turkish': ('data/SemEval_Turkish/raw.pickle', 11),
            'SemEval_Tran_Spanish': ('data/SemEval_Tran_Spanish/raw.pickle', 11),
        }

        self.data_path, self.num_classes = self.datasets[self.cfg["data"]]
        """
        if self.cfg["data"] == "PsychExp":
            encoding = "latin1"
        else:
            encoding = "utf-8"
        self.data = pickle.load(open(self.data_path, "rb"), encoding=encoding, fix_imports=True)
        """
        self.data = pickle.load(open(self.data_path, "rb"))

    def info_to_labels(self, inf):
        labels = []
        for i in range(len(inf)):
            label_lst = inf[i]["label"]
            if type(label_lst) is np.int64 or type(label_lst) is int:
                labels.append([label_lst])
            else:
                trues = [i for i, x in enumerate(label_lst) if x]

                #assert len(trues) == 1

                labels.append(trues)

        n_categories = np.amax(labels)
        if isinstance(n_categories, list):
            n_categories = n_categories[0] + 1
        else:
            n_categories = n_categories + 1

        one_hot = torch.zeros(size=(len(labels), n_categories), dtype=torch.bool)
        for i in range(len(labels)):
            idx = torch.tensor(labels[i], dtype=int)
            one_hot[i, idx] = 1

        return one_hot

    def filter_nonzeros(self, indices, labels):
        """
        res_indices = []
        for idx in indices:
            if torch.any(labels[idx]):
                res_indices.append(idx)

        return res_indices
        """
        return indices

    def get(self):

        texts = self.data["texts"]
          
        if 'SemEval' not in self.cfg["data"]:
            labels = self.info_to_labels(self.data["info"])
        else:
            labels = torch.tensor(self.data["labels"], dtype=torch.bool)
            
        train_ind = self.filter_nonzeros(self.data["train_ind"], labels)
        val_ind = self.filter_nonzeros(self.data["val_ind"], labels)
        test_ind = self.filter_nonzeros(self.data["test_ind"], labels)
        
        if self.cfg['prepro'] == 'all':
            #tokenizer = SocialTokenizer(lowercase=True).tokenize
            text_processor = TextPreProcessor(
                normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                    'time', 'url', 'date', 'number'],
                annotate={"hashtag", "allcaps", "elongated", "repeated",
                    'emphasis', 'censored'},
                fix_html=True,  # fix HTML tokens
                #segmenter="twitter", 
                #corrector="twitter", 
                #unpack_hashtags=True,  # perform word segmentation on hashtags
                #unpack_contractions=True,  # Unpack contractions (can't -> can not)
                spell_correct_elong=False,  # spell correction for elongated words
                tokenizer=SocialTokenizer(lowercase=False).tokenize,
                mode='fast',
                dicts=[emoticons]
            )
            print('Using Social-all')
            import time
            s = time.time()
            #tokens = [tokenizer(t) for t in texts]
            texts = [" ".join(text_processor.pre_process_doc(t)) for t in texts]
            print('Took', time.time() - s)
            #print(tokens)
        elif self.cfg['prepro'] == 'eng':
            text_processor = TextPreProcessor(
                normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                    'time', 'url', 'date', 'number'],
                annotate={"hashtag", "allcaps", "elongated", "repeated",
                    'emphasis', 'censored'},
                fix_html=True,  # fix HTML tokens
                segmenter="twitter", 
                corrector="twitter", 
                unpack_hashtags=True,  # perform word segmentation on hashtags
                unpack_contractions=True,  # Unpack contractions (can't -> can not)
                spell_correct_elong=False,  # spell correction for elongated words
                tokenizer=SocialTokenizer(lowercase=False).tokenize,
                dicts=[emoticons]
            )
            print('Using Social-eng')
            import time
            s = time.time()
            #tokens = [tokenizer(t) for t in texts]
            texts = [" ".join(text_processor.pre_process_doc(t)) for t in texts]
            print('Took', time.time() - s)
            #print(tokens)
            
        print(len(texts))
        return texts, labels, train_ind, val_ind, test_ind


class TextClassificationDataset:

    def __init__(self, cfg, partition):
        self.cfg = cfg
        self.partition = partition
        
        if self.cfg['prepro'] == 'None':
            self.target_path = "data/" + self.cfg['data'] + "-" + self.cfg['pretrained'].replace('/', '_') + ".h5"
        else:
            self.target_path = "data/" + self.cfg['data'] + "-" + self.cfg['pretrained'].replace('/', '_') + "-prepro_" + self.cfg['prepro'] + ".h5" 
            
        f = h5py.File(self.target_path, "r")
        attrs = f["attributes"]
        self.indices = attrs[partition + "_ind"][()]
        self.num_classes = attrs["num_classes"][()]
        self.max_len = attrs["max_len"][()]
        self.embedding_size = attrs["embedding_size"][()]
        self.features = f["features"]

        if partition == "train":
            self.samples_per_cls = torch.from_numpy(attrs["samples_per_cls"][()].astype(np.long))

        self.y_true = attrs[self.partition + "_labels"][()]

    def __len__(self):

        return self.indices.shape[0]

    def __getitem__(self, index):
        idx = self.indices[index]

        y = self.y_true[index].astype(np.double)

        idx = str(idx)
        X = self.features[idx][()]

        mask = len(X)

        return X, y, mask
