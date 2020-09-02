import numpy as np
import h5py
from tqdm import tqdm
import feature_extraction_mp
from dataset import Dataset
from transformers import pipeline
import torch
from torch.multiprocessing import Queue, Value, Process
import time


class FeatureExtraction:

    def __init__(self, cfg, target_path):

        self.cfg = cfg

        self.dataset = Dataset(self.cfg)

        self.target_path = target_path
        
        if self.cfg["device"] != "cpu":
            print('Available devices ', torch.cuda.device_count())
            
            self.n_pipelines = 2
            self.gpus = [int(self.cfg["device"].replace("cuda:", ""))]
            print('Using GPUs ', self.gpus)

            torch.multiprocessing.set_start_method('spawn', force=True)

    """
    def get_gpu_pipeline(self, gpu):
        return pipeline("feature-extraction", model=self.cfg["pretrained"], tokenizer=self.cfg["pretrained"], device=int(gpu))
    """

    def extract(self):
        if self.cfg["device"] != "cpu":
            self.extract_gpu()
        else:
            self.extract_cpu()

    def extract_gpu(self):

        texts, labels, train_ind, val_ind, test_ind = self.dataset.get()
        num_texts = len(texts)

        t0 = time.time()

        queue = Queue()
        producers = []
        num_processes = self.n_pipelines * len(self.gpus)

        if num_processes > 1:
            splits = []
            start_idx = 0
            max_batch_size = num_texts // num_processes
            for pipe_num in range(num_processes):
                gpu_idx = self.gpus[pipe_num//self.n_pipelines]

                if pipe_num == num_processes - 1:
                    end_idx = num_texts
                else:
                    end_idx = start_idx + max_batch_size

                splits.append((pipe_num, gpu_idx, start_idx, end_idx))
                start_idx += max_batch_size
        else:
            pipe_num = 0
            gpu_idx = self.gpus[0]
            start_idx = 0
            end_idx = num_texts
            splits = [(pipe_num, gpu_idx, start_idx, end_idx)]

        # [(0, 0, 0, 535), (1, 0, 535, 1070), (2, 1, 1070, 1605), (3, 1, 1605, 2142)]
        print(f"Number of texts is {num_texts}")
        print(f"Splits: {str(splits)}")

        for pipe_num, gpu_idx, start_idx, end_idx in splits:
            p = Process(target=feature_extraction_mp.producer, args=(queue, texts[start_idx:end_idx], pipe_num, gpu_idx, self.cfg["pretrained"], start_idx))
            p.daemon = True
            producers.append(p)

        max_len = Value('i', 0)
        embedding_size = Value('i', -1)

        consumer = Process(target=feature_extraction_mp.consumer, args=(queue, self.target_path, num_texts, embedding_size, max_len))
        consumer.daemon = True

        consumer.start()
        for p in producers:
            p.start()

        consumer.join()
        for p in producers:
            p.kill()

        print('Parent process exiting...')
        print('Execution time of multiprocessing Code:',  round(time.time() - t0), "seconds")

        max_len = max_len.value
        embedding_size = embedding_size.value
        num_classes = self.dataset.num_classes
        samples_per_cls = [torch.sum(labels[train_ind, cls_idx] > 0) for cls_idx in range(num_classes)]

        with h5py.File(self.target_path, 'a') as f:

            g = f["attributes"] if "attributes" in f else f.create_group("attributes")
            try:
                g.create_dataset("texts", data=np.array(texts, dtype='S'))
            except UnicodeEncodeError or TypeError:
                dt = h5py.special_dtype(vlen=str)
                g.create_dataset("texts", data=np.array(texts, dtype=dt))
            g.create_dataset("train_ind", data=train_ind)
            g.create_dataset("val_ind", data=val_ind)
            g.create_dataset("test_ind", data=test_ind)
            g.create_dataset("num_classes", data=num_classes)
            g.create_dataset("num_texts", data=num_texts)
            g.create_dataset("samples_per_cls", data=samples_per_cls)
            g.create_dataset("max_len", data=max_len)
            g.create_dataset("embedding_size", data=embedding_size)
            g.create_dataset("train_labels", data=labels[train_ind])
            g.create_dataset("val_labels", data=labels[val_ind])
            g.create_dataset("test_labels", data=labels[test_ind])

        torch.multiprocessing.set_start_method('fork', force=True)
        
    def extract_cpu(self):
        
        self.p = pipeline("feature-extraction", model=self.cfg["pretrained"], tokenizer=self.cfg["pretrained"], device=int(-1))

        texts, labels, train_ind, val_ind, test_ind = self.dataset.get()
        num_texts = len(texts)

        num_classes = self.dataset.num_classes

        f = h5py.File(self.target_path, 'a')
        features_g = f["features"] if "features" in f else f.create_group("features")
        #labels_g = f["labels"] if "labels" in f else f.create_group("labels")

        samples_per_cls = [torch.sum(labels[train_ind, cls_idx] > 0) for cls_idx in range(num_classes)]

        max_len = 0
        total_max_len = 0
        embedding_size = -1
        for idx, text in tqdm(enumerate(texts)):
            feats = self.p(text)

            tensor_feats = torch.tensor(feats[0], dtype=torch.double)
            label = labels[idx]
            max_len = max(tensor_feats.shape[0], max_len)
            total_max_len = total_max_len + tensor_feats.shape[0]

            features_g.create_dataset(str(idx), data=tensor_feats)
            if embedding_size < 0:
                embedding_size = tensor_feats.shape[1]
            #labels_g.create_dataset(str(idx), data=label)

        print("total_seq_len=", total_max_len)
        g = f["attributes"] if "attributes" in f else f.create_group("attributes")
        try:
            g.create_dataset("texts", data=np.array(texts, dtype='S'))
        except UnicodeEncodeError or TypeError:
            dt = h5py.special_dtype(vlen=str)
            g.create_dataset("texts", data=np.array(texts, dtype=dt))
        g.create_dataset("train_ind", data=train_ind)
        g.create_dataset("val_ind", data=val_ind)
        g.create_dataset("test_ind", data=test_ind)
        g.create_dataset("num_classes", data=num_classes)
        g.create_dataset("num_texts", data=num_texts)
        g.create_dataset("samples_per_cls", data=samples_per_cls)
        g.create_dataset("max_len", data=max_len)
        g.create_dataset("embedding_size", data=embedding_size)
        g.create_dataset("train_labels", data=labels[train_ind])
        g.create_dataset("val_labels", data=labels[val_ind])
        g.create_dataset("test_labels", data=labels[test_ind])
