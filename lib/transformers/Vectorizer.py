from .ReviewDataset import ReviewDataset

from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import BertModel


import numpy as np
import torch


NUM_WORKERS = 2
MAX_LENGTH = 128
BATCH_SIZE = 1


class Vectorizer():
    def __init__(self, MODEL):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL)
        self.model = BertModel.from_pretrained(MODEL)

    def vectorize(self, data):
        self.vectors = {}

        for entry in data:
            for batch in self.data_loader([entry]):
                
                with torch.no_grad():
                    id = batch["text"][0]
                    
                    self.vectors[id] = self.model(
                        input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"]
                    )[1]

        print("[+] Data vectorized correctly")

    def data_loader(self, data):
        texts, labels = zip(*data)

        entry = ReviewDataset(
            np.array(texts),
            np.array(labels),
            self.tokenizer,
            MAX_LENGTH
        )

        loader = DataLoader(
            entry, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS
        )

        return loader
