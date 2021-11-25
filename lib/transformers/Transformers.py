from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW

import torch
import torch.nn as nn

from tqdm import tqdm


MODEL = 'bert-base-multilingual-uncased'

label_to_key = {"positive": 0, "negative": 1}


class Transformers:

    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL)

    def vectorize(self, data):
        texts, labels = zip(*data)

        texts = []
        labels = []

        for text, label in data:
            texts.append("[CLS] " + " ".join(text))
            labels.append(label_to_key[label])

        vectors = self.tokenizer(texts, 
                                 return_tensors='pt', 
                                 max_length=128, 
                                 truncation=True, 
                                 padding="max_length")

        vectors['labels'] = torch.LongTensor([labels]).T

        return vectors

    def train(self, training_data, validation_data):
        dataset = Dataset(self.vectorize(training_data))
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # We run the computations in the GPU if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.train()

        optim = AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(2):
            for batch in tqdm(loader, leave=True):
                
                optim.zero_grad()

                input_ids      = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)

                outputs = self.model(input_ids, 
                                     token_type_ids=token_type_ids, 
                                     attention_mask=attention_mask,
                                     labels=labels)

                loss = outputs.loss
                loss.backward()
                optim.step()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings.input_ids.shape[0]

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.encodings.items()}
