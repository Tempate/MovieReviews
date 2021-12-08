from .ReviewDataset import ReviewDataset
from .Classifier import Classifier

from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import AdamW

import torch
import torch.nn as nn

import numpy as np


MODEL = 'bert-base-multilingual-uncased'


MAX_LENGTH  = 128
BATCH_SIZE  = 16
NUM_WORKERS = 2

LEARNING_RATE = 2e-5


class Transformers:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL)
        self.model = Classifier(MODEL)

    def create_data_loader(self, data):
        texts, labels = zip(*data)

        dataset = ReviewDataset(
            review=np.array(texts),
            target=np.array(labels),
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH
        )

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        return loader

    def train(self, train_data, valid_data, epochs):
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, correct_bias=False)
        loss_function = nn.BCELoss()

        # We get the model into training mode
        model = self.model.train()

        losses = []
        scores = []

        for epoch in range(epochs):
            for batch in self.create_data_loader(train_data):

                outputs = model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'])

                outputs = torch.flatten(outputs)
                guesses = torch.round(outputs)
                
                targets = batch['targets'].float()
                
                loss = loss_function(outputs, targets)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

            score, loss = self.validate(valid_data, loss_function)

            losses.append(loss)
            scores.append(score)

        return scores[-1], losses[-1]

        def validate(self, data):
            model = self.model.eval()

            losses = []
            scores = []

            with torch.no_grad():
                for batch in self.create_data_loader(data):
                    
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )

                    outputs = torch.flatten(outputs)
                    guesses = torch.round(outputs)

                    targets = batch['targets'].float()
                    loss = loss_function(outputs, targets)

                    losses.append(loss.item())
                    scores.append(torch.sum(guesses == targets))

            return np.mean(scores), np.mean(losses)
