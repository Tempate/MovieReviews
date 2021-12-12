from .Vectorizer import Vectorizer

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


MODEL = 'bert-base-multilingual-uncased'

LEARNING_RATE = 5e-3


class Transformers:

    def __init__(self, data):
        self.vectorizer = Vectorizer(MODEL)
        self.vectorizer.vectorize(data)

        self.model = Classifier(self.vectorizer.model.config.hidden_size)

    def train(self, train_data, valid_data, epochs):
        optimizer = optim.NAdam(self.model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.BCELoss()

        # We pass several times over the training data.
        # Usually there are between 5 and 30 epochs.
        for epoch in range(epochs):
            for review in train_data:
              for batch in self.vectorizer.data_loader([review]):
                  
                  vector = self.vectorizer.vectors[batch["text"][0]]
                  target = batch["targets"].float()

                  # We clear the gradients before each instance
                  self.model.zero_grad()

                  # We run the forward pass
                  log_probs = self.model(vector)

                  # We compute the loss, gradients, and update the parameters
                  loss = loss_function(log_probs, target)
                  loss.backward()

                  optimizer.step()

            score, loss = self.validate(valid_data, loss_function)

            print(f"{epoch + 1}.\tLoss: {loss}\tF1-Score: {score}")

        return score, loss

    def validate(self, data, loss_function):
        targets = []
        guesses = []

        loss = 0

        batches = self.vectorizer.data_loader(data)
        
        for batch in batches:

            vector = self.vectorizer.vectors[batch["text"][0]]
            
            target = batch["targets"].float()
            targets.append(target)

            with torch.no_grad():
                # We run the forward pass
                output = self.model(vector)

                # We save our prediction
                guess = torch.round(torch.flatten(output))
                guesses.append(guess)
                
                # We calculate the loss
                loss += loss_function(output, target).item()

        score = f1_score(targets, guesses, zero_division=1)
        loss /= len(batches)

        return score, loss
