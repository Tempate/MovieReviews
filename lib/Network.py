from lib.Classifier import Classifier
from lib.Chain import Chain

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


LEARNING_RATE = 0.00001


class Network():
    def __init__(self, chain):
        self.chain = chain
        self.model = Classifier(chain)

    def predict(self, sentence):
        return round(self.forward(sentence)[0].item())

    def forward(self, sentence):
        with torch.no_grad():
            vector = self.chain.vectorize(sentence)
            return self.model(vector)

    def run(self, data):
        labels = []
        predictions = []

        for sentence, label in data:
            prediction = self.predict(sentence)
            
            labels.append(self.chain.label_to_key[label])
            predictions.append(prediction)

        return f1_score(labels, predictions)

    def train(self, training_data, validating_data, epochs):
        optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.BCELoss()

        losses = []
        scores = []

        # We pass several times over the training data.
        # Usually there are between 5 and 30 epochs.
        for epoch in range(epochs):
            
            for text, label in training_data:

                vector = self.chain.vectorize(text)
                target = self.chain.make_target(label).float()
                
                # Clear the gradients before each instance
                self.model.zero_grad()
                
                # Run the forward pass
                log_probs = self.model(vector)

                # Compute the loss, gradients, and update the parameters
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()

            loss = self.validate(validating_data, loss_function)
            score = self.run(validating_data)

            losses.append(loss)
            scores.append(score)

            print("%d.\tLoss: %.3f\tF1-score: %.3f" % (epoch+1, loss, score))

        plt.plot(list(range(epochs)), losses, 'o', label='Loss')
        plt.plot(list(range(epochs)), scores, label='F1-score')
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    def validate(self, data, loss_function):
        loss = 0
        
        for text, label in data:
            vector = self.chain.vectorize(text)
            target = self.chain.make_target(label).float()
            
            self.model.zero_grad()
            
            # Run the forward pass
            log_probs = self.model(vector)

            # Compute the loss, gradients, and update the parameters
            loss += loss_function(log_probs, target).item()

        return loss / len(data)
