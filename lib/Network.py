from lib.Classifier import Classifier
from lib.Chain import Chain

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


LEARNING_RATE = 0.00001


class Network():
    def __init__(self, chain, mode):
        self.chain = chain
        self.mode = mode

        self.model = Classifier(chain)

    def eval(self, data):
        texts, labels = zip(*data)
        
        vectors = self.chain.make_vectors(texts, self.mode)
        targets = [self.chain.label_to_key[label] for label in labels]
        
        with torch.no_grad():
            predict = lambda v: round(self.model(v)[0].item())
            guesses = [predict(vector) for vector in vectors]

        return f1_score(guesses, targets)

    def train(self, training_data, validating_data, epochs):
        optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.BCELoss()

        losses = []
        scores = []

        # We pass several times over the training data.
        # Usually there are between 5 and 30 epochs.
        for epoch in range(epochs):

            vectors, targets = self.chain.vectorize(training_data, self.mode)
            
            for vector, target in zip(vectors, targets):
                
                # Clear the gradients before each instance
                self.model.zero_grad()
                
                # Run the forward pass
                log_probs = self.model(vector)

                # Compute the loss, gradients, and update the parameters
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()

            loss = self.validate(validating_data, loss_function)
            score = self.eval(validating_data)

            losses.append(loss)
            scores.append(score)

            print("%d.\tLoss: %.3f\tF1-score: %.3f" % (epoch+1, loss, score))

        xs = list(range(epochs))

        plt.plot(xs, losses, 'o', label='Loss')
        plt.plot(xs, scores, label='F1-score')
        
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    def validate(self, data, loss_function):
        loss = 0

        vectors, targets = self.chain.vectorize(data, self.mode)
        
        for vector, target in zip(vectors, targets):
            self.model.zero_grad()
            
            # Run the forward pass
            log_probs = self.model(vector)

            # Compute the loss, gradients, and update the parameters
            loss += loss_function(log_probs, target).item()

        return loss / len(data)
