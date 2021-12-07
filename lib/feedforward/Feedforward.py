from .Classifier import Classifier
from .Vectorizer import Vectorizer

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


LEARNING_RATE = {
    "bag_of_words": 2e-5,
    "tf_idf": 5e-5
}


class Feedforward():
    def __init__(self, data, mode):
        self.vectorizer = Vectorizer(data)
        self.mode = mode

        self.model = Classifier(self.vectorizer)

    def eval(self, data):
        texts, labels = zip(*data)
        
        vectors = self.vectorizer.make_vectors(texts, self.mode)
        targets = list(map(int, labels))
        
        with torch.no_grad():
            predict = lambda v: round(self.model(v)[0].item())
            guesses = [predict(vector) for vector in vectors]

        return sum([1 for guess, target in zip(guesses, targets) if guess == target]) 
        # f1_score(guesses, targets, zero_division=0)

    def train(self, train_data, valid_data, epochs, patience, verbose=False):
        optimizer = optim.NAdam(self.model.parameters(), lr=LEARNING_RATE[self.mode])
        loss_function = nn.BCELoss()

        losses = []
        scores = []

        # We pass several times over the training data.
        # Usually there are between 5 and 30 epochs.
        for epoch in range(epochs):

            vectors, targets = self.vectorizer.vectorize(train_data, self.mode)
            
            for vector, target in zip(vectors, targets):
                
                # Clear the gradients before each instance
                self.model.zero_grad()
                
                # Run the forward pass
                log_probs = self.model(vector)

                # Compute the loss, gradients, and update the parameters
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()

            loss = self.validate(valid_data, loss_function)
            score = self.eval(valid_data)

            losses.append(loss)
            scores.append(score)

            if verbose:
                print("%d.\tLoss: %.3f\tF1-score: %.3f" % (epoch+1, loss, score))

            # Prevent over-fitting
            if len(losses) >= 2 and losses[-1] - losses[-2] >= 0:
                patience -= 1

            if patience <= 0:
                break

        if verbose:
            self.plot(scores, losses)

        return scores[-1], losses[-1]

    def validate(self, data, loss_function):
        loss = 0

        vectors, targets = self.vectorizer.vectorize(data, self.mode)
        
        for vector, target in zip(vectors, targets):
            self.model.zero_grad()
            
            # Run the forward pass
            log_probs = self.model(vector)

            # Compute the loss, gradients, and update the parameters
            loss += loss_function(log_probs, target).item()

        return loss / len(data)

    def plot(self, scores, losses):
        xs = list(range(len(scores)+1))

        plt.plot(xs, losses, 'o', label='Loss')
        plt.plot(xs, scores, label='F1-score')
        
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()