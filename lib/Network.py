from lib.Classifier import Classifier
from lib.Chain import Chain

import torch
import torch.nn as nn
import torch.optim as optim


class Network():
    def __init__(self, chain):
        self.chain = chain
        self.model = Classifier(chain)

    def predict(self, sentence):
        log_probs = self.forward(sentence).tolist()[0]
        return log_probs.index(max(log_probs))

    def forward(self, sentence):
        with torch.no_grad():
            vector = self.chain.vectorize(sentence)
            return self.model(vector)

    def run(self, data):
        correct = 0

        for sentence, label in data:
            prediction = self.predict(sentence)
            
            if prediction == self.chain.label_to_key[label]:
                correct += 1

        return correct / len(data)

    def train(self, data):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        loss_function = nn.NLLLoss()

        # We pass several times over the training data.
        # 100 is much more than on a real data set, but we only have two instances.
        # Usually there are between 5 and 30 epochs.
        for epoch in range(5):
            for sentence, label in data:

                vector = self.chain.vectorize(sentence)
                target = self.chain.make_target(label)
                
                # Clear the gradients before each instance
                self.model.zero_grad()
                
                # Run the forward pass
                log_probs = self.model(vector)

                # Compute the loss, gradients, and update the parameters
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
