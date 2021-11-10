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
        return round(self.forward(sentence)[0].item())

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

    def train(self, training_data, validating_data):
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001)
        loss_function = nn.BCELoss()

        # We pass several times over the training data.
        # Usually there are between 5 and 30 epochs.
        for epoch in range(10):
            
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

            self.validate(validating_data, loss_function)

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

        print(loss)
