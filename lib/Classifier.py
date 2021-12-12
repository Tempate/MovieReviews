import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()

        self.function1 = nn.Linear(input_size, 16)
        self.activation1 = nn.ReLU()

        self.function2 = nn.Linear(16, 16)
        self.activation2 = nn.ReLU()

        self.function3 = nn.Linear(16, 1)
        self.activation3 = nn.Sigmoid()

    def forward(self, vector):
        layer1 = self.activation1(self.function1(vector))
        layer2 = self.activation2(self.function2(layer1))
        layer3 = self.activation3(self.function3(layer2))

        return (layer3[0]).float()
