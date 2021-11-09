import torch.nn.functional as F
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, chain):
        super(Classifier, self).__init__()

        # Initialize the affine map
        self.linear = nn.Linear(chain.vocab_size, chain.num_labels)

    def forward(self, bow_vec):
        # Pass the input through the linear layer and
        # then through log_softmax for non-linearity
        return F.log_softmax(self.linear(bow_vec), dim=1)
