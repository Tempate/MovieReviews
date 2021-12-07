from transformers import BertModel
from torch import nn


class Classifier(nn.Module):
  def __init__(self, model_name):
    super(Classifier, self).__init__()

    self.bert = BertModel.from_pretrained(model_name)
    
    self.linear = nn.Linear(self.bert.config.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
    output = self.linear(output)

    return self.sigmoid(output).float()
