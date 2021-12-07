from torch.utils.data import Dataset

import torch


class ReviewDataset(Dataset):
  def __init__(self, review, target, tokenizer, max_length):
    self.review = review
    self.target = target
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.review)

  def __getitem__(self, item):
    review = str(self.review[item])

    encoding = self.tokenizer.encode_plus(
      review,
      
      add_special_tokens=True,
      
      truncation=True,
      padding='max_length',
      max_length=self.max_length,
      
      return_token_type_ids=False,
      return_attention_mask=True,
      return_tensors='pt'
    )

    return {
      'review_text': review,
      'targets': torch.tensor(self.target[item], dtype=torch.long),

      'attention_mask': encoding['attention_mask'].flatten(),
      'input_ids': encoding['input_ids'].flatten()
    }
