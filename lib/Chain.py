import torch


class Chain():
    def __init__(self, data):
        self.label_to_key = {"positive": 0, "negative": 1}
        self.words_to_keys(data)

        self.num_labels = len(self.label_to_key)
        self.vocab_size = len(self.word_to_key)

    def words_to_keys(self, data):
        self.word_to_key = {}

        for sentence, _ in data:
            for word in sentence:
                if word in self.word_to_key:
                    continue

                try:
                    self.word_to_key[word] = len(self.word_to_key)
                except KeyError:
                    pass

    def vectorize(self, sentence):
        # Match the number of times a word appears to its key
        count = torch.zeros(len(self.word_to_key))

        for word in sentence:
            count[self.word_to_key[word]] += 1

        return count.view(1, -1)

    def make_target(self, label):
        return torch.LongTensor([self.label_to_key[label]])
