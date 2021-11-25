from sklearn.feature_extraction.text import TfidfVectorizer

import torch


class Vectorizer():
    def __init__(self, data):
        self.label_to_key = {"positive": 0, "negative": 1}
        self.words_to_keys(data)

        self.num_labels = len(self.label_to_key)
        self.vocab_size = len(self.word_to_key)

        self.vectorizer = TfidfVectorizer()

    def words_to_keys(self, data):
        self.word_to_key = {}

        for text, _ in data:
            for word in text:
                if word not in self.word_to_key:
                    self.word_to_key[word] = len(self.word_to_key)

    def vectorize(self, data, mode):
        texts, labels = zip(*data)
        
        vectors = self.make_vectors(texts, mode)
        targets = self.make_targets(labels)

        return vectors, targets

    def make_vectors(self, texts, mode):
        return getattr(self, mode)(texts)

    def bag_of_words(self, texts):
        bag = []

        for text in texts:
            # Count the number of times words appear in a text
            count = torch.zeros(len(self.word_to_key))

            for word in text:
                count[self.word_to_key[word]] += 1

            bag.append(count.view(1, -1))

        return bag

    def tf_idf(self, texts):
        texts = [" ".join(text) for text in texts]
        return self.vectorizer.fit_transform(texts)

    def make_targets(self, labels):
        target = lambda label: torch.LongTensor([self.label_to_key[label]])
        return [target(label).float() for label in labels]
