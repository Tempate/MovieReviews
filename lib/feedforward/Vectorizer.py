from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import torch


mode_to_func = {
    "bag_of_words": "CountVectorizer",
    "tf_idf": "TfidfVectorizer"
}


class Vectorizer():
    def __init__(self, data):
        self.label_to_key = {"positive": 0, "negative": 1}

        self.vocabulary = self.get_unique_words(data)
        self.vocab_size = len(self.vocabulary)

    def vectorize(self, data, mode):
        texts, labels = zip(*data)
        
        vectors = self.make_vectors(texts, mode)
        targets = self.make_targets(labels)

        return vectors, targets

    def make_vectors(self, texts, mode):
        vectorizer = globals()[mode_to_func[mode]]

        texts = [" ".join(text) for text in texts]

        counter = vectorizer(vocabulary=self.vocabulary)
        vectors = counter.fit_transform(texts).toarray()

        to_tensor = lambda vector: torch.from_numpy(vector).float()
        
        return [to_tensor(v).view(1,-1) for v in vectors]

    def make_targets(self, labels):
        to_target = lambda label: torch.LongTensor([self.label_to_key[label]])
        return [to_target(label).float() for label in labels]

    @staticmethod
    def get_unique_words(data):
        vocabulary = set()

        for text, _ in data:
            for word in text:
                vocabulary.add(word)

        return list(vocabulary)
