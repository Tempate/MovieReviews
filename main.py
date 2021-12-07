from lib.feedforward.Feedforward import Feedforward
from lib.transformers.Transformers import Transformers

from optparse import OptionParser

import regex as re
import random


DATASET = "imdb/dataset.csv"
MODES = ["bag_of_words", "tf_idf", "transformers"]


def main():
    options = read_commands()

    # Get an unbiased sample of the data
    data = random.sample(options.data, options.sample)

    train, valid, test = split(data)

    if options.mode == "transformers":
        transformers = Transformers()
        transformers.train(train, valid, options.epochs)
        
    # Feedforward
    elif len(data) <= 50:
        scores = []

        for n in range(5):
            score = 0

            for i in range(len(data)):
                train = data[:i] + data[i+1:]
                valid = [data[i]]
                
                network = Feedforward(data, options.mode)
                score += network.train(train, valid, options.epochs, 10)[0]

            score /= len(data)
            scores.append(score)

            print("[%d]\t%.4f" % (n + 1, score))

        print("Average score: %.4f" % (sum(scores) / 5))

    else:
        network = Feedforward(data, options.mode)

        print("Test-set's F1-score before training: %.3f" % network.eval(test))
        network.train(train, valid, options.epochs, 1, verbose=True)
        print("Test-set's F1-score after training: %.3f" % network.eval(test))


def read_commands():
    parser = OptionParser("%prog -d <dataset>")
    parser.add_option("-d", dest="dataset", help="Dataset to analyze")
    parser.add_option("-e", type="int", default=10, dest="epochs", help="Number of epochs")
    parser.add_option("-s", type="int", default=5000, dest="sample", help="Number of entries to use")
    parser.add_option("-m", type="int", default=0, dest="mode", help="Bag of words (0), Tf-idf (1)")

    options, args = parser.parse_args()

    try:
        options.mode = MODES[options.mode]
    except:
        print("Incorrect mode")
        exit(0)

    with open(options.dataset or DATASET) as dataset:
        options.data = parse(dataset.read().splitlines())

    return options


def parse(dataset):
    data = []

    for entry in dataset:

        text = re.findall('\p{L}+', entry.lower())
        data.append((text, entry[-1]))

    return data


def split(data):
    chunk = len(data) // 5

    train = data[:3 * chunk]
    valid = data[3 * chunk:4 * chunk]
    test  = data[4 * chunk:]

    return train, valid, test


if __name__ == "__main__":
    main()
