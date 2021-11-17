from lib.Chain import Chain
from lib.Network import Network

from optparse import OptionParser

import regex as re
import random


DATASET = "imdb_dataset.csv"


def main():
    options = read_commands()

    # Get an unbiased sample of the data
    data = random.sample(options.data, options.sample)

    train, valid, test = split(data)

    network = Network(Chain(data))

    print("Test-set F1-score before training:", network.run(test))
    network.train(train, valid, options.epochs)
    print("Test-set F1-score after training:", network.run(test))


def read_commands():
    parser = OptionParser("%prog -d <dataset>")
    parser.add_option("-d", dest="dataset", help="Dataset to analyze")
    parser.add_option("-e", type="int", default=10, dest="epochs", help="Number of epochs")
    parser.add_option("-s", type="int", default=5000, dest="sample", help="Number of entries to use")

    options, args = parser.parse_args()

    with open(options.dataset or DATASET) as dataset:
        options.data = parse(dataset.read().splitlines())

    return options


def parse(dataset):
    FORMAT = re.compile('"(.+)",(\S+)')
    
    data = []

    for entry in dataset:
        info = FORMAT.match(entry)
        
        text = info.group(1)
        label = info.group(2)

        # Remove weird symbols and punctuation signs
        text = re.sub('[^\p{L}]+', '', text)
        
        data.append((text.lower(), label))

    return data


def split(data):
    chunk = len(data) // 5

    train = data[:chunk*3]
    valid = data[chunk*3:chunk*4]
    test  = data[chunk*4:]

    return train, valid, test


if __name__ == "__main__":
    main()
