from lib.Chain import Chain
from lib.Network import Network

from optparse import OptionParser

import regex as re
import random


DATASET = "imdb/dataset.csv"
MODES = ["bag_of_words", "tf_idf"]


def main():
    options = read_commands()

    # Get an unbiased sample of the data
    data = random.sample(options.data, options.sample)

    train, valid, test = split(data)

    network = Network(Chain(data), options.mode)

    print("Test-set's F1-score before training:", network.eval(test))
    network.train(train, valid, options.epochs)
    print("Test-set's F1-score after training:", network.eval(test))


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
    FORMAT = re.compile('"(.+)",(\S+)')
    
    data = []

    for entry in dataset:
        info = FORMAT.match(entry)
        
        text  = info.group(1)
        label = info.group(2)

        # Find all the words in the text
        text = re.findall('\p{L}+', text.lower())

        data.append((text, label))

    return data


def split(data):
    chunk = len(data) // 5

    train = data[:chunk*3]
    valid = data[chunk*3:chunk*4]
    test  = data[chunk*4:]

    return train, valid, test


if __name__ == "__main__":
    main()
