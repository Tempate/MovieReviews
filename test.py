from lib.Chain import Chain
from lib.Network import Network

import random
import re


with open("imdb_dataset.csv") as dataset:
    FORMAT = re.compile('"(.+)",(\S+)')
    
    data = []

    for entry in dataset.read().splitlines():
        info = FORMAT.match(entry)
        data.append((info.group(1), info.group(2)))


training = len(data) * 4 // 5

train_data = data[:training]
test_data  = data[training:]

chain = Chain(data)
network = Network(chain)

print("Accuracy before training:", network.run(test_data))
network.train(train_data)
print("Accuracy after training:", network.run(test_data))
