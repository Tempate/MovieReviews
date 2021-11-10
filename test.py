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


random.shuffle(data)
data = data[:5000]

chunk = len(data) // 5

training_data   = data[:chunk * 3]
validating_data = data[chunk * 3:chunk * 4]
testing_data    = data[chunk * 4:]

chain = Chain(data)
network = Network(chain)

print("Accuracy before training:", network.run(testing_data))
network.train(training_data, validating_data)
print("Accuracy after training:", network.run(testing_data))
