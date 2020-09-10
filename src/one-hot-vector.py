from collections import Counter

import numpy as np
import pandas
import pprint

# noinspection PyUnresolvedReferences
from utils import tokenize

# nltk.download('stopwords')

# importing corpus as resume
resume_file = open('../assets/resume.txt', 'r')
resume = resume_file.read().lower()
resume_file.close()

# tokenizing the resume
tokens = tokenize(resume)

# dividing corpus into 6 documents
k = len(tokens) // 6
documents = []
for i in range(5):
    documents.append(tokens[i * k: (i + 1) * k])
documents.append(tokens[5 * k:])

# calculating most common 5 tokens from each document
most_common = set()
for document in documents:
    frequencies = Counter(document)
    for word, frequency in frequencies.most_common(5):
        most_common.add(word)

# creating one hot vector for each word in most common
vectors = {}
for word in most_common:
    vector = [0] * 6
    for index, document in enumerate(documents):
        vector[index] = int(word in document)
    vectors[word] = vector

pprint.pp(vectors)

# one hot vector representation
table = pandas.DataFrame(data=vectors)

# writing the table in a text file to view output
file = open('../assets/one-hot-vector.txt', 'w')
file.write(table.to_string())
file.close()
