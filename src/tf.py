# implementing the term frequency vectors

import pprint
from collections import Counter

import nltk
import numpy as np
import pandas

from src.utils import tokenize

nltk.download('stopwords')

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
documents.append(tokens[4 * k:])

# calculating most common 5 tokens from each document and storing frequency tables for each document
most_common = set()
document_frequencies = []
for document in documents:
    frequencies = Counter(document)
    document_frequencies.append(frequencies)
    for word, frequency in frequencies.most_common(5):
        most_common.add(word)

# Calculating the term frequency vectors
vectors = {}
for word in most_common:
    vector = np.zeros((6), dtype=int)
    for index, frequencies in enumerate(document_frequencies):
        vector[index] = frequencies[word]
    vectors[word] = vector
pprint.pp(vectors)

# creating the table representation of words & vectors
table = pandas.DataFrame(data=vectors)

# storing the table in text file to view output
output_file = open('../assets/tf.txt', 'w')
output_file.write(table.to_string())
output_file.close()
