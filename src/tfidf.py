# Implementing the Term Frequency Inverse Document Frequency (TF-IDF) Vectors

import pprint
from collections import Counter

import nltk
import numpy as np
import pandas

# noinspection PyUnresolvedReferences
from utils import tokenize

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
print(len(most_common))

# calculating the number of documents each word appears in
N_t = 6
N_w = {}
for word in most_common:
    count = 0
    for frequencies in document_frequencies:
        count = count + (word in frequencies)
    N_w[word] = count

# Computing the TF-IDF Vectors
vectors = {}
for word in most_common:
    vector = [0] * 6
    for index, frequencies in enumerate(document_frequencies):
        vector[index] = frequencies[word] * np.log(N_t / N_w[word])
    vectors[word] = vector

# storing the vectors in tabular format
table = pandas.DataFrame(data=vectors)
print(table)

# storing the result in text file
output_file = open('../assets/tfidf.txt', 'w')
output_file.write(table.to_string())
output_file.close()
