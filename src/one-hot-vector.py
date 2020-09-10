from collections import Counter
import pprint
import random

import numpy as np
import pandas
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords


def tokenize(document: str, stopwords_en=stopwords.words('english'), tokenizer=nltk.RegexpTokenizer(r'\w+')):
    document = document.lower()
    return [token for token in tokenizer.tokenize(document) if token not in stopwords_en and token.isalpha()]


# importing corpus as resume
resume_file = open('assets/resume.txt', 'r')
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

# calculating most common 5 tokens from each document
most_common = set()
for document in documents:
    frequencies = Counter(document)
    for word, frequency in frequencies.most_common(5):
        most_common.add(word)

# creating one hot vector for each word in most common
vectors = {}
for word in most_common:
    vector = np.zeros((6), dtype=int)
    for index, document in enumerate(documents):
        vector[index] = word in document
    vectors[word] = vector

print(vectors)

# one hot vector representation
table = pandas.DataFrame(data=vectors)

# writing the table in a text file to view output
file = open('assets/one-hot-vector.txt', 'w')
file.write(table.to_string())
file.close()
