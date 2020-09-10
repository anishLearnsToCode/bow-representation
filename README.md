# BOW Representations
__Anish Sachdeva (DTU/2K16/MC/13)__

__Natural Language Processing - Dr. Seba Susan__

[📕 One Hot Vector](notebooks/one-hot-vector.ipynb) | 
[📕 Term Frequency (TF)](notebooks/term-frequency.ipynb) | 
[📕 Term Frequency - Inverse Document Frequency (TF-IDF)](notebooks/term-frequency-inverse-document-frequency.ipynb) | 
[✒ Report](assets/report.pdf)

![project-image](assets/booster.jpg)

## 📖 Overview
1. [Introduction](#introduction)
1. [Results](#results)
1. [Running it on Your Machine](#running-it-on-your-machine)
1. [Bibliography](#bibliography)

## Introduction
In many applications where we use our words as input in Machine Learning Models or in Deep
Learning etc. we can't directly use our words as text and character input as machines can't 
perform numerical and analytical tasks directly on character sequences and perform better when 
given numerical tasks.

To rectify this we convert words into vectors and then use the techniques of Linear Algebra and
Optimization which are readily available to us to work on our data. We can convert words into 
vectors using many different methods and there are already many different data sources available
online that provide us with pre-computed vectors for words.

In this assignment we compute word vectors from a resume using the following techniques:
1. [One Hot Vectors](notebooks/one-hot-vector.ipynb) 
1. [Term Frequency (TF) Vectors](notebooks/term-frequency.ipynb) 
1. [Term Frequency - Inverse Document Frequency (TF-IDF) Vectors](notebooks/term-frequency-inverse-document-frequency.ipynb) 

## Results
1. ⭐ [One Hot Vector](assets/one-hot-vector.txt)
1. ⭐ [Term Frequency (TF) Vectors](assets/tf.txt)
1. ⭐ [Term Frequency Inverse Document Frequency (TF-IDF) vectors](assets/tfidf.txt)

## Running it on Your Machine
Clone this project on your machine and enter the src directory.
```bash
git clone https://github.com/anishLearnsToCode/bow-representation.git
cd bow-representation/src
```

Install Requirements:
```bash
pip install -r requirements.txt
```

See Vector Outputs as 
```bash
python one-hot-vector.py
python tf.py
python tfidf.py
```

Run the Notebooks and see interactive output:
```bash
cd bow-representation/notebooks
jupyter notebook
```

## Bibliography
1. [Speech & Language Processing ~Jurafsky](https://web.stanford.edu/~jurafsky/slp3/)
1. [nltk](https://www.nltk.org/)
1. [pickle](https://docs.python.org/3/library/pickle.html)
1. [pandas](https://pandas.pydata.org/)
1. [pandas.DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
1. [Indexing and Slicing on Pandas DataFrames](https://datacarpentry.org/python-ecology-lesson/03-index-slice-subset/index.html)
1. [numpy](https://numpy.org/)
