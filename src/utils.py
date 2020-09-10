import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def tokenize(document: str, stopwords_en=stopwords.words('english'), tokenizer=nltk.RegexpTokenizer(r'\w+')):
    document = document.lower()
    return [token for token in tokenizer.tokenize(document) if token not in stopwords_en and token.isalpha()]
