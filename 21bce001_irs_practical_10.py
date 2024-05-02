# -*- coding: utf-8 -*-
"""21BCE111_IRS_Practical_10.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wT6_PNizTUzOQUH4bstzSldWkxXeufO7
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Remove hyperlinks, twitter marks, and styles
def remove_hyperlinks(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

# Remove punctuation
def remove_punctuation(text):
    return ''.join([i for i in text if i not in string.punctuation])

# Tokenize and lower case
def tokenize_and_lower(text):
    tokens = word_tokenize(text)
    return [token.lower() for token in tokens]

# Lemmatize
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Preprocessing function
def preprocess_text(text):
    text = remove_hyperlinks(text)
    text = remove_punctuation(text)
    tokens = tokenize_and_lower(text)
    tokens = lemmatize(tokens)
    return tokens

with open('./Practical-10.txt','r') as f:
  text = f.read()
preprocessed_text = preprocess_text(text)
print(preprocessed_text)

len_vocab = len(preprocessed_text)

dict_words = {}
for i in preprocessed_text:
    if i not in dict_words:
        dict_words[i] = 1
    else:
        dict_words[i] += 1

print(len_vocab)

print(dict_words)

dict_words = {key: value/len_vocab for key, value in dict_words.items() if value!= 0}

dict_words

query = 'the sun flower rose filling intoxicating'

query_vec = query.split()

query_vec

prob = 1

for i in query_vec:
    prob *= dict_words[i]

prob