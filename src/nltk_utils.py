import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# nltk.download('punkt') # only need to run this line of code once
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

# CountVectorizer from sklearn can be also be used for this (instead of this implementation)
def bag_of_words(tokenized_sentence, all_words):
    sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, x in enumerate(all_words):
        if x in sentence:    
            bag[idx] = 1

    return bag