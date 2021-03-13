import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenise(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenised_Sentence,all_words):
    
    bag = np.zeros(len(all_words),dtype = np.float32)
    tokenised_Sentence = [stem(w) for w in tokenised_Sentence]
    
    for ind,w in enumerate(all_words):
        if w in tokenised_Sentence:
            bag[ind] = 1.0
    
    return bag
