#Fitting Scaler
#from magpie import Magpie

import datetime
now = datetime.datetime.now()
print ("Started Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))
import time 
ts = time.time()
print('Program Started by ',ts)

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
import io
import os
import nltk
import string
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

import io
import os
import random
from collections import Counter, defaultdict

from nltk.tokenize import WordPunctTokenizer, sent_tokenize, word_tokenize

w2vmodel_path_in_c  = 'path to word2vec model'
data_dir = 'path to corpus in Complete.txt' # A Single file will have the entire corpus

per_path = 'Path to save the Scaler'

w2vmodel = KeyedVectors.load_word2vec_format(w2vmodel_path_in_c, binary=True)
print('Word2Vec Model loaded')

def save_to_disk(path_to_disk, obj, overwrite=False):
    """ Pickle an object to disk """
    dirname = os.path.dirname(path_to_disk)
    if not os.path.exists(dirname):
        #dirname = os.path.join(os.getcwd(), 
        #print('dirname - ',dirname)
        raise ValueError("Path " + dirname + " does not exist")

    if not overwrite and os.path.exists(path_to_disk):
        
        raise ValueError("File " + path_to_disk + "already exists")

    pickle.dump(obj, open(path_to_disk, 'wb'))

def get_all_words(afile):
    with open(afile) as afl:
        for aline in afl:
            aline = aline.strip() 
            if aline == '':
                continue 
            else:
                #print('aline = ',aline) 
                for w in  [w.lower() for w in word_tokenize(aline) if w not in string.punctuation]:
                    yield w     
    #return w.lower() for w in word_tokenize(self.text) if w not in string.punctuation


def fit_scaler(data_dir, word2vec_model, batch_size=1024, persist_to_path=None):
    """ Get all the word2vec vectors in a 2D matrix and fit the scaler on it.
     This scaler can be used afterwards for normalizing feature matrices. """
    if type(word2vec_model) == str:
        word2vec_model = Word2Vec.load(word2vec_model)
        
        
    #pass
    #doc_generator = get_documents(data_dir)
    
    
    scaler = StandardScaler(copy=False)
    vectors = []
    ithword = 0 
    fitnow = False 
    for word in get_all_words(data_dir):
        try:
            if word in word2vec_model:
                vectors.append(word2vec_model[word]) 
        except:
            print('aline invalid = word ',word)
            continue 
        ithword = ithword + 1
        if ithword == 10000:
            ithword = 0 
            matrix = np.array(vectors)
            print("Fitted to {} vectors".format(matrix.shape[0]))
            scaler.partial_fit(matrix)
            
    matrix = np.array(vectors)
    print("Fitted to {} vectors".format(matrix.shape[0]))
    scaler.partial_fit(matrix)                
    print("Done all vectors fitting")
        #matrix = np.array(vectors)
        #print("Fitted to {} vectors".format(matrix.shape[0]))
    
    #scaler.partial_fit(matrix)

    #no_more_samples = False
    #while not no_more_samples:
    #        #batch = []
    #        #for i in range(batch_size):
    #        #    try:
    #        #        batch.append(six.next(doc_generator))
    #        #    except StopIteration:
    #        #        no_more_samples = True
    #        #        break
    #
    #        #vectors = []
    #        #for doc in batch:
    #        #    for word in doc.get_all_words():
    #        #        if word in word2vec_model:
    #        #            vectors.append(word2vec_model[word])
    #
    #    #matrix = np.array(vectors)
    #    #print("Fitted to {} vectors".format(matrix.shape[0]))
    #        scaler.partial_fit(matrix)

    if persist_to_path:
        save_to_disk(persist_to_path, scaler)

    return scaler


s = fit_scaler(data_dir=data_dir, word2vec_model=w2vmodel, batch_size=1024, persist_to_path=per_path)
    
print('Program Ended')
te = time.time()
print('%2.2f ms' % ((te - ts) * 1000))
print('Ended by ',te)

now = datetime.datetime.now()
print ("Ended date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))

print("Program completed")
