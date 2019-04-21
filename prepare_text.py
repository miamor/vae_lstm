from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from gensim.models import Word2Vec
import numpy as np
from string import punctuation
from os import listdir
from feature_extraction import extract_word2vec


def load_doc(filename, vocab):
    ''' load and turn a doc into clean tokens '''
    # open the file as read only
    f = open(filename, 'r')
    # read all text
    text = (f.read()).strip(' ')
    # seperate to sentences
    sentences = text.split(' . ')
    # print(text)
    # print(sentences)
    # close the file
    f.close()

    return sentences


def pad_doc(doc, max_sentence):
    # pad each document to a max length of (max_sentence) sentences
    if len(doc) < max_sentence:
        doc_ = doc
        for i in range(max_sentence - len(doc)):
            doc_.append('')
    else:
        k = 0
        doc_ = []
        for sentence in doc:
            k += 1
            if k <= max_sentence:
                doc_.append(sentence)
    return doc_

def pad_arr(arr, max_len):
    # pad each document to a max length of (max_len) sentences
    # or each sentence to a max length of (max_len) words
    if len(arr) < max_len:
        arr_ = arr
        for i in range(max_len - len(arr)):
            arr_.append('')
    else:
        k = 0
        arr_ = []
        for el in arr:
            k += 1
            if k <= max_len:
                arr_.append(el)
    return arr_


def clean_doc(doc, vocab):
    # clean the doc
    tokens_doc = []
    for sentence in doc:
        # split into tokens by white space
        tokens = sentence.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        # print(sentence)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        # print(tokens)

        tokens = [w for w in tokens if w in vocab]
        tokens_doc.append(' '.join(tokens))
    return tokens_doc


# encode each word in each sentence into a vector
def word2vec_doc(doc, max_word):
    model_path = 'data/Word2vec_skip_n_gram_100.bin'
    #print(doc)
    encoded_doc = []
    #print('sentence = ', len(doc))
    for sentence in doc:
        words = sentence.split()
        words = pad_arr(words, max_len=max_word)
        #print(words)
        
        encoded_sentence = [extract_word2vec(model_path, word) for word in words]
        encoded_doc.append(encoded_sentence)
    #print(encoded_doc)

    return encoded_doc


# load all docs in a directory
def process_docs(directory, vocab):
    X = []
    labels = []
    # walk through all files in the folder
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc, return an array of sentences
        doc = load_doc(path, vocab)
        # pad the doc to a maximum number of sentences
        doc = pad_arr(doc, max_len=10)
        # clean doc
        doc = clean_doc(doc, vocab)
        # encode doc
        doc = word2vec_doc(doc, max_word=50)

        # add to list
        X.append(doc)

        labels.append('1' if filename.split('_')[0] == 'm' else '0')

    print(labels)

    X = np.array(X)
    labels = to_categorical(labels)

    return X, labels
    # return lines


if __name__ == "__main__":
    # load the vocabulary
    vocab_filename = 'data/vocab.txt'
    f = open(vocab_filename, 'r')
    vocab = f.read()
    f.close()
    vocab = vocab.split()
    vocab = set(vocab)

    # load dataset
    Xtrain, Ytrain = process_docs('data/asm_for_nlp_1', vocab)
    np.save('data/x_train_.npy', Xtrain)
    np.save('data/y_train_.npy', Ytrain)

    # print(Xtrain[0])
    # print(Ytrain)
    print(Xtrain[0].shape)
    print(Xtrain.shape)
    print(Ytrain.shape)

    Xtest, Ytest = process_docs('data/test', vocab)
    np.save('data/x_test_.npy', Xtest)
    np.save('data/y_test_.npy', Ytest)

    '''
    # create the tokenizer
    tokenizer = Tokenizer(num_words=100)
    # fit the tokenizer on the documents
    tokenizer.fit_on_sequences(docs_train)



    # encode training data set
    Xtrain = tokenizer.sequences_to_matrix(docs_train, mode='freq')
    print(Xtrain)
    print(Xtrain.shape)

    # write train data to file
    np.save('data/x_train_.npy', Xtrain)


    # encode test set
    Xtest = tokenizer.texts_to_matrix(docs_test, mode='freq')
    print(Xtest)
    print(Xtest.shape)

    # write test data to file
    np.save('data/x_test.npy', Xtest)
    '''
