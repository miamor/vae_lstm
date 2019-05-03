from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from gensim.models import Word2Vec
import numpy as np
from string import punctuation
import os
from word2vec import extract_word2vec


dataset = 'FLOW016'
op = 'Op'

def load_doc(filename, vocab):
    ''' load and turn a doc into clean tokens '''
    # open the file as read only
    f = open(filename, 'r')
    # read all text
    text = (f.read()).strip(' ')
    # seperate to sentences
    sentences = text.split(' . ')
    # close the file
    f.close()

    return sentences


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

        tokens = [w for w in tokens if w in vocab]
        tokens_doc.append(' '.join(tokens))
    return tokens_doc


# encode each word in each sentence into a vector
def word2vec_doc(doc, max_word):
    model_path = 'data/Word2vec_skip_n_gram_100__'+dataset+'_'+op+'.bin'
    encoded_doc = []
    for sentence in doc:
        words = sentence.split()
        words = pad_arr(words, max_len=max_word)

        encoded_sentence = [extract_word2vec(
            model_path, word) for word in words]
        encoded_doc.append(encoded_sentence)

    return encoded_doc


# load all docs in a directory
def process_docs(directory, vocab):
    X = []
    labels = []
    # walk through all files in the folder
    for filename in sorted(os.listdir(directory)):
        # create the full path of the file to open
        path = directory + '/' + filename
        print('process_docs '+path)

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

        # labels.append('1' if filename.split('_')[0] == 'm' else '0')
        labels.append(int(filename.split('_')[0]))

    X = np.array(X)
    # Y = np.array(labels)
    Y = labels
    Y_onehot = to_categorical(labels)

    return X, Y_onehot, Y


if __name__ == "__main__":
    # load the vocabulary
    vocab_filename = 'data/vocab_'+dataset+'_'+op+'.txt'
    f = open(vocab_filename, 'r')
    vocab = f.read()
    f.close()
    vocab = vocab.split()
    vocab = set(vocab)

    data_save_dir = 'data/'+dataset
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # load dataset
    Xtrain, Ytrain_onehot, Ytrain = process_docs('/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq/'+dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_train', vocab)
    np.save(data_save_dir+'/x_train_'+dataset+'_'+op+'.npy', Xtrain)
    np.save(data_save_dir+'/y_train_'+dataset+'_'+op+'.npy', Ytrain_onehot)
    np.save(data_save_dir+'/y_train_'+dataset+'_'+op+'__.npy', Ytrain)

    Xval, Yval_onehot, Yval = process_docs('/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq/'+dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_test', vocab)
    np.save(data_save_dir+'/x_val_'+dataset+'_'+op+'.npy', Xval)
    np.save(data_save_dir+'/y_val_'+dataset+'_'+op+'.npy', Yval_onehot)
    np.save(data_save_dir+'/y_val_'+dataset+'_'+op+'__.npy', Yval)

    # print(Xtrain[0].shape)
    print(Xtrain.shape)
    print(Ytrain_onehot.shape)
    print(Ytrain)

    # Xtest, Ytest = process_docs('data/test', vocab)
    # np.save('data/x_test_'+dataset+'.npy', Xtest)
    # np.save('data/y_test_'+dataset+'.npy', Ytest)
