from nltk import tokenize
from keras import initializers
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import io
import os
import sys
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
import re
from collections import defaultdict
# import cPickle
import pickle
import pickle  # to save tokenizer
import pandas as pd
import numpy as np
import warnings
from importlib import reload
warnings.filterwarnings('ignore')


dataset = 'Vietnamese'
data_path = '/media/tunguyen/Others/Dataset/assembly_data/Vietnamese'
# data_path = '.'


def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


# set_keras_backend("theano")
set_keras_backend("tensorflow")

seed = 7
np.random.seed(seed)

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
N_CLASSES = 10
content_encoding = "utf-16"
stopwords_encoding = "utf-8"
dictionary_encoding = "utf-8"

data_train = pd.read_csv(data_path+'/data/data_10cat_train_full.csv',
                         escapechar='\\', encoding=content_encoding)
# data_train = pd.read_csv('./data/data_10cat_train_small.csv', escapechar='\\', encoding =content_encoding)
print(data_train.shape)


def clear_text(text):
    text = text.lower()
    text = text.replace("\n", ". ")
    text = text.replace("..", ".")
    text = text.replace("...", ".")
    text = text.replace(". .", ".")
    text = text.replace("?", ".")
    text = text.replace("!", ".")
    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace("/", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace(":", " ")
    text = " ".join(text.split())
    return text


def word_segmentation(text):
    text = ViTokenizer.tokenize(text)
    if not text.startswith(" "):
        text = " " + text
    return text


def remove_stop_word(text):
    new_text = text
    stopwords_path = data_path+"/stopwords/stopwords-vi_full.txt"
    with io.open(stopwords_path, "r", encoding=stopwords_encoding) as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    for w in stopwords:
        word = " " + w + " "
        word = word.lower()
        if text.find(word) != -1:
            new_text = text.replace(word, " ")
            text = new_text
    return new_text


contents = []
labels = []
corpus = []

for idx in range(data_train.content.shape[0]):
    text = data_train.content[idx]
    text = clear_text(text)
    text = word_segmentation(text)
    text = remove_stop_word(text)
    corpus.append(text)
    sentences = tokenize.sent_tokenize(text)
    contents.append(sentences)

    labels.append(data_train.ID[idx])
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(corpus)

# save tokenizer
with open('data/'+dataset+'/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = np.zeros((len(corpus), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(contents):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
nb_test_samples = int(TEST_SPLIT * data.shape[0]) + nb_validation_samples

x_val = data[:nb_validation_samples]
x_test = data[nb_validation_samples:nb_test_samples]
x_train = data[nb_test_samples:]
y_val = labels[:nb_validation_samples]
y_test = labels[nb_validation_samples:nb_test_samples]
y_train = labels[nb_test_samples:]

print('Number of classes in traing, validation and test set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))
print(y_test.sum(axis=0))

GLOVE_DIR = data_path
embeddings_index = {}

f = io.open(os.path.join(GLOVE_DIR, 'word2vec/wiki.vi.vec'),
            'r', encoding=dictionary_encoding)
header = f.readline()
nr_row, nr_dim = header.split()  # nr_dim = EMBEDDING_DIM
for line in f:
    line = line.rstrip()
    pieces = line.rsplit(' ', int(nr_dim))
    ws = pieces[0].split()
    word = ""
    if len(ws) == 0:
        continue
    elif len(ws) == 1:
        word = pieces[0]
    else:
        word = "_".join(ws)
    vector = pieces[1:]
    vector_float = []
    for v in vector:
        v_float = float(v.replace(u'\N{MINUS SIGN}', '-'))
        vector_float.append(v_float)
    coefs = np.asarray(vector_float, dtype='float32')
    embeddings_index[word] = coefs

f.close()

print('Total %s word vectors.' % len(embeddings_index))



'''

# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(
            self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)
dropout = Dropout(0.25)(l_att_sent)
# preds = Dense(N_CLASSES, activation='softmax')(l_att_sent)
preds = Dense(N_CLASSES, activation='softmax')(dropout)
model = Model(review_input, preds)


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        with open("data/"+dataset+"/results/loss.txt", "a") as myfile:
            myfile.write("\n{}".format(loss))
        with open("data/"+dataset+"/results/acc.txt", "a") as myfile:
            myfile.write("\n{}".format(acc))
        print('\nTesting loss: {}, acc: {}'.format(loss, acc))


class F1TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y_true = self.test_data
        # y_pred = self.model.predict_classes(x, verbose=0)
        scores = self.model.predict(x, verbose=0)
        results = scores.argmax(axis=-1)
        y_pred = to_categorical(results)
        _test_f1 = f1_score(y_true, y_pred, average=None)
        _test_precision = precision_score(y_true, y_pred, average=None)
        _test_recall = recall_score(y_true, y_pred, average=None)
        with open("data/"+dataset+"/results/f1.txt", "a") as myfile:
            myfile.write("\n{}".format(_test_f1))
        with open("data/"+dataset+"/results/precision.txt", "a") as myfile:
            myfile.write("\n{}".format(_test_precision))
        with open("data/"+dataset+"/results/recall.txt", "a") as myfile:
            myfile.write("\n{}".format(_test_recall))
        print(" \ntest_f1: {} \ntest_precision: {} \ntest_recall: {}".format(
            _test_f1, _test_precision, _test_recall))


count_epoch = 1


class SaveModelCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        global count_epoch
        count_epoch = count_epoch + 1
        # self.model.save('HAN_'+str(count_epoch)+'.h5')
        self.model.save_weights('data/'+dataset+'/HAN_'+str(count_epoch)+'.h5', overwrite=True)


# create callbacks
ear = EarlyStopping(monitor='val_acc', patience=5)
# mcp = ModelCheckpoint('HANmodel.h5', monitor="val_acc", save_best_only=True, save_weights_only=False)
mcp = ModelCheckpoint('data/'+dataset+'/HANmodel_weights.h5', monitor="val_acc",
                      save_best_only=True, save_weights_only=True)  # only save weights
save_callback = SaveModelCallback()
test_acc = TestCallback((x_test, y_test))
test_f1 = F1TestCallback((x_test, y_test))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=50, callbacks=[test_acc, test_f1, mcp])

model_json = model.to_json()
with open('data/'+dataset+'/HANmodel.json', "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")


# # load json and create model
# json_file = open('HANmodel.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("HANmodel_weights.h5")
# print("Loaded model from disk")
'''