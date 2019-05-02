from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import os
import warnings
from string import punctuation
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")


dataset = 'FLOW016'
op = 'Op'


# def train_word2vec(data_dir, n_feature, model_path, sg):
def train_word2vec(data_dirs, n_feature, model_path, sg):
    # make dictionary
    print("Making document list...")
    documents = []
    
    # train_documents = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    train_documents = []
    for data_dir in data_dirs:
        for f in os.listdir(data_dir):
            train_documents.append(os.path.join(data_dir, f))

    print(train_documents)

    for doc in train_documents:
        doc_content = []
        with open(doc) as d:
            for line in d:
                words = line.split()
                for word in words:
                    doc_content.append(word)
            documents.append(doc_content)

    #print(len(documents))
    #print(documents)
    # clean the doc
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    # print(sentence)
    documents = [[w.translate(table) for w in doc_content] for doc_content in documents]
    # remove remaining tokens that are not alphabetic
    documents = [[word for word in doc_content if word.isalpha()] for doc_content in documents]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    documents = [[w for w in doc_content if not w in stop_words] for doc_content in documents]
    # filter out short tokens
    documents = [[word for word in doc_content if len(word) > 1] for doc_content in documents]
    #print(documents)


    print("Training Word2Vec model...")
    model_word2vec = Word2Vec(
        documents, size=n_feature, window=5, min_count=5, sg=sg)
    model_word2vec.train(documents, total_examples=len(documents), epochs=10)
    model_word2vec.save(model_path)
    print("Training complete!!!")


def extract_word2vec(model_path, word, n_feature=100):
    trained_model = Word2Vec.load(model_path)
    # words = list(trained_model.wv.vocab)
    # print words

    # extract train matrix to csv
    #print("Extracting Word2Vec of word "+word)
    try:
        word_vec = trained_model.wv[word].reshape(n_feature)
        return word_vec
    except:
        word_vec = np.zeros((n_feature))
        return word_vec


if __name__ == "__main__":
    # data_dir = "/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq/"+dataset
    data_dir = ['/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq/'+dataset+'/'+op+'/'+dataset+'_Seq_'+op +
                 '_train',

                 '/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq/'+dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_test']

    # method = "Word2vec_CBoW"
    method = "Word2vec_skip_n_gram"
    n_feature = 100

    delimiter = ","

    if method == "Word2vec_CBoW":
        sg = 0
        model_path = "data/" + method + "_" + str(n_feature) + "__"+dataset+"_"+op+".bin"
        train_word2vec(data_dir, n_feature, model_path, sg)

    if method == "Word2vec_skip_n_gram":
        sg = 1
        model_path = "data/" + method + "_" + str(n_feature) + "__"+dataset+"_"+op+".bin"
        train_word2vec(data_dir, n_feature, model_path, sg)

    # model_path = "data/" + method + "_" + str(n_feature) + ".bin"
    # vec = extract_word2vec(model_path, 'mov')
    # print(vec)
