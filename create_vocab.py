from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import argparse

# dataset = 'FLOW016'
# op = 'NoOp'

''' 
Define a Vocabulary 
'''
# load doc into memory


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def clean_doc(doc):
    ''' turn a doc into clean tokens '''
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    ''' load doc and add to vocab '''
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


def process_docs(directory, vocab):
    ''' load all docs in a directory '''
    # walk through all files in the folder
    for filename in listdir(directory):
        # print(filename)
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


def save_list(lines, filename):
    ''' save list to file '''
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()


def create_vocab(input_dirs, out_path):
    # define vocab
    vocab = Counter()

    for input_dir in input_dirs:
        # add all docs to vocab
        process_docs(input_dir, vocab)
    
    # print the size of the vocab
    print(len(vocab))
    # print the top words in the vocab
    print(vocab.most_common(50))

    # keep tokens with a min occurrence
    min_occurane = 1
    tokens = [k for k, c in vocab.items() if c >= min_occurane]
    print(len(tokens))

    # save tokens to a vocabulary file
    save_list(tokens, out_path)


def main(args):
    dataset = args.dataset.split('_')[0]
    op = args.dataset.split('_')[1]

    data_dirs = ['/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq/'+dataset+'/'+op+'/'+dataset+'_Seq_'+op +
                 '_train',

                 '/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq/'+dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_test']

    out_path = 'data/vocab_'+dataset+'_'+op+'.txt'
    create_vocab(data_dirs, out_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='FLOW016_Op')

    args = parser.parse_args()

    main(args)