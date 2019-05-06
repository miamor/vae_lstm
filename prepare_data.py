from string import punctuation
import os
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import argparse

# dataset = 'FLOW016'
# op = 'NoOp'

def file_to_folder(dDir):
    dataset = args.dataset.split('_')[0]
    op = args.dataset.split('_')[1]

    # list dataset file
    for file in os.listdir(dDir):
        dataset = file.split('_')[0]
        op = file.split('_')[2]
        op = 'NoOp' if op != 'Op' else op
        file_path = os.path.join(dDir, file)
        # folder_path = file_path.split('.txt')[0]
        lastname = file.split('.txt')[0].split('_')[-1]
        folder_path = dDir+'/'+dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_'+lastname
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # read file
        k = 0
        f = open(file_path, "r")
        for line in f:
            label = line.split('		')[0]
            content = line.split('		')[1].replace('<sssss>', ' . ')

            fw = open(folder_path+'/'+label+'_'+str(k)+'.txt', "w")
            fw.write(content)
            fw.close()

            # print(label)
            k += 1
        f.close()


def stat(dirs):
    n_sentences = []
    n_words = []

    # walk through all files in the folder
    for directory in dirs:
        for filename in sorted(os.listdir(directory)):
            file_path = directory + '/' + filename
            # get sentences of this file
            f = open(file_path, "r")
            content = f.read()
            sentences = content.split(' . ')
            n_sentences.append(len(sentences))
            for sentence in sentences:
                # n_words.append(sentence.split(' '))
                words = sentence.split(' ')
                n_words.append(len(words))
            f.close()

    n_sentences_ar = np.array(n_sentences)
    n_words_ar = np.array(n_words)
    mean_sentences = np.mean(n_sentences_ar)
    mean_words = np.mean(n_words_ar)

    print('n_sentences ', n_sentences_ar)
    print('n_words ', n_words_ar)
    print('mean_sentences ', mean_sentences)
    print('n_words ', mean_words)


    plt.figure()
    objects = range(len(n_sentences))
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, n_sentences, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Sentences')


    plt.figure()
    objects = range(len(n_words))
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, n_words, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Words')

    plt.show()




if __name__ == '__main__':
    dDir = '/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='FLOW016_Op')

    args = parser.parse_args()

    dataset = args.dataset.split('_')[0]
    op = args.dataset.split('_')[1]


    # file_to_folder(dDir, args)

    dirs = [os.path.join(dDir, dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_train'),
            os.path.join(dDir, dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_test')]
    # dirs = [os.path.join(dDir, dataset+'/'+op+'/'+dataset+'_Seq_'+op+'_test')]
    stat(dirs)

