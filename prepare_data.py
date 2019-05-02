from string import punctuation
import os
from collections import Counter
from nltk.corpus import stopwords

if __name__ == '__main__':
    dDir = '/media/tunguyen/Others/Dataset/assembly_data/CodeChef_Data_ASM_Seq'
    # list dataset file
    for file in os.listdir(dDir):
        file_path = os.path.join(dDir, file)
        folder_path = file_path.split('.txt')[0]
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