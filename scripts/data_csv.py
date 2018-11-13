#!/usr/bin/env python
# coding: utf-8

# In[12]:


from os import walk
import sys

dirs = ["aviators", "comedians", "cartoonists"]

with open("../data/data_train.csv", "w+", encoding="utf-8") as data_file:
    data_file.write("occupation,category,name,article\n")
    for dir in dirs:
        init_path = "../data/stem/" + dir + "/"
        for (dirpath, dirnames, filenames) in walk(init_path):
            filenames = filenames[:150]
            for file in filenames:
                with open(init_path + file, encoding='utf-8') as f:
                    data = " ".join(f.readlines())
                    data = data.replace("\n", " ")
                    data_file.write(dir+","+str(dirs.index(dir))+","+file[:-4]+","+data.strip()+"\n")
                    print(dir+": "+file)

with open("../data/data_test.csv", "w+", encoding="utf-8") as data_file:
    data_file.write("occupation,category,name,article\n")
    for dir in dirs:
        init_path = "../data/stem/" + dir + "/"
        for (dirpath, dirnames, filenames) in walk(init_path):
            filenames = filenames[150:]
            for file in filenames:
                with open(init_path + file, encoding='utf-8') as f:
                    data = " ".join(f.readlines())
                    data = data.replace("\n", " ")
                    data_file.write(dir+","+str(dirs.index(dir))+","+file[:-4]+","+data.strip()+"\n")
                    print(dir+": "+file)


# In[ ]:




