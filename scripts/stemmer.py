
from os import walk
import sys
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
init_path = "../data/stop/" + "cartoonists" + "/"
dest_path = "../data/stem/" + "cartoonists" + "/"
ps = PorterStemmer()
for (dirpath, dirnames, filenames) in walk(init_path):
	for data_file in filenames:
		print(data_file)
		with open(init_path+data_file,'r',encoding='utf-8') as inFile,open(dest_path+data_file,'w',encoding='utf-8') as outFile:
			for line in inFile.readlines():
				words = word_tokenize(line)
				for word in words:
					outFile.write(ps.stem(word)+" ")