from os import walk
import sys
import nltk
import string
from nltk.corpus import stopwords
init_path = "../data/text/" + str(sys.argv[1]) + "/"
dest_path = "../data/stop/" + str(sys.argv[1]) + "/"
for (dirpath, dirnames, filenames) in walk(init_path):
	for data_file in filenames:
		print(data_file)
		with open(init_path+data_file,'r',encoding='utf-8') as inFile,open(dest_path+data_file,'w',encoding='utf-8') as outFile:
			for line in inFile.readlines():
				print(" ".join([word for word in line.lower().translate(str.maketrans('', '', string.punctuation)).split() if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)