#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
df_train = pd.read_csv('../data/data_train.csv')
df_test = pd.read_csv('../data/data_test.csv')
df_train.head()


# In[70]:


category_id_df = df_train[['occupation', 'category']].drop_duplicates().sort_values('category')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category', 'occupation']].values)
df_train.head()


# In[71]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df_train.groupby('occupation').article.count().plot.bar(ylim=0)
plt.show()


# In[72]:


fig = plt.figure(figsize=(8,6))
df_test.groupby('occupation').article.count().plot.bar(ylim=0)
plt.show()


# In[73]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df_train.article.values.astype('U')).toarray()
labels = df.category
features.shape


# In[74]:


from sklearn.feature_selection import chi2
import numpy as np
N = 2
for occupation, category in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(occupation))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df_train['article'], df_train['occupation'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.values.astype('U'))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[76]:


data = [df_test.article[50]]
print("Name: ",df_test.name[50])
print("Occupation: ",clf.predict(count_vect.transform(data)))
data = [df_test.article[100]]
print("Name: ",df_test.name[100])
print("Occupation: ",clf.predict(count_vect.transform(data)))
data = [df_test.article[1020]]
print("Name: ",df_test.name[1020])
print("Occupation: ",clf.predict(count_vect.transform(data)))


# In[77]:


data = """Ted Key (born Theodore Keyser; August 25, 1912 – May 3, 2008),
was an United States American cartoonist and writer.  He is best known as the creator of the cartoon panel Hazel (comic strip) Hazel, which was later the basis for a Hazel (TV series) television series of the same name, and also the creator of Mister Peabody Peabodys Improbable History. 

==College to cartoons==
Born in Fresno, California, Key was the son of Latvian immigrant Simon Keyser, who had changed his name from Katseff to Keyser, and then to "Key" during World War I.
"""
print(data)
print("Occupation: ",clf.predict(count_vect.transform([data])))


# In[ ]:




