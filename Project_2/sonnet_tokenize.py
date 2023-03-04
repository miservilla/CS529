import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math
from sklearn.feature_extraction.text import CountVectorizer
from bidict import bidict

texts = [
    'Sleep, sleep, beauty bright,',
    'Dreaming in the joys of night;',
    'Sleep, sleep; in thy sleep',
    'Little sorrows sit and weep.',
    'Secret joys and secret smiles,',
    'Little pretty infant wiles.',
    'I have been one acquainted with the night.',
    'I have walked out in rain—and back in rain.',
    'I have outwalked the furthest city light.',
    'I have looked down the saddest city lane.',
    'I have passed by the watchman on his beat',
    'And dropped my eyes, unwilling to explain.',
    'Do not stand at my grave and weep:',
    'I am not there; I do not sleep.',
    'I am a thousand winds that blow,',
    'I am the diamond glints on snow,',
    'I am the sun on ripened grain,',
    'I am the gentle autumn rain.'
]

df = pd.DataFrame({'author': ['Frodo','Frodo','Frodo','Frodo','Frodo','Frodo','Bilbo','Bilbo','Bilbo','Bilbo','Bilbo','Bilbo','Samwise','Samwise','Samwise','Samwise','Samwise','Samwise'], 'text':texts})

cv = CountVectorizer(lowercase = True, token_pattern = '[a-zA-Z]+')
cv_matrix = cv.fit_transform(df['text'])
df_dtm = pd.DataFrame(cv_matrix.toarray(), columns=cv.get_feature_names_out())
df_dtm['author'] = ['Frodo','Frodo','Frodo','Frodo','Frodo','Frodo','Bilbo','Bilbo','Bilbo','Bilbo','Bilbo','Bilbo','Samwise','Samwise','Samwise','Samwise','Samwise','Samwise']
char_names = pd.unique(df_dtm["author"])
label_map = {}
for i, char in enumerate(char_names):
    label_map[char] = i
label_map = bidict(label_map)
print(label_map["Frodo"])
print(label_map.inverse[0])
df_dtm["class_id"] = df_dtm["author"].map(label_map) + 1
print(df_dtm[['author', 'class_id']][:70])
df_dtm.drop(['author'],axis=1, inplace=True)
df_dtm.index = np.arange(1, len(df_dtm) + 1)
df_dtm.to_csv('Project_2/lor.csv', sep=',', header=False)