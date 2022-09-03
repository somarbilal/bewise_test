# -*- coding: utf-8 -*-
"""Somar Bilal Test.ipynb

**Importing the Dependencies**
"""

import numpy as np
import pandas as pd

"""**Use the Saved the Model**"""

import pickle

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

# loading the dataset to a pandas DataFrame
calls_dataset_classification = pd.read_csv('test_data.csv')

calls_dataset_test = calls_dataset_classification['text']
calls_dataset_test

results=[]
for row in calls_dataset_test:
  
  X_new=[row]
  X_new_input=loaded_vectorizer.transform(X_new)
  prediction = loaded_model.predict(X_new_input)
  if (prediction[0]==0):
    results.append('The phrase is Greetings')
  elif(prediction[0]==1):
    results.append('The phrase is Farewell')
  elif(prediction[0]==2):
    results.append('The phrase is Introducing')
  else:
    results.append('The Phrase is Normal Speache')

df1= pd.DataFrame(results)
calls_dataset_result=pd.concat([calls_dataset_classification,df1],axis=1)

calls_dataset_result.columns

calls_dataset_result.rename(columns = {0:'Kind of Phrase'}, inplace = True)
calls_dataset_result

"""# SpaCy"""

import spacy
from spacy import displacy
from collections import Counter

!python -m spacy download ru_core_news_lg

import ru_core_news_lg
rnlp = ru_core_news_lg.load()

calls_dataset_SpaCy = calls_dataset_classification['text']
calls_dataset_SpaCy

Managers=[]
for idx,element in enumerate(calls_dataset_SpaCy):
  doc_frame = rnlp(element)
  for X in doc_frame.ents:
    if X.label_=='PER':
      Managers.append((idx, X.text))

df2= pd.DataFrame(Managers)
df2.rename(columns = {0:"Row's number in dataset",1:"Manager's Name"}, inplace = True)
df2

Organization=[]
for idx,element in enumerate(calls_dataset_SpaCy):
  doc_frame = rnlp(element)
  for X in doc_frame.ents:
    if X.label_=='ORG':
      Organization.append((idx, X.text))

df3= pd.DataFrame(Organization)
df3.rename(columns = {0:"Row's number in dataset",1:"Organization's Name"}, inplace = True)
df3
