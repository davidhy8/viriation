#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize


# In[2]:


# Load data and create example text
df = pd.read_csv('bn_pub_dataset_3.csv')

# Check class distribution
print(df['label'].value_counts())

# Balance classes if needed
df = df.groupby('label').sample(n=min(df['label'].value_counts()), random_state=42)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42)
df = df[["text", "label"]]

subsample = df.sample(n=15, random_state=42)


# In[3]:


def get_label(df, column):
    label_input=[]
    for i in df[column]:
        print(i)
        
        while True:
            label = int(input('Does this relate to viral variants? (0 = No, 1 = Yes)'))
            if label == 0 or label == 1:
                break
            
        label_input.append(label)
        print(" ")
    df['label'] = label_input
    return df


# In[4]:


# Obtain list of sentences 
text = subsample["text"]

# Testing with simple test case
example = ["hisi is cool. But I think I am cooler.", "www.wikipedia.com contains a lot of interesting information. John should definitely visit it."]
text = pd.DataFrame(example, columns=['text'])
text = text["text"]

text_index = 0

# Iterate through each paper
for i in text:
    # Split text into sentences
    sentences = sent_tokenize(i)

    # Create dataframe containing all sentences in text
    sentences_df = pd.DataFrame(sentences, columns=['text'])

    # Manually label each sentence
    sentences_df = get_label(sentences_df, 'text')

    # Save dataframe for current paper
    filename = str(text_index) + "_sentence_labelled.csv"
    sentences_df.to_csv(filename)

    # Point to next paper
    text_index += 1

    print("Moving on to paper #" + str(text_index) + "\n")

print("Finished labelling " + str(text_index) + " papers.")


# In[ ]:




