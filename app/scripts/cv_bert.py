#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TRANSFORMERS_CACHE'] = '/home/david.yang1/.cache/huggingface/'
os.environ['HF_HOME'] = '/home/david.yang1/.cache/huggingface/'


# In[2]:


from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import evaluate
from huggingface_hub import login
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# In[3]:


# login()


# # Split data into K-folds

# In[4]:


# Load dataset consisting of 309 papers that talk about viral variants and 309 papers that do not
df = pd.read_csv('/home/david.yang1/autolit/viriation/script/bert_dataset.csv')

# Check class distribution
print(df['label'].value_counts())

# Balance classes if needed
df = df.groupby('label').sample(n=min(df['label'].value_counts()), random_state=42)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42)
df = df[["text", "label"]]


# In[5]:


# Create 5 folds for cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)


# # Tokenize training data

# In[6]:


# Tokenization with maximum of 512 tokens (padding and truncation)
def tokenize_function(df):
    return tokenizer(
        df['text'],
        padding="longest",
        truncation=True,
        max_length = 512
    )


# In[7]:


# Create DatasetDict for train and validation split
def create_dataset(tds, vds, tokenize_function):
    # Apply the tokenizer to the datasets
    tds = tds.map(tokenize_function, batched=True)
    vds = vds.map(tokenize_function, batched=True)
    
    # Set the format of the datasets to include only the required columns
    tds = tds.rename_column('__index_level_0__', 'index').remove_columns(['text', 'index'])
    vds = vds.rename_column('__index_level_0__', 'index').remove_columns(['text', 'index'])
    
    # Define DatasetDict
    ds = DatasetDict({
        "train": tds,
        "validation": vds
    })

    return ds


# In[8]:


# Compute accuracy, f1, precision, and recall
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# In[9]:


# # BERT models that we will be using
# bert_models = ["NeuML/pubmedbert-base-embeddings","digitalepidemiologylab/covid-twitter-bert", "dmis-lab/biobert-v1.1"]
bert_models = ["digitalepidemiologylab/covid-twitter-bert"]


# In[10]:


# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = "steps",
    eval_steps=500,
    num_train_epochs=3,    # number of training epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.01,
    weight_decay=0.01,
    logging_dir='./logs',
)


# In[11]:


# Variables for saving results
train_results = []
# validation_results = pd.DataFrame()


# In[ ]:


# For each BERT model, perform cross-validation with 5-folds
for bert in bert_models:
    # Specify tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert)

    # Specify sequence classification model
    def model_init():
            return AutoModelForSequenceClassification.from_pretrained(bert, num_labels=2)
    
    # Variable to track CV fold
    trial = 0

    # Dataframe to save model results
    # validation_results = pd.DataFrame()
    
    # Cross-validation
    for train_indices, valid_indices in kfold.split(df, df['label']):
        # Dataframe to save model results
        validation_results = pd.DataFrame()
        
        trial += 1
        df_train = df.iloc[train_indices]
        df_val = df.iloc[valid_indices]
    
        tds = Dataset.from_pandas(df_train)
        vds = Dataset.from_pandas(df_val)
    
        ds = create_dataset(tds, vds, tokenize_function)
    
        # Create the Trainer and start training
        trainer = Trainer(
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            model_init=model_init,
            compute_metrics=compute_metrics,
        )
    
        train = trainer.train()
        train_results.append(train.metrics["train_loss"])
    
        # Evaluate the model
        eval = trainer.evaluate(ds["validation"])

        # Save model metrics
        eval_df = pd.DataFrame(eval, index=[trial,])
        eval_df["model"] = bert
        eval_df["fold"] = str(trial)
        eval_df["training_loss"] = train.metrics["train_loss"]

        validation_results = pd.concat([validation_results, eval_df])

        # Save cross-validation results for current model
        file_name = bert + str(trial) + ".csv"
        validation_results.to_csv(file_name)

    # Save model checkpoint
    path = "./" + bert
    trainer.save_model(path)

    # Save cross-validation results for current model
    # file_name = bert + ".csv"
    # validation_results.to_csv(file_name)


# In[ ]:


print(np.mean(train_results))


# In[ ]:


print(validation_results)


# In[ ]:


# validation_results.to_csv('BERT_model_cv_results.csv')

