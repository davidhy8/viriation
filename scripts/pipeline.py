import os
os.environ['TRANSFORMERS_CACHE'] = '/home/david.yang1/.cache/huggingface/'
os.environ['HF_HOME'] = '/home/david.yang1/.cache/huggingface/'

from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.tokenize import sent_tokenize
from torch.nn import functional as F
import torch
import lightgbm as lgb
from pathlib import Path
import requests
import pickle
import spacy
import argparse
import re
from metapub.convert import pmid2doi
from scripts.data_processor import get_doi_file_name
from scripts.history import History


# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# Load BioBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")


# Tokenize the data
def tokenize_function(df):
    """
    Tokenize and process the text in df with truncation and padding

    Parameters:
        df (pd.DataFrame): Dataframe containing a column called text containing the text data to process
    """

    return tokenizer(
        df['text'],
        padding="longest",
        truncation=True,
        max_length = 512
    )


# Load training and validation dataset
def ds_preparation(df, val_count=0):
    """
    Prepare Dataset dictionary for model fine tuning

    Parameters:
        df (Pandas DataFrame): DataFrame containing a column for text and a column for label
        val_count (int): Number of validation data points to have

    Returns:
        (DatasetDict): Dataset dictionary containing training and validation data
    """

    # Balance classes if needed
    df = df.groupby('label').sample(n=min(df['label'].value_counts()), random_state=42)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)
    df = df[["text", "label"]]
    
    # Split dataset into test & train
    df_train = df[val_count:]
    df_val = df[:val_count]
    
    tds = Dataset.from_pandas(df_train)
    vds = Dataset.from_pandas(df_val)

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


# Compute confusion matrix for data
def compute_metrics(pred):
    """
    Compute the accuracy, F1, precision and recall score given a set of predictions

    Parameters:
        pred (NamedTuple): NamedTuple from trainer.predict() function with label_ids & predictions as keys

    Returns:
        (dict): Dictionary containing accuracy, F1, precision and recall score 
    """

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


# Fine tune model
def fine_tune_model(ds, model_init, train=False):
    """
    Configure and fine-tune (optionally) model with DatasetDict

    Parameters:
        ds (DatasetDict): A Hugging Face `DatasetDict` object that contains the training and validation datasets. Expected keys are `train` and `validation`.
        model_init (PreTrainedModel): A callable function that returns a pre-trained model (such as BERT, GPT-2, etc.) that is fine-tuned during the training process. This is used to initialize the model.

    Returns:
        (Trainer): An instance of the `transformers.Trainer` class, which encapsulates the training and evaluation logic.
    """

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

    # Create the Trainer and start training
    trainer = Trainer(
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    if train:
        trainer.train()

    if ds["validation"]:
        trainer.evaluate()

    return trainer


# Split text into <512 token chunks
def split_text_into_chunks(text, tokenizer, max_tokens=512, overlap_sentences=2):
    """
    Splits a long text into chunks of a maximum token length, ensuring that no chunk exceeds the specified token limit. The function also allows for sentence overlap between chunks for better context preservation in sequence-based tasks (e.g., in NLP models).

    Parameters:
        text (str): The input text that needs to be split into smaller chunks.
    
        tokenizer (PreTrainedTokenizer): A tokenizer (typically a Hugging Face BERT or similar transformer tokenizer) used to tokenize the text. It must have a `tokenize` method that splits sentences into tokens.

        max_tokens (int, optional): The maximum number of tokens allowed in each chunk. Default is 512, which is commonly used in transformer models like BERT.

        overlap_sentences (int, optional): The number of sentences to overlap between consecutive chunks. This helps maintain continuity between chunks. Default is 2 sentences.

    Returns:
        (list): A list of strings, where each string is a chunk of the input text that contains no more than `max_tokens` tokens.
    """

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize variables
    chunks = []
    current_chunk = []
    current_chunk_len = 0

    for i, sentence in enumerate(sentences):
        # Tokenize the sentence using BERT Tokenizer
        tokens = tokenizer.tokenize(sentence)
        token_count = len(tokens)

        # Finalize the current chunk if adding this sentence exceed token limit
        if current_chunk_len + token_count > max_tokens:
            text_chunk = " ".join(current_chunk)
            chunks.append(text_chunk)

            # Create the next chunk with overlap
            overlap_start = max(0, i-overlap_sentences)
            current_chunk = []
            for j in range(overlap_start, i):
                current_chunk.append(sentences[j])
            current_chunk_len = len(current_chunk)

        # Add the current sentence tokens to the chunk
        current_chunk.append(sentence)
        current_chunk_len += token_count

    # Add the last chunk if it has content
    if current_chunk:
        text_chunk = " ".join(current_chunk)
        chunks.append(text_chunk)

    return chunks


# Predict label of dataframe
def prediction_chunks(df, tokenizer, trainer):
    """
    Splits text data into manageable chunks, tokenizes them, and runs predictions using a Hugging Face Trainer.

    This function processes a DataFrame with text data by splitting long text entries into smaller chunks that fit within the token limit, 
    tokenizing each chunk, and using a pre-trained model to predict labels and probabilities for each chunk. The results are returned 
    along with the predicted outputs.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing a column `"text"` with text entries to be split into chunks for prediction. The DataFrame should also contain a `"doi"` column representing unique document identifiers for each row.
        tokenizer (PreTrainedTokenizer):  A Hugging Face tokenizer used for tokenizing the text chunks before prediction.

        trainer (Trainer): An instance of the Hugging Face `Trainer` class, which is used for running predictions on the processed text chunks.

    Returns:
        output (pd.DataFrame): A DataFrame containing the processed chunks, predictions, and associated probabilities. The DataFrame includes the following columns:
            - `"text"`: The text chunks.
            - `"position"`: The index position of each chunk within the original text.
            - `"doi"`: The document identifier (from the original input DataFrame).
            - `"prediction"`: The predicted labels for each chunk.
            - `"probability"`: The confidence score (softmax probability) of the predicted label.

        pred (Object): The raw prediction object returned by the `Trainer.predict()` function, containing the logits and other details from the prediction step.
    """

    output = pd.DataFrame()
    for i, text in enumerate(df["text"]):
        chunks = split_text_into_chunks(text, tokenizer)
        
        chunks_df = pd.DataFrame(chunks, columns=["text"])
        # chunks_df["label"] = df["label"][i]
        chunks_df["position"] = chunks_df.index
        chunks_df["doi"] = df.loc[i, "doi"]
        
        t = Dataset.from_pandas(chunks_df)
        t = t.map(tokenize_function, batched=True)
        ds_t = DatasetDict({
            "test": t
        })

        pred = trainer.predict(ds_t["test"])
        chunks_df["prediction"] = pred.predictions.argmax(-1)

        # convert logit score to torch array
        torch_logits = torch.from_numpy(pred.predictions)

        # get probabilities using softmax from logit score and convert it to numpy array
        probabilities_scores = F.softmax(torch_logits, dim = -1).numpy()

        chunks_df["probability"] = probabilities_scores.max(-1)

        # save into output
        output = pd.concat([output, chunks_df], ignore_index=True)
        
    return output, pred


# Validate model performance
def validate_model(trainer):
    """
    Loads labeled text chunks, tokenizes them, and evaluates the performance of a trained model using the Hugging Face Trainer.

    This function consolidates labeled text chunks from multiple CSV files, processes them into a tokenized dataset, and tests the 
    performance of the model using the provided Trainer. The function returns the computed evaluation metrics.

    Parameters:
        trainer (Trainer): An instance of the Hugging Face `Trainer` class, which is used to perform model evaluation on the processed test dataset.

    Returns:
        (dict): A dictionary containing evaluation metrics (e.g., accuracy, precision, recall, F1-score) computed by the `compute_metrics` function.

    """

    # Load prediction chunks
    pred_chunks_0 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/0_chunks_labelled.csv")
    pred_chunks_1 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/1_chunks_labelled.csv")
    pred_chunks_2 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/2_chunks_labelled.csv")
    pred_chunks_3 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/3_chunks_labelled.csv")
    pred_chunks_4 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/4_chunks_labelled.csv")
    pred_chunks_5 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/5_chunks_labelled.csv")
    pred_chunks_6 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/6_chunks_labelled.csv")
    pred_chunks_7 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/7_chunks_labelled.csv")
    pred_chunks_8 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/8_chunks_labelled.csv")
    pred_chunks_9 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/9_chunks_labelled.csv")
    pred_chunks_10 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/10_chunks_labelled.csv")
    pred_chunks_11 = pd.read_csv("../../data/pipeline_data/paper_flagging_data/11_chunks_labelled.csv")
    
    
    # Concatenate data
    df_test = pd.concat([pred_chunks_0, pred_chunks_1, pred_chunks_2, pred_chunks_3, pred_chunks_4, pred_chunks_5, pred_chunks_6, pred_chunks_7, 
                         pred_chunks_8, pred_chunks_9, pred_chunks_10, pred_chunks_11])
    
    # Load dataframe as dataset
    test = Dataset.from_pandas(df_test)
    
    # Tokenize test dataset
    test = test.map(tokenize_function, batched=True)
    
    # Set the format of the datasets to include only the required columns
    test = test.rename_column('__index_level_0__', 'index').remove_columns(['text', 'index'])
    
    # Define DatasetDict
    ds_test = DatasetDict({
        "test": test
    })
    
    # Test performance of the model on labeled chunks
    pred = trainer.predict(ds_test["test"])
    
    df_test["prediction"] = pred.predictions.argmax(-1)
    
    metrics = compute_metrics(pred)

    return metrics    


# Predict whether papers are relevant or not
def paper_prediction(data, bst, tokenizer, trainer):
    """
    Predicts relevant papers by processing text data into chunks, making predictions using a Hugging Face model, 
    and subsequently predicting paper relevance using a LightGBM model.

    This function first divides the text data into chunks, uses a pre-trained Hugging Face model to generate 
    predictions for each chunk, then groups the chunk-level predictions by paper DOI, and formats them into a 
    feature matrix for LightGBM. The LightGBM model predicts the relevance of each paper based on chunk-level 
    predictions, and relevant papers are returned.

    Parameters:
        data (pd.DataFrame): A pandas DataFrame containing paper text data with at least a "text" and "doi" column.

        bst (lightgbm.Booster): A pre-trained LightGBM Booster model used to classify papers as relevant or not based on chunk-level predictions.

        tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer object used to tokenize the text into chunks.
        
        trainer (Trainer): A Hugging Face Trainer object used to predict chunk-level labels using a pre-trained model.

    Returns:
        (pd.DataFrame): A DataFrame containing the flagged papers (those predicted as relevant) with their respective chunks and predictions.

    """

    chunked_data_df, chunked_data_pred = prediction_chunks(data, tokenizer, trainer)

    # Format lightGBM data 
    data = chunked_data_df
    print(data)
    
    # Format data for lightGBM
    grouped = data.groupby('doi')
    
    # Maximum number of data points in any group
    max_len = 133
    print(max(grouped.size()))

    # Create DataFrame with appropriate number of columns
    columns = [f'prediction_{i}' for i in range(max_len)]
    columns.append("doi")
    
    df = pd.DataFrame(columns=columns)
    
    for name, group in grouped:
        predictions = group["prediction"].values.astype(float)[:133] # truncate any papers with more than 133 chunks
        entry = np.pad(predictions, (0, max_len - len(predictions)), constant_values=np.nan)
        entry = np.append(entry, name)
        df.loc[name] = entry

    predictions = bst.predict(df.drop(columns="doi").astype(float), num_iteration=bst.best_iteration)
    pred = np.where(predictions < 0.5, 0, 1)
    df["prediction"] = pred.T

    # df['pmid'] = df['pmid'].astype(int)
    flagged_papers = df[df['prediction'] == 1]['doi']
    
    relevant_papers = flagged_papers.tolist()
    flagged = chunked_data_df[chunked_data_df["doi"].isin(relevant_papers)]

    return flagged


def query_plain(text, url="http://localhost:8888/plain"):
    """
    Sends a plain text query to a specified server URL and returns the response.

    Parameters:
        text (str): The plain text to be sent in the POST request body.

        url (str, optional): The URL of the server where the request is sent. Defaults to "http://localhost:8888/plain".

    Returns:
        (dict): The response from the server, parsed as a JSON object.
    """

    return requests.post(url, json={'text': text}).json()


def NER(flagged, port):
    """
    Performs Named Entity Recognition (NER) on flagged papers and saves the results to disk.

    This function processes flagged scientific papers by performing Named Entity Recognition (NER)
    on each paper's text using a specified API endpoint. The results are grouped by document identifier 
    (DOI), and the NER results for each paper are saved to a pickle file.

    Parameters:
        flagged (pd.DataFrame): A DataFrame containing flagged papers. It should include columns such as 'doi' (document identifier) and 'text' (the text content of each paper).

        port (str): The URL of the server endpoint where NER queries are sent. This is passed to the `query_plain` function for sending the text to be processed.
    """

    grouped_papers = flagged.groupby('doi')
    
    for name, group in grouped_papers:
        doi_name = get_doi_file_name(name)
        NER_list = list()
        for text in group["text"]:
            NER = query_plain(text, url = port)
            NER_list.append(NER)
        file_name = "../../data/pipeline_data/NER/" + str(doi_name) + "_paper.pkl"
        with open(file_name, 'wb') as f:
            pickle.dump(NER_list, f)


def load_mutations(path = "/home/david.yang1/autolit/viriation/data/pipeline_data/NER"):
    """
    Loads mutation data from pickle files and organizes it into a structured output.

    This function reads all the pickle files in the specified directory that contain Named Entity Recognition (NER)
    results for scientific papers. It extracts mutations and their contexts from the NER data, organizing them
    into a dictionary structure based on their corresponding Document Object Identifiers (DOIs).

    Parameters:
        path (str): The directory path where the NER pickle files are located. Default is "/home/david.yang1/autolit/viriation/data/pipeline_data/NER".

    Returns:
        output (dict): A nested dictionary where the keys are DOIs and the values are dictionaries containing mutations as keys and lists of corresponding context sentences as values. 

    """

    files = Path(path).glob("*.pkl")
    # Initialize output dictionary
    # output = defaultdict(lambda:{"mutation": [], "text": [], "doi": None})
    output = defaultdict(lambda: defaultdict(list))

    for file in files:

        with open(file, 'rb') as f:
            # Load NERs
            ner = pickle.load(f)

        # Obtain DOI
        basename = os.path.basename(file)
        match = re.search(r'(.+)_paper\.pkl$', basename)
        doi = match.group(1)
        doi = doi.replace("_", "/")

        # doi = pmid2doi(pmid) 

        for ner_chunk in ner:
            text = ner_chunk['text']

            # Process text chunks for increased readability
            chunk = nlp(text)
            
            # Split text into sentences
            sentences = [sent.text for sent in chunk.sents]
            annotations = ner_chunk['annotations']
            
            for annotation in annotations:
                if annotation['obj'] == 'mutation':
                    mutation = annotation["mention"]
                    for count, sent in enumerate(sentences):
                        if mutation in sent:
                            context = []

                            # Save sentence before and after mutation
                            if count != 0:
                                context.append(sentences[count-1])

                            context.append(sent)

                            if count != (len(sentences)-1):
                                context.append(sentences[count+1])

                            context = " ".join(context)

                            # output[pmid]["text"].append(context)
                            output[doi][mutation].append(context)
                            # output[pmid]["doi"] = doi
        
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viriation Pipeline")

    # Define arguments
    parser.add_argument('--data', type=str, required=True, help='File name containing data to parse')
    parser.add_argument('--url', type=str, required=True, help='URL for NER processing')
    
    # Parse arguments
    args = parser.parse_args()

    # Load data
    df = pd.read_csv('../../data/pipeline_data/paper_flagging_data/bert_dataset.csv')
    ds = ds_preparation(df, val_count=128)
    
    # Load model
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained("../../models/chunks-pubmed-bert-v2", num_labels=2)
    
    trainer = fine_tune_model(ds, model_init, train=False)

    # Load lightbgm model
    bst = lgb.Booster(model_file='../../models/lightgbm_model.txt')

    # Load new papers
    data = pd.read_csv(args.data)

    results = paper_prediction(data, bst, tokenizer, trainer)
    flagged_papers = results["doi"].tolist()

    # Update screened papers
    with open('../data/database/history.pkl', 'rb') as f:
        h = pickle.load(f)

    # Add papers that have been screened as irrelevant    
    for doi in data["doi"].tolist():
        if doi not in flagged_papers:
            h.addPaper(doi)

    h.updatePapers(irrelevant_papers = irrelevant)

    with open('../data/database/history.pkl', 'wb') as f:
        pickle.dump(h, f)

    lhost = str(args.url) + "/plain"
    
    NER(results, lhost)




