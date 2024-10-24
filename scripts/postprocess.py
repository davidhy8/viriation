import pandas as pd
from pathlib import Path
import dill

# Load data
with open("../data/database/history.pkl", "rb") as f:
    history = dill.load(f)

with open("../data/database/self_train.pkl", "rb") as f:
    self_train = dill.load(f) # keys: relevant papers, irrelevant papers, relevant text, irrelevant text


# STEP 2: Paper level feedback

# IRRELEVANT PAPERS
irrelevant_df = pd.DataFrame(
    self_train["irrelevant papers"],  # Convert dictionary to list of tuples
    columns=['DOI']  # Specify column names
)

# Set as irrelevant
irrelevant_df["Classification"] = 0

# RELEVANT PAPERS
relevant_df = pd.DataFrame(
    self_train["relevant papers"],
    columns=["DOI"]
)

# Set as relevant
relevant_df["Classification"] = 1
papers_df = pd.concat([relevant_df,irrelevant_df], ignore_index=True)
papers_df.to_csv("../data/train/papers_retrain.csv")


# STEP 2: Chunk level feedback
# IRRELEVANT TEXT
irrelevant_df = pd.DataFrame(
    self_train["irrelevant text"],  # Convert dictionary to list of tuples
    columns=['text']  # Specify column names
)

# Set as irrelevant
irrelevant_df["Classification"] = 0

# RELEVANT TEXT
relevant_df = pd.DataFrame(
    self_train["relevant text"],
    columns=["text"]
)

# Set as relevant
relevant_df["Classification"] = 1
chunks_df = pd.concat([relevant_df,irrelevant_df], ignore_index=True)
chunks_df.to_csv("../data/train/chunks_retrain.csv")