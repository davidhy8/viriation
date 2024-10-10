# Read in all the data and format it in a manner that can be read and saved in the database

# Phase 1: PubmedBERT fine-tuning
# Reading in & processing the data including irrelevant chunks & relevant chunks -> separate file locations

# Fine-tune the pubmedbert model --> easy to fine tune 


# Phase 2: LightGBM fine-tuning
# Reading in data where papers are irrelevant or relevant
with open('data/database/irrelevant_papers.pkl', 'rb') as f:
        papers = pickle.load(f)


# Fine tune lightGBM model