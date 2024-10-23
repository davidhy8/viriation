from flask import render_template, request, redirect, url_for, jsonify
from app import app
from scripts.pipeline import load_mutations
from metapub.convert import pmid2doi
import pickle
import time
from scripts.data_processor import get_doi_file_name
import pandas as pd
import json
import os
from scripts.history import History
import dill

current_timestamp = time.time()
print("Initialization:", current_timestamp)

# Loading mutations 
my_dict = load_mutations()
current_timestamp = time.time()
print("Loading timestamp:", current_timestamp)
data = {k: my_dict[k] for i, k in enumerate(my_dict) if i < 10}

# Load metadata
metadata = pd.read_csv("data/scraper/info.csv")

# Load mutation data from pokay
with open("submodules/pokay/output_2.json") as f:
    mutations_data = json.load(f)

@app.route('/get_mutations/<mutation_id>')
def get_mutations(mutation_id):
    # Filter data based on whether the mutation_id is in the URL
    relevant_mutations = [
        mutation for mutation in mutations_data
        if mutation_id in mutation["url"] # filter for relevant mutations
    ]
    # return jsonify(relevant_mutations)
    return render_template('mutation.html', mutations=relevant_mutations, mutation_id=mutation_id)

@app.route('/')
def index():
    print("data: " + str(len(data)))
    filtered_data = update_data(data)
    print("filtered data: " + str(len(filtered_data)))
    print(data)
    print(filtered_data.keys())
    meta = metadata[metadata["doi"].isin(filtered_data.keys())] # Filter metadata so you only keep the rows that are in dictionary
    print("Meta: " + str(len(meta)))
    current_timestamp = time.time()
    print("Loading index:", current_timestamp)
    print(request.base_url)
    return render_template('index.html', info=meta) 


@app.route('/paper/<doi_id>', methods=['GET', 'POST'])
def paper(doi_id):
    paper_id = doi_id.replace("_", "/")

    doi = next((p for p in data.keys() if str(p) == str(paper_id)), None)
    # doi = pmid2doi(pmid)

    if not doi:
        return redirect(url_for('index'))

    meta = metadata[metadata["doi"] == doi]
    title = meta["title"].values[0]
    doi = meta["doi"].values[0]
    authors = meta["authors"].values[0]

    if request.method == 'POST':
        labels = []
        irrelevant_chunks = request.form.get('irrelevant_chunks')
        if irrelevant_chunks:
            irrelevant_chunks = json.loads(irrelevant_chunks) 
            print(irrelevant_chunks)
        else:
            irelevant_chunks = None

        for mutation, text in data[doi].items():
            effect = []

            # Protein that is mutated
            protein = request.form.get(f'protein_{doi_id}_{mutation}')

            # Effects of mutation
            if request.form.get(f'invasion_{doi_id}_{mutation}'):
                effect.append("Host invasion")
            if request.form.get(f'neutralization_{doi_id}_{mutation}'):
                effect.append("Serum neutralization")
            if request.form.get(f'transmission_{doi_id}_{mutation}'):
                effect.append("Transmissibility")
            if request.form.get(f'homoplasy_{doi_id}_{mutation}'):
                effect.append("Homoplasy")
            if request.form.get(f'irrelevant_paper'):
                irrelevant_text = True
            else:
                irrelevant_text = False

            # IMPLEMENT LOGIC TO REMOVE TEXT CHUNK HERE IF IT IS ALSO FOUND IN IRRELEVANT_CHUNKS
            if irrelevant_chunks:
                for chunk in text:
                    if mutation in irrelevant_chunks and chunk in irrelevant_chunks[mutation]:
                        text.remove(chunk)
            
            # only save if needed
            if len(text) > 0 and (protein != "None" or len(effect) > 0):
                labels.append([mutation, doi, protein, effect, text])
        
        save_data(labels, doi, doi_id, irrelevant_text, irrelevant_chunks)
        # save_labels(labels, doi_id, irrelevant_text)

         # Save the irrelevant text chunks
        # if irrelevant_chunks:
        #     save_irrelevant_chunks(doi_id, irrelevant_chunks)

        # update_completion_list(doi, irrelevant_text)
        return redirect(url_for('index'))
    return render_template('paper.html', data = data[doi], title = title, authors=authors, doi = doi, doi_id = doi_id)


def update_data(data):
    with open('data/database/history.pkl', 'rb') as f:
        history = dill.load(f)
    
    # res = {k: v for k, v in data.items() if k not in papers.keys()}
    res = {k: v for k, v in data.items() if not history.checkPaper(k)}
    return res


def save_data(labels, doi, doi_id, irrelevant_text, irrelevant_chunks=None):
    # load
    with open('data/database/history.pkl', 'rb') as f:
        history = dill.load(f)
    with open('data/database/self_train.pkl', 'rb') as f:
        retrain = pickle.load(f)
    
    history.addPaper(doi, relevance=not irrelevant_text) # update history

    if irrelevant_text:
        retrain["irrelevant papers"].add(doi) # update irrelevant self-train papers
    
    elif not irrelevant_text and len(labels) > 0:
        retrain["relevant papers"].add(doi) # update relevant self-train papers

        # save labels
        filename = 'data/database/annotations/' + str(doi_id) + '.txt'
        with open(filename, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

    # update relevant self-train text chunks
    for label in labels:
        for chunk in label[-1]:
            retrain["relevant text"].add(chunk)

    # update irrelevant self-train text chunks
    if irrelevant_chunks:
        for mutation, chunks in irrelevant_chunks.items():
            for chunk in chunks:
                # chunk = chunk[3:] # get rid of the numbering
                retrain["irrelevant text"].add(chunk)

    # save
    with open('data/database/history.pkl', 'wb') as f:
        dill.dump(history, f)
    with open('data/database/self_train.pkl', 'wb') as f:
        pickle.dump(retrain, f)


# def update_completion_list(doi, is_relevant):
#     with open('data/database/history.pkl', 'rb') as f:
#         h = pickle.load(f)

#     h.addPaper(doi, relevance=is_relevant)

#     with open('data/database/screened_papers.pkl', 'wb') as f:
#         pickle.dump(h, f)


# def save_labels(labels, key, irrelevant):
#     filename = 'data/database/annotations/' + str(key) + '.txt'
#     with open(filename, 'w') as f:
#         for label in labels:
#             f.write(f"{label}\n")

#     if not os.path.exists('data/database/self-train/papers_retrain_data.pkl'):
#         with open('data/database/self-train/papers_retrain_data.pkl', 'wb') as file:
#             papers = {}
#             pickle.dump(papers, file) 

#     with open('data/database/self-train/papers_retrain_data.pkl', 'rb') as f:
#         papers = pickle.load(f)
#         if key not in papers:
#             papers[key] = "irrelevant" if irrelevant else "relevant"

#     print(papers)

#     with open('data/database/papers_retrain_data.pkl', 'wb') as f:
#         pickle.dump(papers, f)


# Save the irrelevant chunks to a separate file
# def save_irrelevant_chunks(doi, irrelevant_chunks):
#     filename = f'data/database/self-train/chunks_retrain_data.txt'
#     with open(filename, 'w') as f:
#         for mutation, chunks in irrelevant_chunks.items():
#             # f.write(f'Mutation: {mutation}\n')
#             for chunk in chunks:
#                 chunk = chunk[3:]
#                 f.write(f'Irrelevant\t{chunk}\n')
        # for mutation
            # f.write('\n')


# Features to implement:
# 1. Removing irrelevant chunks from all chunks that get passed through
# 2. Unify the functions so that one function will perform everything including saving annotations, updating history and updating the self-train data