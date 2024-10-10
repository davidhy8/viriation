from flask import render_template, request, redirect, url_for, jsonify
# from . import create_app
from app import app
from .scripts import pipeline
from metapub.convert import pmid2doi
import pickle
import time
from .scripts.data_processor import get_doi_file_name
import pandas as pd
import json
import os

current_timestamp = time.time()
print("Initialization:", current_timestamp)

# Loading mutations 
my_dict = pipeline.load_mutations()
current_timestamp = time.time()
print("Loading timestamp:", current_timestamp)
data = {k: my_dict[k] for i, k in enumerate(my_dict) if i < 10}

# Load metadata
metadata = pd.read_csv("data/scraper/info.csv")

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


            labels.append([mutation, doi, protein, effect, text])


        save_labels(labels, doi_id, irrelevant_text)

         # Save the irrelevant text chunks
        if irrelevant_chunks:
            save_irrelevant_chunks(doi_id, irrelevant_chunks)

        update_completion_list(doi)
        return redirect(url_for('index'))
    return render_template('paper.html', data = data[doi], title = title, authors=authors, doi = doi, doi_id = doi_id)

# @app.route('/paper/<int:paper_id>', methods=['GET', 'POST'])
# def paper(paper_id):
#     pmid = next((p for p in data.keys() if int(p) == int(paper_id)), None)
#     doi = pmid2doi(pmid)

#     if not pmid:
#         return redirect(url_for('index'))
#     if request.method == 'POST':
#         labels = []
#         for mutation, text in data[pmid].items():
#             effect = []

#             # Protein that is mutated
#             protein = request.form.get(f'protein_{pmid}_{mutation}')

#             # Effects of mutation
#             if request.form.get(f'invasion_{pmid}_{mutation}'):
#                 effect.append("Host invasion")
#             if request.form.get(f'neutralization_{pmid}_{mutation}'):
#                 effect.append("Serum neutralization")
#             if request.form.get(f'transmission_{pmid}_{mutation}'):
#                 effect.append("Transmissibility")
#             if request.form.get(f'homoplasy_{pmid}_{mutation}'):
#                 effect.append("Homoplasy")

#             labels.append([mutation, text, pmid, doi, protein, effect])

#         save_labels(labels, pmid)
#         update_completion_list(pmid)
#         return redirect(url_for('index'))
#     return render_template('paper.html', pmid = pmid, data = data[pmid], doi = doi)


def update_completion_list(doi):
    with open('data/database/screened_papers.pkl', 'rb') as f:
        papers = pickle.load(f)

    # papers[pmid] = "labelled" # indicate the paper has been labelled
    papers = set()
    papers.add(doi)

    with open('data/database/screened_papers.pkl', 'wb') as f:
        pickle.dump(papers, f)

def update_data(data):
    with open('data/database/screened_papers.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    papers = set()
    
    # res = {k: v for k, v in data.items() if k not in papers.keys()}
    res = {k: v for k, v in data.items() if k not in papers}
    return res

def save_labels(labels, key, irrelevant):
    filename = 'data/database/annotations/' + str(key) + '.txt'
    with open(filename, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    if not os.path.exists('data/database/self-train/papers_retrain_data.pkl'):
        with open('data/database/self-train/papers_retrain_data.pkl', 'wb') as file:
            papers = {}
            pickle.dump(papers, file)

    with open('data/database/self-train/papers_retrain_data.pkl', 'rb') as f:
        papers = pickle.load(f)
        if key not in papers:
            papers[key] = "irrelevant" if irrelevant else "relevant"

    print(papers)

    with open('data/database/papers_retrain_data.pkl', 'wb') as f:
        pickle.dump(papers, f)

# Save the irrelevant chunks to a separate file
def save_irrelevant_chunks(doi, irrelevant_chunks):
    filename = f'data/database/self-train/chunks_retrain_data.txt'
    with open(filename, 'w') as f:
        for mutation, chunks in irrelevant_chunks.items():
            # f.write(f'Mutation: {mutation}\n')
            for chunk in chunks:
                chunk = chunk[3:]
                f.write(f'Irrelevant\t{chunk}\n')
            
        # for mutation
            # f.write('\n')


