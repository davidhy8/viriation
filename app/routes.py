from flask import render_template, request, redirect, url_for, jsonify
# from . import create_app
from app import app
from .models import pipeline
from metapub.convert import pmid2doi
import pickle


data = pipeline.load_mutations()

@app.route('/')
def index():
    filtered_data = update_data(data)
    return render_template('index.html', data=filtered_data)

@app.route('/paper/<int:paper_id>', methods=['GET', 'POST'])
def paper(paper_id):
    pmid = next((p for p in data.keys() if int(p) == int(paper_id)), None)
    doi = pmid2doi(pmid)

    if not pmid:
        return redirect(url_for('index'))
    if request.method == 'POST':
        labels = []
        for mutation, text in data[pmid].items():
            effect = []

            # Protein that is mutated
            protein = request.form.get(f'protein_{pmid}_{mutation}')

            # Effects of mutation
            if request.form.get(f'invasion_{pmid}_{mutation}'):
                effect.append("Host invasion")
            if request.form.get(f'neutralization_{pmid}_{mutation}'):
                effect.append("Serum neutralization")
            if request.form.get(f'transmission_{pmid}_{mutation}'):
                effect.append("Transmissibility")
            if request.form.get(f'homoplasy_{pmid}_{mutation}'):
                effect.append("Homoplasy")

            labels.append([mutation, text, pmid, doi, protein, effect])

        save_labels(labels, pmid)
        update_completion_list(pmid)
        return redirect(url_for('index'))
    return render_template('paper.html', pmid = pmid, data = data[pmid], doi = doi)


def update_completion_list(pmid):
    with open('data/database/completed_papers.txt', 'rb') as f:
        papers = pickle.load(f)
    
    papers.add(pmid)

    with open('data/database/completed_papers.txt', 'wb') as f:
        pickle.dump(papers, f)

def update_data(data):
    with open('data/database/completed_papers.txt', 'rb') as f:
        papers = pickle.load(f)
    
    res = {k: v for k, v in data.items() if k not in papers}

    return res

def save_labels(labels, key):
    filename = 'data/database/' + str(key) + '.txt'
    with open(filename, 'w') as f:
        for label in labels:
            f.write(f"{label}\n") 


