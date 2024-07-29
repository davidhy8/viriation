from flask import render_template, request, redirect, url_for
# from . import create_app
from app import app
from .models import pipeline

# app = create_app()

@app.route('/', methods=['GET', 'POST'])
def index():
    data = pipeline.load_mutations()
    
    if request.method == 'POST':
        labels = []
        for key, value in data.items():
            for (doi, text) in zip(value["doi"], value["text"]):
                effect = request.form.get(f'effect_{key}_{doi}')
                protein = request.form.get(f'protein_{key}_{doi}')
                labels.append([key, doi, text, effect, protein])
        save_labels(labels)
        # labels = [[key, value, request.form.get(key)] for key, value in data.items()]
        # save_labels(labels)
        return redirect(url_for('success'))

    return render_template('label.html', data=data)

def save_labels(labels):
    with open('data/labels.txt', 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

@app.route('/success')
def success():
    return "Labels saved successfully! Pending approval to update GitHub repository."

# a simple page that says hello
@app.route('/label')
def label():
    return 'Hello, World!'
