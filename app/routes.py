from flask import render_template, request, redirect, url_for, jsonify
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
            for index, (doi, text) in enumerate(zip(value["doi"], value["text"]), start=1):
                effect = request.form.get(f'effect_{key}_{doi}_{index}')
                protein = request.form.get(f'protein_{key}_{doi}_{index}')
                print(f"Received: key={key}, doi={doi}, effect={effect}, protein={protein}")  # Debugging print
                labels.append([key, doi, text, effect, protein])
            save_labels(labels, key)
        # return redirect(url_for('success'))
        return jsonify({'message': 'Labels saved successfully'})

    return render_template('label.html', data=data)

def save_labels(labels, key):
    # print(f"Saving labels: {labels}")  # Debugging print
    filename = 'data/pipeline_data/label/labels_' + key + '.txt'
    with open(filename, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

@app.route('/success')
def success():
    return "Labels saved successfully! Pending approval to update GitHub repository."

# a simple page that says hello
@app.route('/label')
def label():
    return 'Hello, World!'
