# Viriation
This project leverages state-of-the-art BERT models to monitor the latest scientific publications for discussions on new variants of the virus under study. It identifies, flags, and summarizes mutations and their effects, consolidating the information into a comprehensive summary file.


## Table of Contents 
- [Introduction](#introduction) 
- [Features](#features) 
- [Installation](#installation) 
- [Usage](#usage) 
- [Results](#results) 
- [License](#license) 

## Introduction 
As the viral landscape evolves, keeping up-to-date with the latest mutations and their implications is critical for research and public health. This project automates the surveillance of recent publications, identifying and summarizing novel variants and their potential impact on the virus. This repository currently focuses on tracking SARS-CoV-2 variants, but our ultimate goal is to create a versatile framework that can be applied to monitor and analyze variants of any viral type.

## Features

- **Automated Literature Monitoring**: Scans and analyzes the latest publications for relevant content. 
- **Mutation Identification**: Flags papers discussing new viral mutations. 
- **Summary Generation**: Produces concise summaries of the mutations and their effects.
- **Comprehensive Database**: Updates a summary file with all identified variants and their details.


## Installation 
To run this project, ensure you have Python installed. Follow these steps to set up the project:
1. Clone the repository:

```git clone https://github.com/davidhy8/viriation.git```

```cd viriation ``` 

2. Create and activate a virtual environment: 

```python -m venv venv``` 

```source venv/bin/activate```, on Windows, use `venv\Scripts\activate` 

3. Install the required dependencies: 

```pip install -r requirements.txt ``` 

## Usage
To use the project, follow these steps: 
1. **Run the notebook**:

```jupyter notebook cv_bert.ipynb ``` 

2. **View results**: Check the `output/` folder for the latest summaries of variants.

## Results 
The `output/` folder provides a consolidated view of all identified variants, including:  
- Mutation details
- Effects on the virus 
- Relevant publication links 

These variants are categorized into files according to the specific protein affected by the mutation and its overall impact.

## License 
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
