# Viriation
This project combines state-of-the-art BERT models with gradient boosting to monitor the latest scientific publications for discussions on new variants of the virus under study. It identifies, flags, and summarizes mutations and their effects, consolidating the information into a comprehensive summary file.

# Introduction 
As the viral landscape evolves, keeping up-to-date with the latest mutations and their implications is critical for research and public health. To date, over 600,000 papers discussing COVID-19 have been published, with an additional 100 to 400 papers released daily. This sheer volume of literature highlights the urgent need for an automated solution to efficiently identify and flag relevant studies, ensuring that researchers can stay informed about new viral mutations amidst the overwhelming influx of information. Viriation automates the surveillance of recent publications, identifying and summarizing novel variants and their potential impact on the virus. This repository currently focuses on tracking SARS-CoV-2 variants, but our ultimate goal is to create a versatile framework that can be applied to monitor and analyze variants of any viral type.

Features:
- **Automated Literature Monitoring**: Scans and analyzes the latest publications and preprints for relevant content. 
- **Mutation Identification and Annotation**: Extracts discussion of new viral mutations and provides an interface for users to label its effect and location. 
- **Comprehensive Database**: Updates a summary file with all identified variants and their details.

# Installation guide
Running Viriation requires installing various dependencies, both for Viriation itself and for the submodules it utilizes. As a result, multiple environments need to be set up.

# Prerequisites
Before you start, ensure you have the following installed on your machine:
- [Python 3.11.5](https://www.python.org/downloads/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda or Miniconda)
- [Git](https://git-scm.com/downloads)
- [pip](https://pip.pypa.io/en/stable/installation/)
  
Optional:
- [Virtual Environment](https://docs.python.org/3/library/venv.html) for isolating dependencies

## Step 1: Clone the repository
Open your terminal and run the following command to clone the project repository:
```bash
git clone https://github.com/davidhy8/viriation.git
cd viriation
```

## Step 2: Set up environments
Create Conda environment for submodule dependencies and python virtual environment for viriation:

### BERN2 environment
```
# Install torch with conda (please check your CUDA version)
conda create -n bern2 python=3.7
conda activate bern2
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
conda install faiss-gpu libfaiss-avx2 -c conda-forge

# Check if cuda is available
python -c "import torch;print(torch.cuda.is_available())"

# Install BERN2
cd submodules/BERN2
pip install -r requirements.txt
```
### Auto-Corpus environment
```
conda create -n autocorpus python=3.8.19
cd submodules/autocorpus
pip install .
```
You might get an error here ModuleNotFoundError: No module named 'skbuild' if you do then run
```
$ pip install --upgrade pip
```

### Viriation environment
```
# Goto base directory
cd ../../
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 3: Download pre-trained model
Download this [folder](https://drive.google.com/drive/folders/1qW7AAjQoAgopW4FVELnfDyCa_Y9Olv7l?usp=sharing) and put it inside `models/`

## Step 4: Setup directory
This includes creating the necessary directories and downloading pre-trained PubMedBERT models
```
bash setup.sh
```

## Step 5: Running the program
To run the program, you need to specify the date range that you would like to study. Replace the `$start` and `$end` (format=YYYY-MM-DD)
```
bash run.sh --start_date $start --end_date $end
```

# License 
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

See wiki docs for more details.
