#!/bin/bash

# need to add mkdir in the script and rm directories as well

# Read arguments
while getopts s:e: flag
do
        case "${flag}" in
                s) start_date=${OPTARG};;
                e) end_date=${OPTARG};;
        esac
done

# Activate conda
CONDA_PATH=~/miniconda3/etc/profile.d/conda.sh
source $CONDA_PATH

# Activate environments
BASE_PATH=/home/david.yang1/autolit/viriation/
conda activate viriation
cd $BASE_PATH
source .venv/bin/activate

# Scrape pubmed papers
start_pm="${start_date//-/\/}"
end_pm="${end_date//-/\/}"
esearch -db pubmed -query "('coronavirus'[All Fields] OR 'ncov'[All Fields] OR 'cov'[All Fields] OR '2019-nCoV'[All Fields] OR 'SARS-CoV-2'[All Fields] OR 'COVID19'[All Fields] OR 'COVID-19'[All Fields] OR 'COVID'[All Fields]) AND (\"${start_pm}\"[CRDT] : \"${end_pm}\"[CRDT]) NOT preprint[pt]" | efetch -format docsum > data/scraper/pubmed/litcovid.xml 

# Scrape new pubmed and rxiv papers
cd app/scripts/
python scrape_papers.py --start $start_date --end $end_date --path $BASE_PATH
# Process xmls with multiple root structures

# Process rxiv papers to BioC format
conda activate autocorpus
cd $BASE_PATH/submodules/autocorpus
python run_app.py -c "${BASE_PATH}/data/other/config_allen.json" -t "${BASE_PATH}/data/scraper/rxiv/bioc/" -f "${BASE_PATH}/data/scraper/rxiv/html/" -o JSON

# Preprocess the papers
conda activate viriation
cd $BASE_PATH/app/scripts 
python preprocessing.py --data ${BASE_PATH}/data/scraper/scraped_papers.txt --out ${BASE_PATH}/data/scraper/papers.csv

# Activate the first conda environment and start the localhost server in the background
BERN2_PATH=/home/david.yang1/autolit/BERN2/scripts
conda activate bern2
cd $BERN2_PATH
bash stop_bern2.sh
bash run_bern2_cpu.sh > output.txt &
sleep 200

# Capture the process url of the localhost server
LOCALHOST_PID=$(cat output.txt| grep -a "Running on" | tail -n 1| cut -c15-)
echo $LOCALHOST_PID

# Screen papers
conda activate viriation
cd ${BASE_PATH}/app/scripts
python pipeline.py --data "${BASE_PATH}/data/scraper/papers.csv" --url $LOCALHOST_PID