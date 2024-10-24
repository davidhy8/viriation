BASE_PATH="$(dirname "$0")"
cd $BASE_PATH

# pipeline directory
cd data/pipeline_data
mkdir NER

# scraper directories
cd ../
mkdir scraper
cd scraper
mkdir pubmed rxiv
cd rxiv
mkdir bioc html server_dumps

# database directory
cd ../../
mkdir database
