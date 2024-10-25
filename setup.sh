BASE_PATH="$(cd "$(dirname "$0")" && pwd)"
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
