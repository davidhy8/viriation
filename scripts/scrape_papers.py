import pandas as pd
import dill
import data_processor as processor
import json
from paperscraper.pubmed import get_and_dump_pubmed_papers
from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv
from paperscraper.xrxiv.xrxiv_query import XRXivQuery
import argparse
import pandas as pd
import os
import xml.etree.ElementTree as ET
import re

import sys
sys.path.append('/home/david.yang1/autolit/virus-trial/viriation') # replace with your folder location

from scripts.data_processor import get_doi_file_name
from history import History


def wrap_xml_with_root(input_path, output_path, new_root="all_roots"):
    """
    Wraps all root elements in an XML file with a new single root element,
    preserving the XML declaration and DOCTYPE.

    Parameters:
        input_path (str): The path to the input XML file.
        output_path (str): The path where the wrapped XML file will be saved.
        new_root (str): The name of the new root element to wrap around the original roots.
    """
    # Read the original XML content
    with open(input_path, 'r') as file:
        content = file.read()

    # Extract the XML declaration if it exists
    xml_declaration = ""
    doctype_declaration = ""
    
    # Match and extract the XML declaration
    if content.startswith("<?xml"):
        xml_match = re.match(r'^<\?xml.*?\?>', content)
        if xml_match:
            xml_declaration = xml_match.group()
            content = content[len(xml_declaration):].strip()
    
    # Match and extract the DOCTYPE declaration
    if content.startswith("<!DOCTYPE"):
        doctype_match = re.match(r'<!DOCTYPE.*?>', content)
        if doctype_match:
            doctype_declaration = doctype_match.group()
            content = content[len(doctype_declaration):].strip()
    
    # Wrap the remaining content with the new root element
    wrapped_content = f"{xml_declaration}\n{doctype_declaration}\n<{new_root}>\n{content}\n</{new_root}>"

    # Save the wrapped content to the output file
    with open(output_path, 'w') as file:
        file.write(wrapped_content)

    print(f"File saved successfully to {output_path}.")

def process_xml_file(file_path: str, fields: list) -> pd.DataFrame:
    """
    Processes the esearch XML file and extracts the specified fields into a DataFrame.

    Parameters:
        file_path (str): Path to the XML file.
        fields (list): List of fields to extract.

    Returns:
        pd.DataFrame: A DataFrame with the extracted fields.
    """

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    # Define a field mapper to map XML tags to desired field names
    field_mapper = {
        "Title": "title",
        "Authors": "authors",
        "Journal": "journal",
        "ELocationID": "doi",
        "EPubDate": "date",
        "SortPubDate": "other_date"
        # Add more mappings as needed
    }

    # Define a processing function for each field if needed
    def process_authors(authors_element):
        """
        Processes author list
        
        Returns:
            String of author list separated by comma
        """
        return ", ".join([author.find('Name').text for author in authors_element.findall('Author')])

    process_fields = {
        "authors": process_authors,
        # Add more processing functions as needed
    }

    # Extract and process fields
    # processed_data = {
    #     field_mapper.get(field.tag, field.tag): process_fields.get(
    #         field_mapper.get(field.tag, field.tag), lambda x: x.text
    #     )(field)
    #     for field in root
    #     if field_mapper.get(field.tag, field.tag) in fields
    # }
    all_processed_data = []

    for docset in root:
        for doc in docset:
            processed_data = {}
            for field in doc:
                field_name = field_mapper.get(field.tag, field.tag)
                # print(field_name)
                if field_name in fields:
                    if field_name == "authors":
                        # Special case for handling nested <Authors> element
                        if field_name is not None:
                            author_names = [author.find('Name').text for author in field.findall('Author')]
                            processed_data[field_name] = ", ".join(author_names)
                        else:
                            processed_data[field_name] = None
                    else:
                        # Default processing for other fields
                        processing_function = process_fields.get(field_name, lambda x: x.text)
                        processed_data[field_name] = processing_function(field)

            all_processed_data.append(processed_data)


    # Convert to DataFrame
    df = pd.DataFrame(all_processed_data)

    return df

if __name__ == "__main__":
    bioc = {}
    parser = argparse.ArgumentParser(description="Viriation Pipeline")

    # Define arguments
    parser.add_argument('--start', type=str, required=True, help='Start date for Rxiv filtering YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=False, default='3000-01-01', help='End date for Rxiv filtering YYYY-MM-DD')
    parser.add_argument('--path', type=str, required=True, help='Path to working directory')
    
    # Parse arguments
    args = parser.parse_args()

    # Load list of papers previously looked at
    with open(args.path + '/data/database/history.pkl', 'rb') as f:
        history = dill.load(f)

    # LitCovid search terms
    covid_terms = ['coronavirus', 'ncov', 'cov', '2019-nCoV', 'SARS-CoV-2', 'COVID19', 'COVID']
    query = [covid_terms]

    # Format LitCovid data
    wrap_xml_with_root('../data/scraper/pubmed/litcovid.xml', '../data/scraper/pubmed/wrapped_litcovid.xml')
    
    # Load LitCovid data
    fields = ["title", "authors", "date", "doi", "other_date"]  # Define the fields you want to extract
    litcovid_data = process_xml_file(args.path + "/data/scraper/pubmed/wrapped_litcovid.xml", fields)

    # Filter LitCovid data for repetitive papers
    # litcovid_data = litcovid_data[~litcovid_data["doi"].isin(papers)]
    litcovid_data = litcovid_data[~litcovid_data["doi"].apply(history.checkPaper)]

    # Fill NAs for missing dates
    litcovid_data["date"] = litcovid_data["date"].fillna(litcovid_data["other_date"])
    litcovid_data = litcovid_data.drop('other_date', axis=1)

    # Format date
    litcovid_data["date"] = pd.to_datetime(litcovid_data["date"], format='mixed')

    # Format doi
    litcovid_data["doi"] = litcovid_data["doi"].astype(str)
    litcovid_data["doi"] = litcovid_data["doi"].apply(lambda x: x.split('doi: ')[-1] if 'doi: ' in x else x)    
    
    # Create empty BioC JSON dictionary
    dois = litcovid_data["doi"].tolist()
    litcovid_papers = {}
    for doi in dois:
        litcovid_papers[str(doi)] = None

    # Fetch BioC JSON data  
    litcovid_bioc, __ = processor.get_journal_publication_bioc(litcovid_papers)
    bioc.update(litcovid_bioc)
    print(len(litcovid_bioc))
    print(len(__))

    # Load Rxiv data
    medrxiv(begin_date=args.start, end_date = args.end, save_path=args.path + "/data/scraper/rxiv/server_dumps/medrxiv.jsonl")
    biorxiv(begin_date=args.start, end_date = args.end, save_path=args.path + "/data/scraper/rxiv/server_dumps/biorxiv.jsonl")
    
    # Filter MedRxiv
    querier = XRXivQuery(args.path + '/data/scraper/rxiv/server_dumps/medrxiv.jsonl')
    querier.search_keywords(query, output_filepath = args.path + '/data/scraper/rxiv/covid19_medrxiv.jsonl')
    
    # Filter ioRxiv
    querier = XRXivQuery(args.path + '/data/scraper/rxiv/server_dumps/biorxiv.jsonl')
    querier.search_keywords(query, output_filepath=args.path + '/data/scraper/rxiv/covid19_biorxiv.jsonl')

    # Load and process
    medrxiv = pd.read_json(args.path + "/data/scraper/rxiv/covid19_medrxiv.jsonl",lines=True)
    biorxiv = pd.read_json(args.path + "/data/scraper/rxiv/covid19_biorxiv.jsonl",lines=True)

    rxiv = pd.concat([medrxiv, biorxiv], ignore_index = True, sort = False)
    
    # Filter rxiv data for repetitive papers
    # rxiv = rxiv[~rxiv["doi"].isin(papers)]
    rxiv = rxiv[~rxiv["doi"].apply(history.checkPaper)]

    # Format date
    rxiv["date"] = pd.to_datetime(rxiv["date"])

    # rxiv.to_csv(args.path + '/data/processed/rxiv_info.csv')

    dois = rxiv["doi"].tolist()

    rxiv_papers = {}
    for doi in dois:
        print(doi)
        rxiv_papers[str(doi)] = None

    rxiv_bioc, __ = processor.get_rxiv_bioc(rxiv_papers)
    print(str(len(rxiv_bioc)) + "rxiv_bioc")
    print(len(__))

    bioc.update(rxiv_bioc)

    with open(args.path + '/data/scraper/scraped_papers.txt', 'w') as file:
        file.write(json.dumps(bioc))

    # Create metadata
    rxiv_subset = rxiv[['title', 'doi', 'date', 'authors']]
    info = pd.concat([rxiv_subset, litcovid_data], ignore_index = True, sort = False)
    info["doi_id"] = info["doi"].astype(str).apply(get_doi_file_name)

    info.to_csv(args.path + '/data/scraper/info.csv')

    history.addDateRange((args.start, args.end))
    history.updateTree()

    with open('data/database/history.pkl', 'wb') as f:
        dill.dump(history, f)     