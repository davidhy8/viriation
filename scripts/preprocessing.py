import re
import json
import random
import copy
from bioc import biocjson
import pandas as pd
import pypdf
import os
import argparse
from datetime import date

import sys
sys.path.append('/home/david.yang1/autolit/virus-trial/viriation') # replace with your folder location

from scripts.data_processor import get_doi_file_name
from pathlib import Path
 

def check_dictionary(d):
    """
    Checks the contents of a dictionary and prints information about its size 
    and the status of its keys.

    This function iterates through the provided dictionary, checking the value 
    of each key. It prints the size of the dictionary, and for each key, it 
    checks if the value is `None` or the string "converting". If either 
    condition is met, it prints the corresponding key with a message.

    Parameters:
        d (dict): A dictionary to check. Keys can be of any hashable type, and values can be of any type.
    """

    print("size: " + str(len(d)))
    for key in d:
        if d[key] is None:
            print("None: " + key)
    
        if d[key] == "converting":
            print("Converting: " + key)


def get_file_name(key):
    """
    Generates a sanitized file name from a given DOI or URL string.

    This function checks if the provided key is a DOI link and extracts the 
    relevant portion to create a file name. The function replaces certain 
    characters (periods and slashes) in the DOI with safe alternatives 
    (dashes and underscores, respectively) to ensure the resulting 
    file name is valid.

    Parameters:
        key (str): A DOI or URL string from which to generate a file name.

    Returns:
        (str): A sanitized file name derived from the input string.
    """

    doi_pattern = r'https:\/\/doi\.org\/[\w/.-]+'
    doi = re.search(doi_pattern, key)

    if doi is not None:
        file_name = key.split('doi.org/')[-1]
    else:
        key = key.split('https://')[-1]
        file_name = key

    # Replace . in DOI with -
    file_name = file_name.replace(".", "-")
    # Replace / in DOI with _
    file_name = file_name.replace("/", "_")
    # file_name += ".pdf"

    return file_name


def extract_nested_elements(input_string):
    """
    Extracts nested elements enclosed in curly braces from a string.

    This function scans the input string for nested elements, defined as 
    substrings enclosed in matching curly braces (`{}`). It keeps track 
    of the current nesting level and captures complete elements whenever 
    a matching closing brace is found.

    Parameters:
        input_string (str): A string potentially containing nested elements enclosed in curly braces.

    Returns:
        (list): A list of extracted nested elements as strings.
    """
    
    elements = []
    start = 0
    brace_count = 0
    inside_element = False

    for i, char in enumerate(input_string):
        if char == '{':
            if brace_count == 0:
                start = i
                inside_element = True
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and inside_element:
                elements.append(input_string[start:i+1])
                inside_element = False

    return elements


def pubtator_extract(paper):
    """
    Extracts text from a BioC JSON formatted paper originating from pubtator API.

    Parameters:
        paper (str): A string representation of the BioC JSON formatted paper.

    Returns:
        (str): Extracted text from the paper, or None if extraction fails.
    """

    text = ""
    paper = paper[1:-1]

    try:
        bioc_list = extract_nested_elements(paper)
        
        bioc_collection = biocjson.loads(bioc_list[-1])
        
    except:
        return None

    for document in bioc_collection.documents:    
        for passage in document.passages:
            try:
                text += passage.text
                
            except:
                pass

            if text[-1].isalnum(): 
                text += ". "
            else:
                text += " "
   
    if text == "":
        return None

    return text


def pdf_extract(data):
    """
    Extracts text from a BioC JSON formatted paper originating from PDF conversions.

    Parameters:
        data (str): A string representation of the BioC JSON formatted paper.

    Returns:
        str: Extracted text from the paper, or None if extraction fails.
    """

    text = ""

    try:
        bioc_collection = biocjson.loads(data)

    except:
        return None
        
    for document in bioc_collection.documents:    
        for passage in document.passages:
            try:
                text += passage.text
            except:
                pass

            if text[-1].isalnum(): 
                text += ". "
            else:
                text += " "

    if text == "":
        return None

    return text


def jats_extract(paper):
    """
    Extracts text from a BioC JSON formatted paper originating from JATS XML conversions.

    Parameters:
        paper (str): A string representation of the BioC JSON formatted paper.

    Returns:
        (str): Extracted text from the paper, or None if extraction fails.
    """
    text = ""
    
    try:
        # paper_copy = paper[1:-1]
        paper_copy = paper
        bioc_collection = biocjson.loads(paper_copy)

    except:
        try:
            bioc_collection = biocjson.loads(paper)
        except:
            return None

    for document in bioc_collection.documents:    
        for passage in document.passages:
            try:
                text += passage.text
            except:
                pass

            if text[-1].isalnum(): 
                text += ". "
            else:
                text += " "

    if text == "":
        return None

    return text
    

def text_extract(paper):
    """
    Attempts to extract text from a BioC JSON paper obtained from various sources and conversions (PubTator, JATS, PDF).

    Parameters:
        paper (str or None): The BioC JSON paper to extract text from.

    Returns:
        (str): Extracted text, or the text from a PDF file (manually downloaded) if the paper cannot be obtained through API links.
    """
    text_extracted = False
    text = ""
    
    if paper is not None:
        # Try to extract as pubtator
        try:
            text = pubtator_extract(paper)

            if text is not None:
                text_extracted = True
                pokay_text.append(text)
        except:
            pass

        if text_extracted:
            return text

        # Try to extract as JATS
        try:
            text = jats_extract(paper)

            if text is not None:
                text_extracted = True
                pokay_text.append(text)
        except:
            pass

        if text_extracted:
            return text

        # Try to extract as PDF
        try:
            text = pdf_extract(paper)

            if text is not None:
                text_extracted = True
                pokay_text.append(text)
        except:
            pass

    else:
        file = get_file_name(key) # file name of paper PDF
        file = "../data/raw/pdf/unconverted/" + file + ".pdf" # file location of PDF 
        isExist = os.path.exists(file) 
        if isExist:
            print(file)
            reader = pypdf.PdfReader(file)
    
            for page in reader.pages:
                text += page.extract_text()

    return text


def related_paper(paper):
    """
    Checks if a paper was previously scraped and annotated in the pokay database (training data)

    Parameters:
        paper (str): DOI link of paper

    Returns:
        (bool): True if the paper is in the pokay dataset, False otherwise.
    """
    try:
        doi = paper["passages"][0]['infons']['article-id_doi']
        
        if doi in pokay_data:
            return True
            
    except:
        return False

    return False


def regex_filtering(data):
    """
    Filter papers based on mutation-related patterns found in the text.

    This function applies regular expressions to extract genetic mutation 
    information from provided papers. It identifies one-letter and three-letter 
    amino acid changes as well as genome changes. Papers containing any 
    identified mutations are stored in a filtered dictionary.

    Args:
        data (dict): A dictionary where keys are DOIs and values are paper data (typically in a structured format).

    Returns:
        dict: A dictionary containing only the papers that have identified mutations. Each entry has the DOI as the key and the original paper text as the value.

    Raises:
        Exception: If any errors occur during text extraction or regular expression matching.
    """
    
    # Regular expressions
    one_letter_aa_change = r'\b([ARNDCQEGHILKMFPSTWYV])([1-9]+\d*)(del|(?!\1)[ARNDCQEGHILKMFPSTWYV])\b'
    three_letter_aa_change = r'\b((?:ALA|ARG|ASN|ASP|CYS|GLN|GLU|GLY|HIS|ILE|LEU|LYS|MET|PHE|PRO|SER|THR|TRP|TYR|VAL))([1-9]+\d*)(?!(\1))(ALA|ARG|ASN|ASP|CYS|DEL|GLN|GLU|GLY|HIS|ILE|LEU|LYS|MET|PHE|PRO|SER|THR|TRP|TYR|VAL)\b'
    genome_change = r'\bg\.[ATGCU][1-9]+\d*[ATGCU]\b'
    genome_change_alt =  r'\bg\.[1-9]+\d*[ATGCU]\>[ATGCU]\b'

    filtered_papers = {}

    for doi, paper in data.items():
        text = text_extract(paper)
        # print(text)

        if text is None:
            print(doi)
            continue

        mutations = []
        mutations += ["".join(x) for x in re.findall(one_letter_aa_change, text, re.IGNORECASE)]
        mutations += ["".join(x) for x in re.findall(three_letter_aa_change, text, re.IGNORECASE)]
        mutations += re.findall(genome_change, text, re.IGNORECASE)
        mutations += re.findall(genome_change_alt, text, re.IGNORECASE)
        mutations = set(mutations)

        if len(mutations) > 0:
            filtered_papers[doi] = paper

    return filtered_papers


def date_filtering(data, date_cutoff):
    """
    Filter papers based on a specified publication date cutoff.

    This function checks the publication date of each paper and retains 
    only those published after the provided date cutoff. The papers are 
    stored in a filtered dictionary.

    Args:
        data (dict): A dictionary where keys are DOIs and values are paper data (typically in a structured format).
        date_cutoff (str): The cutoff date in ISO format (YYYY-MM-DD) for filtering papers.

    Returns:
        dict: A dictionary containing only the papers published after the specified date cutoff. Each entry has the DOI as the key and the original paper text as the value.

    Raises:
        Exception: If any errors occur during date extraction or processing.
    """
    filtered_papers = {}
    for doi, paper in data.items():
        print(paper)
        p = paper[1:-1] if paper[0] == "[" else paper
        bioc_collection = biocjson.loads(p)
        date_cutoff = date.fromisoformat(str(date_cutoff))

        try:
            d = bioc_collection.date
            d = date.fromisoformat(str(d))
            if d > date_cutoff:
                filtered_papers[doi] = (paper)
        except:
            continue
    
    return filtered_papers


def subset_sample(original, n):
    """
    Create a random subset of papers from the original dataset.

    This function shuffles the original list of papers and extracts 
    a subset of size `n`. The remaining papers are returned alongside 
    the sampled subset.

    Args:
        original (pd.DataFrame): A dataframe of paper data.
        n (int): The number of samples to extract.

    Returns:
        df (list): The dataframe containing the remaining papers after sampling.
        sub (list): The sampled subset of papers.

    Raises:
        ValueError: If `n` is greater than the length of the original list.
    """
    sub = []
    df = copy.deepcopy(original)
    random.seed(42)
    random.shuffle(df)
    
    for i in range(n):
        entry = df.pop(-1)
        sub.append(entry)

    return df, sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing")

    # Define arguments
    parser.add_argument('--data', type=str, required=True, help='File to preprocess')
    parser.add_argument('--cutoff', type=str, required=False, default='20000101' ,help='Oldest date to process paper: YYYYMMDD')
    parser.add_argument('--out', type=str, required=True, help="Output file for processed dataframe (csv file)")

    # Parse arguments
    args = parser.parse_args()

    # Load new papers
    with open(args.data) as file:
        data = json.loads(file.read())
    print(len(data))

    # Load converted rxiv doi
    for key in data:
        if data[key] == "converting":
            file = get_doi_file_name(key)
            file = "../data/scraper/rxiv/bioc/" + file + "_bioc.json"
            try:
                data[key] = Path(file).read_text().replace('\n', '')
            except:
                print(file)
                pass

    print(f"Original: {len(data)}")

    # Filter by date 
    # filtered_data = date_filtering(data, args.cutoff)
    # print(f"Date filtered: {len(filtered_data)}")
    
    filtered_data = data
    
    # Filter by regex
    filtered_data = regex_filtering(data)
    print(f"Regex filtered: {len(filtered_data)}")

    # Remove papers also found in Pokay
    # with open('../../data/processed/pokay/data_bioc.txt') as file:
    #     pokay_data = json.loads(file.read())
    
    # filtered_data = {x:k for x in filtered_data.values() if not related_paper(x)}
    # print(f"Pokay filtered: {len(filtered_data)}")

    # Create output dataframe
    df = pd.DataFrame.from_dict(filtered_data, orient='index', columns=["text"]).reset_index(names="doi")
    df["text"] = df["text"].map(text_extract)
    df.to_csv(args.out)
