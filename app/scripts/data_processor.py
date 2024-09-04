import re
import urllib.request
from pathlib import Path
from collections import defaultdict
import os
from metapub.convert import doi2pmid, pmid2doi
import requests
from ratelimit import limits, sleep_and_retry
from bs4 import BeautifulSoup
import json
import lxml.etree as ET
import subprocess
import copy
from metapub import PubMedFetcher
import pandas as pd


# Regex format of DOI links, mutations, blocks, and literature type
doi_pattern = r'https:\/\/doi\.org\/[\w/.-]+'
mutation_pattern = r'(.*)?\n\n'
block_pattern = r'(?:(?<=\n\n)|^)(.+?)(?=\n\n|\Z)'
literature_pattern = r'(?<=\[)(.*?)(?=\])'
url_pattern_alone = r'https:\/\/[^\s]+'
url_pattern = r'https?:\/\/[\w\/.%()-]+(?=\s*\[[^\]]*\])'
url_and_lit_pattern = r'https?:\/\/[\w\/.%()-]+\s+\[(.*?)\]'


# Obtain BioC JSON file from PMID or PMC with a maximum of 3 API calls per second
@sleep_and_retry
@limits(calls=3, period=1)
def get_pubtator_bioc_json(id):
    # API link for BioC
    url = "https://www-ncbi-nlm-nih-gov.ezproxy.lib.ucalgary.ca/research/bionlp/RESTful/pubmed.cgi/BioC_json/" + str(id) + "/unicode"
    bioc = requests.get(url, allow_redirects=True)

    if bioc.status_code != 200:
        raise ConnectionError('could not download {}\nerror code: {}'.format(url, bioc.status_code))
        return None

    if bioc.content.decode('utf-8') == '[]':
        return None
    
    return (bioc.content.decode('utf-8'))


# Obtain PMID ID from DOI link with a maximum of 3 API calls per second
@sleep_and_retry
@limits(calls=3, period=1)
def get_pmid(doi):
    # pmid = doi2pmid(doi)
    doi_part = doi.split('doi.org/')[-1] if 'doi.org/' in doi else doi

    # Api link for paper details
    api_link = 'https://www-ncbi-nlm-nih-gov.ezproxy.lib.ucalgary.ca/pmc/utils/idconv/v1.0/?tool=doi2pmid&email=david.yang1@ucalgary.ca&ids=' + doi_part
    paper = requests.get(api_link)
    soup = BeautifulSoup(paper.content, "xml")
        
    pmid = soup.find('record')['pmid']
    return pmid


# Get metadata of Rxiv paper
@sleep_and_retry
@limits(calls=1, period=1)
def get_rxiv_details(doi, is_biorxiv):
    doi_part = doi.split('doi.org/')[-1] if 'doi.org/' in doi else doi
    
    if is_biorxiv:
        api_link = 'https://api.biorxiv.org/details/biorxiv/' + doi_part
    else:
        api_link = 'https://api.medrxiv.org/details/medrxiv/' + doi_part
    
    preprint_details = requests.get(api_link)
    
    if preprint_details.status_code != 200:
        raise ConnectionError('could not download {}\nerror code: {}'.format(api_link, preprint_details.status_code))
        return None
    
    return preprint_details.content


# Get PMID of Rxiv paper
@sleep_and_retry
@limits(calls=3, period=1)
def get_rxiv_pmid(doi, is_biorxiv):
    details = get_rxiv_details(doi,is_biorxiv).decode('utf-8')
    pmid = None
    
    # Load the JSON data
    data = json.loads(details)
    title = data['collection'][0]['title']
    modified_title = title.replace(" ", "%20")
    pubmed_link = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=1000&term=" + modified_title + "&field=title"
    
    data_json = requests.get(pubmed_link).content.decode('utf-8')
    data = json.loads(data_json)
    pmid = data['esearchresult']['idlist'][0]

    return pmid


def get_rxiv_published_doi(details):
    data = json.loads(details)

    # Check if it is published
    if "published" in data["collection"][0]:
        doi = "https://doi.org/" + data['collection'][0]['published']
        return doi
    else:
        return None 


def get_rxiv_jats_xml(details):
    data = json.loads(details)
    
    # Grab the JATS XML
    jatsxml_url = data['collection'][0]['jatsxml']
    jats_xml = requests.get(jatsxml_url).content.decode('utf-8')
    return jats_xml


def convert_jatsxml_to_html(input_file, output_file):
    # dom = ET.parse(input_file)
    dom = ET.fromstring(input_file)

    # XSL style sheet
    xslt = ET.parse('../../data/other/jats-to-html.xsl')
    transform = ET.XSLT(xslt)
    newdom = transform(dom)
    newdom.write_output(output_file)


def command_line_call(call):
    x = call.split(" ")
    subprocess.run(x)


def get_journal_publication_bioc(dict, isPmidDict = False):
    unk_dict = {}
    for key in dict:
        rxiv = False
        
        bioc = None
        
        try:
            # Convert to PMID
            pmid = key if isPmidDict else get_pmid(key)
            bioc = get_pubtator_bioc_json(pmid)
            if bioc == '[]':
                bioc = None
        except:
            # Alternative way to get bioc
            bioc = None
            pass

        if bioc is None:
            try:
                pmid = doi2pmid(key)
                bioc = get_pubtator_bioc_json(pmid)
                if bioc == '[]':
                    bioc = None
            except:
                # Alternative way to get bioc
                bioc = None
                pass

        # Check if it is a preprint
        if bioc is None or bioc == '[]':
            unk_dict[key] = None
            continue

        dict[key] = bioc

    for key in unk_dict:
        dict.pop(key, None)
        
    return dict, unk_dict


def get_rxiv_bioc(dict):
    unk_dict = {}
    for key in dict:
        bioc = None
        converted = False
        biorxiv = True
        details = None
        
        # Get PMID as BioRxiv
        try:
            # Convert to PMID
            pmid = get_rxiv_pmid(key, is_biorxiv=True)
            bioc_temp = get_pubtator_bioc_json(pmid)
            if bioc_temp != '[]' and bioc_temp is not None:
                    bioc = bioc_temp
                    converted = True
        except:
            # cannot_convert_m1 += 1
            converted = False
            # print("fail bio pmid")
            pass
    
        # Get PMID as MedRxiv
        if converted == False:
            try:
                # Convert to PMID
                pmid = get_rxiv_pmid(key, is_biorxiv=False)
                bioc_temp = get_pubtator_bioc_json(pmid)
                if bioc_temp != '[]' and bioc_temp is not None:
                    bioc = bioc_temp
                    converted = True
            except:
                # cannot_convert_m1 += 1
                converted = False
                # print("fail med pmid")
                pass

        if converted == False:
        # Convert to PMID then BioC using another tool
            try:
                # Convert to DOI then PMID then BioC
                pmid = get_pmid(key)
                bioc_temp = get_pubtator_bioc_json(pmid)
                if bioc_temp != '[]' and bioc_temp is not None:
                    bioc = bioc_temp
                    converted = True
            except:
                # cannot_convert_m2 += 1
                converted = False
                # print("no pmid")
                pass
    
        # Successful, goto next key
        if bioc and bioc != "[]":
            dict[key] = bioc
            # print("done")
            continue
    
        # Try to get details as BioRxiv
        try: 
            details = get_rxiv_details(key, is_biorxiv=True).decode('utf-8')
    
        except:
            biorxiv = False
            converted = False
            print("fail bio detail")
    
        if details:
            status = json.loads(details)
            status = status['messages'][0]['status']
    
            if status != 'ok': 
                biorxiv = False
        
        # Try to get details as MedRxiv
        if biorxiv == False:
            try:
                details = get_rxiv_details(key, is_biorxiv=False).decode('utf-8')
            except:
                print("fail med detail")
                continue

        if converted == False:
        # Convert to PMID then BioC using another tool
            try:
                # Convert to DOI then PMID then BioC
                doi = get_rxiv_published_doi(details)
                pmid = get_pmid(doi)
                bioc_temp = get_pubtator_bioc_json(pmid)
                if bioc_temp != '[]' and bioc_temp is not None:
                    bioc = bioc_temp
                    converted = True
            except:
                # cannot_convert_m2 += 1
                converted = False
                # print("no published doi")
                pass
    
        # Retreive JATS XML then convert to HTML for later conversions
        if converted == False or bioc == "[]":
            try:
                jats_xml = get_rxiv_jats_xml(details)
                file_name = key.split('doi.org/')[-1]
                # Replace . in DOI with -
                # file_name = file_name.replace(".", "-")
                # # Replace / in DOI with _
                # file_name = file_name.replace("/", "_")
                file_name = get_doi_file_name(file_name)
                output_file = '../../data/scraper/rxiv/html/' + file_name + ".html"
                print("checkpoint")
                convert_jatsxml_to_html(jats_xml, output_file)
                bioc = "converting"
            except:
                bioc = None
                print("fail jats")
        
        # Check if it is a preprint
        if bioc is None or bioc == '[]':
            unk_dict[key] = None
            continue
    
        dict[key] = bioc

    for key in unk_dict:
        print(key)
        dict.pop(key, None)

    return dict, unk_dict


def get_file_name(key):
    doi = re.search(doi_pattern, key)

    if doi is not None:
        file_name = key.split('doi.org/')[-1]
    else:
        key = key.split('https://')[-1]
        file_name = key

    # Replace . in DOI with -
    # file_name = file_name.replace(".", "-")
    # Replace / in DOI with _
    file_name = file_name.replace("/", "_")
    # file_name += ".pdf"

    return file_name

def get_doi_file_name(key):
    doi = re.search(doi_pattern, key)

    if doi is not None:
        file_name = key.split('doi.org/')[-1]
    else:
        file_name = key

    # Replace . in DOI with -
    # file_name = file_name.replace(".", "-")
    # Replace / in DOI with _
    file_name = file_name.replace("/", "_")
    # file_name += ".pdf"

    return file_name

@sleep_and_retry
@limits(calls=3, period=1)
def get_doi(pmid):
    doi = pmid2doi(pmid)
    return doi

@sleep_and_retry
@limits(calls=3, period=1)
def fetcher(pmid):
    fetch = PubMedFetcher()
    article = fetch.article_by_pmid(pmid)
    author = article.author_list[0]

    return article, author

def fetch_info(key, df):
    info = df.loc[key]
    return info
