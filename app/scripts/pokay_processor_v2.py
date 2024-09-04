import re
import urllib.request
from pathlib import Path
from collections import defaultdict
import os
from metapub.convert import doi2pmid
import requests
from ratelimit import limits, sleep_and_retry
from bs4 import BeautifulSoup
import json
import lxml.etree as ET
import subprocess
import copy
from data_processor import *


# Regex format of DOI links, mutations, blocks, and literature type
doi_pattern = r'https:\/\/doi\.org\/[\w/.-]+'
mutation_pattern = r'(.*)?\n\n'
block_pattern = r'(?:(?<=\n\n)|^)(.+?)(?=\n\n|\Z)'
literature_pattern = r'(?<=\[)(.*?)(?=\])'
url_pattern_alone = r'https:\/\/[^\s]+'
url_pattern = r'https?:\/\/[\w\/.%()-]+(?=\s*\[[^\]]*\])'
url_and_lit_pattern = r'https?:\/\/[\w\/.%()-]+\s+\[(.*?)\]'

# # Helper functions

# Categorize entries in Pokay
def recategorize_pokay(directory):
    # Dictionary of doi to BioC JSON files 
    publication_bioc = {}
    grey_bioc = {}
    rxiv_bioc = {}

    # Dictionary of doi to pokay mutation summaries from that article
    publication_key = defaultdict(list)
    grey_key = defaultdict(list)
    rxiv_key = defaultdict(list)
    
    # Retrieve all files from Pokay directory
    files = Path(directory).glob('*/*')
    
    # Iterate through all files in the pokay directory
    for file in files:
        with open(file, 'r') as f:
        
            # Read file
            file_contents = f.read()
    
            # Find all mutations
            # mutations = re.findall(mutation_pattern, file_contents)
    
            # Find all text blocks
            text_blocks = re.findall(block_pattern, file_contents, re.DOTALL)
    
            # Iterate through all text blocks
            for text in text_blocks:
                
                # Find article types
                article_type = re.findall(url_and_lit_pattern, text)

                # Find url links
                matches = re.findall(url_pattern, text)

                # If no article type provided, check format of the link
                if len(article_type)==0:
                    url = re.search(url_pattern_alone, text)

                    if url:
                        doi = re.search(doi_pattern, url.group())
                        
                        # Check if it is preprint
                        if doi:
                            rxiv_key[doi.group()].append(text)
                            rxiv_bioc[doi.group()] = None
                        # Otherwise, grey literature
                        else:
                            grey_key[url.group()].append(text)
                            grey_bioc[url.group()] = None
                    continue
                
                for i in range(len(article_type)):
    
                    if "Journal publication" in article_type[i]:
                        # Search for the DOI of the publication
                        # doi = re.search(doi_pattern, text).group()
                        doi = matches[i]
                        publication_key[doi].append(text)
                        publication_bioc[doi] = None
                        
                    elif "Preprint" in article_type[i]:
                        # Check if new DOI is provided
                        # doi = re.search(doi_pattern, article_type[-1])
                        doi = re.search(doi_pattern, article_type[i])
                        
                        # Check if Rxiv is now published
                        if doi is not None:
                            publication_key[doi.group()].append(text)
                            publication_bioc[doi.group()] = None
        
                        # Store as Rxiv
                        else:
                            # doi = re.search(doi_pattern, text)
                            doi = matches[i]
                            # DOI link provided
                            if doi is not None:
                                # rxiv_key[doi.group()].append(text)
                                # rxiv_bioc[doi.group()] = None
                                rxiv_key[doi].append(text)
                                rxiv_bioc[doi] = None
                            # DOI link not provided
                            else: 
                                # rxiv_key[re.search(url_pattern, text).group()].append(text)
                                # rxiv_bioc[re.search(url_pattern, text).group()] = None
                                print("special case")
    
                    # Check if the article is grey literature
                    elif "Grey literature" in article_type[i]:
                        # Search for url link
                        # url = re.search(url_pattern, text)
                        url = matches[i]
                        if url is not None:
                            # grey_key[url.group()].append(text)
                            # grey_bioc[url.group()] = None
                            grey_key[url].append(text)
                            grey_bioc[url] = None
                    
                    # All other groups categorize as grey literature
                    # else:
                    #     url = re.search(url_pattern, text)
                    #     if url is not None:
                    #         grey_key[url.group()].append(text)
                    #         grey_bioc[url.group()] = None

    return publication_bioc, publication_key, rxiv_bioc, rxiv_key, grey_bioc, grey_key


# Obtain BioC JSON file from PMID or PMC with a maximum of 3 API calls per second
@sleep_and_retry
@limits(calls=3, period=1)
def get_pubtator_bioc_json(id):
    # API link for BioC
    url = "https://www-ncbi-nlm-nih-gov.ezproxy.lib.ucalgary.ca/research/bionlp/RESTful/pmcoa.cgi/BioC_json/" + str(id) + "/unicode"
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
    doi_part = doi.split('doi.org/')[-1]

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
    doi_part = doi.split('doi.org/')[-1]
    
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
    xslt = ET.parse('../data/other/jats-to-html.xsl')
    transform = ET.XSLT(xslt)
    newdom = transform(dom)
    newdom.write_output(output_file)


def command_line_call(call):
    x = call.split(" ")
    subprocess.run(x)


def get_journal_publication_bioc(dict, isPmidDict = False):
    count = 0
    unk_dict = {}
    for key in dict:
        rxiv = False
        count += 1 

        if count > 1000:
            break
        
        bioc = None
        
        try:
            # Convert to PMID
            if isPmidDict:
                pmid = key
            else:
                pmid = get_pmid(key)
            bioc = get_pubtator_bioc_json(pmid)
            if bioc == '[]':
                bioc = None
        except:
            # Alternative way to get bioc
            # print(key)
            bioc = None
            pass

        if bioc is None:
            try:
                pmid = doi2pmid(doi)
                bioc = get_pubtator_bioc_json(pmid)
                if bioc == '[]':
                    bioc = None
            except:
                # Alternative way to get bioc
                # print(key)
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
    count = 0
    unk_dict = {}
    for key in dict:
        bioc = None
        converted = False
        biorxiv = True
        details = None
        
        count += 1

        if count > 100:
            break
        
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
                # print("no doi")
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
            # print("fail bio detail")
    
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
                # print("fail med detail")
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
                file_name = file_name.replace(".", "-")
                # Replace / in DOI with _
                file_name = file_name.replace("/", "_")
                output_file = '../data/processed/html/' + file_name + ".html"
                convert_jatsxml_to_html(jats_xml, output_file)
                bioc = "converting"
            except:
                bioc = None
                # print("fail jats")
        
        # Check if it is a preprint
        if bioc is None or bioc == '[]':
            unk_dict[key] = None
            continue
    
        dict[key] = bioc

    for key in unk_dict:
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
    file_name = file_name.replace(".", "-")
    # Replace / in DOI with _
    file_name = file_name.replace("/", "_")
    # file_name += ".pdf"

    return file_name


# # Sort through Pokay data files

# publication_bioc, publication_key, rxiv_bioc, rxiv_key, grey_bioc, grey_key = categorize_pokay('../data/raw/pokay')
# publication_bioc_2, publication_key_2, rxiv_bioc_2, rxiv_key_2, grey_bioc_2, grey_key_2 = recategorize_pokay('../data/raw/pokay')
publication_bioc, publication_key, rxiv_bioc, rxiv_key, grey_bioc, grey_key = recategorize_pokay('../data/raw/pokay')


# # Load local dictionary

# In[108]:


# with open('rxiv_bioc.txt') as file:
#         rxiv_bioc = json.loads(file.read())

# with open('rxiv_unk_bioc.txt') as file:
#         rxiv_unk_bioc = json.loads(file.read())

# with open('publication_bioc.txt') as file:
#         publication_bioc = json.loads(file.read())

# with open('publication_unk_bioc.txt') as file:
#         publication_unk_bioc = json.loads(file.read())

# with open('grey_bioc.txt') as file:
#         grey_bioc = json.loads(file.read())


# # Retrieve BioC JSON for Journal Publications (don't run)

# In[ ]:


# publication_bioc, publication_unk_bioc = get_journal_publication_bioc(publication_bioc)


# # Retrieve BioC JSON for preprints (don't run)

# In[ ]:


to_pop = []
for key in rxiv_bioc:
    if rxiv_bioc[key] is None:
        rxiv_unk_bioc[key] = None
        to_pop.append(key)

for p in to_pop:
    rxiv_bioc.pop(key, None)


# ### Manually convert some preprints that cannot be converted

# In[ ]:


# Preprint 1 
key = 'https://doi.org/10.1101/2021.02.03.429164v4'

# Not found manually search up online
# Special doi
doi = 'https://doi-org.ezproxy.lib.ucalgary.ca/10.1038/s41587-022-01382-3'

doi_part = doi.split('.ca/')[-1]

# Api link for paper details
api_link = 'https://www-ncbi-nlm-nih-gov.ezproxy.lib.ucalgary.ca/pmc/utils/idconv/v1.0/?tool=doi2pmid&email=david.yang1@ucalgary.ca&ids=' + doi_part
paper = requests.get(api_link)
soup = BeautifulSoup(paper.content, "xml")        
pmid = soup.find('record')['pmid']

bioc = get_pubtator_bioc_json(pmid)

rxiv_unk_bioc[key] = bioc


# In[ ]:


# Preprint 2
key = 'https://doi.org/10.21203/rs.3.rs-226857/v1'

# Special doi
doi = 'https://doi-org.ezproxy.lib.ucalgary.ca/10.1038/s41467-021-25167-5'

doi_part = doi.split('.ca/')[-1]

# Api link for paper details
api_link = 'https://www-ncbi-nlm-nih-gov.ezproxy.lib.ucalgary.ca/pmc/utils/idconv/v1.0/?tool=doi2pmid&email=david.yang1@ucalgary.ca&ids=' + doi_part
paper = requests.get(api_link)
soup = BeautifulSoup(paper.content, "xml")        
pmid = soup.find('record')['pmid']

bioc = get_pubtator_bioc_json(pmid)

rxiv_unk_bioc[key] = bioc


# In[ ]:


# Preprint 4
key = 'https://doi.org/10.1101/2021.12.24.21268382v1.full'

doi = 'https://doi.org/10.1101/2021.12.24.21268382'

details = get_rxiv_details(doi, False).decode('utf-8')
jats_xml = get_rxiv_jats_xml(details)
file_name = key.split('doi.org/')[-1]
# Replace . in DOI with -
file_name = file_name.replace(".", "-")
# Replace / in DOI with _
file_name = file_name.replace("/", "_")
output_file = '../data/processed/html/' + file_name + ".html"
convert_jatsxml_to_html(jats_xml, output_file)

rxiv_unk_bioc[key] = "converting"


# In[ ]:


# Preprint 5
key = 'https://doi.org/10.1101/2021.03.09.434607v9'

# Not found manually search up online
doi = 'https://doi.org/10.1101/2021.03.09.434607'

details = get_rxiv_details(doi, True).decode('utf-8')

jats_xml = get_rxiv_jats_xml(details)
file_name = key.split('doi.org/')[-1]
# Replace . in DOI with -
file_name = file_name.replace(".", "-")
# Replace / in DOI with _
file_name = file_name.replace("/", "_")
output_file = '../data/processed/html/' + file_name + ".html"
convert_jatsxml_to_html(jats_xml, output_file)

rxiv_unk_bioc[key] = "converting"


# In[ ]:


# Special case
key = "https://doi.org/10.1101/2021.03.24.436850"
details = get_rxiv_details(key, True).decode('utf-8')
jats_xml = get_rxiv_jats_xml(details)
file_name = key.split('doi.org/')[-1]
# Replace . in DOI with -
file_name = file_name.replace(".", "-")
# Replace / in DOI with _
file_name = file_name.replace("/", "_")
output_file = '../data/processed/html/' + file_name + ".html"
convert_jatsxml_to_html(jats_xml, output_file)

rxiv_unk_bioc[key] = "converting"

# print(file_name)


# # Retrieve BioC JSON for grey literature (don't run)

# In[ ]:


grey_bioc


# In[ ]:


# Grey 1
key = 'https://www.fda.gov/media/155050/download'
# Cannot convert (PDF)


# In[ ]:


# Grey 2
key = 'https://observablehq.com/@aglucaci/sc2-omicron'
# Cannot convert (web)


# In[ ]:


# Grey 3
key = 'https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1009243/Technical_Briefing_20.pdf'
# Cannot convert (PDF)


# In[ ]:


# Grey 4
key = 'https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/961042/S1095_NERVTAG_update_note_on_B.1.1.7_severity_20210211.pdf'
# Cannot convert (PDF)


# In[ ]:


# Grey 5
key = 'https://doi.org/10.47326/ocsat.dashboard.2021.1.0'
# Cannot convert (web)


# In[ ]:


# Grey 6
key = 'https://www.covid19genomics.dk/2021-05-08_data-overview.html#b1525'
# Cannot convert (web)


# In[ ]:


# Grey 7
key = 'https://cmmid.github.io/topics/covid19/reports/sa-novel-variant/2021_01_11_Transmissibility_and_severity_of_501Y_V2_in_SA.pdf'
# Cannot convert (PDF)


# In[ ]:


# Grey 8
key = 'https://drive.google.com/file/d/1CuxmNYj5cpIuxWXhjjVmuDqntxXwlfXQ/view'
# Cannot convert (PDF)


# In[ ]:


# Grey 9
key = 'https://doi.org/10.15585/mmwr.mm7034e4'
pmid = get_pmid(key)
bioc = get_pubtator_bioc_json(pmid)
grey_bioc[key] = bioc


# In[ ]:


# Grey 10
key = 'https://doi.org/10.15585/mmwr.mm7034e3'
pmid = get_pmid(key)
bioc = get_pubtator_bioc_json(pmid)
grey_bioc[key] = bioc


# In[ ]:


# Grey 11
key = 'https://www.fda.gov/media/146217/download'
# Cannot convert (PDF)


# In[ ]:


# Grey 12
key = 'https://doi.org/10.15585/mmwr.mm7017e2'
pmid = get_pmid(key)
bioc = get_pubtator_bioc_json(pmid)
grey_bioc[key] = bioc


# In[ ]:


# Grey 13
key = 'https://www.moh.gov.sg/news-highlights/details/3-new-cases-of-locally-transmitted-covid-19-infection-28apr2021-update'
# Cannot convert (PDF)


# In[ ]:


# Grey 14
key = 'https://mg.co.za/coronavirus-essentials/2021-03-24-single-dose-jj-janssen-covid-19-vaccine-hopes-to-speed-up-sas-vaccination-programme/'
# Cannot convert (web)


# In[ ]:


# Grey 15
key = 'https://github.com/cov-lineages/pango-designation/issues/4'
# Cannot convert (web)


# # Manual conversion

# ### Publication conversion

# In[111]:


cannot_convert_publication = {}

for key in publication_unk_bioc:
    if publication_unk_bioc[key] is None:
        cannot_convert_publication[key] = None


# In[112]:


# Fetch manual conversions: converted these by downloading pdf -> html -> bioc_json format
for key in cannot_convert_publication:
    file = get_file_name(key)
    file = "/home/david.yang1/autolit/viriation/data/processed/bioc_paper/" + file + "_bioc.json"
    try:
        # print(file)
        cannot_convert_publication[key] = Path(file).read_text().replace('\n', '')
        
    except:
        print(file)
        pass


# ### Rxiv conversion
# 
# Changing "converting" in dictionary to BioC

# In[113]:


# Fetch automated conversion from earlier for rxiv_bioc
for key in rxiv_bioc:
    if rxiv_bioc[key] == "converting":
        file = get_file_name(key)
        file = "/home/david.yang1/autolit/viriation/data/processed/bioc/" + file + "_bioc.json"
        try:
            rxiv_bioc[key] = Path(file).read_text().replace('\n', '')
        except:
            print(file)
            pass


# In[114]:


# Fetch automated conversion from earlier for rxiv_unk_bioc
for key in rxiv_unk_bioc:
    if rxiv_unk_bioc[key] == "converting":
        file = get_file_name(key)
        file = "/home/david.yang1/autolit/viriation/data/processed/bioc/" + file + "_bioc.json"
        try:
            rxiv_unk_bioc[key] = Path(file).read_text().replace('\n', '')
        except:
            print(file)
            print(key)
            pass


# **Convert "None" in dictionary to BioC**

# In[115]:


cannot_convert_rxiv = {}

for key in rxiv_unk_bioc:
    if rxiv_unk_bioc[key] is None:
        cannot_convert_rxiv[key] = None


# In[116]:


print(len(cannot_convert_rxiv))
print(sum(x is None for x in rxiv_unk_bioc.values()))
print(sum(x == 'converting' for x in rxiv_unk_bioc.values()))


# In[117]:


# Manually download all rxiv pdfs (Already done, so commented out)
# for key in cannot_convert_rxiv:
#     print(key)
#     print(get_file_name(key))
#     print(" ")


# In[118]:


# Fetch manual conversions. pdf -> html -> bioc json
for key in cannot_convert_rxiv:
    file = get_file_name(key)
    file = "/home/david.yang1/autolit/viriation/data/processed/bioc_other/" + file + "_bioc.json"
    try:
        # print(file)
        cannot_convert_rxiv[key] = Path(file).read_text().replace('\n', '')
    except:
        print(file)
        pass


# ### Grey conversion

# In[119]:


cannot_convert_grey = {}

for key in grey_bioc:
    if grey_bioc[key] is None:
        cannot_convert_grey[key] = None


# In[120]:


# Manually download all grey pdfs (Already done, so commented out)
# for key in cannot_convert_grey:
#     print(key)
#     print(get_file_name(key))
#     print(" ")


# In[121]:


# Fetch manual conversion
for key in cannot_convert_grey:
    file = get_file_name(key)
    file = "/home/david.yang1/autolit/viriation/data/processed/bioc_other/" + file + "_bioc.json"
    try:
        # print(file)
        cannot_convert_grey[key] = Path(file).read_text().replace('\n', '')
    except:
        print(file)
        pass


# # Combine dictionaries for final results

# In[122]:


# Update publications bioc dictionary
print("length of publication_bioc: " + str(len(publication_bioc)))
print("length of publication_unk_bioc: " + str(len(publication_unk_bioc)))
publication_unk_bioc.update(cannot_convert_publication)
publication_bioc.update(publication_unk_bioc)
print("length after: " + str(len(publication_bioc)))


# In[123]:


# publications
count = 0
# for key in publication
print(sum(x is None for x in publication_bioc.values()))

for key in publication_bioc:
    if publication_bioc[key] is None:
        print(key)


# In[124]:


print(sum(x is None for x in rxiv_bioc.values()))

for key in rxiv_bioc:
    if rxiv_bioc[key] is None:
        print(key)
    elif rxiv_bioc[key] == "converting":
        print(key)

print("https://virological.org/t/emergence-of-y453f-and-69-70hv-mutations-in-a-lymphoma-patient-with-long-term-covid-19/580" in rxiv_unk_bioc)
print("https://doi.org/10.21203/rs.3.rs-318392/v1" in rxiv_unk_bioc)


# In[125]:


for key in rxiv_unk_bioc:
    if rxiv_unk_bioc[key] is None:
        print(key)
    if rxiv_unk_bioc[key] == "converting":
        print(key)


# In[126]:


# Update rxiv bioc dictionary
print("length of rxiv_bioc: " + str(len(rxiv_bioc)))
print("length of rxiv_unk_bioc: " + str(len(rxiv_unk_bioc)))

rxiv_unk_bioc.update(cannot_convert_rxiv)
rxiv_bioc.update(rxiv_unk_bioc)
print("length after: " + str(len(rxiv_bioc)))


# In[127]:


sum(x is None for x in rxiv_bioc.values())


# In[128]:


# Update grey bioc dictionary
print("length of grey_bioc: " + str(len(grey_bioc)))
grey_bioc.update(cannot_convert_grey)
print("length after: " + str(len(grey_bioc)))


# In[129]:


sum(x is None for x in grey_bioc.values())


# In[130]:


sum(x in publication_bioc for x in rxiv_bioc)


# In[131]:


len(rxiv_bioc) + len(publication_bioc) + len(grey_bioc)


# In[132]:


# Combine all dictionaries together
data_bioc = copy.deepcopy(publication_bioc)
print(sum(x is None for x in data_bioc.values()))
# print(len(data_bioc))
data_bioc.update(rxiv_bioc)
# print(len(data_bioc))
data_bioc.update(grey_bioc)
print(len(data_bioc))

data_keys = copy.deepcopy(publication_key)
data_keys.update(rxiv_key)
data_keys.update(grey_key)
print(len(data_keys))


# In[133]:


sum(x is None for x in data_bioc.values())


# In[134]:


sum(x == "converting" for x in data_bioc.values())


# # Save dictionary locally

# In[137]:


# with open('../data/processed/pokay/rxiv_bioc.txt', 'w') as file:
#      file.write(json.dumps(rxiv_bioc))

# with open('../data/processed/pokay/rxiv_unk_bioc.txt', 'w') as file:
#      file.write(json.dumps(rxiv_unk_bioc))

# with open('../data/processed/pokay/publication_bioc.txt', 'w') as file:
#      file.write(json.dumps(publication_bioc))
    
# with open('../data/processed/pokay/publication_unk_bioc.txt', 'w') as file:
#      file.write(json.dumps(cannot_convert_publication))

# with open('../data/processed/pokay/grey_bioc.txt', 'w') as file:
#      file.write(json.dumps(grey_bioc))


# In[146]:


# with open('../data/processed/pokay/data_bioc.txt', 'w') as file:
#      file.write(json.dumps(data_bioc))

# with open('data_keys.txt', 'w') as file:
#      file.write(json.dumps(data_keys))


# # Autocorpus commands

# In[139]:


# python run_app.py -c "../../data/other/config_allen.json" -t "../../data/processed/bioc_paper/" -f "../../data/processed/html/10-1101_2021-03-24-436850" -o JSON

# python submodules/autocorpus/run_app.py -c "data/other/config_biorxiv.json" -t "data/processed/bioc/" -f "data/processed/html/" -o JSON

# python run_app.py -c "../../data/other/config_biorxiv.json" -t "../../data/processed/bioc/" -f "../../data/processed/html/" -o JSON

# python run_app.py -c "../../data/other/config_allen.json" -t "../../data/processed/bioc_paper/" -f "../../data/processed/html_papers/" -o JSON

