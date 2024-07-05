#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install PyPDF2
# !pip install textract


# In[1]:


import PyPDF2
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[2]:


filename = '../data/raw/pdf/unconverted/10-21203_rs-3-rs-318392_v1.pdf'
open_filename = open(filename, 'rb')

file = PyPDF2.PdfReader(open_filename)


# In[3]:


file.metadata


# In[4]:


total_pages = len(file.pages)
total_pages


# In[5]:


import textract


# In[6]:


count = 0
text  = ''

# Lets loop through, to read each page from the pdf file
while(count < total_pages):
    # Get the specified number of pages in the document
    file_page  = file.pages[count]
    # Process the next page
    count += 1
    # Extract the text from the page
    text += file_page.extract_text()


# In[7]:


if text != '':
    text = text
    
else:
    textract.process(open_filename, method='tesseract', encoding='utf-8', langauge='eng' )    


# In[8]:


text


# In[26]:


# !pip install autocorrect


# In[28]:


# !pip install nltk


# In[33]:


from autocorrect import Speller
from nltk.tokenize import word_tokenize
import nltk


# In[44]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[37]:


def to_lower(text):

    """
    Converting text to lower case as in, converting "Hello" to  "hello" or "HELLO" to "hello".
    """
    
    # Specll check the words
    spell  = Speller(lang='en')
    
    texts = spell(text)
    
    return ' '.join([w.lower() for w in word_tokenize(text)])


# In[ ]:


lower_case = to_lower(text)
# print(lower_case)


# In[39]:


import nltk
import re
import string
from nltk.corpus import stopwords, brown
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from autocorrect import spell


# In[40]:


def clean_text(lower_case):
    # split text phrases into words
    words  = nltk.word_tokenize(lower_case)
    
    
    # Create a list of all the punctuations we wish to remove
    punctuations = ['.', ',', '/', '!', '?', ';', ':', '(',')', '[',']', '-', '_', '%']
    
    # Remove all the special characters
    punctuations = re.sub(r'\W', ' ', str(lower_case))
    
    # Initialize the stopwords variable, which is a list of words ('and', 'the', 'i', 'yourself', 'is') that do not hold much values as key words
    stop_words  = stopwords.words('english')
    
    # Getting rid of all the words that contain numbers in them
    w_num = re.sub('\w*\d\w*', '', lower_case).strip()
    
    # remove all single characters
    lower_case = re.sub(r'\s+[a-zA-Z]\s+', ' ', lower_case)
    
    # Substituting multiple spaces with single space
    lower_case = re.sub(r'\s+', ' ', lower_case, flags=re.I)
    
    # Removing prefixed 'b'
    lower_case = re.sub(r'^b\s+', '', lower_case)
    
    
    
    # Removing non-english characters
    lower_case = re.sub(r'^b\s+', '', lower_case)
    
    # Return keywords which are not in stop words 
    keywords = [word for word in words if not word in stop_words  and word in punctuations and  word in w_num]
    
    return keywords


# In[ ]:


# Lemmatize the words
wordnet_lemmatizer = WordNetLemmatizer()

lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in clean_text(lower_case)]

# lets print out the output from our function above and see how the data looks like
clean_data = ' '.join(lemmatized_word)
# print(clean_data)


# In[ ]:




