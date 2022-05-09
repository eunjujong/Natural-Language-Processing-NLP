# import libraries
import os 
import pandas as pd
import numpy as np
import re
import csv
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import coo_matrix
import logging


# returns a dictionary of words extracted from a url
def word_frequency(url):
    req = Request(url, headers={'User-Agent' : "Magic Browser"})
    page = urlopen(req) #timeout=5
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')

    tag_list = html_tags(soup) # generate a list of tags

    text_all = soup.get_text() # all text inside the soup object
    text_all = text_all.replace("\n", "") # remove line spacing 
    text_all = ''.join(text_all).split() # a list of words
    text_all = [re.sub('[^A-Za-z]+', '', word) for word in text_all] # remove special characters and numbers
    text_all = [word.lower() for word in text_all if word] # remove empty space and convert to lower case

    stop_words = set(nltk.corpus.stopwords.words("english")) # stop words
    lem = nltk.stem.wordnet.WordNetLemmatizer() # grouping the inflected forms of a word

    text_all = [lem.lemmatize(word) for word in text_all if not word in stop_words]

    text_dict = {} # words and frequencies stored in a dictionary
    for i in text_all:
        if i not in text_dict:
            text_dict[i] = 1
        else:
            text_dict[i] += 1

    return text_dict

# returns space-separated strings of words extracted from a url
def url_text(url):
    req = Request(url, headers={'User-Agent' : "Magic Browser"})
    page = urlopen(req) #timeout = 5
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')

    tag_list = html_tags(soup) # generate a list of tags

    text_all = soup.get_text() # all text inside the soup object
    text_all = text_all.replace("\n", "") # remove line spacing 
    text_all = ''.join(text_all).split() # a list of words
    text_all = [re.sub('[^A-Za-z]+', '', word) for word in text_all] # remove special characters and numbers
    text_all = [word.lower() for word in text_all if word] # remove empty space and convert to lower case

    stop_words = set(nltk.corpus.stopwords.words("english")) # stop words
    lem = nltk.stem.wordnet.WordNetLemmatizer() # grouping the inflected forms of a word

    text_all = [lem.lemmatize(word) for word in text_all if not word in stop_words]
    trext_all = ' '.join(text_all)

    return text_all

# returns a list of html tags to be excluded from text extraction
def html_tags(soup_obj):
    tags = []
    for tag in soup_obj.find_all(True):
        tag_name = tag.name
        if tag_name not in tags:
            tags.append(tag.name)
    return tags
 
# returns a list containing TF-IDF scores for words in each sector
def get_tfidf(corpus, stop_words):
    # calculate tf_idf scores using CountVectorizer and TfidfTransformer functions
    vector_count = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=1000, ngram_range=(1,3))
    vocab = vector_count.fit_transform(corpus)
    vector_names = vector_count.get_feature_names()

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(vocab)

    # for each sector, generate a set of words and their tf_idf scores
    results = []
    for doc in corpus:
        tf_idf_vector = tfidf_transformer.transform(vector_count.transform([doc]))
        matrix_coo = tf_idf_vector.tocoo() # get matrix coordinates
        tuples = zip(matrix_coo.col, matrix_coo.data) 
        sorted_tuples = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True) # sorted tf-idf in descending order

        vals = []
        words = []
        for i, val in sorted_tuples:
            vals.append(round(val, 4))
            words.append(vector_names[i])

        results.append(list(zip(words, vals)))

    return results

# returns the sum of idf's from matching keywords by sector
def match_idf_sum(idf_list, text):
    results = []
    for item in idf_list:
        total = 0
        keywords = [i[0] for i in item]
        for word in text:
            if word in keywords:
                index = [i[0] for i in item].index(word) # index of the matching word
                total += item[index][1]
        results.append(total)

    return results

# returns the classification of the data urls based on the idf scores of the sectors
# urls classified based on the highest tf-idf scores
def url_classify(sectors, scores):
    results = []
    for item in scores:
        if len(set(item)) == 1: # if url with error
            results.append(sectors[-1])
        else:
            max_val = max(item)
            max_index = item.index(max_val)
            sec_class = sectors[max_index]
            results.append(sec_class)
    
    return results

def save_file(data, labels, file_name):
    data["Sector"] = labels
    name = file_name + '_New.xlsx'
    data.to_excel(name, index = False) # write DataFrame to excel
    
def main():
    # data info
    data_file = "Data Science Project.xlsx"
    sheet_name = "Company Data"
    
    # directory where data is stored
    dir_path = os.path.dirname(os.path.realpath(data_file))
    # load data
    data = pd.read_excel(dir_path+"/"+data_file, sheet_name=sheet_name)
    
    print("Data Loaded")
    
    # sectors
    sectors = ["Healthcare", "Education", "Environment", "Other Sector"]
    # URLs by sector
    health_web = ["https://www.nlm.nih.gov/", 
              "https://health.gov/", 
              "https://www.nachc.org/"]
    ed_web = ["https://ies.ed.gov/ncee/projects/nle/", 
              "https://obamaadministration.archives.performance.gov/agency/department-education.html", 
              "https://www.si.edu/"]
    env_web = ["https://www.nal.usda.gov/", 
               "https://www.epa.gov/", 
               "https://www.state.gov/policy-issues/climate-and-environment/"]
    general_news = ["https://www.nytimes.com/", 
                    "https://www.washingtonpost.com/", 
                    "https://www.cnn.com/"]
    
    #nltk.download() # downlaod nltk packages
    stop_words = set(nltk.corpus.stopwords.words("english")) # stop words
    
    sector_urls = health_web+ed_web+env_web+general_news
    corpus = []
    sub_corpus = []
    i = 0 # keeps track of urls in each sector
    for url in sector_urls:
        i += 1
        text = url_text(url) # get space-seprataed strings of words
        sub_corpus += text
        if i == 3:      
            corpus.append(' '.join(sub_corpus))
            sub_corpus = []
            i = 0
    
    print("Calculating Sector TF-IDF...")
    
    tf_idf_list = get_tfidf(corpus, stop_words) # tuples of words and their tf-idf scores for three sectors
    
    #avg_tfidf_by_sector = [] # avg tf-idf by sectors as classification thresholds
    #for doc in tf_idf_list:
        #vals = [i[1] for i in doc]
        #avg = sum(vals) / len(vals)
        #avg_tfidf_by_sector.append(avg)
    
    print("Calculating idf scores of the Company URLs...")

    # for each company url, classify them into one of the sectors based on 
    # IDF score of 10 most frequent words
    scores = [] # stores idf sectors of data urls
    for url in data.iloc[:,1]:
        try:
            text = url_text(url) # get space-seprataed strings of words
            
        except Exception as e:
            # log the full error message
            #logging.exception(e) 
            scores.append([0,0,0]) # zero idf scores if error occurs while opening a url
            continue
        else:
            idf_scores = match_idf_sum(tf_idf_list, text)
            scores.append(idf_scores)
    
    print("Classifying URLs...")
    
    labels = url_classify(sectors, scores) # classify urls based on max idf scores
    save_file(data, labels, sheet_name) # save a new csv file with sector labels
    
    print("New CSV File with Classified Labels Saved")
    
    
    
if __name__ == "__main__":
    main()
    
