#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:07:47 2020

This file will allow to import and analyze via Dynamic topic modelling, the abstract and files created via
CC_DataPrep.py. 
 
The papers are all part of the Web of Science repository and where selected with the following key-word search in all fields:
    "Climate Change" OR "Global Warming" OR "Climatic Change" OR "Climatic Changes"

@author: Jacopo Baggio

Some code was heavily based on gensim tutorials and nltk tutorials 

"""

#packages to change directory and load datta from Meta_DataLoad_andPrep.py
import os
import pickle
import csv

#usual suspects
import pandas as pd
import numpy as np

#packages for figures and graphs
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS

#Import for natural language processing and topic modeelling
import re
#import nltk and nltk sub-models
import nltk
from nltk import FreqDist
nltk.download('punkt')
#import spacy and load spacy pre-trained model on englis core web for lemmatization. 
import spacy
#sp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
sp1 = spacy.load('en_core_web_lg') 

#import gensim and submodels from gensim
import gensim
from gensim import corpora
from gensim import models
from gensim.models import CoherenceModel
from gensim.models import ldaseqmodel #for Dhynamic topic analysis we take the whole corpus

#import submodels from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


#import sentence-encoder for using BERT and/or RoBERTa 
#both cpould be word-embeddings used to calculate documnt similarity btw docs.
from sentence_transformers import SentenceTransformer
#load pre-trained bert model: this model was specifically trained for semantic similarity (stsb) or natural language in general
embedsemantic = SentenceTransformer ('bert-base-nli-stsb-mean-tokens')
embedsemanticr = SentenceTransformer ('roberta-large-nli-stsb-mean-tokens')


#Preprocessing and cleaning again to bettere understand topics and words
#remove stopwords added such as climate change and global warming as they are part of the search 
#and each paper should refer to them and further words that do not define much. Also in specific words
#there are words added that were raised as issue in previous runs of the text cleaning. Some relate to full tokens
#some to stem tokens. The words added are also to avoide the inconsistency exception that was prevoiusly raised.
stopwords = nltk.corpus.stopwords.words('english')
specificwords =['climate','change', 'climate-change', 'global-warming','global warming','use', 'potential','potent', 'article','articles',
                'studi','studying','studies', 'study', 'analysis','anal','analy','analyzing','results','result','response','responses','respons',
                'large','small','paper', 'papers','possible', 'possibilty', 'ie','eg', 'article'
                'rates','rate','rating','may','could','should', 'would','high','low','let', 'must', 
                "'d", "'m", "'s", 'ca', 'els','henc', 'howev', 'let', 'must', "n't", 'otherwis','sha', 'sinc', 
                'therefor', 'wo','nan','-', 'ltd','elsevier', '-PRON-','et','al']
bistop =['climate_change','global_warming','climatic_change', 'global_warm','global_warme']

#extend stopword list to include specific words.
stopwords.extend(specificwords)
stopwords.extend(list(STOPWORDS))
#now remove duplicates in the list.
stopwords = list(set(stopwords))

def remstop(text):
    #function that re-performs stopwords elimination
    #return [word for word in text if word not in stopwords]
   return [[word for word in doc if word not in stopwords]for doc in text]


def rembigrams(text):
    #function thatperfroms eliminatin of common bigrams
    return [[word for word in doc if word not in bistop]for doc in text]

def stw(text):
    #function that tokenizes a text
    for sentence in text:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=False, min_len=2))  # deacc=True removes punctuations

#two ways of lemmatizing, however, Co2 and other important 
#words are not retained with if we restrict the tags to noung, adj, verb and adverbs.
def lemmatize(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    tout = []
    for word in text:
        doc = sp1(" ".join(word)) 
        tout.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return tout
    
def lemma_all(text):
    tok = []
    for word in text:
        tokprep = sp1(' '.join(word))
        tok.append([token.lemma_ for token in tokprep])
    return tok

#Change directory and start working on the actual data.
   
wdir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis/Data')
os.chdir(wdir)
#import files from CC_DataPrep.py using pickle
#rcall = the whole record collection
#rcy = collection divideed by yeaer and in dict format
#rtop = rcy but with initial cleaning for common stopwords, whitespaces, numbers and all lower case

rcall = pickle.load(open('fullcollection.p', 'rb'))
rcy   = pickle.load(open('yearlycollection.p', 'rb'))
rtop  = pickle.load(open('yearlytopic.p', 'rb'))

#import category classification csv file in which we map WoS research areas to macro-research areas modified
#once imported we use a dictionary to substitote WoS research areas with the modified macro-research areas
with open('CatClassDict.csv',newline='') as f:
    reader = csv.reader(f)
    next(reader)
    ccdict = dict(reader)

#set general context and colors for figures and graphs
hfont = {'fontname':'Times'}
sns.set(context='paper', style='white', palette='colorblind', font_scale=3)

#General parameters
#Use vectorization of tokens to assess cosine similarity between papers based on word tokenization.
#define vectorizer parameters
#max_df = frequency of word affter which it carries little meaning as it is almost always present
#min_df = frequency of word in documents to be considered
#i use the same stopwords already used
def identityTok(text):
   return text
tfidveclem  = TfidfVectorizer(analyzer='word',tokenizer=identityTok, preprocessor=identityTok, token_pattern=None,
                              max_df=0.8, max_features=200000, min_df=0.1, stop_words=None, use_idf=True)
#General K-Means cluster parameters#Number of clusters 
#for clustering papers based on simiilarity, eithre stemmed or lemmatized words
num_clusters = 30
km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)

#Number of topics to look for using LDA topic modelling
#and number of words that will be reported for each topic in stored results
# we use 30 topics for each, as that would make it easier to compare. 
#However, it is known that the number of topcis is tricky to assess.
#Also, we use 20 words per topic to facilitate topic nuances and interpretation.
ntopics = 30
nwords = 20

#result storing:
avgcslem = {} # where to store average similarity per 3year time-span based on Tf-Idf matrix based on lemming
lemlda  = {} #where to store LDA topic modelling results based on lemmatized words
nodeprop = {} #dictionary tof dataframes to be passed on the network analysis in which we define clusters based on TF-IDF and LDA and LSI for both, stemmed and lemmatized text
wordfreq = {} #dictionary to store the 100 most common words per time-period
bertprop = {} #dictionary for potentially using bert or roberta
coherence = {} #dictionary to store coherence valus
topicvecs = {} #dictionary to store vectors to calculate hellinger (not for DTM)

#Figure specific dictionaries 
cloudim = {} #dictionary to store word cloud data for figure
tsneim = {} #dictionary to store word research area colored tsne data for figures
wordim = {} #dictionary to store top word bar graph data for figure
resim = {} #dictionary to store research area bar graph data for figure


#Analyze the 3year blocks. no Rolling window is a choice to avoid over-counting topics. we think that 3 year interval
#is ok. 3 years is also the benchmark for impact factor, but we admit that the choice is somehwat arbitrary.
for key in rtop:
    idx = str(key)
    print(idx)
    tempfile = rtop[key]
    #create dataframee, sort by year and re-index the sorted datafram
    dft = pd.DataFrame(tempfile)
    dft = dft.sort_values(['PY'])    
    dft = dft.reset_index(drop=True)
    #eliminate non alphanumeric charactrers from keywords
    keyw = dft.ID.str.replace('[^a-zA-Z]', ' ')
    dft['keyw'] = keyw
    #use title and keywords as added information to the abstracts
    #clean the list from being nan. Here we add keywords and title to abstracts. In some cases
    #some papers may do not report abstract but do so with title and keywords, somtimes they do not hav keeywords tc.. 
    #overall, title, keywords and abstract contain the main topic informations we need, hence we analyze them jointly
    dft['ab2'] = dft['AB'].astype(str)+' '+dft['TI'].astype(str)+' '+dft['ID'].astype(str)
    #clean ab2  first by eliminating characters that are not alphanumeric, then by lowering the case
    dft['abclean'] = dft['ab2'].map(lambda x: re.sub(r'_+|[^\w-]+', ' ', x))
    dft['abclean'] = dft['abclean'].map(lambda x: x.lower())
    dft['abclean'] = dft['abclean'].map(lambda x: re.sub(r'sea level rise', 'sea_level_rise', x))
    dft['abclean'] = dft['abclean'].map(lambda x: re.sub(r' c ', 'carbon', x))
    dft['abclean'] = dft['abclean'].map(lambda x: re.sub(r' n p z ', 'npz', x))
    cleanabst = dft.abclean.values.tolist()
    
    templem2 = list(stw(cleanabst))
    #make sure that C = carbon and CO = Co2 as it is better for comprehension and calculations.
    templem2 = [['co2'if word =='co' else word for word in doc]for doc in templem2]

    #templem2 = [['co2'if word =='co' else word for word in doc]for doc in templem2]
    templem3 = remstop(templem2)
    #Build the bigram model. Here we tuned the threshold to include some clear bigram in the 
    bigram = gensim.models.Phrases(templem2, min_count=5, threshold=50)
    bimodel = gensim.models.phrases.Phraser(bigram)
    #make bigrams based on the bigram model defined above
    totbig = [bimodel[doc] for doc in templem3]
    #lemmatize avoiding anuything that is not a noun, adjective, verb or adverb.
    totlem = lemma_all(totbig)
    #remove lemmas 
    totlem = rembigrams(totlem)
    totlem = remstop(totlem)
    #reduce dataframe size.
    dft =  dft[['AB','AF', 'AU', 'ID', 'PY', 'TI', 'UT', 'SC','keyw', 'abclean']]
    print('Finished preprocessing')

    #Generate a WordCloud!
    wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='black',
                          collocations=False, colormap='tab10')
    #generate a word cloud on lemmatized words
    toklem = []
    for w in totlem:
        toklem.extend(w)
    fdist = FreqDist(toklem)
    cloud = wordcloud.generate_from_frequencies(fdist)
    cloudim[key] = cloud
    
    #get top 30 words and assess frequencies
    k   = 100 #overall words we want in th list... the first N words
    ttk = 30 # top N we want to put in the histogram
    k_words = fdist.most_common(k)
    topkw = k_words[0:ttk]
    dftop = pd.DataFrame(topkw, columns=['words','freq'])
    dftop['frac']= dftop['freq'] / len(toklem)    
    wordim [key] = dftop 
    
    #store the most common 100 words per time-period
    wordfreq[key] = k_words  
    print ('Finished Wordclouds')

    #Calculate TF-IDF on lemmatized words
    tfidmatlem  = tfidveclem.fit_transform(list(totlem))
    #Cosine similarity 
    simillem  = cosine_similarity(tfidmatlem)
    #Calculate average similarity per time period
    avgsimillem  = np.mean(simillem)
    #Store average similarity scores
    avgcslem[key] = avgsimillem
    #Clustering of papers based on TF-IDF on stemmed and lemmatized matrix.
    #first convert similarity to distance:
    distlem  = 1 - simillem
    print('Cosine Distance Matrices  on TF-IDF')
    
    #generate colors according to WC categories
    cat = dft.SC
    cat = cat.fillna('Not Available')
    colcat = []
    for i in cat:
        ccat = [ccdict.get(item,item) for item in i]
        ccat = list(set(ccat))
        ccat.sort()
        ccat = str(ccat)
        #eliminatre non alphamnumric characters nand th word Sciences and then substitute for 3letter codes
        ccat = re.sub(r'_+|[^\w-]+',r'', ccat)
        ccat = re.sub(r'Sciences',r'', ccat)
        ccat = re.sub(r'EconomicsandManagement',r'ECm', ccat)
        ccat = re.sub(r'ArtsHumanities', 'AHu', ccat)
        ccat = re.sub(r'Social',r'SSc', ccat)
        ccat = re.sub(r'Natural',r'NSc', ccat)
        ccat = re.sub(r'Health',r'HHs', ccat)
        ccat = re.sub(r'Technology',r'TEn', ccat)
        ccat = re.sub(r'Physical',r'PYs', ccat)
        colcat.append(ccat)
    dft['macroRA'] = colcat
    #now calculate the most frequnt topics, used to color the t-SNE plots below.
    radist = FreqDist(dft.macroRA)
    dft['freqRA']=dft.macroRA.map(radist)
    dft['rankRA']=dft.freqRA.rank(method='dense', ascending=False)
    #now assess distribution of topic and topic combinations per time-period. Use the first 20 words
    ra1 = radist.most_common(20)
    dfra = pd.DataFrame(ra1, columns=['resarea','freq'])
    dfra['frac']= dfra['freq'] / len(dft.macroRA)
    resim[key] = dfra
    print('Finished plotting, Re-naming and classifying Macro RA')
    
    #Do t-SNE embedding for reducing dimensionality on the distance matrix from TF-IDF
    #for clusters / documents visualization -> takes a long time, about 4 to 6 hours for the last two time-periods
    dft['tocolor'] = np.where(dft['rankRA'] < 9, dft['macroRA'], 'Other')
    #use only the first 10 areas to color for the T-SNE graph
    tocolor = dft.tocolor.to_list() 
    pos = TSNE().fit_transform(distlem)  
    # TSNE with 2 components based on the distance matrix
    xs, ys = pos[:, 0], pos[:, 1]
    print('Finished tSNE calculations')
    #Use Macro Research Areas as colors,save results for plotting later
    df1 = pd.DataFrame(dict(x=xs, y=ys, label=tocolor)) 
    groups = df1.groupby('label')
    tsneim[key] = groups
    print('Finished Word Data, prevalence of MacroAras and T-SNE')

"""
Figurs: Wordclouds, Word proportion in bar graph, Research area proportion in bar graph, and t-SNE visualization
of documents based on TF-IDF and colored based on research areas
"""
wdir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis/Figures')
os.chdir(wdir)

#set general context and colors for figures and graphs
hfont = {'fontname':'Times'}
sns.set(context='paper', style='white', palette='colorblind', font_scale=2)

#Figure to assess topic evolution every 3 year avg without the same scaling (better for visualizing trends)
fig, ax = plt.subplots(figsize = (30, 20), sharey=True, sharex=True)
i = 1
for key in cloudim:
    if i == 9:
        i = 10
    plt.subplot(3, 4, i)
    plt.imshow(cloudim[key]) 
    plt.title(key)
    plt.axis ('off')
    i += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)
fig.savefig('wordclouds.pdf')

#Figure for top words 
fbar, ax = plt.subplots(figsize = (40, 30), sharey=True, sharex=True)
i = 1
for key in wordim:
    if i == 9:
        i == 10
    dftop = wordim[key]
    plt.subplot(3, 4, i)
    sns.barplot(dftop.words, dftop.frac, palette='Blues_r')
    plt.title(key, fontsize=30, color ='black')
    plt.xticks(rotation=90, ha='center')
    plt.tick_params(axis='x', which='major', labelsize=28)
    plt.ylabel('Fraction of Words', color = 'black', fontsize = 36)
    plt.xlabel('Lemmas', color = 'black', fontsize = 36)
    i+=1
fbar.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)
fbar.savefig('topwords.pdf')
              
#Figure for top research areas
fres, ax = plt.subplots(figsize = (40, 30), sharey=True, sharex=True)
i = 1
for key in resim:
    if i == 9:
        i == 10
    dfra = resim[key]
    plt.subplot(3, 4, i)
    sns.barplot(dfra.resarea, dfra.frac, palette='Blues_r')
    plt.title(key, fontsize=30, color ='black')
    plt.xticks(rotation=90, ha='center')
    plt.tick_params(axis='x', which='major', labelsize=32)
    plt.ylabel('Fraction of Documents', color = 'black', fontsize = 36)
    plt.xlabel('Research Area', color = 'black', fontsize = 36)
    i+=1
fres.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)
fres.savefig('topresarea.pdf')


#Figure for t-SNE clustering and coloring based on research area           
fsne, ax = plt.subplots(figsize = (50, 30), sharey=True, sharex=True)
i = 1
for key in tsneim:
    if i == 8:
        i == 9
    groups = tsneim[key]
    plt.subplot(3, 4, i)
#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        plt.scatter(group.x, group.y, s=5, label=name)
        plt.ylabel('t-SNE 1', color = 'black', fontsize = 36)
        plt.xlabel('t-SNE 2', color = 'black', fontsize = 36)
        plt.box(False)        
        plt.legend(ncol=1, loc='center right', labelspacing=0.3, bbox_to_anchor=(1.3, 0.4), frameon=False, 
            markerscale=5, fontsize = 24)
    i+=1
fsne.tight_layout(pad=0.4, w_pad=1, h_pad=0)
fsne.savefig('tsne.pdf')

   
"""
DYNAMIC TOPIC MODELLING

"""
#Now, on the whole corpus assess topic evolution via DTM (Dynamic Topic Modelling).
#Preprocess the data as we have done for TF-IDF, word frequencis and LDA topic modelliing.
dft = pd.DataFrame(rcall)
dft = dft.sort_values(['PY'])    
dft = dft.reset_index(drop=True)
keyw = dft.ID.str.replace('[^a-zA-Z]', ' ')
dft['keyw'] = keyw
dft['ab2'] = dft['AB'].astype(str)+' '+dft['TI'].astype(str)+' '+dft['ID'].astype(str)
dft['abclean'] = dft['ab2'].map(lambda x: re.sub(r'_+|[^\w-]+', ' ', x))
dft['abclean'] = dft['abclean'].map(lambda x: x.lower())
dft['abclean'] = dft['abclean'].map(lambda x: re.sub(r'sea level rise', 'sea_level_rise', x))
dft['abclean'] = dft['abclean'].map(lambda x: re.sub(r' c ', 'carbon', x))
dft['abclean'] = dft['abclean'].map(lambda x: re.sub(r' n p z ', 'npz', x))
cleanabst = dft.abclean.values.tolist()
templem2 = list(stw(cleanabst))
templem2 = [['co2'if word =='co' else word for word in doc]for doc in templem2]
templem3 = remstop(templem2)
bigram = gensim.models.Phrases(templem2, min_count=5, threshold=50)
bimodel = gensim.models.phrases.Phraser(bigram)
totbig = [bimodel[doc] for doc in templem3]
totlem = lemma_all(totbig)
totlem = rembigrams(totlem)
totlem = remstop(totlem)
dft['tottxt'] = totlem
#reduce number dataframe size
dft =  dft[['AB','AF', 'AU', 'ID', 'PY', 'TI', 'UT', 'keyw', 'abclean','tottxt']]
print('Finished preprocessing')

#Now check for topic modelling with LDA to calculate baseline to be used in DTM
#create a Gensim dictionary from the lemmatized words 
dictlem = corpora.Dictionary(totlem)
#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
#for the dynamic topic modelling, we do not want words that appear 80% or above in the overall corpus
dictlem.filter_extremes(no_below=15, no_above=1, keep_n=None)    #convert the dictionary to a bag of words corpus for reference
corpuslem = [dictlem.doc2bow(tx) for tx in totlem]  
#do lda model on all to have a baseline that is easier than the gensim one for the dynamic ttopic modelling.
ldaall = models.LdaModel(corpuslem, num_topics=30, 
                    id2word=dictlem, 
                    alpha=0.01, eta=0.005,
                    update_every=10, 
                    chunksize= 6000, 
                    passes=100)

pickle.dump(ldaall, open('ldabaseline.p', 'wb')) #in case something happens one can restart loading ldaall directly
#calculate base coherence to assess the validity of the modl, if it is too low, then rerun ldaall
coherenceModel= CoherenceModel(model=ldaall, texts=totlem, dictionary=dictlem, coherence='c_v')
clda = coherenceModel.get_coherence() #0.548  

#Get number of documents per time-slice, niumber of docs per 3 year period
tslice = []
for key in rcy:
    a = len(rcy[key])
    tslice.append(a)

#modified if old_bound == 0  to old_bound = 0.00000001 for convergence issues
ldaseq3 = ldaseqmodel.LdaSeqModel(corpus=corpuslem, id2word=dictlem, time_slice=tslice, num_topics=ntopics, chunksize = 6000, alphas=0.01)

#dump (save) model result and corpus
ldaseq3.save('ldadyn')
pickle.dump(ldaseq3, open('dynlda3.p', 'wb'))
pickle.dump(dft, open('nodedyntop.p', 'wb'))
pickle.dump(corpuslem, open('corpuslem.p', 'wb'))
pickle.dump(dictlem, open('dictlem.p', 'wb'))

topevo = []
for i in range(0,29):
    top1 = ldaseq3.print_topic_times(topic = i)
    topevo.append(top1)

timetop = []
for i in range(0,10):
    tim1 = ldaseq3.print_topics(time = i)
    timetop.append(tim1)

doctop = []
for i in range(0,len(dft)):
    dc1 = ldaseq3.doc_topics(i)
    doctop.append(dc1)
    
pickle.dump(topevo, open('topevo.p', 'wb'))
pickle.dump(timetop, open('timetop.p', 'wb'))
pickle.dump(doctop, open('docdyntop.p', 'wb'))
    
