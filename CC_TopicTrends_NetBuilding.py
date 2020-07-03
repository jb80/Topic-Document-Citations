#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:07:47 2020

This script does 2 main things:
    1) Trend analysis of topics
    2) Builds the necessary networks for the integrated topic-doc-citation analysis

This script should be used after CC_Topic_Bigram_Analysis.py. 
The final network format is a dataframe. ids are also preserved and changed to integer.


@author: Jacopo Baggio
"""

#Import usual suspects:
#packages to change directory and load datta from Meta_DataLoad_andPrep.py, and clean up
import os
import pickle

#packages for figures and graphs
import seaborn as sns
import matplotlib.pyplot as plt

#usual suspects
import pandas as pd
import numpy as np
import math
import networkx as nx
import itertools

#import gensim module to assess similarity between topics (using hellinger or cosine distance/similarity)
from gensim.matutils import hellinger
from gensim.matutils import cossim
from gensim.models.coherencemodel import CoherenceModel
from gensim import models


#bipartite community detection algorithm based LPAwb+Dirt by Becket in R
from rpy2.robjects.packages import importr
#import numpy conversion to R object
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
base = importr('base')
rlpa = importr('bipartite')

 
# we create now a function to mak sure that what is returned for indexing unique values is sequential
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
#function to make sure that nodes that do not have WoS or other type of id are still identified
def dict_clean(items):
    result = {}
    for key, value in items:
        if value is None:
            value = key
        result[key] = value
    return result

#calculate gini coefficient for network degree
def gini_coeff(x):
    # requires all values in x to be zero or positive numbers,
    # otherwise results are undefined
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x)) # calculates zero-based ranks
    return 1 - (2.0 * (r*x).sum() + s)/(n*s)


#change directory to where the analysis and files genrated from CC_topic_Bigram_Analysis.py 
#i changed where i store topics, but can be all in the same directory.
ndir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis/Networks')
tdir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis/NLP')
ddir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis/Data')
wdir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis')

#load all data needed for further analysis 
#load the whole collection
os.chdir(ddir)
rcall = pickle.load(open('fullcollection.p', 'rb'))
#load network evoolution in 3 year period intervals and general dataframe in 3 year interval
os.chdir(ndir)
#use citation or co-citation.co-citatoin betteer for simplicial complexes
rnet = pickle.load(open('yearlycitenet.p', 'rb'))

#load topics per 3 year interval
os.chdir(tdir)
nodetopic = pickle.load(open('nodeTopics.p', 'rb'))
cv = pickle.load(open('topicCoherence.p', 'rb'))
topics = pickle.load(open('topicslemmed.p', 'rb'))
tvecs = pickle.load(open('topicVectors.p', 'rb'))
#load file for TF-IDF cosine similarity
avgcslem = pickle.load(open('avglemmedcluster.p', 'rb'))

#load file for dynamic topic analysis (DTM) 
#load the model
dyntop = pickle.load(open('dynlda3.p','rb'))
ldadyn   = models.wrappers.dtmmodel.DtmModel.load('ldadyn')
#load the overall corpus and dictioinary used for DTM
corpuslem = pickle.load(open('corpuslem.p','rb'))
dictlem = pickle.load(open('dictlem.p','rb'))
#load the dataframe that will be used for assessing topics and citation network
nodetop = pickle.load(open('nodedyntop.p', 'rb'))

#change to analysis director
os.chdir(wdir)

#set general figure context
hfont = {'fontname':'Times'}
sns.set(context='paper', style='white', palette='colorblind', font_scale=3)

"""
TOPIC MODELLING ANALYSIS FOR EACH TIME-SLICE

#first lets build topics networsk or better the square matrix between topics whose entries
#represent the hellinger similarity (1-hellinger distance) between them. 
#This is based on each topic being defined by a word vector and probab iltiy of each word
#being part of a topic. 

"""
#function to calculate hellinger distance between two probability distributions
def myhelling(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / (np.sqrt(2))

#topic "networks" are devised for each time-step.
topichel = {}
topiccos = {}
avghelsim = {}
avgcossim = {}

for key in tvecs:
    temp1 = tvecs[key]
    #assign name to topics...     
    maxind = len(temp1)
    tmat = np.zeros ((maxind, maxind))
    tmat2 = np.zeros ((maxind, maxind))
    for i in range (0,maxind):
        for j in range (0,maxind):
            tmat[i,j] = 1 - hellinger(temp1[i],temp1[j])
            tmat2[i,j] = cossim(temp1[i],temp1[j])
    avghelsim[key] = np.mean(tmat)            
    topichel[key] = tmat
    avgcossim [key]= np.mean(tmat2)
    topiccos[key] = tmat2

#Generate Bar graph of avg hellinger and cosine topic similarity and coherence
m1 = len(avghelsim)
reind = np.arange(1,2,1)
dfcs = pd.DataFrame(avghelsim, index=reind).T
dfcs.columns = ['hsim']
fbar, (ax) = plt.subplots(1,1, sharey=False, sharex=False, figsize=(10,10))
sns.barplot(dfcs.index, dfcs.hsim, palette='Blues', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, ha='center')
ax.set_ylabel('Avg Hellinger Similarity', color = 'black', fontsize = 30)
fbar.tight_layout()
fbar.savefig('AvgTopicHsim.pdf')

m1 = len(avgcossim)
reind = np.arange(1,2,1)
dfcs = pd.DataFrame(avgcossim, index=reind).T
dfcs.columns = ['csim']
fbar, (ax) = plt.subplots(1,1, sharey=False, sharex=False, figsize=(10,10))
sns.barplot(dfcs.index, dfcs.csim, palette='Blues', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, ha='center')
ax.set_ylabel('Avg Cosine Similarity', color = 'black', fontsize = 30)
fbar.tight_layout()
fbar.savefig('AvgTopicCsim.pdf')

m1 = len(cv)
reind = np.arange(1,2,1)
dfcs = pd.DataFrame(cv, index=reind).T
dfcs.columns = ['coherence']
fbar, (ax) = plt.subplots(1,1, sharey=False, sharex=False, figsize=(10,10))
sns.barplot(dfcs.index, dfcs.coherence, palette='Blues', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, ha='center')
ax.set_ylabel('Avg Topic Coherence', color = 'black', fontsize = 30)
fbar.tight_layout()
fbar.savefig('AvgTopicCv.pdf')
   

"""
DYNAMIC TOPIC MODELLING ANALYSIS
"""
#give topic names following the list (30 topics)


#get the topic evolution overt time 
topevo = []
for i in range(0,30):
    top1 = dyntop.print_topic_times(topic = i)
    topevo.append(top1)

timetop = []
for i in range(0,10):
    tim1 = dyntop.print_topics(time = i)
    timetop.append(tim1)

#get topic distribution, term per topic, overall term frequency and vocabulary for DTM
#generate time slice list for graphs and to add to the analysis
#these will also be used as keys
tslice = ['1990-1992','1993-1995','1996-1998','1999-2001','2002-2004','2005-2007','2008-2010','2011-2013','2014-2016','2017-2019']
tslice2= ['1992','1995','1998','2001','2004','2007','2010','2013','2016','2019']

topname =['H-E system', 'bird migration','risk and public perception', 'impact and adaptation', 'water resources',
          'carbon cost and policies', 'species and habitat range','model method','soil and carbon','futur temp and precip',
          'crop and yields','fire and livestock','warming temperatures','agriculture and food production','genetic selection','lakes and sediments', 'local policies',
          'co2 concentration','arctic','marine habitat','adaptation and vulnerability','energy',
          'country development and mitigation policies', 'land and ecosystem conservation','flooding and coastal areas','ghg emissions and mitigation',
          'health','urban infrastructure','plants and droughts','forests']

doctop = {}
tt = 0
for ti in tslice:
    dtop = dyntop.dtm_vis(time=tt, corpus = corpuslem)
    doctop.update({ti:dtop})
    tt =tt + 1

#get distribution of DTM per document.
#as toic distribution does not vary with time (only topics and term within topic do), we ause a random key to chck for topic distribution
nodetop['doctop'] = list(doctop['2017-2019'][0])

#calculate hellinger and cosine similarity (1-cosine or hellinger distance) between topics pet time slice
maxind = 30 #number of topics
tmat = np.zeros ((maxind, maxind))
tmat2 = np.zeros ((maxind, maxind))
#generate figure where we add hellinger and cosine distance iteratively
dhtop = plt.figure(figsize = (20,15))
dctop = plt.figure(figsize =(20,15))
k1 = 0 #for indexing subplots
for key in doctop:
    #transpose the term-frequency matrix
    #and then calculate hellinger and cosine distances
    tvec = doctop[key][1].T #
    for i in range (0,maxind):
        for j in range (0,maxind):
            tmat[i,j] = 1 - myhelling (tvec[:,i],tvec[:,j])
    print('Avg Hellinger') 
    print(np.mean(tmat))
    #set titles for plots
    tit1 = str(key) + ' Avg(h)= ' + str(round(np.mean(tmat),3))
    #do individual clustergrams based on hellinger distance
    sns.set(context='paper', style='white', palette='colorblind', font_scale=1.5)
    sns.clustermap(tmat, cmap='coolwarm', row_cluster=True,
                    col_cluster=True, linewidths=0.1,  yticklabels=topname, xticklabels=topname)
    ftit = str(key)+'_clumaps.pdf'
    plt.tight_layout()
    plt.savefig(ftit)
    sns.set(context='paper', style='white', palette='colorblind', font_scale=2)
    #do subplots one per hellinger distance
    k1 += 1
    if k1 == 9:
        k1 = 10
    #add subplots, one colormap per hellingr/cosine time period
    a1 = dhtop.add_subplot(3,4,k1)  
    a1.pcolor(tmat, cmap='Blues', vmin=0,vmax=1)
    a1.set_title(tit1, fontsize=20)
im = plt.pcolor(tmat, cmap='Blues')
im2 = plt.pcolor(tmat2, cmap='Blues')
cb_ax1 = dhtop.add_axes([0.8, 0.1, 0.02, 0.25])
cbar1 = dhtop.colorbar(im, cax=cb_ax1, cmap='coolwarm')
dhtop.savefig('Dyn_hellDist.pdf')

#prepare text for coherenc for dynamic topics
#first get the overall lemmatized text in a list format
tottxt = list(nodetop.tottxt)
#then calculate coherence per 3-year time periods aka time-slices
cvdyn = []
for i in range (0,10):
    tcv = ldadyn.dtm_coherence(time=i)
    cv1 = CoherenceModel(topics=tcv, texts=tottxt, dictionary=dictlem, coherence='c_v')
    cvd = cv1.get_coherence()
    cvdyn.append(cvd)   
#generate evolving coherence per time bar graph
m1 = len(cvdyn)
reind = np.arange(1,2,1)
dfcs = pd.DataFrame(cvdyn)
dfcs.columns = ['cvd']
fbar, (ax) = plt.subplots(1,1, sharey=False, sharex=False, figsize=(10,10))
sns.barplot(dfcs.index, dfcs.cvd, palette='Blues', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, ha='center')
ax.set_ylabel('Avg Coherence Similarity', color = 'black', fontsize = 24)
ax.set_xlabel('Time', color='black',fontsize = 24)
fbar.tight_layout()
fbar.savefig('DynCoherence.pdf')    

#assess topic distribution relatd to n of papers per year
#first build topic dataframe, using doclist (the distribution of topic per paper, and naming them)
#then add year and unique id
doclist = nodetop.doctop
dftop = pd.DataFrame.from_records(doclist, columns=topname)
dftop['year'] = nodetop.PY
dftop['uid'] = nodetop.UT
ndyntop = pd.merge(nodetop, dftop, left_on='UT',right_on='uid')
ndyntop = ndyntop.drop(columns=['year','uid'])
pickle.dump(ndyntop, open('nodeattr.p','wb'))

#group by the 3 year time-slices and calculate mean of the probabilty of a topic being part of a specific document
#this may not bee perfect, but allows for a better understanding of topic evolution over time.
#here we use an arithmetic mean, but othr measure of central tendency may be more appropriate (to check)
topivot = dftop.groupby(['year']).mean()
#Figure to assess yearly topic evolution, on average
fig, ax = plt.subplots(figsize = (25, 20), sharey=True, sharex=True)
for i, col in enumerate(topivot.columns):
    plt.subplot(6, 5, i+1)
    plt.plot(topivot.iloc[:,i])
    plt.title(col)
plt.tight_layout()
fig.savefig('topicEvolution.pdf')

#figure for average of 3 year time-slices
#now use the tslice and repeeat every for number of papers so one can take the overall mean.
tp1 = [105, 215, 492, 675, 967, 1984, 5988, 11473, 17451, 24753]
repslice2 = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(tslice2, tp1))))
dftop['times'] = repslice2
slicepiv = dftop.groupby(['times']).mean()
slicepiv = slicepiv.drop(columns=['year'])

ds1 = 25 #figsize
#Figure to assess topic evolution every 3 year avg on same scale (better to assess relativ importance)
fig, ax = plt.subplots(figsize = (ds1, ds1), sharey=True, sharex=True)
for i, col in enumerate(slicepiv.columns):
    plt.subplot(6, 5, i+1)
    plt.plot(slicepiv.iloc[:,i])
    plt.xticks(rotation=45)
    plt.ylim(0, 0.12)
    plt.title(col)
plt.tight_layout()
fig.savefig('topicEvolution3s.pdf')

#Figure to assess topic evolution every 3 year avg without the same scaling (better for visualizing trends)
fig, ax = plt.subplots(figsize = (ds1, ds1), sharey=True, sharex=True)
for i, col in enumerate(slicepiv.columns):
    plt.subplot(6, 5, i+1)
    plt.plot(slicepiv.iloc[:,i])
    plt.xticks(rotation=90)
    plt.title(col)
plt.tight_layout()
fig.savefig('topicEvolution3.pdf')


#make sure we do evolution of topics, so cumulative, not as they were diffrenrent in different time-periods

#cumulative evolution
slsum= dftop.groupby(['times']).sum()
slcnt = dftop.groupby(['times']).count()
slsum = slsum.drop(columns=['year'])
slcnt = slcnt.drop(columns=['year', 'uid'])
slsum= slsum.cumsum()
slcnt = slcnt.cumsum()
cumpiv = slsum.div(slcnt)
ds1 = 25 #figsize
#Figure to assess topic evolution every 3 year avg on same scale (better to assess relativ importance)
fig, ax = plt.subplots(figsize = (ds1, ds1), sharey=True, sharex=True)
for i, col in enumerate(cumpiv.columns):
    plt.subplot(6, 5, i+1)
    plt.plot(slicepiv.iloc[:,i])
    plt.xticks(rotation=45)
    plt.ylim(0, 0.12)
    plt.title(col)
plt.tight_layout()
fig.savefig('CumtopicEvolution3s.pdf')

#Figure to assess topic evolution every 3 year avg without the same scaling (better for visualizing trends)
fig, ax = plt.subplots(figsize = (ds1, ds1), sharey=True, sharex=True)
for i, col in enumerate(cumpiv.columns):
    plt.subplot(6, 5, i+1)
    plt.plot(slicepiv.iloc[:,i])
    plt.xticks(rotation=90)
    plt.title(col)
plt.tight_layout()
fig.savefig('CumtopicEvolution3.pdf')


slicesum = topivot.groupby(['times']).sum()
#Figure to assess topic evolution every 3 year avg on same scale (better to assess relativ importance)
fig, ax = plt.subplots(figsize = (ds1, ds1), sharey=True, sharex=True)
for i, col in enumerate(slicesum.columns):
    plt.subplot(6, 5, i+1)
    plt.plot(slicepiv.iloc[:,i])
    plt.xticks(rotation=45)
    plt.ylim(0, 10000)
    plt.title(col)
plt.tight_layout()
fig.savefig('topicEvolution_Sum_scaled.pdf')

#Figure to assess topic evolution every 3 year avg without the same scaling (better for visualizing trends)
fig, ax = plt.subplots(figsize = (25, 25), sharey=True, sharex=True)
for i, col in enumerate(slicesum.columns):
    plt.subplot(6, 5, i+1)
    plt.plot(slicepiv.iloc[:,i])
    plt.xticks(rotation=45)
    plt.title(col)
plt.tight_layout()
fig.savefig('topicEvolution_Sum_noscale.pdf')

#assess topic word evolution (looking only at ranks)
#first create a variable with topic names
namelist =  list(itertools.chain.from_iterable(itertools.repeat(x, 10) for x in topname))
#generate dataframe for topic evolution
dfevo = pd.DataFrame()
dfevo['topics'] = namelist
for i in range (0,10):
    year = tslice[i]
    topwords = []
    for j in range(0,30):
        a,b = zip(*topevo[j][i])
        tw = list(a[0:10])
        topwords = topwords + tw  
    dfevo[year]=topwords
    
#download results to csv to make a table of word rank evolution per topic
dfevo.to_csv('topevo.csv')      
  
"""
Bipartite Topic-Document network and Citation Network co-evolution!

Now we have checked the evolution of topics, their relative importance per time-slice etc... 
#question is: does the citation network and bipartite citation-topic network at t-1 
"""

#load main record collection and sort it to be the same as the order of the bipartite matrix (row=docs)
dfall =  pd.DataFrame(rcall)
dfall = dfall.sort_values(['PY'])    
dfall = dfall.reset_index(drop=True)
#generate dataframe with only essential node info
nprop = pd.DataFrame()
#add column forr WoS id,  year of publication, joournal and first author for node properties (importang to link bipartite to citation networks)
nprop['id'] = dfall.UT
nprop['yr']= dfall.PY
nprop['jo'] = dfall.J9
au = []
for idx,name in enumerate(dfall.AU):
    if name == '' or (isinstance(name, float) and  math.isnan(name)):
        au.append('nan')  
    else:
        au.append(name[0])
nprop['au'] = au

#add time slice variable that needs to be generated
tl1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #here avoid strings for now
#then generate list of number of documents pr time-slice
tp1 = [105, 215, 492, 675, 967, 1984, 5988, 11473, 17451, 24753]
tp0 = [0, 105, 215, 492, 675, 967, 1984, 5988, 11473, 17451]

# itertools.repeat(elem, n) - repeat elem n times
# zip(a, b) Make a list of 2-tuples out of the two lists, pairing each element with the corresponding element in the other list. This gives you exactly what you need to pass to itertools.repeat in your use case.
# itertools.chain - flattens the resulting list of iterators into a single list of values. You can either chain(*iterable) as I have done or chain.from_iterable(iterable) as Martijn Peters does.
repslice = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(tl1, tp1))))
# nprop['tslice']= repslice

#now we create bipartite dataframe doc-topics and use tp1 to divide the biadjacency matrix
doclist = nodetop.doctop
bmat = pd.DataFrame.from_records(doclist, columns=topname) #create only the bi-adjacency matrix, then add node attributes to nprop, useful fo gnerating networks  
bmat['split'] = repslice
bmat = bmat.set_index(nprop.id)
#define now sets of nodes for the bipartite networks, index = documents, columns = topics, useful for later
U = bmat.index
V = bmat.columns
#first we need to create the bipartite networks per time slice.
#use tp1 to define the rows that relate to each time-slice and create a dictionary of bipartite networks
nodeid = [] #create a list important for later re-assigining node indices thatt are unique and sequential for the whole network
bnodeid = [] #create a list important for later re-assigining node indices thatt are unique and sequential for the doc-topic net
sliced_bmat = bmat.groupby(bmat.split)
btemp = pd.DataFrame(columns=['i','j','t','w'])
t = 0
for (tslice, data) in sliced_bmat:
     bslice = data
     #drop the groupby column
     bslice = bslice.drop(['split'], axis =1)
     b1 = bslice.stack().reset_index()
     t1 = [t]*len(b1)
     b1['t'] = t1
     #rename columns
     b1.columns = list('ijwt') 
     #change order w, t
     b1 = b1[['i', 'j', 't', 'w']]
     nodeid.extend(b1.j) #make sure 0-29 are topics, rest documents
     nodeid.extend(b1.i)
     bnodeid.extend(b1.i)
     bnodeid.extend(b1.j)
     btemp = btemp.append(b1)
     t += 1
#we create citation networks per time-slice, to then add togther in order to assess the evolution of the 
#citation networks. (rnet is already the citation network per time-period)
   # d1 = dict(g.nodes(data='id'))
tlist = []
dftemp = pd.DataFrame(columns=['i','j','t','weight'])
cnodeid = [] #create a list important for later re-assigining node indices thatt are unique and sequential for the citation nt
t = 0
for key in rnet:
    g = rnet[key]
    #rename nodes with WoS id
    d1 = dict(g.nodes(data='id'))
    d2 = dict_clean(d1.items())
    g = g.copy()
    g = nx.relabel_nodes(g, d2, copy=False)
    t1 = [t]*len(g.edges)
    tn1 = g.edges
    dt = pd.DataFrame(tn1, columns = ['i','j'])
    nodeid.extend(dt.i)
    nodeid.extend(dt.j)
    cnodeid.extend(dt.i)
    cnodeid.extend(dt.j)
    dftemp = dftemp.append(dt)
    tlist.extend(t1)
    t+=1
dftemp['t'] = tlist
#add weight = 1 for conformity with bipartit networks
w1 = [1]*len(dftemp)
dftemp['weight']=w1
  # orgs2 = set(n for n,d in g.nodes(data=True) if d['bipartite']==0)
  # projs2 = set(g)-orgs2  
   
#merge the bipartite network with the citation network
ttemp = btemp.append(dftemp)
ttemp2 = ttemp.copy()
ttemp2 = ttemp.fillna(0)
ttemp2['w2'] = ttemp2['w']+ttemp2['weight']
ttemp2 = ttemp2.drop(columns=['w', 'weight'])
ttemp2 = ttemp2.rename(columns={'w2': 'weight'})
ttemp = ttemp2.copy()
#now we have a ntwork ready to be timestamped with tlist.
#However, the nodes namee are not indices, but are strings, we need to convert all strings to indices
# for mapping the dataframe on to a temporal network object in teneto
alldid = {ni: indi for indi, ni in enumerate(unique(nodeid))}
cdid = {ni: indi for indi, ni in enumerate(unique(cnodeid))}
bdid = {ni: indi for indi, ni in enumerate(unique(bnodeid))}

#now map i and j of dfnet to their ids and we have the dataframe structure of a temporal network, where i and j are nodes and t = time period
totnet = ttemp
totnet['i']= ttemp['i'].map(alldid)
totnet['j']= ttemp['j'].map(alldid)

dfnet = dftemp
dfnet['i']= dftemp['i'].map(cdid)
dfnet['j']= dftemp['j'].map(cdid)


bfnet = btemp
bfnet['i']= btemp['i'].map(bdid)
bfnet['j']= btemp['j'].map(bdid)

#save some files in case there are issues with kernel (as they have been so far)
pickle.dump(totnet, open('tf.p', 'wb'))
pickle.dump(dfnet, open('cf.p', 'wb'))
pickle.dump(bfnet, open('bf.p', 'wb'))
pickle.dump(alldid, open('nodeid.p','wb'))
pickle.dump(cdid, open('citeid.p','wb'))
pickle.dump(bdid, open('topdocid.p','wb'))

