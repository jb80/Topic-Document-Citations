#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:00:22 2020

This file is used to merge the data collecteed from WoS or othr sources into one single 
record collection. From theen, we divid the record collection according to specific time intervals (3 years here)

@author: Jacopo Baggio
"""

#Import packages for directory changes and saving output
import os
import pickle

#import packages for datta-processing (non-NLP)
import metaknowledge as mk
import pandas as pd

#import packages for graphs/figures
import seaborn as sns
import matplotlib.pyplot as plt

#import packages for specific operations (network data cleaning, NLP data cleaning)
import networkx as nx
import nltk


#First lets go in the directory where the data are
wdir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/CCRecords')
wdir2 = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis')
os.chdir(wdir)
            
#now create a collection for the citation analysis and topic analysis. 
#make record collection allows to make a collection from all
#database files in a dirctory (wdir in our case)
rc = mk.RecordCollection(wdir)

#eliminate 2020 records - keep 30 years
rc = rc.yearSplit(1990, 2019)

print(len(rc))
#check and eliminate bad entries
lb = len(rc.badEntries())
print(lb)
if lb > 0:
  rc.dropBadEntries()
#double check that all is good.
print (len(rc))

os.chdir(wdir2) #dirctory where the analysis will be done and for pickle dump

#Now assess growth of papers per year
growth = pd.DataFrame(rc.timeSeries('year'))
growth['cum_gr']=  growth.loc[::-1,'count'].cumsum()   
#Graph number of papers per year both cumulative and as bar per year

hfont = {'fontname':'Times'}
sns.set(context='paper', style='white', palette='colorblind', font_scale=2)
fgn, ((ax1, ax2)) = plt.subplots(1, 2,  sharey=False, sharex=True, figsize=(20,10))
ax1.plot(growth['year'], growth['cum_gr'], linewidth=2, color = 'black')
ax1.set_title('Cumulative Published Papres', fontsize=24, color = 'black')
ax1.set_ylabel('Number of Papers', fontsize=24, color = 'black')
ax1.set_xlabel('Year', fontsize=24, color = 'black')
ax2.bar(growth['year'], growth['count'])
ax2.set_title('Yearly Published Papers', fontsize=24, color = 'black')
#ax2.set_ylabel('Number of Papers', fontsize=24, color = 'black')
ax2.set_xlabel('Year', fontsize=24, color = 'black')
plt.tight_layout()
plt.savefig('Lit_Growth.pdf')
  


#Split the dataset to have time-diemension, hence the topic analysis and co-ciation 
#and 2 mode network should all be yearly to assess the evolution of the topic and co-citation ntwork
 
# #split the records per year create unique list of years or time-frames - not used as we use a 3 years interval
# years= rcall.PY.unique()
# sortyear = np.arange(1990,2020,1)

"""
the First ipcc assesssmnts date is in 1990, starting also of the rcords. 
Then we parse records evry 3 years. 3 years is arbiratry, but also allows for 10 time-intervals to assess
the evolution of the co-citation network as well as the topics within climate change and related to
adaptation, transformation and mitigation. To gather the records the following search string was used:
    
(TS = "climate change" AND TS = (adapt* OR transform* OR mitigat*)) 
OR (TS = "global warming" AND TS = (adapt* OR transform* OR mitigat*)) 
OR (TS = "climatic changes" AND TS = (adapt* OR transform* OR mitigat*)) 

Timespan: 1990-2019. 
Indexes from WebOfScience: 
SCI-EXPANDED, SSCI, A&HCI, CPCI-S, CPCI-SSH, BKCI-S, BKCI-SSH, ESCI, CCR-EXPANDED, IC.


"""
ipc1 =[1990,1993,1996,1999,2002,2005,2008,2011,2014,2017] 
ipc2 =[1992, 1995, 1998, 2001,2004,2007,2010,2013,2016,2019]


#For all networks, we are interested only in co-ciations, journal citation and author citation of papers that ar 
#part of the collection. We do not take into account citations to papers outside the WoS search.
#full co-citation networks

#assess overall difference when useing only core papers (searched) or all papers cited
netall = rc.networkCitation(weighted=False, coreOnly=True, directed=False,detailedCore=True, detailedCoreAttributes=True)
netall.remove_edges_from(nx.selfloop_edges(netall))
netall2 = rc.networkCoCitation(weighted=False, coreOnly=True, detailedCore=True, detailedCoreAttributes=True)
netall2.remove_edges_from(nx.selfloop_edges(netall2))


bipall = rc.networkTwoMode('AF','ID') #this is not really working as it does not build a bipartite graph...
#createe network of years (how much papers in the prevoius thre years ahv been cited by thee second three-years span)
yearcite= rc.networkCitation(nodeType ='year', coreOnly=True, directed=False,detailedCore=True, detailedCoreAttributes=True)
journcite= rc.networkCitation(nodeType ='journal', coreOnly=True, directed=False,detailedCore=True, detailedCoreAttributes=True)
authorcite= rc.networkCitation(nodeType ='author', coreOnly=True, directed=False,detailedCore=True, detailedCoreAttributes=True)

#create three empty dictionaries to contain the yearly data. One is the full collection, one contains 
#co-citation networks and one the bipartite author-keyword network and one network eliciting citations between years
rcy = {}
rnet = {}
rmulti = {}

#words we want to eliminate for topic analysis.
stopwords = nltk.corpus.stopwords.words('english')
#where to store resulting topic modelling first cleaning passag
rtop = {}

for i in range (0,len(ipc1),1):
    s1 = ipc1[i]
    s2 = ipc2[i]
    #genrate key for dictionary entries
    dn = str(s1) + '_' + str(s2)
    print ('Starting ' + dn)
    #create 3 year time-span collections and create a temp file for building dictionaries
    # rcy[dn]=rc.yearSplit(s1, s2)
    # tempfil = rcy[dn]
    #create 3 year time-span citation network 
    subyear = [x for x,y in netall.nodes(data=True) if y['year'] <=s2]
    rnet[dn] =  netall.subgraph(subyear)
    rnet[dn] = rnet[dn].copy()
    #eliminate self-loop edges 
    rnet[dn].remove_edges_from(nx.selfloop_edges(rnet[dn]))

    #create 3 years time-span bipartite networks with authors and keywords
    rmulti[dn] = tempfil.networkTwoMode('AF','ID')
    #eliminate self-loop edges 
    rmulti[dn].remove_edges_from(nx.selfloop_edges(rmulti[dn]))
    #create 3 year time-span abstract cleaening for natural language processing (also done in CC_Topic_Analysis.py)
    #for NLP is inbuild in metaknowledge; in CC_Topic_Analysis we  clean the text in any case.
    rtop[dn]= tempfil
    #forNLP(dropList=stopwords, lower = True, removeNumbers = True, removeNonWords=True,removeWhitespace=True
    print('Finished '+ dn)


#save files generated for analysis via CC_Topic_Analysis.py and then via CC_Network_Analysis.py
os.chdir(wdir2 + '/Data')

pickle.dump(growth, open('npapersyear.p','wb'))

pickle.dump(rc, open('fullcollection.p', 'wb'))
pickle.dump(netall, open('fullcitenet.p', 'wb'))
pickle.dump(bipall, open('fullbipnet.p', 'wb'))

pickle.dump(yearcite, open('yeartoyearcitations.p','wb'))
pickle.dump(journcite, open('journalcitations.p','wb'))
pickle.dump(authorcite, open('authorcitations.p','wb'))

#export files at 3 year time-span
pickle.dump(rcy, open('yearlycollection.p', 'wb'))
pickle.dump(rtop, open('yearlytopic.p', 'wb'))
pickle.dump(rmulti, open('yearlybipnet.p', 'wb'))

#make sure that if one changes cocite to cite one changes the name heere too.
pickle.dump(rnet, open('yearlycitenet.p', 'wb'))



"""
issues when downloading or building the full citation network without the core-Only... super-big networks!
#issues, on my machine, to pickle dum the yearly cite net file... 

net_9092 = rnet['1990_1992']
net_9395 = rnet['1993_1995']
net_9698 = rnet['1996_1998']
net_9901 = rnet['1999_2001']
net_0204 = rnet['2002_2004']
net_0507 = rnet['2005_2007']
net_0810 = rnet['2008_2010']
net_1113 = rnet['2011_2013']
net_1416 = rnet['2014_2016']
net_1719 = rnet['2017_2019'] -> this network is too big... 



#issues, on my machine, to pickle dum the yearly cite net file... 
pickle.dump(net_9092 , open('rnet1.p','wb'))
pickle.dump(net_9395 , open('rnet2.p','wb')) 
pickle.dump(net_9698 , open('rnet3.p','wb'))
pickle.dump(net_9901 , open('rnet4.p','wb'))
pickle.dump(net_0204 , open('rnet5.p','wb'))
pickle.dump(net_0507 , open('rnet6.p','wb'))
pickle.dump(net_0810 , open('rnet7.p','wb'))
pickle.dump(net_1113 , open('rnet8.p','wb'))
pickle.dump(net_1416 , open('rnet9.p','wb'))
pickle.dump(net_1719 , open('rnet10.p','wb'))

"""
