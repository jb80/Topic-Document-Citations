#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:03:44 2020

This script takes the results of the node-network and the  implicial complexes analysis 
in CC_NetMetrics_Simpl_Cmplx and looks at the co-evolution of the doc-topic and citatioin network.


@author: Jacopo Baggio
"""
#import utilities
import os
import pickle
import itertools

#matrices operations and data manipulation
import pandas as pd
import scipy as sp
from scipy import NINF
import numpy as np
from sklearn import preprocessing
#temporal network
import pathpy as ppy

#figures
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#transform dataframe into tuple list
def records(df): return df.to_records(index=False).tolist()


#figure general characterisics
#set general figure context
# matplotlib.use('Agg')
hfont = {'fontname':'Times'}
sns.set(context='paper', style='white', palette='colorblind', font_scale=2)

#working directory
wdir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis')
#change to working directory
os.chdir(wdir)

#load simplicial complex and network data dervied from CC_Simpl_Cmplx.py. 
#note that names are given by me, one can use different names. Also you can run this after CC_SimplCmplx
#without saving the results first.
bipsimp  = pickle.load(open('bipsimpc.p', 'rb'))  #top-doc simplicial complex analysis
citesimp = pickle.load(open('citsimpc.p', 'rb')) #doc-doc simplicial complex analysis
bipnet   = pickle.load(open('bipnet.p', 'rb'))    #top-doc network and node analysis (density, degree, clustering)
citenet  = pickle.load(open('citnet.p', 'rb'))   #top-doc network and node analysis (density, degree, clustering and pagerank)
dfpz     = pickle.load(open('partzscore.p','rb'))
simpid   = pickle.load(open('simpdict.p', 'rb'))     #id of nodes in citsimp
nodeattr = pickle.load(open('nodeattr.p','rb'))       #attributes of nodes - topics, author, macro research area
betti    = pickle.load(open('betti.p','rb'))          #betti numbers and euler characteristics of doc-doc networks
idnode   = pickle.load(open('nodeid.p','rb'))         #from numeric node id to WoS and cited paper ID

#switch key values for nodeid and ids, better for assignment 
nodeid = dict((v,k) for k,v in idnode.items())
for key in simpid:
    simpid[key] = dict((v,k) for k,v in simpid[key].items())
    
#get list of names useful for figures and such, for time-periods and topics
tslice2 = ['1990-1992','1993-1995','1996-1998','1999-2001','2002-2004','2005-2007','2008-2010','2011-2013','2014-2016','2017-2019']


tslice = ['1992','1995','1998','2001','2004','2007','2010','2013','2016','2019']
tnofirst = ['1995','1998','2001','2004','2007','2010','2013','2016','2019']


topname =['H-E system', 'bird migration','risk and public perception', 'impact and adaptation', 'water resources',
          'carbon cost and policies', 'species and habitat range','model method','soil and carbon','futur temp and precip',
          'crop and yields','fire and livestock','warming temperatures','agriculture and food production','genetic selection','lakes and sediments', 'local policies',
          'co2 concentration','arctic','marine habitat','adaptation and vulnerability','energy',
          'country development and mitigation policies', 'land and ecosystem conservation','flooding and coastal areas','ghg emissions and mitigation',
          'health','urban infrastructure','plants and droughts','forests']
#get abbreviated topic names
topabbr = ['HmEnSy', 'BrdMgr','RskPpr', 'ImpApt', 'WatRes', 'CrbCst', 'SpcHtR', 'MdlMtd', 'SolCrb',
           'TmpPrc', 'CrpYld', 'FirLvs', 'WrmTmp', 'AgFdPr', 'GnSlc','LkSdm', 
           'LocPol', 'Co2Cnc','Arctic', 'MarHab', 'AptVln', 'Energy', 'DevMtg', 'LndEcs', 
           'FldCst', 'GhgMtg', 'Health', 'Urban', 'PltDrg', 'Forest']
   

#then generate list of number of documents pr time-slice
tp1 = [105, 215, 492, 675, 967, 1984, 5988, 11473, 17451, 24753]
repslice = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(tslice, tp1))))

counth = 0.25 #threshold that defines strong association if we want to dichotomize the networks

#get first 10 most cited papers per period and topic they refer to
#build reference dataframe for only topics
doclist = nodeattr.doctop
dftop = pd.DataFrame.from_records(doclist, columns=topname)
dftop2 = dftop.copy()
#eliminate -> change to 0 -> all values < threshold (0.1)
dftop.where((dftop > 0.1), inplace=True)
dftop = dftop.fillna(0)
#do the same for count of strong association with values < 0.25
dftop2.where((dftop > counth), inplace=True)
dftop2 = dftop2.notnull().astype('int')


#add tiime slices to dftop
dftop['tslice'] = repslice
dftop['UT'] = nodeattr.UT
#add tiime slices to topivot
dftop2['tslice'] = repslice
dftop2['UT'] = nodeattr.UT

#prepare dataframe including also author and the like
top10node = nodeattr.copy()
top10node= top10node.drop(['AB', 'AF', 'tottxt','ID','keyw','abclean','doctop'], axis=1)
#dictionaries to store final results
topdictnode = {}
topdicttop = {}
topcount = {} 
topdicttop[0] = dftop.groupby(['tslice']).mean()
topcount[0] = dftop2.groupby(['tslice']).sum()

for key in citenet:
    if key > 0:
        da = dict(citenet[key]['indeg'])
        dfda = pd.DataFrame.from_records([da]).T #,
        dfda.columns=['indeg']
        k1, v1 = max(da.items(), key=lambda x:x[1])
        idmax = nodeid[k1]
        print ('Period: ' + str(key) +' MaxInDegNode: ' + str(k1) +' InDeg: '+str(v1)  +' IDNodeMax: ' + str(idmax))
        top10 = dfda['indeg'].nlargest(10, keep ='all')
        toplistnode=[]
        for x in list(top10.index):
            toplistnode.append(nodeid[x])
        topdictnode[key] = top10node.loc[top10node['UT'].isin(toplistnode)]
        temp = dftop.loc[dftop['UT'].isin(toplistnode)]
        #now from the top 10 papers, only list the ones clearly related to a specific topic x. We define this as
        #the probabilty or a topic-doc relation > 0.25 (1/4)
        temp = temp.drop(columns=['tslice', 'UT'])
        #one can either use the average probability 
        topdicttop[key] = temp
        #or the number of ppaers strongly related to as specific topic, where strongly is, 
        #however, somewhat arbitrarily defined. Here we use the 1/4 threshold: 0.25
        temp2 = temp[temp > counth]
        topcount[key] = temp2.notnull().astype('int')


"""
Bar Graph of top 10 papers per period, both as 
Avg Probability and Number of papers strongly associated to a specific topic
"""
#bar graph of top10 paper topics per period - AVG
fig, axs = plt.subplots(figsize = (25, 25), sharey=True, sharex=False)
for key in topdicttop:
    if key > 0:
        t1 = topdicttop[key].sum()
        t1 = t1.rename(index=dict(zip(topname,topabbr)))
        #t1 = t1.drop(index=['tslice', 'UT'])
        t1 = t1[t1 > 0] / len(topdicttop[key])
        t1 = t1.sort_values(ascending=False)
        plt.subplot(3,3,key)
        t1.plot(kind='bar')
        plt.title(tslice[key])
fig.tight_layout()
fig.savefig('Toppapertopic_Avg.pdf')

#bar graph of top10 paper topics per period - COUNT
fig, axs = plt.subplots(figsize = (25, 25), sharey=True, sharex=False)
for key in topcount:
    if key > 0:
        t1 = topcount[key].sum()
        t1 = t1.rename(index=dict(zip(topname,topabbr)))
        #t1 = t1.drop(index=['tslice', 'UT'])
        t1 = t1[t1 > 0] #/ len(topdicttop[key])
        t1 = t1.sort_values(ascending=False)
        plt.subplot(3,3,key)
        t1.plot(kind='bar')
        plt.title(tslice[key])
fig.tight_layout()
fig.savefig('Toppapertopic_Count.pdf')


"Relationship between top 10 cited paper topic prevalence and overall topic prevalence"
#create topics of the most cited paper, and then topics for top-doc documents
cavg = {}
tavg = {}
temptop = topdicttop[0]
for key in topdicttop:
    if key == 0:
        c0 = temptop[0:1].T.squeeze()
        ls0 = [x == 0 if x == 0  else 0 for x in c0]
        ls0 = pd.Series(ls0, index = topname)
        cavg[key] = ls0
    if key > 0:
        citetop = topdicttop[key].fillna(0)
        citetop = topdicttop[key].mean()
        cavg[key] = citetop.fillna(0)
        tmp1 = temptop[key-1:key].T
        tavg[key-1] = tmp1
tavg[9]= temptop[9:10].T
topicavg = tavg

correl ={}
for key in cavg:
    if key == 9:
        pass
    else:
        k1 = key 
        v1 = cavg[key+1]
        v2 = tavg[key].squeeze()
        #standardize for euclidean distance
        s1 = preprocessing.scale(np.array(v1))
        s2 = preprocessing.scale(np.array(v2))
        rel, pval = sp.stats.spearmanr(v1,v2)
        drho = key, key+1, round(rel,3), round(pval,3)
        correl[key] = drho
        print (str(key))
        print (rel)
        
#from dictionaries create list of lists
clist = ([[k]+list(v) for k,v in cavg.items()])
tlist = ([[k]+list(v.squeeze()) for k,v in tavg.items()])

n=len(tslice)
distMavg=np.zeros([n, n]) 
pval=np.zeros([n, n]) 
distV=[]

for i in range(0,n):
    tc = clist[i]
    for j in range (0,n):
       tt = tlist[j]
       dist, p = sp.stats.spearmanr(tc,tt) # dist = nc.dfdist(tc,tt, 30) #
       distV.append(dist)
       pval[i,j] = round(p,3)
       distMavg[i,j]=dist
distMavg = np.nan_to_num(distMavg)

#create topics of the most cited paper, and then topics for top-doc documents for countrs (dichotomized)
cavg = {}
tavg = {}
temptop = topcount[0]
for key in topcount:
    if key == 0:
        c0 = temptop[0:1].T.squeeze()
        ls0 = [x == 0 if x == 0  else 0 for x in c0]
        ls0 = pd.Series(ls0, index = topname)
        cavg[key] = ls0
    if key > 0:
        citetop = topcount[key].fillna(0)
        citetop = topcount[key].mean()
        cavg[key] = citetop.fillna(0)
        tmp1 = temptop[key-1:key].T
        tavg[key-1] = tmp1
tavg[9]= temptop[9:10].T
topiccnt = tavg

#from dictionaries create list of lists
clist = ([[k]+list(v) for k,v in cavg.items()])
tlist = ([[k]+list(v.squeeze()) for k,v in tavg.items()])

n=len(tslice)
distMcnt=np.zeros([n, n]) 
pval=np.zeros([n, n]) 
distV=[]

for i in range(0,n):
    tc = clist[i]
    for j in range (0,n):
       tt = tlist[j]
       dist, p = sp.stats.spearmanr(tc,tt) # dist = nc.dfdist(tc,tt, 30) #
       distV.append(dist)
       pval[i,j] = round(p,3)
       distMcnt[i,j]=dist
distMcnt = np.nan_to_num(distMcnt)


#now assess clustering of time-slices - not to use for now... better doing MDS
sns.clustermap(distMavg, method='complete', metric='euclidean', cmap='coolwarm', row_cluster=True,
               col_cluster=True, linewidths=0.1,  yticklabels=tslice, xticklabels=tslice)
plt.savefig('ClusterSpearman_Avg.pdf')
plt.close()
sns.clustermap(distMcnt, method='complete', metric='euclidean', cmap='coolwarm', row_cluster=True,
                    col_cluster=True, linewidths=0.1,  yticklabels=tslice, xticklabels=tslice)
plt.savefig('ClusterSpearman_count.pdf')
plt.close()


im = plt.imshow(distMavg, vmin=-0.6, vmax=0.8, cmap='coolwarm')
                
#Combine count and avg topic 10 and all fgures
fco, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,12))
cbar_ax = fco.add_axes([0.99, .3, .03, .4])
heat = sns.heatmap(distMavg, cmap = 'coolwarm', vmin=-0.6, vmax=0.8, ax = ax1, yticklabels=tslice, xticklabels=tslice, square = True, cbar=False)
ax1.set_xticklabels(tslice, fontsize=28, color='black', rotation = 90)
ax1.set_yticklabels(tslice, fontsize=28, color='black', rotation = 0)
heat = sns.heatmap(distMcnt, ax = ax2, cmap = 'coolwarm',vmin=-0.6, vmax=0.8, yticklabels=tslice, xticklabels=tslice, square = True, cbar=False)
ax2.set_xticklabels(tslice,fontsize=28, color='black', rotation = 90)
ax2.set_yticklabels(tslice, fontsize=28, color='black', rotation = 0)
ax1.text(-0.05, 1.02, 'a)', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
ax2.text(-0.05, 1.02, 'b)', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
fco.text(0.5, -0.02, 'All Documents', fontsize=36, ha='center', va='center')
fco.text(-0.02, 0.5, 'Top 10 Cited Documents', ha='center',  fontsize = 36,va='center', rotation='vertical')
fco.colorbar(im, cax=cbar_ax)
fco.tight_layout()
plt.savefig('Co-EvolutionSpearman.pdf', bbox_inches='tight')
plt.close()

#Check persistency in ranking in both avg and count topics

dfa = pd.DataFrame(([[k]+list(v.squeeze()) for k,v in topicavg.items()])).T
dfc = pd.DataFrame(([[k]+list(v.squeeze()) for k,v in topiccnt.items()])).T
dfa = dfa.drop([0,0]) #drop first row
dfc = dfc.drop([0,0]) #drop first row

autdistMavg = dfa.corr(method='spearman')
autdistMcnt = dfc.corr(method='spearman')

im = plt.imshow(distMavg, vmin=-0.1, vmax=1, cmap='coolwarm')
                
#Combine count and avg topic 10 and all fgures
ftc, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,12))
cbar_ax = ftc.add_axes([0.99, .3, .03, .4])
heat = sns.heatmap(autdistMavg, cmap = 'coolwarm', vmin=-0.1, vmax=1, ax = ax1, yticklabels=tslice, xticklabels=tslice, square = True, cbar=False)
ax1.set_xticklabels(tslice, fontsize=28, color='black', rotation = 90)
ax1.set_yticklabels(tslice, fontsize=28, color='black', rotation = 0)
heat = sns.heatmap(autdistMcnt, ax = ax2, cmap = 'coolwarm',vmin=-0.1, vmax=1, yticklabels=tslice, xticklabels=tslice, square = True, cbar=False)
ax2.set_xticklabels(tslice,fontsize=28, color='black', rotation = 90)
ax2.set_yticklabels(tslice, fontsize=28, color='black', rotation = 0)
ax1.text(-0.05, 1.02, 'a)', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
ax2.text(-0.05, 1.02, 'b)', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
ftc.colorbar(im, cax=cbar_ax)
ftc.tight_layout()
plt.savefig('TopicPersistentpearman.pdf', bbox_inches='tight')
plt.close()

#assess degree centrality, clustering and page rank throughout time
#First, get time-evolution fof network metrics
citedata = {}
for key in citenet:
    if key > 0:
        nodes = citenet[key]['nodes']
        dens = citenet[key]['density']
        deg     = np.mean(list(dict(citenet[key]['indeg']).values()))
        vardeg  = np.var(list(dict(citenet[key]['indeg']).values()))
        skdeg   = sp.stats.skew(list(dict(citenet[key]['indeg']).values()))
        prank   = np.mean(list(dict(citenet[key]['pagerank']).values()))
        varprank= np.var(list(dict(citenet[key]['pagerank']).values()))
        skprank = sp.stats.skew(list(dict(citenet[key]['pagerank']).values()))
        clus    = np.mean(list(dict(citenet[key]['clus']).values()))
        dirclus = np.mean(list(dict(citenet[key]['dirclus']).values()))
        tkey = tslice[key]
        citedata[tkey] = nodes, round(dens,5), round(deg,5), round(vardeg,5), round(skdeg,5), round(prank,5), round(varprank,5),  round(skprank,5), round(clus,5), round(dirclus,5) 
dfnet = pd.DataFrame.from_dict(citedata)
dfnet = dfnet.T 
dfnet.columns = ['nodes','dens','avgdeg','vardeg', 'skdeg','avgprank','varprank','skprank','avgclus','avgdirclus']

#now graph evolution of general network metrics per time period.
tnofirst = ['1990-1995','1998','2001','2004','2007','2010','2013','2016','2019']
fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3, figsize = (15, 5))
ax1.plot(dfnet.dens, color="black", marker="o",ms=8)
ax2.plot(dfnet.avgdeg, color="black", marker="o",ms=8)
ax3.plot(dfnet.avgdirclus, color="black", marker="o",ms=8)
ax1.set_xticklabels(tnofirst, rotation = 90)
ax2.set_xticklabels(tnofirst, rotation = 90)
ax3.set_xticklabels(tnofirst, rotation = 90)
ax1.set_xlabel('Time Period')
ax2.set_xlabel('Time Period')
ax3.set_xlabel('Time Period')
ax1.set_ylabel('Density')
ax2.set_ylabel('Avg In-Degree')
ax3.set_ylabel('Avg Clusteriing')
fig.tight_layout()
fig.savefig('NetworkClassics.pdf')
plt.close()

#now graph degree distribution of cited docs and topics.
counts = []
bins = []
for i in citenet:
    [c1,b1,patches]=plt.hist(dict(citenet[i]['indeg']).values(),bins=100)
    counts.append(c1)
    bins.append(b1)
    
bcounts = []
bbins = []
for i in bipnet:
    [cb,bb,patches]=plt.hist(dict(bipnet[i]['topdeg'][1]).values(),bins=100)
    bcounts.append(cb)
    bbins.append(bb)

sns.set(context='paper', style='white', palette='colorblind', font_scale=2)
fcd, axs = plt.subplots(figsize = (10, 10), sharey=True, sharex = True)
axs.axis([0, 100000, 0, 1])
for i in citenet:
    if i > 0:
        cnt = counts[i]
        bns = bins[i]
        countsnozero=cnt*1.
        countsnozero[cnt==0]=NINF
        plt.subplot(3,3,i)
        plt.scatter(bns[:-1],countsnozero/float(np.sum(cnt)),s=10, color = 'black')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(0.00008,1.1)
        plt.xlim(0.8,100000)
        plt.title(tslice[i])
fcd.text(0.5, 0.02, 'Degree', ha='center', va='center')
fcd.text(0.02, 0.5, 'Fraction of Nodes', ha='center', va='center', rotation='vertical')
fcd.tight_layout()
fcd.savefig('Cite_DegreeDistr.pdf')
plt.close()

fbd, axs = plt.subplots(nrows=3, ncols=3, figsize = (10, 10))
for i in bipnet:
    if i > 0:
        bnt = bcounts[i]
        bbs = bbins[i]
        countsnozero=bnt*1.
        countsnozero[bnt==0]=NINF
        plt.subplot(3,3,i)
        plt.scatter(bbs[:-1],countsnozero/float(np.sum(bnt)),s=10, color = 'red')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(0.00008,1.1)
        plt.xlim(0.8,100000)
        plt.title(tslice[i])
fbd.text(0.5, 0.02, 'Degree', ha='center', va='center')
fbd.text(0.02, 0.5, 'Fraction of Nodes', ha='center', va='center', rotation='vertical')
fbd.tight_layout()
fbd.savefig('Topic_Distr.pdf')
plt.close()


#first get top ten subject overall, biased towards present given the exponential (hyper)
#growth of papers
dcol = dfpz[9].copy() #using the last evolved network.
rafr = dcol['mod'].value_counts()
dcol['freqRA']=dcol['mod'].map(rafr)
dcol['rankRA']=dcol.freqRA.rank(method='dense', ascending=False)
dcol['tocolor'] = np.where(dcol['rankRA'] < 9, dcol['mod'], 'Other')
sns.set(context='paper', style='white', palette='colorblind', font_scale=2)
#plot participation coefficient and degree to asses interdisciplinarity
fpz, ax = plt.subplots(ncols = 3, nrows = 3, figsize = (13, 10), sharey=True, sharex=True)
for i in dfpz:
    if i > 0:
        #now map colors to top 10 research areas (or combinations)
        d1 = pd.merge(dfpz[i],dcol, on = 'node')
        grcol = d1.groupby('tocolor')
        plt.subplot(3, 3, i)
        for nm, gr in grcol:
            plt.scatter(gr.partin_x, gr.zin_x, s=5, label=nm)
            plt.title(tslice[i])       
#   labelspacing=0.1, ncol=5)
labels_handles = {label: handle for ax in fpz.axes for handle, label in zip(*ax.get_legend_handles_labels())}
ax1 = fpz.add_subplot(1,1,1)
ax1.axis('off')
ax1.legend(labels_handles.values(),  labels_handles.keys(), bbox_to_anchor=(1.2, 0.7), markerscale=2)
fpz.text(0.5, 0.02, 'Participation Coefficient', ha='center', va='center')
fpz.text(0.02, 0.5, 'Within-RA Z-Score', ha='center', va='center', rotation='vertical')
fpz.tight_layout()
fpz.savefig('PartZScore.pdf')
plt.close()


#top 5 documents topological degree within the max dimension (connected to most of the network)
topolmax = nodeattr.copy()
topolmax= topolmax.drop(['AB', 'AF', 'tottxt','ID','keyw','abclean','doctop'], axis=1)

topoldegavg = {}
topoldegcnt = {}
topoldegavg[0] = dftop.groupby(['tslice']).mean()
topoldegcnt[0] = dftop2.groupby(['tslice']).sum()

for key in citesimp:
    if key > 0:
        tdeg = citesimp[key]['TopDeg']
        dftdeg = pd.DataFrame.from_dict(tdeg).T
        maxdim = dftdeg.iloc[:,-1]
        maxdim = maxdim[maxdim > 0]
        topdocmaxdim = maxdim.nlargest(10, keep='all')
        topmaxdim = []
        for i in list(topdocmaxdim.index):
            s2 = simpid[key]
            tempkey = s2[i]
            topmaxdim.append(nodeid[tempkey])
            topoldegavg[key] = topolmax.loc[topolmax['UT'].isin(topmaxdim)]
            temp = dftop.loc[dftop['UT'].isin(topmaxdim)]
            #now from the top 5 papers, only list the ones clearly related to a specific topic x. We define this as
            #the probabilty or a topic-doc relation > 0.25 (1/4)
            temp = temp.drop(columns=['tslice', 'UT'])
            #one can either use the average probability 
        topoldegavg[key] = temp
        #or the number of ppaers strongly related to as specific topic, where strongly is, 
        #however, somewhat arbitrarily defined. Here we use the 1/4 threshold: 0.25
        temp2 = temp[temp > counth]
        topoldegcnt[key] = temp2.notnull().astype('int')

#bar graph of topological max dimension max degree paper topics per period - AVG
fig, axs = plt.subplots(figsize = (25, 25), sharey=True, sharex=False)
for key in topoldegavg:
    if key > 0:
        t1 = topoldegavg[key].sum()
        t1 = t1.rename(index=dict(zip(topname,topabbr)))
        #t1 = t1.drop(index=['tslice', 'UT'])
        t1 = t1[t1 > 0] / len(topoldegavg[key])
        t1 = t1.sort_values(ascending=False)
        plt.subplot(3,3,key)
        t1.plot(kind='bar')
        plt.title(tslice[key])
fig.tight_layout()
fig.savefig('Topological_Deg_Topic_Avg.pdf')

#bar graph of top 5 paper topics per topological degree in max dimension - COUNT
fig, axs = plt.subplots(figsize = (25, 25), sharey=True, sharex=False)
for key in topoldegcnt:
    if key > 0:
        t1 = topoldegcnt[key].sum()
        t1 = t1.rename(index=dict(zip(topname,topabbr)))
        #t1 = t1.drop(index=['tslice', 'UT'])
        t1 = t1[t1 > 0] #/ len(topdicttop[key])
        t1 = t1.sort_values(ascending=False)
        plt.subplot(3,3,key)
        t1.plot(kind='bar')
        plt.title(tslice[key])
fig.tight_layout()
fig.savefig('Topological_Deg_Topic_Count.pdf')

"Relationship between top 10 documents in max topological dimension and overall topic prevalence"
#create topics of the most cited paper, and then topics for top-doc documents
cavg = {}
tavg = {}
temptop = topoldegavg[0]
for key in topoldegavg:
    if key == 0:
        c0 = temptop[0:1].T.squeeze()
        ls0 = [x == 0 if x == 0  else 0 for x in c0]
        ls0 = pd.Series(ls0, index = topname)
        cavg[key] = ls0
    if key > 0:
        citetop = topoldegavg[key].fillna(0)
        citetop = topoldegavg[key].mean()
        cavg[key] = citetop.fillna(0)
        tmp1 = temptop[key-1:key].T
        tavg[key-1] = tmp1
tavg[9]= temptop[9:10].T
topicavg = tavg

#from dictionaries create list of lists
clist = ([[k]+list(v) for k,v in cavg.items()])
tlist = ([[k]+list(v.squeeze()) for k,v in tavg.items()])

n=len(tslice)
distTDavg=np.zeros([n, n]) 
pval=np.zeros([n, n]) 
distV=[]

for i in range(0,n):
    tc = clist[i]
    for j in range (0,n):
       tt = tlist[j]
       dist, p = sp.stats.spearmanr(tc,tt) # dist = nc.dfdist(tc,tt, 30) #
       distV.append(dist)
       pval[i,j] = round(p,3)
       distTDavg[i,j]=dist
distTDavg = np.nan_to_num(distTDavg)

#create topics of the most cited paper, and then topics for top-doc documents for countrs (dichotomized)
cavg = {}
tavg = {}
temptop = topoldegcnt[0]
for key in topoldegcnt:
    if key == 0:
        c0 = temptop[0:1].T.squeeze()
        ls0 = [x == 0 if x == 0  else 0 for x in c0]
        ls0 = pd.Series(ls0, index = topname)
        cavg[key] = ls0
    if key > 0:
        citetop = topoldegcnt[key].fillna(0)
        citetop = topoldegcnt[key].mean()
        cavg[key] = citetop.fillna(0)
        tmp1 = temptop[key-1:key].T
        tavg[key-1] = tmp1
tavg[9]= temptop[9:10].T

#from dictionaries create list of lists
clist = ([[k]+list(v) for k,v in cavg.items()])
tlist = ([[k]+list(v.squeeze()) for k,v in tavg.items()])

n=len(tslice)
distTDcnt=np.zeros([n, n]) 
pval=np.zeros([n, n]) 
distV=[]

for i in range(0,n):
    tc = clist[i]
    for j in range (0,n):
       tt = tlist[j]
       dist, p = sp.stats.spearmanr(tc,tt) # dist = nc.dfdist(tc,tt, 30) #
       distV.append(dist)
       pval[i,j] = round(p,3)
       distTDcnt[i,j]=dist
distTDcnt = np.nan_to_num(distTDcnt)

#now assess clustering of time-slices - not to use for now... better doing MDS
sns.clustermap(distTDavg, method='complete', metric='euclidean', cmap='coolwarm', row_cluster=True,
               col_cluster=True, linewidths=0.1,  yticklabels=tslice, xticklabels=tslice)
plt.savefig('TopolClusterSpearman_Avg.pdf')
plt.close()
sns.clustermap(distTDcnt, method='complete', metric='euclidean', cmap='coolwarm', row_cluster=True,
                    col_cluster=True, linewidths=0.1,  yticklabels=tslice, xticklabels=tslice)
plt.savefig('TopolClusterSpearman_count.pdf')
plt.close()


im = plt.imshow(distTDavg, vmin=-0.6, vmax=0.8, cmap='coolwarm')
              
#Combine count and avg topic 10 and all fgures
fco, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,12))
cbar_ax = fco.add_axes([0.99, .3, .03, .4])
heat = sns.heatmap(distTDavg, cmap = 'coolwarm', vmin=-0.6, vmax=0.8, ax = ax1, yticklabels=tslice, xticklabels=tslice, square = True, cbar=False)
ax1.set_xticklabels(tslice, fontsize=28, color='black', rotation = 90)
ax1.set_yticklabels(tslice, fontsize=28, color='black', rotation = 0)
heat = sns.heatmap(distTDcnt, ax = ax2, cmap = 'coolwarm',vmin=-0.6, vmax=0.8, yticklabels=tslice, xticklabels=tslice, square = True, cbar=False)
ax2.set_xticklabels(tslice,fontsize=28, color='black', rotation = 90)
ax2.set_yticklabels(tslice, fontsize=28, color='black', rotation = 0)
ax1.text(-0.05, 1.02, 'a)', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
ax2.text(-0.05, 1.02, 'b)', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
fco.text(0.5, -0.02, 'All Documents', fontsize=36, ha='center', va='center')
fco.text(-0.02, 0.5, 'Top 10 Degree in Highest Dimension', ha='center',  fontsize = 36,va='center', rotation='vertical')
fco.colorbar(im, cax=cbar_ax)
fco.tight_layout()
plt.savefig('Topol-Co-EvolutionSpearman.pdf', bbox_inches='tight')
plt.close()


#Complexity of the Socal-Knowledge domain
#assess co-evolution topic-citation network
#cumulative topic evolution and keep only top 5

slsum= dftop.groupby(['tslice']).sum()
slcnt = dftop.groupby(['tslice']).count()
slsum= slsum.cumsum()
slcnt = slcnt.cumsum()
slcnt = slcnt.drop(columns=['UT'])
cumpiv = slsum.div(slcnt)
cumpiv.columns = topabbr
toptop = cumpiv.sum().nlargest(5)
dowtop = cumpiv.sum().nsmallest(5)
tplist = list(toptop.index) + list(dowtop.index)
tp5 =  cumpiv.filter(items=tplist)

#Evolution of Network complexity of the networks as defined by Casti 1979. 
cpxCit = []
for key in citesimp:
    if key > 0:
        qv = citesimp[key]['Qvec']
        kmax = citesimp[key]['K-dim']
        kd =  list(range(kmax+1))
        kd.sort(reverse=True)
        denom = len(kd) * (len(kd) +1) / 2
        numer = np.sum([qi * ki  for qi,ki in zip(qv, kd)])
        cpxCit.append (numer/denom)

cpxBip = []
for key in bipsimp:
    qv = bipsimp[key]['Qvec']
    kmax = bipsimp[key]['K-dim']
    kd =  list(range(kmax+1))
    kd.sort(reverse=True)
    denom = len(kd) * (len(kd) +1)
    numer = np.sum([qi * ki  for qi,ki in zip(qv, kd)]) * 2
    cpxBip.append (numer/denom)

dfcx = pd.DataFrame()
dfbx = pd.DataFrame()
dfcx['Time'] = tnofirst
dfcx['Complexity Score'] = cpxCit
dfbx['Time'] = tslice
dfbx['Complexity Score'] = cpxBip
#add row to citation Complexity - as citation in 1990 is empty (no cited doc before 1992 to papers between 1990-1992)
dfcx.loc[-1] = ['1992', np.nan]  # adding a row
dfcx.index = dfcx.index + 1  # shifting index
dfcx.sort_index(inplace=True) 

#figures, on either entropy or coimplexity score and top 5 prevalent topics with the topic-doc 
#net and topic-doc complexity
#COMPLEXITY
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        
sns.set(context='paper', style='white', palette='colorblind', font_scale=3)

fevo, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
fevo.subplots_adjust(right=0.75)
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.1))
make_patch_spines_invisible(ax3)
ax3.spines["right"].set_visible(True)
tp5.plot(ax=ax1, linewidth=3, marker='o', markersize=12)
a2, = ax2.plot(dfbx.Time, dfbx['Complexity Score'], marker = 'o', linewidth=3, markersize=12, linestyle='dashed', color = 'black', label='Doc-Topic Complexity')
a3, = ax3.plot(dfcx.Time, dfcx['Complexity Score'], marker = 'o', linewidth=3, markersize=12, linestyle='dashed', color = 'red', label ='Citation Complexty')
ax1.set_ylabel('Average Topic Prevalence', fontsize=32, color = 'black')
ax2.set_ylabel('Topic Cpx Score', fontsize=32, color ='black')
ax3.set_ylabel('Citation Cpx Score', fontsize=32, color ='black')
ax2.yaxis.label.set_color('black')
ax3.yaxis.label.set_color('red')
ax3.tick_params(axis='y', colors='red')
ax1.set_xlabel('Time', fontsize=32, color = 'black')
ax1.set_xticklabels(tslice, color='black', rotation = 90)
#legend, as always a pain...
ccpx = Line2D([0], [0], color='red', linewidth=3, linestyle='--', label=r'$C_{cpx}$')
tcpx = Line2D([0], [0], color='black', linewidth=3, linestyle='--', label=r'$T_{cpx}$')
handles, labels = ax1.get_legend_handles_labels()
handles.append(tcpx) 
handles.append(ccpx) 
ax1.legend(handles=handles, loc='upper center', ncol=6, bbox_to_anchor=(0.5,1.2))

fevo.tight_layout()
fevo.savefig('Complexity.pdf', bbox_inches='tight')
plt.close()


# #Use PathPy for temporal paths
# #load dataframe representing all doc-doc and doc-topic relationship. This can be named differently 
# #and it comes from CC_Net_Building.py
# tf = pickle.load(open('tf.p','rb'))
# #make sure that topics are all in j 
# cond = tf.i < 30
# tf.loc[cond, ['j', 'i']] = tf.loc[cond, ['i', 'j']].values
# #use citation as topic-doc are bipartite
# cdf = tf[tf['j'] >= 30] #citation part of network doc matrix
# cdf = cdf.drop(columns=['weight'])
# templist = records(cdf)
# #generate temporal network
# ctemp = ppy.TemporalNetwork()
# for e1 in templist:
#     ctemp.add_edge(e1[0],e1[1],e1[2])
# sumstats = ctemp.summary()
# #same results but better output with
# print(ctemp)
# #find all causal paths in the citation network:
# causal_paths = ppy.path_extraction.temporal_paths.paths_from_temporal_network_dag(ctemp, delta=1)
# print(causal_paths)
# causal_paths5 = ppy.path_extraction.temporal_paths.paths_from_temporal_network_dag(ctemp, delta=5)
# print(causal_paths5)
# causal_paths9 = ppy.path_extraction.temporal_paths.paths_from_temporal_network_dag(ctemp, delta=9)
# print(causal_paths9)

# style = {    
#   'ts_per_frame': 1, 
#   'ms_per_frame': 2000,
#   'look_ahead': 2, 
#   'look_behind': 2, 
#   'node_size': 15, 
#   'inactive_edge_width': 2,
#   'active_edge_width': 4, 
#   'label_color' : '#ffffff',
#   'label_size' : '24px',
#   'label_offset': [0,5]
#   }
# ppy.visualisation.plot(ctemp, **style)
# ppy.visualisation.export_html(ctemp, 'tempCiteNetwork.html', **style)
