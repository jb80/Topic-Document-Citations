#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:50:25 2020

This script is to use after CC_TopicTrends_NetBuilding. Net Building gives out a dataframe, 
the dataframe encodes all doc-doc and doc-topic associations with time slice and weight.

To recap briefly
Topic-Document relationhip represent the probability of a topic to be part of a specific document.
Topics were elicited by using dynamic topic modelling. (See CC_Topic_Bigram_Analysis). Topic change over 
time and change their prevalence in the documents per 3-year time period (see CC_TopicTrends_NetBuilding).

Document-Document relationship represent citations.

This script will derive th networks from the dataframe and then analyze, separately, the 
bipartite top-doc network and the doc-doc citatoin network. The analysis is done to reveal mesocscale structures
via simplciial complexes, and nodal and network structure via selected, standard, network metrics. 

@author: Jacopo Baggio

"""
#import utilities
import os
import pickle

#import networks related
import networkx as nx
import networkx.algorithms.bipartite as bipartite
#import igraph as ig ->only needed if igraph is preferred to networkx

#matrices operations and data manipulation
import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.stats import entropy

#simplicial complexes analysis
import pyflagser as pfl

#create quick function to delete isolates
def isoledges(K):
    lnc=[len(c) for c in nx.connected_components(K)]
    isedg = lnc.count(2)
    return isedg

#credit for code of participation coef and Z-score to BCT toolbox. Z-score was modified to slghtly modified sparse matrices.
def partcoeff(W, ci, degree='undirected'):
	'''
	Participation coefficient is a measure of diversity of intermodular
	connections of individual nodes.
	Parameters
	----------
	W : NxN np.ndarray
		binary/weighted directed/undirected connection
		must be as scipy.sparse.csr matrix
	ci : Nx1 np.ndarray
		community affiliation vector
	degree : str
		Flag to describe nature of graph 'undirected': For undirected graphs
										 'in': Uses the in-degree
										 'out': Uses the out-degree
	Returns
	-------
	P : Nx1 np.ndarray
		participation coefficient
	'''
	if degree == 'in':
		W = W.T

	_, ci = np.unique(ci, return_inverse=True)
	ci += 1

	n = W.shape[0]  # number of vertices
	Ko = np.array(W.sum(axis=1)).flatten().astype(float)  # (out) degree
	Gc = W.copy().astype('int16')
	Gc[Gc!=0] = 1 
	Gc = Gc * np.diag(ci)# neighbor community affiliation
	
	P = np.zeros((n))
	for i in range(1, int(np.max(ci)) + 1):
		P = P + (np.array((W.multiply(Gc == i).astype(int)).sum(axis=1)).flatten() / Ko)**2
	P = 1 - P
	# P=0 if for nodes with no (out) neighbors
	P[np.where(np.logical_not(Ko))] = 0

	return P

def mdzscore(W, ci, flag=0):
    '''
    The within-module degree z-score is a within-module version of degree
    centrality.
    Parameters
    ----------
    W : NxN np.narray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.array_like
        community affiliation vector
    flag : int
        Graph type. 0: undirected graph (default)
                    1: directed graph in degree
                    2: directed graph out degree
                    3: directed graph in and out degree
    Returns
    -------
    Z : Nx1 np.ndarray
        within-module degree Z-score
    '''
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    if flag == 2:
        W = W.copy()
        W = W.T
    elif flag == 3:
        W = W.copy()
        W = W + W.T

    n =  np.shape(W)[0]
    Z = np.zeros((n,1))  # number of vertices
    for i in range(1, int(np.max(ci) + 1)):
        Koi = np.sum(W[np.ix_(ci == i, ci == i)], axis=1)
        Z[np.where(ci==i)] = ((Koi - np.mean(Koi)) / np.std(Koi))[0:]

    Z[np.where(np.isnan(Z))] = 0
    return Z


#working directory
wdir = ('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/CC_Topic_Cites/Analysis')
#change to working directory
os.chdir(wdir)
#load node attributes:
nodeattr = pickle.load(open('nodeattr.p','rb'))       #attributes of nodes - topics, author, macro research area
idnode   = pickle.load(open('nodeid.p','rb'))         #from numeric node id to WoS and cited paper ID
   

#load dataframe representing all doc-doc and doc-topic relationship. This can be named differently 
#and it comes from CC_Net_Building.py
tf = pickle.load(open('tf.p','rb')) 
#make sure that topics are all in j 
cond = tf.i < 30
tf.loc[cond, ['j', 'i']] = tf.loc[cond, ['i', 'j']].values
#eliminate weights topic-docs < th, that is, a document to be considered connected to a topic, needs to 
#have at least th probability of represeenting that topic. 
#if below, we do not considered a document linked to a specfici topic. Heere th = 0.1
tf = tf[tf.weight > 0.1] 
#slice it by time to get the temporal network slices
sliced_tf = tf.groupby(tf.t)
#create lists to contain the networks, in nx and ig format for future analysis, and for docs and tops projections
nxcite =[]
nxcitedir = []
nxbip = []
bipmat = []
nxbipcite = []

for tslice, data in sliced_tf:
    print(tslice)
    data = data.drop(columns=['t'])
    if tslice  == 0:
        tempdf = data        
    else:
        tempdf = pd.concat([tempdf, data]).groupby(['i', 'j'], as_index=False)['weight'].sum()
    #split the overall data into the doc topic network and citation network
    tdf = tempdf[tempdf['j'] < 30] #top-doc part of network
    ddf = tempdf[tempdf['j'] >= 30] #citation part of network doc matrix
    tdf['type'] = 'intra'
    #generate bipartite network and the incidence matrix of doc-topc
    tdn = nx.DiGraph()
    tdn.add_nodes_from(tdf['i'], docs=1)
    tdn.add_nodes_from(tdf['j'], docs=0)
    topics = {n for n, d in tdn.nodes(data=True) if n < 30}
    docs = set(tdn) - topics
    tdn.add_weighted_edges_from([(row['i'], row['j'], row['weight']) for idx, row in tdf.iterrows()])
    nxbip.append(tdn)
    #build incidence matrix ... so all weights are = 1 hjer, that is, there is an association beetween
    #topic and document if the doc has 0.1 probability of containing a specific topic.
    bm1 = bipartite.biadjacency_matrix(tdn, docs, topics, format='csr')
    #make sure that columsn are nods and rows are topics, to use in q-analysis
    bm1 = bm1.T
    bipmat.append(bm1)
    #create undirected graph for simplicial complexes based on association
    #create directed graph for paper importance 
    gci = nx.Graph()
    gcidir = nx.DiGraph()
    if len(ddf) > 0:
        mn = min(ddf.i)
        mx = max(ddf.i)
        #define edges as inter or intra temporal layer
        ddf['type'] = np.where((ddf['j'] >= mn) & (ddf['j'] <= mx), 'intra', 'inter')
        gci = nx.from_pandas_edgelist(ddf, source = 'i', target = 'j', edge_attr = ['type','weight'], create_using=nx.Graph())
        gcidir = nx.from_pandas_edgelist(ddf, source = 'i', target = 'j', edge_attr = ['type','weight'], create_using=nx.DiGraph())
    nxcite.append(gci)
    nxcitedir.append(gcidir)
    print('finished citation networks')
    #now we merge citation and top-doc network, having a multilevel evolving network
    nx.set_node_attributes(gci, 1,'docs')
    nx.set_edge_attributes(tdn, 'type','intra')
    nxcb = nx.compose(gci, tdn)
    nxbipcite.append(nxcb)
    print('finished merged network')

#get adjacency matrices  and 
#and assess Betti numbers and euler charactersitics and bar-graph homology via pyflagser
# for directed -weighted networks. use pmean filtration. We do this for both, citation only and 
#full network.
sccite = []
for nt in nxcite:
    if len (nt.nodes()) > 0:
        #get adjacency matrix and calculat betti numbers, persiistence and euler
        adji = nx.adjacency_matrix(nt)
        s1 = pfl.flagser_unweighted(adji, directed=False)
        sccite.append (s1)
        print('Net with vertices = ' + str(len(nt.nodes)) +' done')
    else:
        sccite.append({})
        
sccitedir = []
for nt in nxcitedir:
    if len (nt.nodes()) > 0:
        #get adjacency matrix and calculat betti numbers, persiistence and euler
        adji = nx.adjacency_matrix(nt)
        s1 = pfl.flagser_unweighted(adji, directed=True)
        sccitedir.append (s1)
        print('Net with vertices = ' + str(len(nt.nodes)) +' done')
    else:
        sccitedir.append({})
betti = {}
betti['directed'] = sccitedir
betti['undirected'] = sccite
        

"""
First we analyze the incidence matrix between documents and topics alone.
This will allow us to assess the structure of the doc-topic network. Underluying assumtpion is that
topics are considereed simplicial complexes on their own (bipartite representation)
"""

bmqlow = {} #to store n component per dimension
bmnet = {} #to store topological characteristics per time-slice
tslice = 0 #timeslice
#choosing threshold for doc-topic relationshipi for Q-analysis.
#for example, 0.25 means that doc-topic will be related if there is a 1/4 probability 
#of them being related etc..That is that one doc in this configuration can be related to max 4 topics.
thq = 0.25
for net in nxbip:
    #get bi-adjacencymatrix
    bm = bipmat[tslice]
    print ('TimeSlice: ' + str(tslice))
    qlower = {}
    qavect ={}
    tg2 = nx.Graph()
    tg1 = nx.Graph()
    tg = nx.Graph()
    #filter according to threshold. hre w use 0.332 as a document will be related to a topic if at least 30% of probability of the topic and document being linked.
    bm1 = bm.multiply(bm >= thq)
    bm1.data[:] = 1
    Q1 = np.dot(bm1, bm1.T)
    sz = Q1.shape
    ones = np.ones(sz, dtype=int)
    Qconn = Q1-ones
    utcq = np.triu(Qconn, k=0)
    
    cqlsz = list(utcq.diagonal())
    cqlmx=int(max(cqlsz))
    dims=list(range(cqlmx+1)) # so is : 0,1,3,...k-max are all k-dims
    scdim=max(dims)   # K-dimension
    dims.sort(reverse=True)
    f0vect=[cqlsz.count(s) for s in dims]  # no. of k-dim simplices at k-dim
    f1vect=list(np.flip(np.flip(f0vect,0).cumsum(),0)) # complem.cumul.sum q-dims
    #start calculating and storing qlow and values into dictionaris
    for i in dims:
        if len(tg2.nodes) > 0:
            tg1 = tg2.copy()
        else:
            tg1 = nx.Graph()
        ii = np.nonzero(utcq == i)
        ii = np.where(utcq == i)
        c1 = tuple(zip(*ii))
        tg = nx.Graph()
        tg.add_edges_from(c1)
        tg2 = nx.compose(tg,tg1)
        qi = nx.number_connected_components(tg2)
        ql = {i:qi}
        qlower.update(ql)
        #print('Finished count simplicial of dimension = ' + str(i))
        M = sparse.csr_matrix(utcq)
        indices = utcq.diagonal()==i
        if np.sum(indices) !=0:
            M -= sparse.dia_matrix((M.diagonal()[sp.newaxis, :], [0]), shape=M.shape)
            qast = np.max(M[indices])
            qavect[i]  = qast
        else:
            qavect[i] = qast
    #now calculate the key features of the network (together with f0 and f1)
    Qvect = list(qlower.values())
    #Qvect.reverse()
    Qhat=[round(1-(a/b),3) for a,b in zip(Qvect,f1vect)]
    ecc = [round((qh-qa)/(qa+1),3) for qh,qa in zip(dims,list(qavect.values()))]
    print('Finished Calculating Strucutre Vectors and Eccentricity')
    #calc node topological dimension
    #first define cliques as topics
    cql = (list(nx.generate_adjlist(net)))
    #total topological degree
    nqdim = np.sum(bm1, axis=1)
    #topological degree / dimension
    qdic=dict()
    #find a way to do this without double loops...
    for nn in net.nodes():
        nvec=[0]*(scdim+1)
        for cq in cql:
            cq = [i for i in cq.split()]
            cq.pop(0)
            if len(cq) > 0:
                if str(nn) in cq:
                    dq=len(cq)-1
                    nvec[dq]=nvec[dq]+1
        qdic[nn]=nvec
    print('Finshed calculating topological degree')
    #calc topological entropy
    #generate array from dictionary
    qtop = sparse.csr_matrix([*qdic.values()])
    qtop = qtop.T
    #calculate probabilities
    p = qtop/qtop.sum(axis=1) #normalize(qtop, norm='l1', axis=1) #using sklearn preprocessing
    #count non-zero entries per row
    nq = np.diff(qtop.tocsr().indptr)
    #use entropy scipy
    e1 = entropy(p.T, base=2)
    #calculate topological entropy
    entr = [eni / nqi for eni , nqi in zip(e1,nq)]
    #store results
    bmnet[tslice] = {'K-dim':scdim, 'Qvec': Qvect, 'f0vec':f0vect,'f1vec':f1vect,'Qhat':Qhat, 'Entropy':entr,'Eccentricity':ecc, 'TopDeg':qdic, 'TotDeg':nqdim}
    bmqlow[tslice] = qlower
    tslice += 1


#generate main dictionaries to store results
fullqlow = {} #count of simplicial complexes of dimension dim
iddict = {}
shfac = [] #shared face matrix list
fullnet = {} #store topological characteristic per time-slice
#fill first time-slice of dictionaries with empty, as the network has 66 nodes and 0 edges, so all isolates
fullqlow[0]= {}
fullnet[0] = {}
#now calcualte Q vector; start at 1 as nxcite at time 0 has 0 nodes/edges
for net in range(1, len(nxcite)):
    print(net)
    #first calculate simplicial complexes for directed/undirected networks
    G = nxcite[net]
    #add id attribute for building incidence matrix
    gn = list(G.nodes())
    gid = list(range(len(gn)+1))
    gnid = dict(zip(gn, gid))
    iddict[net] = gnid
    nx.set_node_attributes(G, gnid, 'id')
    #order size and isolates elimination
    ordr = G.order()
    sizr = G.size()
    isln = nx.number_of_isolates(G)
    isedg = isoledges(G)
    #find cliques, if directed graphs only, then use neighborhood method
    G1 = nx.relabel_nodes(G, gnid)
    cql = list(nx.clique.find_cliques(G1))
    cqlsz=[len(itm)-1 for itm in cql]
    cqlmx=int(max(cqlsz))
    dims=list(range(cqlmx+1)) # so is : 0,1,3,...k-max are all k-dims
    scdim=max(dims)   # K-dimension
    f0vect=[cqlsz.count(s) for s in dims]  # no. of k-dim simplices at k-dim or f~ vector
    f1vect=list(np.flip(np.flip(f0vect,0).cumsum(),0)) # complem.cumul.sum q-dims - second structure vector
    dims.sort(reverse=True)
    print('Finished Clique Calculation')
    #calculate incidence matrix
    #first create dictionary akin to adjacency list
    dcql = dict(enumerate(cql))
    #from dictionary to sparse matrix = incidence matrix
    row_ind = [k for k, v in dcql.items() for _ in range(len(v))]
    col_ind = [i for ids in dcql.values() for i in ids]
    incmat = sparse.csr_matrix(([1]*len(row_ind), (row_ind, col_ind))) # sparse csr matrix
    print('Finished Incidence Matrix of size = ' + str(np.size(incmat)))
    #calculate shared-facet matrix (upper triangle of the incmat * incmat.T - 
    #np.ones same size of incmat)
    Q1 = np.dot(incmat,incmat.T)
    sz = Q1.shape
    Qconn = Q1.copy()
    ones = Qconn.data - 1
    Qconn.data[:] = ones
    utcq = sparse.triu(Qconn, k=0)  
    shfac.append(utcq)
    print('Finished Shared-Face Matrix Building of size = ' + str(sz))
    #total topological degree
    nqdim = np.sum(incmat, axis=1)
    #topological degree / dimension
    qdic=dict()
    #find a way to get something similar to cliques....
    for nn in G1.nodes():
        nvec=[0]*(scdim+1)
        for cq in cql:
            if nn in cq:
                dq=len(cq)-1
                nvec[dq]=nvec[dq]+1
        qdic[nn]=nvec
    print('Finished calculating topological degree per dimension')
    #start calculating and storing qlow and values into dictionaris
    qavect = {}
    qlower = {}
    tg2 = nx.Graph()
    tg1 = nx.Graph()
    tg = nx.Graph()
    for i in dims:
        if len(tg2.nodes) > 0:
            tg1 = tg2.copy()
        else:
            tg1 = nx.Graph()
        if i > 0:      
            c1 = utcq.multiply(utcq == i)
            tg = nx.Graph()
            tg = nx.from_scipy_sparse_matrix(c1)
            tg.remove_nodes_from(list(nx.isolates(tg)))
            tg2 = nx.compose(tg,tg1)
            qi = nx.number_connected_components(tg2)
            ql = {i:qi}
        #get the max simplicial to simplicial connectivity per dimension, for calculating eccentricity
        qlower.update(ql)
        M = sparse.csr_matrix(utcq)
        indices = utcq.diagonal()==i
        if np.sum(indices) !=0:
            M -= sparse.dia_matrix((M.diagonal()[sp.newaxis, :], [0]), shape=M.shape)
            qast = np.max(M[indices])
            qavect[i]  = qast
        else:
            qavect[i] = qast
    print('Finished calculating qlower')
    #now calculate the key features of the network (together with f0 and f1)
    Qvect = list(qlower.values()) #Q vector - first structure vector so we have all from lower to higher dims
    Qhat=[round(1-(a/b),3) for a,b in zip(Qvect,f1vect)] # QHat = third structure vector
    #calculate eccentricity
    ecc = [round((qh-qa)/(qa+1),3) for qh,qa in zip(dims,list(qavect.values()))]
    #calc entropy
    #generate array from dictionary
    qtop = sparse.csr_matrix([*qdic.values()])
    qtop = qtop.T
    #calculate probabilities
    p = qtop/qtop.sum(axis=1) #gives problems for division by zero.
    #count non-zero entries per row
    nq = np.diff(qtop.tocsr().indptr)
    #use entropy scipy
    e1 = entropy(p.T, base=2)
    #calculate topological entropy
    entr = [round(eni / nqi,3) for eni , nqi in zip(e1,nq)]
    entr.sort(reverse=True)
    #store results
    fullnet[net] = {'K-dim':scdim, 'Qvec': Qvect, 'f0vec':f0vect,'f1vec':f1vect,'Qhat':Qhat, 'Entropy':entr, 'Eccentricity':ecc, 'TopDeg':qdic, 'TotDeg':nqdim}
    fullqlow[net] = qlower
    print('Finished working on net = ' + str(net))

#notes: asssess top 10 cited papers per time-slice and the topics they look at, see if they create path dependencies
#path depdendence is: Processes that are non-ergodic, and thus unable to shake free of their history, are said to yield path dependent outcomes.    
#here we assess standard citation network metrics
citemetrics = {}
bipmetrics = {}
period = 0
for i in range(0, len(nxbip)):
    print(period)
    cnet = nxcite[i]
    cnetd = nxcitedir[i]
    bnet = nxbip[i]
    topics = set(n for n,d in bnet.nodes(data=True) if d['docs']==0)
    papers = set(bnet) - topics
    #citation network metrix
    if len(cnet.nodes) < 1:
        cnn     = 0
        cmm     = 0
        indistr =0
        din     = 0
        den     = 0 
        prd     = 0
        clus    = 0
        clusd   = 0
    else:
        cnn   = cnet.order()
        cmm   = cnet.size()
        din   = cnetd.in_degree()
        den   = nx.density(cnet)
        print("Calcualted Degree and Density for Citation Networks")
        clus  = nx.clustering(cnet)
        clusd = nx.clustering(cnetd)
        print('Calculated clustering on Citation Networks')
        prd   = nx.pagerank(cnetd, alpha=0.85)
        print("Calculated Page Rank for Citation Networks")
        citemetrics[period] = {'nodes':cnn, 'edges':cmm, 'indeg': din, 'density': den,  'clus':clus, 'dirclus': clusd, 'pagerank':prd}
    #bipartitte top-doc metrcs ( t = topics set, p = documents set)
    bnn = len(papers)
    den_t  = bipartite.density(bnet, topics)
    den_p  = bipartite.density(bnet, papers)
    deg_t  = bipartite.degrees(bnet,topics, weight = 'weight')
    deg_p  = bipartite.degrees(bnet,papers, weight = 'weight')
    topdistr = sorted(deg_t[1],reverse=True) 
    print('Calculatted Degree and Density for Bipartite network')
    clus_t = bipartite.clustering(bnet,topics)
    clus_p = bipartite.clustering(bnet,papers)
    print('Calculated Clustering for Bipartite Network')
    bipmetrics[period] = {'nodes':bnn, 'docdeg': deg_p, 'topdeg':deg_t, 'docden': den_p, 'topden':den_t, 'clusdoc':clus_p, 'clustop': clus_t}
    period +=1
    
#assess potential interdisciplinarity by looking at the participation coefficient of documents
#for macro research areas and then based on toipics:
mod = pd.concat([nodeattr['UT'], nodeattr['mra']], axis=1, keys=['ut', 'ra'])  
mod = mod.dropna()  
mod['node'] = ([idnode[x] for x in mod.ut])
nodeid = dict((v,k) for k,v in idnode.items())
dfpz = {}
for i in range(0, len(nxcitedir)):
    if i == 0:
        dfpart = pd.DataFrame([0, 0, 0]).T
        zra = pd.DataFrame ([0, 0, 0]).T
    else:
        print(i)
        g = nxcitedir[i]
        n1 = ([nodeid[x] for x in g.nodes()])
        mdl = mod[mod['ut'].isin(n1)]
        nlist = list(mdl.node)
        md1 = list(mdl.ra)
        cmat = nx.adjacency_matrix(g, nlist)
        dmat = cmat.copy()
        zin  = mdzscore(dmat, md1, 1)
        zout = mdzscore(dmat, md1, 2)
        prain = partcoeff(cmat, md1, 'in')
        praout = partcoeff(cmat, md1, 'out')
        dfpart = pd.DataFrame([nlist, list(prain), list(praout), list(zin), list(zout), md1]).T
        dfpart.columns = ['node','partin','partout', 'zin', 'zout','mod']
    dfpz[i] = dfpart
    
    
    
#saving files - node.netowrk and simplicial analysis
pickle.dump(bipmetrics, open('bipnet.p','wb'))
pickle.dump(citemetrics, open('citnet.p','wb'))
pickle.dump(bmnet, open('bipsimpc.p','wb'))
pickle.dump(fullnet, open('citsimpc.p','wb'))
pickle.dump(betti, open('betti.p','wb'))
pickle.dump(iddict, open('simpdict.p', 'wb'))
pickle.dump(dfpz, open('partzscore.p','wb'))
    
    

"""
add following code to tf loops if you want igraph networks
where to store igraph nets
  igcite = []
  igbip = []
  igbipcite = []
  
add this after print ('finished merged networks') to convert nx to ig graphs.
      nx.write_graphml(nxcb,'rncit.graphml') # Export nx graph to file
      igbc = ig.read('rncit.graphml',format="graphml") # Create new IG graph from file
      igbipcite.append(igbc)
      nx.write_graphml(gci,'rncit.graphml') # Export nx graph to file
      igc = ig.read('rncit.graphml',format="graphml") # Create new IG graph from file
      igcite.append(igc)
      nx.write_graphml(tdn,'rncit.graphml') # Export nx graph to file
      igb = ig.read('rncit.graphml',format="graphml") # Create new IG graph from file
      igbip.append(igb)
    print('finished igraph networks')

"""