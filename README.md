# Topic-Document-Citations
Repository for the analysis of social-knowledge networks based on bibliometrics

This repository contains the python scripts used for the analysis of topic-document and citation networks.
The scripts contained here should be used in order.

1. CC_DataPrep.py takes the downloaded files from Web of Science (full records, abstracts and citations included) and generates fiiles using metaknowledge.
1. CC_Topic_Bigram_Analysis.py uses generated files in CC_Data_Prep.py,  generates wordclouds and model the topic dynamically (via Dynamic Topic Modelling).
1. CC_TopicTrends_NetBuilding.py generates figure based on  Topic Modelling in CC_Topic_Bigram_Analysis.py and builds a full edgelist of the temporal network relating topics and docuemnts (bipartite, weighted) and the citation network (undirected, unweighted)
1. CC_NetMetrics_Simpl_Cmplx.py analyses the bipartite network and the citation network. It performs a network analysis with common metrics, as well as an analysis of higher order structures (Q-analysis and Betti numbers)
1. CC_NetAnalysis.py uses results in CC_NetMetrics_Simpl_Cmplx and derives figures.

#Pickled Data

Pickled data only contains only the files used in NetAnalysis, NetMetrics and TopicTrends, due to size constrains. 
If you are interested in the other data, please contact me.
