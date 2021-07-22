# Topic-Document-Citations
Analysis code for the followinig paper: Baggio, J. A. (2021). Knowledge generation via social-knowledge network co-evolution: 30 years (1990–2019) of adaptation, mitigation and transformation related to climate change. Climatic Change, 167(1–2), 13. https://doi.org/10.1007/s10584-021-03146-5for 

This repository contains the python scripts used for the analysis of topic-document and citation networks.
The scripts contained here should be used in order.

1. CC_DataPrep.py takes the downloaded files from Web of Science (full records, abstracts and citations included) and generates fiiles using metaknowledge.
1. CC_Topic_Bigram_Analysis.py uses generated files in CC_Data_Prep.py,  generates wordclouds and model the topic dynamically (via Dynamic Topic Modelling).
1. CC_TopicTrends_NetBuilding.py generates figure based on  Topic Modelling in CC_Topic_Bigram_Analysis.py and builds a full edgelist of the temporal network relating topics and docuemnts (bipartite, weighted) and the citation network (undirected, unweighted)
1. CC_NetMetrics_Simpl_Cmplx.py analyses the bipartite network and the citation network. It performs a network analysis with common metrics, as well as an analysis of higher order structures (Q-analysis and Betti numbers)
1. CC_NetAnalysis.py uses results in CC_NetMetrics_Simpl_Cmplx and derives figures.

# Pickle Data Folder

Pickled data is now empty, and a signpost. Files are too large to be uploaded. 
For raw data download full records from the WebOfScience using the search terms in the article
