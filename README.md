# aspect-analysis
Inferential analysis of consumer perception's of product aspects

The objective of this project is to leverage free text product reviews to better understand which aspects contribute most to person likelihood to purchase one product over the other. A summary of our work is as follows:
* Aspect identification: We utilize linguistic and grammar heuristics to traverse dependency trees to extract aspects independent of text content
* Clustering and identify a common name: We cluster semantically similar aspects such that we can assign a single representative name for further analysis
* Compute reviewer sentiment of individual aspects: leveraging an in-house developed sentiment analysis model, we extract the reviewer's opinion of each individual aspect. 
* Ranking product aspect: a regression model and expectation maximization variant are then applied to compute the importance of each aspect such that we can rank them.  The two models are compared



### Video Presentation and Report
https://www.youtube.com/watch?v=jHlf1E2q7kU

Report is found at Report/Report.docx

### Script Details
All code can be found in programs/ folder. Scripts are listed in the order that they should be run. 

extractData.py
 * extracts metadata and product reviews
 
AspectExtraction.py
 * applies linguistic rules to extract product aspects from free text reviews
 * utilizes SpaCy dependency parsers 

ClusterAspects.py
 * applies K-Means clustering on averaged GloVe embeddings
 * identifies optimal K through simple grid search
 * assigns representative name to each semantic cluster via TF-IDF weights

TrainSentiment.py
 * Trains and evaluates a sentiment analysis model
 * CNN + BiLSTM network

CreateOpinion.py
 * computes sentiment towards individual aspects
 * Applies trained sentiment model above
 
GetWeightsLinReg.py
 * applies linear regression to compute importance weights by aspects
 
ProbabilisticRanking.py
 * applies EM variant to comput importance weights by aspects
 
Visualization code found in visualizations/ folder
Exploratory and draft code found in explore/ folder


### [[Complete README to be added once project is complete]]
