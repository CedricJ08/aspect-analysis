# aspect-analysis
Inferential analysis of consumer perception's of product aspects

The objective of this project is to leverage free text product reviews to better understand which aspects contribute most to person likelihood to purchase one product over the other. A summary of our work is as follows:
* Aspect identification: We utilize linguistic and grammar heuristics to traverse dependency trees to extract aspects independent of text content
* Clustering and identify a common name: We cluster semantically similar aspects such that we can assign a single representative name for further analysis
* Compute reviewer sentiment of individual aspects: leveraging an in-house developed sentiment analysis model, we extract the reviewer's opinion of each individual aspect. 
* Ranking product aspect: a regression model and expectation maximization variant are then applied to compute the importance of each aspect such that we can rank them.  The two models are compared

All code can be found in programs/ folder

### Video Presentation and Report
https://www.youtube.com/watch?v=jHlf1E2q7kU

Report is found at Report/Report.docx

### [[Complete README to be added once project is complete]]
