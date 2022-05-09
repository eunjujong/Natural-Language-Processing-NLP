

- Company URL Evaluation and Classification:

We extract keywords from 3 reputative websites for each of the Healthcare, Education, Environment industries to classify the companies. We also include 3 general news websites to generate keywords for the "Other Sector". 

- Strategy: 

The classification of the company urls is based on the TF-IDF (term frequency-inverse document frequency) scores measured from three reputative websites from each sector. We use the TfidfTransformer function from the from sklearn.feature_extraction.text package to calculate the TF-IDF scores. 

1) Healthcare:
- https://www.nlm.nih.gov/
- https://nathealthcare.com/
- https://www.nachc.org/
2) Education: 
- https://ies.ed.gov/ncee/projects/nle/
- https://bensguide.gpo.gov/parent-ed-u-s-government-web-sites-for-educators
- https://www.si.edu/
3) Environment: 
- https://www.nal.usda.gov/
- https://www.epa.gov/
- https://www.state.gov/policy-issues/climate-and-environment/
4) General News:
- https://www.nytimes.com/ 
- https://www.washingtonpost.com/ 
- https://www.cnn.com/

- Data: 

The data is in an excel file containing two columns: a column of company names and a column of their urls. 

- Requirements:

This code has been developped python 3.8.5 and requires the packages given in main.py.

- src:

main.py contains the python code. To run the functions, run python3 main.py from the src working directory.


