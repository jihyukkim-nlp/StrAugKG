

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.corpus import stopwords  
stopword_set = set(stopwords.words('english'))
import string
puctuations = set(list(string.punctuation))
stopword_set.update(puctuations)
stopword_set.add('<digit>')
stopword_set.add('<num>')
import pprint
pp = pprint.PrettyPrinter(indent=4)



#!@