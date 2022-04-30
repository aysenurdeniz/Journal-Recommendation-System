import json
from urllib.request import urlopen

from urllib3 import *

connection = urlopen('http://localhost:8983/solr/reviews/select?q=App%3AFood')
response = json.load(connection)

print(response['response']['numFound'], "documents found.")

# Print the name of each document.

for document in response['response']['docs']:
    print("Name =", document['App'], document['Sentiment'])
