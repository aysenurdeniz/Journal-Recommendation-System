import json
from urllib.request import urlopen


class SolrCon:
    def __init__(self, solr_url):
        self.solr_url = urlopen(solr_url)

    def print_response(self):
        response = json.load(self.solr_url)
        numresults = response['response']['numFound']
        results = response['response']['docs']
        return numresults, results

    # def print_each(self):
    #     # Print the name of each document
    #     for document in self.response['response']['docs']:
    #         print("Name =", document['App'], document['Sentiment'])
