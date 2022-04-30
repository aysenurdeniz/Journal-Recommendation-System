import json
from urllib.request import urlopen


class SolrCon:
    def __init__(self, solr_url):
        self.solr_url = urlopen(solr_url)

    def print_response(self):
        response = json.load(self.solr_url)
        print(response['response']['numFound'], "documents found.")

    # def print_each(self):
    #     # Print the name of each document
    #     for document in self.response['response']['docs']:
    #         print("Name =", document['App'], document['Sentiment'])


my_url = 'http://localhost:8983/solr/reviews/select?q=App%3AFood'

object1 = SolrCon(my_url)
object1.print_response()
