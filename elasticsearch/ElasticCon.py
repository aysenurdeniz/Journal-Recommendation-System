from elasticsearch import Elasticsearch


class ElasticCon:
    def __init__(self, es_url, my_query):
        self.es_url = Elasticsearch(es_url)
        self.my_query = my_query

    def get_response(self):
        resp = self.es_url.search(query=self.my_query)
        print(resp)


my_query = {"match_all": {}}
my_url = 'http://localhost:9200/reviews/'

object1 = ElasticCon(my_url, my_query)
object1.get_response()
