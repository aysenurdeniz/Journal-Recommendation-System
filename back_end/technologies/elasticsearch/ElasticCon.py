from elasticsearch import Elasticsearch
from back_end.get_timer.Timer import Timer
timerr = Timer()


class ElasticCon:

    elastic_url = Elasticsearch('http://localhost:9200/papers/')

    def elastic_search(self, fields, search_word, row_size):
        """
         A method to search in Elasticsearch
        :param fields: string
        :param search_word: string
        :param row_size: int
        :return: int, response, document
        """
        timerr.start_time()
        response = self.elastic_url.search(size=row_size, track_total_hits=True, query={"match": {fields: search_word}})
        finish_time = timerr.finish_time()
        return finish_time, response['hits']['total']['value'], response['hits']['hits']