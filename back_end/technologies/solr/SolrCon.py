import json
from urllib.request import urlopen
import simplejson
from back_end.get_timer.Timer import Timer
timerr = Timer()


class SolrCon:

    solr_url = 'http://localhost:8983/solr/wos/select?'

    def print_response(self):
        response = json.load(self.solr_url)
        num_results = response['response']['numFound']
        results = response['response']['docs']
        return num_results, results

    def solr_search(self, fields, search_word, row_size):
        """
        A method to search in Apache Solr
        :param fields: string
        :param search_word: string
        :param row_size: int
        :return: int, response, document
        """
        timerr.start_time()
        # defType=dismax iken qf ile istenilen fieldlerin hepsinde arama işlemi gerçekleştirilebilir.
        response = simplejson.load(urlopen("{}rows={}&q={}:{}".format(self.solr_url, row_size, fields, search_word)))
        finish_time = timerr.finish_time()
        return finish_time, response['response']['numFound'], response['response']['docs']
