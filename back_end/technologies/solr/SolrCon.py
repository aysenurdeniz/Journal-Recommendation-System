from urllib.request import urlopen
import simplejson
import pysolr
from back_end.get_timer.Timer import Timer

timerr = Timer()


class SolrCon:
    solr_url = 'http://localhost:8983/solr/wos2/select?'
    conn = pysolr.Solr(solr_url)

    def solrSearch(self, row_size=200, query=""):
        """
        A method to search in Apache Solr
        :param query:
        :param row_size: int
        :return: int, response, document
        """
        timerr.start_time()
        # defType=dismax iken qf ile istenilen fieldlerin hepsinde arama işlemi gerçekleştirilebilir.
        response = simplejson.load(urlopen("{}rows={}&q={}".format(self.solr_url, row_size, query)))
        finish_time = timerr.finish_time()

        return finish_time, response['response']['numFound'], response['response']['docs']



# solrc = SolrCon()
# finish_time, cont, doc = solrc.solrSearch(100, "Aims_and_Scope:Systems%20AND%20Indexing_and_Abstracting:Scopus")
#
# ids = ["0"]*1655
#
# for i in range(0, results['response']['numFound']):
#     ids[i] = results["response"]['docs'][i]["id"]
#
# def solr_doc_update(self, id, comment_doc):
#     doc = [{'id': id, 'comments': {'set': [comment_doc]}}]
#
#
# solr_url = 'http://localhost:8983/solr/wos'
# conn = pysolr.Solr(solr_url)

# def solr_doc_update(id, comment_doc):
#     doc = {'id': id, 'comments': {'set': [comment_doc]}}
#     conn.add(doc)
#     conn.commit()
#
#
# solr_doc_update(7, "Best Journal!")