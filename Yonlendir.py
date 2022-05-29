from urllib.request import urlopen

import simplejson
from elasticsearch import Elasticsearch
from flask import Flask, redirect, url_for, render_template, request
from get_timer.Timer import Timer

app = Flask(__name__)

solr_url = 'http://localhost:8983/solr/papers/select?q='
elastic_url = Elasticsearch('http://localhost:9200/papers/')
timerr = Timer()


@app.route("/", methods=["GET", "POST"])
def index():
    my_title = "Full Text Search"
    solr_time, solr_count_results, solr_results = [None, None, None]
    es_time, es_count_results, es_results = [None, None, None]
    fields = "keywords"
    if request.method == "POST":
        search_word = request.form["searchWord"]
        field_keywords = request.form.get("keywords", False)
        field_abstract = request.form.get("abstract", False)

        if field_keywords == "True":
            fields = "keywords"
        if field_abstract == "True":
            fields = "Abstract"

        solr_time, solr_count_results, solr_results = SolrSearch(fields, search_word)
        es_time, es_count_results, es_results = ElasticSearch(fields, search_word)

    return render_template('index.html', my_title=my_title, numresults=solr_count_results, results=solr_results,
                           timeFin=solr_time, es_count_results=es_count_results, es_results=es_results,
                           es_finTime=es_time)


def SolrSearch(fields, search_word):
    timerr.startTime()
    connection = urlopen("{}{}:{}".format(solr_url, fields, search_word))
    response = simplejson.load(connection)
    finish_time = timerr.finishTime()
    return finish_time, response['response']['numFound'], response['response']['docs']


def ElasticSearch(fields, search_word):
    body = {
        "query": {
            "match": {
                fields: search_word
            }
        }
    }
    timerr.startTime()
    res = elastic_url.search(body=body)
    finish_time = timerr.finishTime()
    return finish_time, res['hits']['total']['value'], res['hits']['hits']


#
# def LuceneSearch():
#     pass
#
#
# def MongoDBSearch():
#     pass


if __name__ == '__main__':
    app.run(debug=True)
