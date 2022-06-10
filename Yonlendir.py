from urllib.request import urlopen

import simplejson
from elasticsearch import Elasticsearch
from flask import Flask, redirect, url_for, render_template, request
from get_timer.Timer import Timer

app = Flask(__name__)

solr_url = 'http://localhost:8983/solr/papers/select?'
elastic_url = Elasticsearch('http://localhost:9200/papers/')
timerr = Timer()


@app.route("/", methods=["GET", "POST"])
def index():
    my_title = "Full Text Search"
    solr_array, es_array = [0.0] * 6, [0.0] * 6
    solr_time, solr_count_results, solr_results = [None, None, None]
    es_time, es_count_results, es_results = [None, None, None]
    fields = "keywords"
    rowlist = [{'name': '10'}, {'name': '25'}, {'name': '50'}, {'name': '100'}]
    if request.method == "POST":
        search_word = request.form["searchWord"]
        rowsize = request.form.get('row_select')
        field_keywords = request.form.get("keywords", False)
        field_abstract = request.form.get("abstract", False)
        field_domain = request.form.get("domain", False)

        if field_keywords == "True":
            fields = "keywords"
        if field_abstract == "True":
            fields = "Abstract"
        if field_domain == "True":
            fields = "Domain"

        # solr_time, solr_count_results, solr_results = SolrSearch(fields, search_word)
        # es_time, es_count_results, es_results = ElasticSearch(fields, search_word)

        for i in range(6):
            solr_time, solr_count_results, solr_results = SolrSearch(fields, search_word, rowsize)
            es_time, es_count_results, es_results = ElasticSearch(fields, search_word, rowsize)
            solr_array[i] = solr_time
            es_array[i] = es_time

        print(solr_array)
        print(es_array)
        print("Solr Average:" + array_average(solr_array) + "\n" +
              "ES Average:" + array_average(es_array))

    return render_template('index.html', my_title=my_title, numresults=solr_count_results, results=solr_results,
                           timeFin=solr_time, es_count_results=es_count_results, es_results=es_results,
                           es_finTime=es_time, rowlist=rowlist)


def array_average(arr):
    arr_avg = 0.0
    for i in range(1, len(arr)):
        arr_avg += arr[i]
    return str(arr_avg / len(arr) - 1)


def SolrSearch(fields, search_word, rowsize):
    timerr.startTime()
    connection = urlopen("{}rows={}&q={}:{}".format(solr_url, rowsize, fields, search_word))
    response = simplejson.load(connection)
    finish_time = timerr.finishTime()
    return finish_time, response['response']['numFound'], response['response']['docs']


def ElasticSearch(fields, search_word, rowsize):
    timerr.startTime()
    res = elastic_url.search(size=rowsize, track_total_hits=True, query={"match": {fields: search_word}})
    finish_time = timerr.finishTime()
    return finish_time, res['hits']['total']['value'], res['hits']['hits']


# def LuceneSearch():
#     pass
#
#
# def MongoDBSearch():
#     pass


if __name__ == '__main__':
    app.run(debug=True)
