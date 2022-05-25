from urllib.request import urlopen

import simplejson
from elasticsearch import Elasticsearch
from flask import Flask, redirect, url_for, render_template, request, flash, jsonify
from technologies.solr import SolrCon
from get_timer.Timer import Timer

app = Flask(__name__)

solr_url = 'http://localhost:8983/solr/papers/select?q='
elastic_url = Elasticsearch('http://localhost:9200/papers/')
timerr = Timer()


@app.route("/")
def index():
    my_title = "Full Text Search"
    return render_template('index.html', my_title=my_title)


@app.route('/', methods=["GET", "POST"])
def SolrSearch():
    query = None
    numresults = None
    results = None
    timeFin = None
    if request.method == "POST":
        query = request.form["searchWord"]

        timerr.startTime()
        connection = urlopen("{}keywords:{}".format(solr_url, query))
        response = simplejson.load(connection)
        timeFin = timerr.finishTime()
        numresults = response['response']['numFound']
        results = response['response']['docs']

    return render_template('index.html', query=query, numresults=numresults, timeFin=timeFin,
                           results=results)


@app.route('/', methods=["GET", "POST"])
def ElasticSearch():
    es_query = None
    es_numresults = None
    es_results = None
    es_timeFin = None

    if request.method == "POST":
        query = request.form["searchWord"]
        print(query)

        # timerr.startTime()
        # res = elastic_url.search(query={"match_all": {"query": query}})
        # print(res)
        # es_numresults = res['hits']['total']
        # es_results = jsonify(res['hits']['hits'])
        # es_timeFin = timerr.finishTime()


    return render_template('index.html', es_query=es_query, es_numresults=es_numresults, es_timeFin=es_timeFin,
                       es_results=es_results)

#
# def LuceneSearch():
#     pass
#
#
# def MongoDBSearch():
#     pass


if __name__ == '__main__':
    app.run(debug=True)
