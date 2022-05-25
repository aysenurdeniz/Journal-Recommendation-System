from urllib.request import urlopen

import simplejson
from flask import Flask, redirect, url_for, render_template, request, flash
from technologies.solr import SolrCon
from get_timer.Timer import Timer

app = Flask(__name__)

my_url = 'http://localhost:8983/solr/papers/select?q='
timerr = Timer()


@app.route("/", methods=["GET", "POST"])
def index():
    my_title = "Full Text Search"
    query = None
    numresults = None
    results = None
    timeFin = None

    if request.method == "POST":
        query = request.form["searchWord"]
        radio1 = request.form.get("keywords", False)
        radio2 = request.form.get("domain", False)
        radio3 = request.form.get("abstract", False)

        print(radio3, radio2, radio1)

        if query is None or query == "":
            query = "*:*"

        if radio1 == "True":
            timerr.startTime()
            connection = urlopen("{}keywords:{}".format(my_url, query))
            response = simplejson.load(connection)
            timeFin = timerr.finishTime()
            numresults = response['response']['numFound']
            results = response['response']['docs']

        if radio2 == "True":
            timerr.startTime()
            connection = urlopen("{}Domain:{}".format(my_url, query))
            response = simplejson.load(connection)
            timeFin = timerr.finishTime()
            numresults = response['response']['numFound']
            results = response['response']['docs']

        if radio3 == "True":
            timerr.startTime()
            connection = urlopen("{}Abstract:{}".format(my_url, query))
            response = simplejson.load(connection)
            timeFin = timerr.finishTime()
            numresults = response['response']['numFound']
            results = response['response']['docs']

        # solr_con = SolrCon("{}{}".format(my_url, query))
        # numresults, results = solr_con.print_response()

    return render_template('index.html', my_title=my_title, query=query, numresults=numresults, timeFin = timeFin,
                           results=results)


#
# @app.route('/', methods=["GET", "POST"])
# def SolrSearch():
#     if request.method == "POST":
#         query = request.form["searchWord"]
#
#         if query is None or query == "":
#             query = "*:*"
#
#         solr_con = SolrCon("{}{}".format(my_url, query))
#         numresults, results = solr_con.print_response()
#
#     return render_template('index.html', query=query, numresults=numresults, results=results)
#
# #
# def ElasticSearch():
#     pass
#
#
# def LuceneSearch():
#     pass
#
#
# def MongoDBSearch():
#     pass


if __name__ == '__main__':
    app.run(debug=True)
