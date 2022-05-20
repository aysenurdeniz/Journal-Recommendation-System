from urllib import request
from urllib.request import urlopen
import simplejson as simplejson
from flask import Flask, render_template
from technologies.solr import SolrCon
from timer import Timer

app = Flask(__name__)

BASE_PATH ='http://localhost:8983/solr/reviews/select?wt=json&q='


@app.route('/', methods=["GET", "POST"])
def index():
    query = None
    numresults = None
    results = None

    # get the search term if entered, and attempt
    # to gather results to be displayed
    if request.method == "POST":
        query = request.form["searchWord"]

        # return all results if no data was provided
        if query is None or query == "":
            query = "*:*"

        # query for information and return results
        connection = urlopen("{}{}".format(BASE_PATH, query))
        response = simplejson.load(connection)
        numresults = response['response']['numFound']
        results = response['response']['docs']

    return render_template('index.html', query=query, numresults=numresults, results=results)


def SolrSearch():
    pass


def ElasticSearch():
    pass


def LuceneSearch():
    pass


def MongoDBSearch():
    pass


if __name__ == '__main__':
    app.run(debug=True)
