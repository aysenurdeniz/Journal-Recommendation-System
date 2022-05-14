from urllib import request
from urllib.request import urlopen
import simplejson as simplejson
from flask import Flask, render_template
from technologies.solr import SolrCon
from timer import Timer

app = Flask(__name__)

BASE_PATH = SolrCon.my_url
print(BASE_PATH)


@app.route('/', methods=["GET", "POST"])
def index():
    query = None
    numresults = None
    results = None

    if request.method == "POST":
        query = request.form["searchWord"]

    if query is None or query == "":
        query = "*:*"

    # query for information and return results
    connection = urlopen("{}{}".format(BASE_PATH, "den"))
    print(connection)
    response = simplejson.load(connection)
    print(response)
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
