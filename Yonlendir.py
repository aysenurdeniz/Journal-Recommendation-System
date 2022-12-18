from urllib.request import urlopen
import simplejson
from elasticsearch import Elasticsearch
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_login import login_required, logout_user
import os
import psycopg2
from back_end.get_timer.Timer import Timer

app = Flask(__name__)

solr_url = 'http://localhost:8983/solr/wos/select?'
elastic_url = Elasticsearch('http://localhost:9200/papers/')
timerr = Timer()


@app.route("/", methods=["GET", "POST"])
def index():
    my_title = "Full Text Search"
    # solr_array, es_array = [0.0] * 6, [0.0] * 6
    # solr_sec, solr_time, solr_count_results, solr_results, results = [None, None, None, None, None]
    # es_time, es_count_results, es_results = [None, None, None]
    fields = "*"
    search_word = "*"

    if request.method == "POST":
        fields = "Aims_and_Scope"
        search_word = request.form["searchWord"]
        # rowsize = request.form.get('row_select')
        # field_keywords = request.form.get("keywords", False)
        # field_abstract = request.form.get("abstract", False)
        # field_domain = request.form.get("domain", False)

        # if field_keywords == "True":
        # fields = "keywords"
        # if field_abstract == "True":
        # fields = "Abstract"
        # if field_domain == "True":
        # fields = "Domain"

        # es_time, es_count_results, es_results = elastic_search(fields, search_word, "10")
    solr_time, solr_count_results, solr_results = solr_search(fields, search_word, "10")
    solr_sec = float((solr_time / 1000) % 60)
    # for i in range(6):
    #     solr_time, solr_count_results, solr_results = SolrSearch(fields, search_word, rowsize)
    #     es_time, es_count_results, es_results = ElasticSearch(fields, search_word, rowsize)
    #     solr_array[i] = solr_time
    #     es_array[i] = es_time
    #
    # print("Solr Average:" + array_average(solr_array) + "\n" +
    #       "ES Average:" + array_average(es_array))

    return render_template('index.html', my_title=my_title, numresults=solr_count_results, results=solr_results,
                           timeFin=solr_sec)


@app.route('/general/about_us')
def about():
    return render_template('/general/about_us.html')


@app.route('/general/contact')
def contact():
    return render_template('/general/contact.html')


@app.route('/user/login', methods=['GET', 'POST'])
def login():
    error = None
    next = request.args.get('next')
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # if authenticate(app.config['AUTH_SERVER'], username, password):
        #     user = User.query.filter_by(username=username).first()
        #     if user:
        #         if login_user(DbUser(user)):
        #             # do stuff
        #             flash("You have logged in")
        #             return redirect(next or url_for('index', error=error))
        error = "Login failed"
    return render_template('login.html', login=True, next=next, error=error)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have logged out')
    return redirect(url_for('login'))


def get_db_connection():
    con_postgres = psycopg2.connect(host='localhost',
                                    database='recommendation',
                                    user=os.environ['user'],
                                    password=os.environ['password'])
    return con_postgres


def array_average(arr):
    """
    Average of values in a list
    :param arr: list, array
    :return: string
    """
    # print(arr)
    arr_avg = 0.0
    for i in range(1, len(arr)):
        arr_avg += arr[i]
    return str(arr_avg / len(arr) - 1)


def solr_search(fields, search_word, row_size):
    """
    A method to search in Apache Solr
    :param fields: string
    :param search_word: string
    :param row_size: int
    :return: int, response, document
    """
    timerr.start_time()
    # defType=dismax iken qf ile istenilen fieldlerin hepsinde arama işlemi gerçekleştirilebilir.
    response = simplejson.load(urlopen("{}rows={}&q={}:{}".format(solr_url, row_size, fields, search_word)))
    finish_time = timerr.finish_time()
    return finish_time, response['response']['numFound'], response['response']['docs']


def elastic_search(fields, search_word, row_size):
    """
     A method to search in Elasticsearch
    :param fields: string
    :param search_word: string
    :param row_size: int
    :return: int, response, document
    """
    timerr.start_time()
    response = elastic_url.search(size=row_size, track_total_hits=True, query={"match": {fields: search_word}})
    finish_time = timerr.finish_time()
    return finish_time, response['hits']['total']['value'], response['hits']['hits']


def pagination(page=1, total=100, per_page=5):
    offset = total - ((page - 1) * per_page) + 1


if __name__ == '__main__':
    app.run(debug=True)
