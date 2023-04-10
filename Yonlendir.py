import uuid
from urllib.request import urlopen

import bcrypt as bcrypt
import pymongo
from flask_mail import Mail, Message

import simplejson
from elasticsearch import Elasticsearch
from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_login import login_required, logout_user, login_user
import os
import psycopg2
from pymongo.auth import authenticate

from back_end.get_timer.Timer import Timer
from back_end.technologies.mongodb.MongoDBCon import MongoDBCon
from back_end.user.User import DbUser

app = Flask(__name__)
app.secret_key = os.urandom(20)

solr_url = 'http://localhost:8983/solr/wos/select?'
elastic_url = Elasticsearch('http://localhost:9200/papers/')
timerr = Timer()

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.get_database('local')
records = db["local"]


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


# -------------------------- Login Operations ---------------------------------
# Ref: https://medium.com/codex/simple-registration-login-system-with-flask-mongodb-and-bootstrap-8872b16ef915


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if "email" in session:
        return redirect(url_for("logged_in"))

    if request.method == "POST":
        email = request.form.get("login_email")
        password = request.form.get("login_password")
        print("email: {}, password: {}".format(email, password))  # check
        user_found = records.find_one({"email": email})
        print(user_found)  # check
        if user_found:
            email_val = user_found.get("email")
            passwordcheck = user_found.get("password")

            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('/user/login.html', message=message)
        else:
            message = 'email not found'
            return render_template('/user/login.html', message=message)
    return render_template('/user/login.html', message=message)


@app.route('/user/profile', methods=["POST", "GET"])
def logged_in():
    if "email" in session:
        email = session["email"]
        user_found = records.find_one({"email": email})
        return render_template('base.html', email=email, user_name=user_found["user_name"])
    else:
        return redirect(url_for("login"))


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("/user/login.html")
    else:
        return render_template('index.html')


@app.route("/register", methods=["POST", "GET"])
def register():
    message = ''
    if "email" in session:
        return redirect(url_for("logged_in"))

    if request.method == "POST":
        user_name = request.form.get("user_name")
        email = request.form.get("email")

        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        user_found = records.find_one({"user_name": user_name})
        email_found = records.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('/user/login.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('/user/login.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('/user/login.html', message=message)
        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            user_input = {'user_name': user_name, 'email': email, 'password': hashed}
            records.insert_one(user_input)

            user_data = records.find_one({"email": email})
            new_email = user_data['email']

            return render_template('index.html', email=new_email)
    return render_template('/user/login.html')


@app.route('/user/forgot_password', methods=["POST", "GET"])
def forgot_password():
    # recipient = request.form['recipient']
    try:
        new_pass = uuid.uuid4()
        msg = Message(subject="New Password - JRS",
                      body="Hello!\n This new password:{}".format(new_pass),
                      sender="anurdenizz@gmail.com",
                      recipients=["anurdenizz@gmail.com"]
                      )
        Mail.send(msg)
        return 'Mail successfully send!'

    except Exception as e:
        return str(e)

    return render_template('/user/forgot_password.html')


# -----------------------------------------------------------


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


if __name__ == '__main__':
    app.run(debug=True)
