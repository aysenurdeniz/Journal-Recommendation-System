from datetime import datetime

import bcrypt as bcrypt
from flask import Flask, render_template, request, redirect, url_for, session
import os

from flask_paginate import Pagination
from matplotlib import __getattr__

from back_end.technologies.mongodb.MongoDBCon import MongoDBCon
from back_end.technologies.solr.SolrCon import SolrCon
from back_end.technologies.elasticsearch.ElasticCon import ElasticCon

mongoDBCon = MongoDBCon()
solrCon = SolrCon()
elasticsearchCon = ElasticCon()

app = Flask(__name__)
app.secret_key = os.urandom(20)


@app.route("/", methods=["GET", "POST"])
def index():
    index_title = "Content & Feedback JRS"
    fields = "*"
    search_word = "*"
    journal_id = ""

    if request.method == "POST":
        fields = "Aims_and_Scope"
        search_word = request.form["searchWord"]
        journal_id = request.form.get("journal_id")

    print(journal_id)

    # es_time, es_count_results, es_results = elastic_search(fields, search_word, "10")
    solr_time, solr_count_results, solr_results = solrCon.solr_search(fields, search_word, "200")
    solr_sec = float((solr_time / 1000) % 60)

    pagination, items_pagination = paginate(solr_results, 10)

    comments = [('Ayşenur Deniz', 'Great Journal', 8),
                ('Esra Deniz', 'Great Journal', 8),
                ('Ahmet Er', 'Great Journal', 8),
                ('Ömer Türk', 'Great Journal', 8),
                ('Talha Türkmen', 'Great Journal', 8),
                ('Yusuf Koç', 'Great Journal', 8),
                ('Ayşenur Deniz', 'Great Journal', 8),
                ('Nedime Gul', 'Great Journal', 8),
                ('Mehmet Ulusoy', 'Great Journal', 8)]
    for comment in comments:
        print(comment[2])

    return render_template('index.html', index_title=index_title, numresults=solr_count_results, results=solr_results,
                           timeFin=solr_sec, pagination=pagination, items=items_pagination, comments=comments)


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
        user_found = mongoDBCon.find_user({"email": email})
        if user_found:
            email_val = user_found.get("email")
            password_check = user_found.get("password")
            if bcrypt.checkpw(password.encode('utf-8'), password_check):
                session["email"] = email_val
                session["full_name"] = user_found["full_name"]
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('/user/login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('/user/login.html', message=message)
    return render_template('/user/login.html', message=message)


@app.route('/user/profile', methods=["POST", "GET"])
def logged_in():
    if "email" in session:
        return redirect(url_for("profile"))
    else:
        return redirect(url_for("login"))


@app.route('/profile', methods=["POST", "GET"])
def profile():
    if "email" in session:
        email = session["email"]
        user = mongoDBCon.find_user({"email": email})
        all_user = mongoDBCon.find_all()
        return render_template("/user/profile.html", user=user, all_user=all_user)
    else:
        return redirect(url_for("login"))


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return redirect(url_for("login"))
    else:
        return redirect(url_for("index"))


@app.route("/register", methods=["POST", "GET"])
def register():
    message = ''
    if "email" in session:
        return redirect(url_for("logged_in"))

    if request.method == "POST":
        user_name = request.form.get("user_name")
        email = request.form.get("email")
        full_name = request.form.get("full_name")
        department = request.form.get("department")

        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        user_found = mongoDBCon.find_user({"user_name": user_name})
        email_found = mongoDBCon.find_user({"email": email})
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
            user_input = {'user_name': user_name, 'full_name': full_name, 'email': email, 'password': hashed,
                          'department': department, 'role': 'user', 'created_date': datetime.now(),
                          'updated_date': datetime.now()}
            mongoDBCon.insert(user_input)

            user_data = mongoDBCon.find_user({"email": email})
            new_email = user_data['email']

            return render_template('index.html', email=new_email)
    return render_template('/user/login.html')


@app.route('/user/forgot_password', methods=["POST"])
def forgot_password():
    pass


@app.route('/comment', methods=["POST"])
def comment():
    journal_id, comment_text, rating_range = "", "", ""
    if "email" in session:
        email = session["email"]
        user = mongoDBCon.find_user({"email": email})

        if request.method == "POST":
            journal_id = request.form.get("journal_id")
            comment_text = request.form.get("comment_text")
            rating_range = request.form.get("rating_range")

            mongoDBCon.my_col.update_one({"_id": user["_id"]},
                                         {"$set": {"comments.{}".format(journal_id): {"com": comment_text,
                                                                                      "rating": rating_range}}})
    return redirect(url_for('index'))


@app.route('/get_comments')
def get_comments():
    if request.method == "GET":
        journal_id = request.form.get("id")
        cursor = mongoDBCon.my_col.find({"comments.{}".format(journal_id): {"$exists": "true"}})
        comments = [(cur["full_name"], cur["comments"][journal_id]["com"],
                     cur["comments"][journal_id]["rating"]) for cur in cursor]
        return render_template('index.html', comments=comments)


def paginate(results, per_page):
    page = int(request.args.get('page', 1))
    offset = (page - 1) * per_page
    items_pagination = results[offset:offset + per_page]
    total = len(results)
    pagination = Pagination(page=page, per_page=per_page, offset=offset, total=total)
    return pagination, items_pagination


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


# ------------------- ERROR PAGES ----------------------------

# @app.errorhandler(404)
# def page_not_found(e):
#     return render_template('404.html'), 404  # Not Found
#
# @app.errorhandler(500)
# def internal_server_error(e):
#     return render_template('500.html'), 500  # Internal Server Error


if __name__ == '__main__':
    app.run(debug=True)
