import bcrypt as bcrypt
from flask import Flask, render_template, request, redirect, url_for, session
import os

from back_end.technologies.mongodb.MongoDBCon import MongoDBCon
from back_end.mail.ForgotPassword import ForgotPassword
from back_end.technologies.solr.SolrCon import SolrCon
from back_end.technologies.elasticsearch.ElasticCon import ElasticCon

mongoDBCon = MongoDBCon()
forgotPassword = ForgotPassword()
solrCon = SolrCon()
elasticsearchCon = ElasticCon()

app = Flask(__name__)
app.secret_key = os.urandom(20)


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
    solr_time, solr_count_results, solr_results = solrCon.solr_search(fields, search_word, "10")
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
                          'department': department, 'role': 'user'}
            mongoDBCon.insert(user_input)

            user_data = mongoDBCon.find_user({"email": email})
            new_email = user_data['email']

            return render_template('index.html', email=new_email)
    return render_template('/user/login.html')


@app.route('/user/forgot_password', methods=["POST"])
def forgot_password():
    return forgotPassword.mail_send()


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


if __name__ == '__main__':
    app.run(debug=True)
