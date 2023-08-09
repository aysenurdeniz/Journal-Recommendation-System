import uuid
from crypt import methods
from datetime import datetime

import bcrypt
from flask import Flask, session, request
from flask_mail import Mail, Message

from Yonlendir import mongoDBCon

app = Flask(__name__)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'anurdenizz@gmail.com'
app.config['MAIL_PASSWORD'] = ''
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)


@app.route('/mail-gonder', methods="POST")
def sendMail():
    if request.method == 'POST':
        email = request.form.get('email')
        user = mongoDBCon.find_user({"email": email})
        if user:
            tmp_pas = uuid.uuid4()
            hashed = bcrypt.hashpw(tmp_pas.encode('utf-8'), bcrypt.gensalt())
            mongoDBCon.my_col.update_one({"email": user["email"]},
                                         {"$set": {"password": hashed, "updated_date": datetime.now()}})

    try:
        msg = Message("Merhaba yeni şifreniz!",
          sender="anurdenizz@gmail.com",
          recipients=["anurdenizz@gmail.com"])
        tmp = uuid.uuid4()
        msg.body = "Yeni şifre:\n".format(hashed.encode('utf-8'))
        mail.send(msg)
        return 'Mail başarıyla gönderildi!'
    except Exception as e:
        return(str(e))


if __name__ == '__main__':
   app.run(debug = True)

