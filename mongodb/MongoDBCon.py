"""
--author: <aysenurdeniz>
--date: <11.03.2022>
"""

import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["test"]
mycol = mydb["kisiler"]


class MongoDBCon:
    def __init__(self, my_client, my_db, my_col, my_query):
        self.my_client = my_client
        self.my_db = my_db
        self.my_col = my_col
        self.my_query = my_query

    def find_all(self):
        for x in self.my_col.find(self.my_query):
            print(x)

    def update(self, new_values):
        self.my_col.update(self.my_query, new_values)

    def insert(self):
        self.my_col.insert_many(self.my_query)

    def delete(self):
        self.my_col.delete_one(self.my_query)


mydict = [{"name": "John", "surname": "Highway", "age": "20"},
          {"name": "Ayse", "surname": "Nur", "age": "20"},
          {"name": "Aynur", "surname": "Gul", "age": "41"},
          {"name": "Sultan", "surname": "Ay", "age": "32"},
          {"name": "Mehmet", "surname": "Dunya", "age": "20"},
          {"name": "Merve", "surname": "Kul", "age": "20"},
          {"name": "Murat", "surname": "Koc", "age": "28"},
          {"name": "Serhat", "surname": "Demir", "age": "20"},
          {"name": "Esma", "surname": "Bilen", "age": "21"},
          {"name": "Kader", "surname": "Sur", "age": "21"},
          {"name": "Alp", "surname": "Deniz", "age": "18"}
          ]

myquery = {}
myquery2 = {"surname": "Dunya"}
myquery3 = {"name": {"$regex": "^A"}}

prequery = {"name": "Ayse"}
new_values = {"$set": {"name": "Aysenur", "age": "18"}}

mongoObject = MongoDBCon(myclient, mydb, mycol, myquery)
# mongoObject.delete()
# mongoObject.find_all()
# mongoObject.update(new_values)
mongoObject.find_all()

