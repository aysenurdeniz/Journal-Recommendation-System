import pymongo


class MongoDBCon:
    def __init__(self):
        self.my_client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.my_db = self.my_client["local"]
        self.my_col = self.my_db["local"]

    def find_user(self, my_query):
        result = self.my_col.find(my_query)
        for i in result:
            print(i)

    def update(self, new_values):
        self.my_col.update(new_values)

    def insert(self, my_query):
        self.my_col.insert_many(my_query)

    def delete(self, my_query):
        self.my_col.delete_one(my_query)

    def text_searching(self, search_text):
        result = self.my_col.find({"$text": {"$search": search_text}})
        for doc in result:
            print(doc)


myquery = {"user_name": "evelyn.gay"}

mongoObject = MongoDBCon(myquery)
# mongoObject.index_for_search()
# mongoObject.index_for_search("araceli.wilder")
mongoObject.find_user()

