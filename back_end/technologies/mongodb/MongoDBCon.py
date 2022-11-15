import pymongo


class MongoDBCon:
    def __init__(self, my_query):
        self.my_client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.my_db = self.my_client["test"]
        self.my_col = self.my_db["reviews"]
        self.my_query = my_query

    # def find_all(self):
    #     for x in self.my_col.find(self.my_query):
    #         print(x)
    #
    # def update(self, new_values):
    #     self.my_col.update(self.my_query, new_values)
    #
    # def insert(self):
    #     self.my_col.insert_many(self.my_query)
    #
    # def delete(self):
    #     self.my_col.delete_one(self.my_query)

    def index_for_search(self):
        self.my_col.create_index([('Translated_Review', 'text')])

    def text_searching(self, search_text):
        result = self.my_col.find({"$text": {"$search": search_text}})
        for doc in result:
            print(doc)


myquery = {}

mongoObject = MongoDBCon(myquery)
# mongoObject.index_for_search()
mongoObject.text_searching("Good")

