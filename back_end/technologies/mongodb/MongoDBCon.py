import pymongo


class MongoDBCon:
    my_client = pymongo.MongoClient("mongodb://localhost:27017/")
    my_db = my_client["local"]
    my_col = my_db["local"]

    def find_user(self, my_query):
        result = self.my_col.find_one(my_query)
        return result

    def find_all(self):
        result = self.my_col.find({})
        return result

    def updateOne(self, old_values, new_values):
        self.my_col.update_one(old_values, new_values)

    def updateMany(self, old_values, new_values):
        self.my_col.update_many(old_values, new_values)

    def insert(self, my_query):
        self.my_col.insert_one(my_query)

    def delete(self, my_query):
        self.my_col.delete_one(my_query)

    def text_searching(self, index_name, field, search_text):
        self.my_col.create_index([(field, index_name)], default_language='english')
        result = self.my_col.find({"${}".format(index_name): {"$search": search_text}})
        for doc in result:
            print(doc)
