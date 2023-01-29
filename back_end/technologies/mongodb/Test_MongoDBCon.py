import pymongo

# ----------------------------   TEST  ---------------------------

# ---- Connection -----
my_client = pymongo.MongoClient("mongodb://localhost:27017/")
my_db = my_client["local"]
my_col = my_db["local"]

# ---- Queries for Test -----
my_query = {'mail': 'anurdenizz@gmail.com',
            'name': 'aysenur',
            'surname': 'deniz',
            'password': 'deneme'}

my_query1 = {'user_name': 'consuelo.eaton'}

my_query2 = {'name': 'aysenur'}

my_query3 = ({'mail': 'anurdenizz@gmail.com',
            'name': 'aysenur',
            'surname': 'deniz',
            'password': 'deneme'},
             {'mail': 'anurdenizz1@gmail.com',
              'name': 'aysenur1',
              'surname': 'deniz1',
              'password': 'deneme1'})

# ---- Insertion Method -----
my_col.insert_one(my_query)

my_col.insert_many(my_query3)
my_col.find({"name":{"$regex": "^ayse"}}).limit(2).__getitem__(1)

# ---- Deletion Method -----
my_col.delete_one(my_query)

my_col.find(my_query2).limit(1).next()  # not found
my_col.find(my_query1).limit(1).next()  # it found

# ---- Update Method -----
myquery = {'user_name': 'consuelo.eaton'}
newvalues = {"$set": {'user_name': 'consuelo'}}

my_col.update_one(myquery, newvalues)
my_col.find({'user_name': 'consuelo'}).next()
# {'_id': ObjectId('63d0410cb8f8c6fced821c32'), 'id': 2, 'user_name': 'consuelo', 'password': '0869347314'}


# $regex - şartları sağlayan bütün verileri güncelleyecek
# ^, bir dizenin belirli bir karakterle başladığını belirtmek için kullanılır
# $, bir dizenin belirli bir karakterle bitmesini sağlamak için kullanılır.

myquery1 = {"user_name": {"$regex": "^c"}}
newvalues1 = {"$set": {'password': '1111'}}

my_col.update_many(myquery1, newvalues1)
my_col.find({'user_name': 'consuelo'}).next()
# {'_id': ObjectId('63d0410cb8f8c6fced821c32'), 'id': 2, 'user_name': 'consuelo', 'password': '1111'}

# ---- Find Method -----
my_col.find(my_query1).next()

# ---- Find All -----
n = 5

for i in range(n):
    print(my_col.find({}).limit(n).__getitem__(i))

liste = my_col.find({}).limit(n)

for i in range(n):
    print(liste.__getitem__(i))

