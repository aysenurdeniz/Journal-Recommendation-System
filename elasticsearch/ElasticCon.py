from elasticsearch import Elasticsearch

es = Elasticsearch('http://localhost:9200/')

es.indices.refresh(index="reviews")

resp = es.search(index="reviews", query={"match_all": {}})

# print(resp)

print("Got %d Hits:" % resp['hits']['total']['value'])
for hit in resp['hits']['hits']:
    print("%(App)s %(Sentiment_Subjectivity)s %(Translated_Review)s "
          "%(Sentiment_Polarity)s %(Sentiment)s"
          % hit["_source"])
