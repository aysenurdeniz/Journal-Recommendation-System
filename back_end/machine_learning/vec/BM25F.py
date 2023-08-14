import math
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class BM25F:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b

        self.doc_lengths = [len(doc.split()) for doc in self.docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.docs)
        self.doc_term_counts = [Counter(doc.split()) for doc in self.docs]

    def idf(self, term, idx):
        doc_count_with_term = sum(1 for doc in self.docs if term in doc)
        return math.log((len(self.docs) - doc_count_with_term + 0.5) / (doc_count_with_term + 0.5) + 1)

    def score(self, query):
        scores = []
        query_terms = query.split()

        for idx, doc in enumerate(self.docs):
            doc_score = 0.0
            doc_term_count = self.doc_term_counts[idx]

            for term in query_terms:
                term_idf = self.idf(term, idx)
                term_tf = doc_term_count[term] if term in doc_term_count else 0
                doc_score += term_idf * ((term_tf * (self.k1 + 1)) / (term_tf + self.k1 * (1 - self.b + self.b * (len(doc.split()) / self.avg_doc_length))))

            scores.append(doc_score)

        return scores


# JSON dosyasını okuma
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Veri kümesinden belge içeriklerini ve journal isimlerini alma
documents = [entry["Aims and Scope"] for entry in data]
journal_names = [entry["Journal Name"] for entry in data]

# BM25F modelini oluşturma
bm25f = BM25F(documents)

# Kullanıcının sorgusu
# query = "information systems retrieval knowledge discovery and data mining"
query = "Novel manufacturing processes" \
        "Machine or low-resource Translation languages involving" \
        "Operating Performance optimization Secure System management"

# BM25F skorlarını hesaplama
scores = bm25f.score(query)

# En yüksek benzerlik skorlarına sahip belgelerin indekslerini alın
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

# En yüksek skorlu belgelerin journal isimlerini ve benzerlik skorlarını yazdırma
print("En Yüksek Benzerlik Skoruna Sahip 5 Belge:")
for idx in top_indices:
    print(f"Journal Name: {journal_names[idx]}")
    print(f"Belge {idx+1} - BM25F Skoru: {scores[idx]:.3f}")
    print("-" * 50)

