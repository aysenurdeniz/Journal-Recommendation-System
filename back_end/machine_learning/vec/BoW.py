import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Örnek kullanıcı girdisi
user_input = "information systems retrieval knowledge discovery and data mining"

# Verileri ayıklayın
texts = [entry["Aims and Scope"] for entry in data]

# CountVectorizer ile Bag-of-Words matrisini oluşturun (N-gram desteği ekleyin)
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))

# İşlem süresini başlat
start_time = time.time()

bow_matrix = vectorizer.fit_transform(texts + [user_input])

# Kullanıcı girdisinin vektör temsili
user_input_vector = bow_matrix[-1]

# Kozinüs benzerliği hesaplayın
cosine_similarities = cosine_similarity(user_input_vector, bow_matrix[:-1])

# İşlem süresini hesapla
end_time = time.time()
process_time = end_time - start_time

# En benzer 5 sonucu alın
top_indices = cosine_similarities.argsort()[0][-5:][::-1]
for i, index in enumerate(top_indices, start=1):
    journal_name = data[index]['Journal Name']
    similarity_score = cosine_similarities[0][index]
    print(f"{i}. Journal: {journal_name}\nBenzerlik Skoru: {similarity_score:.3f}")

# İşlem süresini yazdır
print(f"İşlem Süresi: {process_time:.4f} saniye")
