# Journal-Recommendation-System

**Flask projeleri için notlar:**

---
- Tüm html sayfaları için ortak bir html sayfası (yani template) yapılmak istense root dizin içerisinde **templates** klasörünü kullanmak gerekir.
- Tüm sayfalarda değişkenlik gösteren bir alanın dinamik yapılması gövde (body) kısmı için şöyle örneklendirilebilir.

``` html
{% block body %} 
{% endblock %}
``` 
- **body** kelimesi değişkendir. Burada body olarak kullandık, farklı şekilde de adlandırılabilir.

---
- Sayfalar arası geçiş için **Yönlendir.py** dosyasında şu **@app.route**ların tanımlanması gerekir. Örneğin, about sayfası için şu şekilde yazılabilir:

``` python
@app.route('/general/about_us')
def about():
    return render_template('/general/about_us.html')
``` 

``` html
<a class="nav_a" href="{{ url_for('about') }}">About Us</a>
``` 

or

``` html
<a class="nav_a" href="/general/about_us">About Us</a>
``` 


---
- CSS dosyası import edilmek istendiğinde varsayılan olarak **static** *(projenin ana dizininde olmalı)* klasörü oluşturulur. Bu dizinde *stylesheets* içerisine CSS dosyaları eklenmelidir.

``` html
<link type="text/css" rel="stylesheet" href="{{ url_for('static', filename='stylesheets/style.css') }}">
```

or

``` html
<link type="text/css" rel="stylesheet" href="/static/stylesheets/style.css">
```

- static dizini ihtiyaca göre özel olarak da konumlandırılabilir.

---
- Navbar'da aktif sayfayının linkini renklendirmek için **a** taginin class özelliği şu şekillerde yazılabilir.

``` html
<li class="nav-item">
      <a class="nav_a {{ 'nav_a_active' if request.path == '/general/about_us' }}" href="{{ url_for('about') }}">About Us</a>
</li>
```

or

``` html
<li class="nav-item">
      <a class="nav_a {{ 'nav_a_active' if request.path == url_for('about') }}" href="{{ url_for('about') }}">About Us</a>
</li>
```

- *nav_a* özelliğinin yanında {{ ... }} arasında belirtilen şartı sağlaması durumunda *nav_a_active* stili de eklenmektedir.

---
- **float** tipindeki değerleri anlamlı ifade sınırı ile yazma formatı:

```
{{ '%0.2f'|format(timeFin|float) }}
```

---
- Dinamik model için contenti içeren id'lerin unique olması gerekir.
- Bu proje için **read-more{{ document.id }}** yapısı kullanılabilir. Döngü içerisinde her içeriğin document id değeri unique bir yapıya sahiptir. 
``` html
{% for document in results %}
<p class="read-more">
   <a data-bs-toggle="modal" data-bs-target="#read-more{{ document.id }}">Read More</a>
</p>
   
<!-- Modal -->
<div class="modal fade" id="read-more{{ document.id }}" data-bs-backdrop="static" data-bs-keyboard="false" tabindex="-1"
             aria-labelledby="read-more" aria-hidden="true">
   <div class="modal-dialog modal-xl modal-dialog-centered">
        <div class="modal-content" role="document">
        ...
        </div>
   </div>
</div>
{% endfor %}
```
---
