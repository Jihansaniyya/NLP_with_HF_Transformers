<div align="center">

# Natural Language Processing with Hugging Face Transformers

<img src="https://media.tenor.com/Jm7uD9rCkS8AAAAC/anime-anime-girl.gif" width="220" alt="anime gif" />

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=1000&center=true&vCenter=true&width=800&lines=Natural+Language+Processing;Hugging+Face+Transformers;Generative+AI+Guided+Project;Cognitive+Class+by+IBM" alt="Typing SVG" />

Generative AI Guided Project on Cognitive Class by IBM

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

</div>

## Name : Jihan Saniyya Pudaliba

---

## My todo :

## 1. Exercise 1 - Sentiment Analysis

### TODO :

```python
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")

tweet_text = "This new phone is amazing! The camera quality is superb, but the battery life is a bit disappointing. #tech #review"

specific_model(tweet_text)
```

### Result :

```python
[{'label': 'LABEL_2', 'score': 0.9162421226501465}]
```

### Analysis :

Model memberikan hasil `LABEL_2`, yang berarti sentimen positif. Tweet tersebut memiliki kalimat positif seperti "This new phone is amazing" dan "The camera quality is superb". Walaupun ada bagian negatif tentang battery life yang sedikit mengecewakan, model tetap membaca keseluruhan tweet sebagai positif.

---

## 2. Exercise 2 - Topic Classification

### TODO :

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier(
    "Buku ini membahas tentang sejarah dan budaya Indonesia dari zaman kuno hingga modern.",
    candidate_labels=["sejarah", "budaya", "teknologi", "ekonomi", "sains"]
)
```

### Result :

```python
{'sequence': 'Buku ini membahas tentang sejarah dan budaya Indonesia dari zaman kuno hingga modern.',
 'labels': ['budaya', 'sejarah', 'teknologi', 'sains', 'ekonomi'],
 'scores': [0.6817455291748047,
  0.2404392808675766,
  0.0456598624587059,
  0.025178344920277596,
  0.006976999808102846]}
```

### Analysis :

Kalimat yang digunakan membahas buku tentang sejarah dan budaya Indonesia. Karena itu, label yang paling sesuai kemungkinan besar adalah `sejarah` dan `budaya`. Model zero-shot classification dapat menentukan topik berdasarkan kandidat label yang diberikan tanpa perlu training ulang.

---

## 3. Exercise 3 - Text Generation Models

### TODO :

```python
generator = pipeline('text-generation', model='gpt2')

generator(
    "Indonesia adalah negara kepulauan yang indah dengan beragam budaya dan kekayaan alam yang melimpah.",
    max_length=50,
    num_return_sequences=2
)
```

### Result :

```python
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
[{'generated_text': 'Indonesia adalah negara kepulauan yang indah dengan beragam budaya dan kekayaan alam yang melimpah.\n\nP.O. Box 6712, Room 3, Honolulu'},
 {'generated_text': "Indonesia adalah negara kepulauan yang indah dengan beragam budaya dan kekayaan alam yang melimpah. It's the best!\n\nI'll tell you, we are"}]
```

### Analysis :

Model GPT-2 mencoba melanjutkan kalimat berdasarkan prompt tentang Indonesia sebagai negara kepulauan yang indah. Hasil teks yang dibuat dapat berbeda setiap kali kode dijalankan karena proses text generation bersifat probabilistik.

---

## 4. Exercise 4 - Named Entity Recognition

### TODO :

```python
nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)

example = "Her name is Anjela and she lives in Seoul."

ner_results = nlp(example)
print(ner_results)
```

### Result :

```python
[{'entity_group': 'PER', 'score': np.float32(0.94814444), 'word': 'Anjela', 'start': 11, 'end': 18}, {'entity_group': 'LOC', 'score': np.float32(0.99861133), 'word': 'Seoul', 'start': 35, 'end': 41}]
```

### Analysis :

Kalimat tersebut berisi nama orang dan lokasi. Model NER digunakan untuk mengenali entitas seperti `Anjela` sebagai person dan `Seoul` sebagai location. Dengan task ini, informasi penting dalam teks dapat diambil secara otomatis.

---

## 5. Exercise 5 - Question Answering

### TODO :

```python
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

question_answerer(
    question="Where is the Eiffel Tower located?",
    context="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
)
```

### Result :

```python
{'score': 0.3864145576953888,
 'start': 56,
 'end': 86,
 'answer': 'Champ de Mars in Paris, France'}
```

### Analysis :

Pertanyaan yang diberikan adalah lokasi Eiffel Tower. Pada context, lokasi Eiffel Tower disebutkan berada di Champ de Mars in Paris, France. Model question answering mencari bagian context yang paling relevan untuk menjawab pertanyaan tersebut.

---

## 6. Exercise 6 - Text Summarization

### TODO :

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer(
    """
Perkembangan teknologi kecerdasan buatan telah mengubah cara kita berinteraksi dengan dunia sehari-hari. Dari asisten virtual hingga mobil otonom, AI telah merasuk ke berbagai aspek kehidupan. Namun, seiring dengan kemajuan ini, muncul pula tantangan etika dan privasi yang perlu ditangani. Penting bagi kita untuk mengembangkan regulasi yang tepat dan memastikan penggunaan AI yang bertanggung jawab, demi kemajuan yang berkelanjutan dan bermanfaat bagi seluruh umat manusia. Edukasi publik tentang AI juga krusial agar masyarakat dapat memahami potensi dan batasannya, serta berpartisipasi aktif dalam pembentukan masa depan berbasis AI.
"""
)
```

### Result :

```python
[{'summary_text': ' Perkembangan teknologi kecerdasan buatan telah mengubah cara kita berinteraksi dengan dunia sehari-hari . Edukasi publik tentang AI juga krusial agar masyarakat dapat memahami potensi .'}]
```

### Analysis :

Model berhasil membuat ringkasan dari paragraf tentang perkembangan AI. Ringkasan mengambil inti pembahasan bahwa kecerdasan buatan mengubah kehidupan sehari-hari dan edukasi publik tentang AI sangat penting. Namun, hasil ringkasan masih kurang sempurna karena model ini lebih cocok untuk teks bahasa Inggris, sedangkan input yang digunakan adalah bahasa Indonesia.

---

## 7. Exercise 7 - Translation

### TODO :

```python
translator = pipeline("translation_en_to_de", model="t5-small")

print(translator("Hello, how are you today?", max_length=40))
```

### Result :

```python
[{'translation_text': 'Hallo, wie sind Sie heute?'}]
```

### Analysis :

Model digunakan untuk menerjemahkan kalimat bahasa Inggris ke bahasa Jerman. Kalimat "Hello, how are you today?" diterjemahkan ke dalam bahasa Jerman. Task ini menunjukkan bahwa Hugging Face Transformers dapat digunakan untuk translation antarbahasa.

---

## Overall Analysis

Dari project ini, saya belajar menggunakan beberapa pipeline Hugging Face Transformers untuk task Natural Language Processing. Pipeline yang digunakan antara lain sentiment analysis, zero-shot classification, text generation, named entity recognition, question answering, text summarization, dan translation.

Saya juga belajar bahwa setiap task membutuhkan model yang sesuai. Contohnya, model `cardiffnlp/twitter-roberta-base-sentiment` cocok untuk analisis sentimen tweet karena dilatih menggunakan data Twitter. Untuk summarization, model `sshleifer/distilbart-cnn-12-6` dapat digunakan untuk meringkas teks, tetapi hasilnya lebih optimal jika input menggunakan bahasa Inggris.

Selain itu, saya sempat mengalami error pada pipeline summarization karena versi `transformers` yang digunakan tidak cocok. Setelah menggunakan versi yang sesuai, model berhasil berjalan dan menghasilkan output summary.

---

## Conclusion

Hugging Face Transformers memudahkan pengguna untuk menjalankan berbagai task NLP hanya dengan beberapa baris kode. Dengan model pre-trained, saya dapat melakukan analisis sentimen, klasifikasi topik, pembuatan teks, ekstraksi entitas, tanya jawab, ringkasan teks, dan terjemahan tanpa perlu melatih model dari awal.
