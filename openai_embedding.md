Embedding Model: text-embedding-ada-002

#### Install colorama for colored outputs
```python
pip install colorama
```

#### Get Embeddings and Save as CSV File
- read dataset(csv)
- openai API KEY
- get embeddings
- save as embeddings.csv

```python
import openai
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df = pd.read_csv('text_data.csv')
print(df)
openai.api_key = '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'
# Embedding işlemi
def get_embeddings(texts):
    response = openai.embeddings.create(model="text-embedding-ada-002", input=texts)
    # response.data öğesi, doğrudan erişim için 'embedding' özelliği kullanılır.
    embeddings = [embedding.embedding for embedding in response.data]
    return embeddings

# DataFrame'e embedding'leri ekleyin
embeddings = get_embeddings(df["title"].tolist())  # title sütunundan embeddings alınıyor
df["embedding"] = embeddings

# Embedding'leri CSV'ye kaydedin
df.to_csv("embeddings.csv", index=False)
print('-------------------------------------------------------------')
print(df)
```
OUTPUT<br><br>
dataset(text_data.csv):<br><br>
<img width="400" alt="Screenshot 2024-12-24 at 10 42 44" src="https://github.com/user-attachments/assets/27fbac75-b840-4ee6-a522-87e8d30cbdb6" /><br><br><br>
embeddings.csv:<br><br>
<img width="400" alt="Screenshot 2024-12-24 at 10 43 07" src="https://github.com/user-attachments/assets/f801f8b3-94ea-4820-b749-4ec1402b1bde" /><br>
<img width="400" alt="Screenshot 2024-12-24 at 10 43 17" src="https://github.com/user-attachments/assets/fb5c92dc-5f13-459c-9c37-4495750f8186" />

#### Chat Completion
kullanıcı sorusunun threshold değeri küçükse (embedding değerine göre veri setindeki değerlere yakın değilse) > <b>chat completions</b>
kullanıcı sorusunun threshold değeri büyükse > <b>embeddings.csv'deki cevap</b>
```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, init
import openai

openai.api_key = "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"

# Embedding'leri içeren CSV'yi yükle
df = pd.read_csv("embeddings.csv")
df["embedding"] = df["embedding"].apply(eval)  # Liste formatına dönüştür.

def get_gpt4_mini_response(user_question):
    # OpenAI API çağrısı
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {'role': 'system', 'content': "Sen OPAK ERP programının canlı destek asistanısın. Fakat ERP programlarıyla ilgili teknik bir bilgi sorulursa 'Sorunuzu anlayamadım..Yardımcı olabileceğim farklı bir konu varsa lütfen yazmaktan çekinmeyin.' cevabını vereceksin."},
        {'role': 'user', 'content': user_question},
        ],
        temperature=0,
)
    # Yanıtın içeriğine erişim
    return response.choices[0].message.content

while True:
    user_question = input(f"{Fore.CYAN}KULLANICI(çıkmak için 'exit' yazın): ")

    if user_question.lower() == 'exit':
        print(f"{Fore.GREEN}Çıkılıyor...")
        print(f"{Fore.BLACK}Çıkıldı!")
        break

    # Kullanıcı sorusunun embedding'ini al
    user_embedding = get_embeddings([user_question])[0]

    # Benzerlik hesapla
    similarities = cosine_similarity([user_embedding], df["embedding"].tolist())
    best_match_idx = similarities.argmax()
    best_similarity = similarities[0][best_match_idx]

    # Eşik değeri
    threshold = 0.84

    if best_similarity < threshold:
        # GPT-4 Mini ile cevap oluştur
        best_answer = get_gpt4_mini_response(user_question)
    else:
        best_answer = df.iloc[best_match_idx]["description"]

    # Best Cevap
    print(f"{Fore.RED}OPAK ERP CANLI DESTEK SİSTEMİ: {Fore.RESET}{best_answer}")
```
<img width="800" alt="chat" src="https://github.com/user-attachments/assets/e5d880b7-c98b-4d34-9d42-61e7fa2337ad" />
