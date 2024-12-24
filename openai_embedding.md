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
<img width="400" alt="Screenshot 2024-12-24 at 10 42 44" src="https://github.com/user-attachments/assets/27fbac75-b840-4ee6-a522-87e8d30cbdb6" /><br>
embeddings.csv:<br><br>
<img width="400" alt="Screenshot 2024-12-24 at 10 43 07" src="https://github.com/user-attachments/assets/f801f8b3-94ea-4820-b749-4ec1402b1bde" /><br>
<img width="400" alt="Screenshot 2024-12-24 at 10 43 17" src="https://github.com/user-attachments/assets/fb5c92dc-5f13-459c-9c37-4495750f8186" />
