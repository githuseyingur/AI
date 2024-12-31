#### READ FILE FROM DRIVE
```python
from google.colab import drive
drive.mount('/content/gdrive')

## read file:
df=pd.read_csv('gdrive/My Drive/text_data.csv')
```
#### UNRAR/UNZIP
```python
get_ipython().system_raw("unrar x ArcFace.rar")
!unrar x /content/ArcFace.rar
```
#### INSTALL SPECIAL VERSION
```python
!pip install openai==0.28
```
