import pandas as pd
import re
import string 
import emoji
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words_set = set(stopwords.words('english'))

def preprocessing_clean_text(text):
    if pd.isnull(text): return ""

    # 0. Xử lý xuống dòng
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

    # 1. Xử lý Emoji (Dịch sang tiếng Anh)
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 2. Lowercase
    text = str(text).lower()

    # 3. Xóa IP, URLS, Username
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)

    # 4. Chuẩn hóa ký tự lặp (hateeeee -> hatee)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 5. Xử lý dấu câu: Tách ! và ? ra (What? -> What ?)
    text = re.sub(r'([!?])', r' \1 ', text)
    
    # 6. Xóa số
    text = re.sub(r'[0-9]', ' ', text)

    # 7. Xóa ký tự lạ (Giữ lại chữ cái, ! và ?), giữ lại _ do đây là từ sau khi xử lý emoji tạo ra emoji -> A_B_C
    text = re.sub(r"[^a-zA-Z!?_']", ' ', text)

    # 8. Xử lý khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    word = text.split()
    word = [w for w in word if w not in stop_words_set]

    text = ' '.join(word)
    return text