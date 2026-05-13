import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
p=re.compile(r"[^\w\s]")
df=pd.read_csv(r'D:\Adaptive-Mail-Filtration-System\Data\Raw\mail_data.csv')
df.dropna(inplace=True)
df=df.drop_duplicates()
df['Message']=df['Message'].apply(lambda x: p.sub('',x))
df['Message']=df['Message'].str.lower()
vector=TfidfVectorizer(stop_words='english')
v=vector.fit_transform(df['Message'])
print(v)
