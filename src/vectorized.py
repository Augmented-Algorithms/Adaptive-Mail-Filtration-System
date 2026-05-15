from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('..')
from Data.Processed.processed_data import Processed
class Vector:
    def __init__(self):
        a=Processed()
        self.df=a.msg()
    def token(self):
        vector = TfidfVectorizer(stop_words='english')
        v = vector.fit_transform(self.df)
        return v