import pandas as pd
import re
import os
class Processed:
    def __init__(self):
        current_dir=os.path.dirname(__file__)
        file_path=os.path.join(current_dir,"..","Raw","mail_data.csv")
        self.pattern=r"[^\w\s]"
        self.df=pd.read_csv(file_path)
    def clean(self):
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
    def label(self):
        self.clean()
        df1=self.df["Category"]
        y=list((map(lambda x:int(x=='spam'),df1)))
        return y
    def msg(self):
        self.clean()
        self.df['Message'] = self.df['Message'].apply(lambda x: re.sub(self.pattern,'', x))
        self.df['Message'] = self.df['Message'].str.lower()
        df2=self.df["Message"]
        return df2