import pandas as pd
try:
    df = pd.read_csv('fake_or_real_news.csv')
    print(df.head())
    print(df.columns)
except Exception as e:
    print(e)
