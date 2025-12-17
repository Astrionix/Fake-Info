import pandas as pd
df = pd.read_csv('fake_or_real_news.csv')
print("--- FAKE EXAMPLE START ---")
print(df[df['label'] == 'FAKE']['title'].iloc[0])
print(df[df['label'] == 'FAKE']['text'].iloc[0][:300])
print("--- FAKE EXAMPLE END ---")

print("\n--- REAL EXAMPLE START ---")
print(df[df['label'] == 'REAL']['title'].iloc[0])
print(df[df['label'] == 'REAL']['text'].iloc[0][:300])
print("--- REAL EXAMPLE END ---")
