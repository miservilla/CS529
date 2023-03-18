import pandas as pd

df = pd.read_csv('/home/michaelservilla/CS529/Project_2/framingham.csv', header=None)
df = df.fillna(df.median())
df.to_csv('/home/michaelservilla/CS529/Project_2/framingham.csv', sep=',')


